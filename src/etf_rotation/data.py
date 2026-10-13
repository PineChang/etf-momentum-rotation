"""行情数据获取与本地缓存(tushare 中转站 ETF 日线为首选,东财 akshare 兜底,前复权)。

缓存为 CSV:data/{symbol}.csv。首次全量下载,之后增量更新。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_COL_MAP = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}


def _load_ts_token() -> str | None:
    """tushare 中转站 token:优先环境变量 TUSHARE_RELAY_TOKEN,否则读项目根 .env。"""
    token = os.environ.get("TUSHARE_RELAY_TOKEN")
    if token:
        return token
    env_f = Path(__file__).resolve().parents[2] / ".env"
    if env_f.exists():
        for line in env_f.read_text().splitlines():
            if line.startswith("TUSHARE_RELAY_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _ts_pro():
    import tushare as ts

    token = _load_ts_token()
    if not token:
        raise RuntimeError("未配置 TUSHARE_RELAY_TOKEN(环境变量或项目根 .env)")
    pro = ts.pro_api(token, timeout=30)
    pro._DataApi__http_url = "https://tl.kaixin8.top/tushare/sdk"
    return pro


def _ts_code(symbol: str) -> str:
    return symbol + (".SH" if symbol.startswith(("5", "6")) else ".SZ")


def _fetch_ts(symbol: str, start: str, end: str, adjust: str, retries: int = 4) -> pd.DataFrame:
    """tushare 中转站 ETF 日线(首选数据源,2026-10-12 起)。

    fund_daily 为交易所原始价(3 位小数,与富途/支付宝逐分一致,实证 512890
    10-09 收 1.208);qfq/hfq 用 fund_adj 复权因子自行计算并保留 3 位小数,
    与东财缓存口径一致(重叠区实证逐分不差,不触发锚点重拉)。复权因子自行
    计算意味着永不退回未复权序列——因子缺失直接抛错走兜底,不静默放行。
    中转站上游池偶发限流(upstream_pool_exhausted,文档建议等 30s),带重试。
    """
    import time

    pro = _ts_pro()
    ts_code = _ts_code(symbol)
    for attempt in range(retries):
        try:
            daily = pro.fund_daily(ts_code=ts_code, start_date=start, end_date=end,
                                   fields="trade_date,open,high,low,close,vol")
            if daily.empty:
                return pd.DataFrame(columns=list(_COL_MAP))
            k = 1.0
            if adjust in ("qfq", "hfq"):
                adj = pro.fund_adj(ts_code=ts_code, start_date=start, end_date=end,
                                   fields="trade_date,adj_factor")
                if adj.empty:
                    raise RuntimeError(f"{symbol} fund_adj 复权因子为空,无法计算 {adjust}")
                daily = daily.merge(adj, on="trade_date", how="left")
                if daily["adj_factor"].isna().any():
                    raise RuntimeError(f"{symbol} fund_adj 复权因子缺交易日,无法计算 {adjust}")
                f = daily["adj_factor"]
                # qfq 锚定窗口最新交易日因子(窗口 end=今日,即全局最新锚点)
                k = f / f[daily["trade_date"].idxmax()] if adjust == "qfq" else f
            return pd.DataFrame({
                "日期": daily["trade_date"],
                "开盘": (daily["open"] * k).round(3),
                "收盘": (daily["close"] * k).round(3),
                "最高": (daily["high"] * k).round(3),
                "最低": (daily["low"] * k).round(3),
                "成交量": daily["vol"],
            })
        except Exception as e:
            if attempt == retries - 1:
                raise
            # 上游池耗尽按文档等 30s;其余断连/超时短退避
            time.sleep(30 if "pool_exhausted" in str(e) else 2 * (attempt + 1))


def _fetch_em(symbol: str, start: str, end: str, adjust: str, retries: int = 4) -> pd.DataFrame:
    """带重试的东财接口调用(偶发断连/限流)。"""
    import time

    for attempt in range(retries):
        try:
            return ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                       start_date=start, end_date=end, adjust=adjust)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _fetch_daily(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """tushare 中转站优先(交易所原始精度);断连/限流时兜底东财。

    2026-10-12 起移除腾讯兜底:腾讯 fqkline 会把最新 bar 圆整到 2 位小数
    (512890 真实收 1.208 被写成 1.21),违反"与交易所精确相等"的硬要求。
    两源均失败时抛错,由 cron 下一时隙重试,不写入可疑数据。
    """
    try:
        df = _fetch_ts(symbol, start, end, adjust)
        if not df.empty:
            return df
        warnings.warn(f"{symbol} tushare 中转站返回空,回退东财。", stacklevel=2)
    except Exception as e:
        warnings.warn(f"{symbol} tushare 中转站取数失败({e}),回退东财。", stacklevel=2)
    return _fetch_em(symbol, start, end, adjust)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_COL_MAP)[list(_COL_MAP.values())]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _needs_refresh(latest: pd.Timestamp, now: pd.Timestamp) -> bool:
    """缓存是否需要增量刷新。

    - 已含今日 bar:不刷;
    - 缺昨日及更早的 bar(休眠唤醒/隔夜补跑):任何时刻都刷——那些 bar 已收盘定型;
    - 只缺今日 bar:仅收盘后(≥16:00)刷,盘中刷会把未定型的半根 bar(收盘=最新价)
      追加进缓存,污染当日之后的信号。

    2026-10 审计修复:旧条件"落后超 4 天才刷"使正常每日运行从不刷新,
    晚间 plan 的信号永远晚一个交易日(trade_log 实证 09-29 晚用了 09-28 数据)。
    """
    today = now.normalize()
    if latest >= today:
        return False
    if latest < today - pd.Timedelta(days=1):
        return True
    return now.hour >= 16


def _drop_rounded_latest(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """丢弃被数据源圆整到 2 位小数的最新 bar(2026-10-09 实证:腾讯兜底把
    512890 真实收 1.208 写成 1.21、518880 真实收 8.617 写成 8.62;ETF 最小
    价位 0.001,真实 4 价恰好全是 0.01 整数倍的概率约 1e-4)。

    判定:最新行开/收/高/低全部等于自身保留 2 位小数的值,且近 5 行历史
    存在 3 位小数精度(排除整段历史本就 2 位精度的标的)。命中则丢弃该行
    并告警——缓存停留前一交易日,plan 走 data_lag 告警,下次运行自动重试,
    且该 bar 次日进入重叠区后会被正常并入(检查只针对最后一行)。
    """
    if len(df) < 3:
        return df
    ohlc = ["open", "close", "high", "low"]
    last = df.iloc[-1][ohlc].astype(float)
    hist = df.iloc[-6:-1][ohlc].astype(float)
    last_rounded = bool(np.isclose(last, last.round(2)).all())
    hist_precise = bool((~np.isclose(hist, hist.round(2))).any(axis=None))
    if last_rounded and hist_precise:
        warnings.warn(
            f"{symbol} 最新 bar({df['date'].iloc[-1].date()})价格疑似被数据源"
            f"圆整到 2 位小数(开{last['open']}/收{last['close']}/高{last['high']}"
            f"/低{last['low']}),与真实成交价不符——已丢弃该行,缓存停留前一交易日,"
            f"待数据源给出精确值后自动补上。", stacklevel=2)
        return df.iloc[:-1].reset_index(drop=True)
    return df


def fetch_etf(symbol: str, start: str = "20190101", end: str | None = None,
              use_cache: bool = True, adjust: str = "qfq") -> pd.DataFrame:
    """获取单只 ETF 复权日线。返回 date/open/high/low/close/volume 列。

    adjust: "qfq" 前复权(默认,价格序列连续便于看图,但除息后历史价格会被改写,
            含轻微未来信息);"hfq" 后复权(锚定上市首日,历史价格不被改写,
            无未来信息,严谨的因子计算应以此验证)。
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / (f"{symbol}.csv" if adjust == "qfq" else f"{symbol}.{adjust}.csv")
    now = pd.Timestamp.now()
    today = now.normalize()
    end = end or now.strftime("%Y%m%d")

    def _drop_partial_today(df: pd.DataFrame) -> pd.DataFrame:
        """盘中(<16:00)拉到的今日 bar 未定型,不落缓存、不进返回值。"""
        return df if now.hour >= 16 else df[df["date"] < today]

    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        latest = df["date"].max()
        if _needs_refresh(latest, now):
            # 带 5 个交易日重叠增量拉取:重叠区用于校验复权锚点。
            # 前复权(qfq)在分红后整条历史价格会被数据源改写,若重叠区价格
            # 与缓存不一致,说明锚点已变——直接追加会在除息日产生假跳空,
            # 必须丢弃缓存全量重拉(后复权锚定上市日,不受影响)。
            ov_start = (latest - pd.Timedelta(days=9)).strftime("%Y%m%d")
            inc = _fetch_daily(symbol, ov_start, end, adjust)
            if not inc.empty:
                inc = _normalize(inc)
                overlap = inc[inc["date"] <= latest].set_index("date")["close"]
                cached = df[df["date"].isin(overlap.index)].set_index("date")["close"]
                reanchored = (not overlap.empty
                              and not np.allclose(overlap.sort_index(), cached.sort_index(), atol=1e-4))
                if reanchored:
                    df = _normalize(_fetch_daily(symbol, start, end, adjust))
                else:
                    df = pd.concat([df, inc]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
                df = _drop_partial_today(df)
                df = _drop_rounded_latest(df, symbol)
                df.to_csv(cache, index=False)
        return df

    fresh = _fetch_daily(symbol, start, end, adjust)
    df = _drop_rounded_latest(_drop_partial_today(_normalize(fresh)), symbol)
    df.to_csv(cache, index=False)
    return df


def fetch_universe(symbols: list[str], start: str = "20190101", adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    """批量获取股票池数据(逐个拉取,间隔 1s 避免限流)。"""
    import time

    out = {}
    for s in symbols:
        out[s] = fetch_etf(s, start=start, adjust=adjust)
        time.sleep(1)
    return out


def fetch_premium(symbol: str) -> pd.Series:
    """溢价率序列(index=date):未复权收盘 / 单位净值 - 1。

    依赖缓存 data/{symbol}.raw.csv(tushare 未复权日线)与 {symbol}.nav.csv(东财单位净值),
    由 scripts/eval_zhihu.py 抓取。任一缺失时返回空序列(调用方按无数据处理)。

    严格 PIT:信号日 t 收盘做决策时,当日净值(t 晚间)尚未公布,故每个交易日
    取的是**前一交易日已知的溢价**(整体 shift 1),不引入未来信息。
    """
    raw_f = DATA_DIR / f"{symbol}.raw.csv"
    nav_f = DATA_DIR / f"{symbol}.nav.csv"
    if not (raw_f.exists() and nav_f.exists()):
        return pd.Series(dtype=float)
    raw = pd.read_csv(raw_f, parse_dates=["date"])
    nav = pd.read_csv(nav_f, parse_dates=["date"])
    m = raw.merge(nav, on="date")
    if m.empty:
        return pd.Series(dtype=float)
    prem = pd.Series((m["close"] / m["nav"] - 1).to_numpy(), index=m["date"])
    return prem.sort_index().ffill().shift(1).dropna()


def attach_premium(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """给每只 ETF 的行情 df 注入 premium 列(该日最新已知溢价,缺失日 ffill)。

    premium_max 过滤(backtest._passes_risk_filters)只在行情 df 含 premium 列时生效;
    无缓存数据的标的自动放行,不会因数据缺失而误杀——但这也意味着规则会**静默失效**,
    调用方应配合 premium_status() 做缺失/过期警告(cli 已做)。
    """
    out = {}
    for sym, df in data.items():
        prem = fetch_premium(sym)
        if prem.empty:
            out[sym] = df
            continue
        d = df.merge(prem.rename("premium"), left_on="date", right_index=True, how="left")
        d["premium"] = d["premium"].ffill()
        out[sym] = d
    return out


def premium_status(symbols: list[str], max_lag_days: int = 1,
                   max_age_days: int = 7) -> dict[str, dict]:
    """溢价数据健康检查:{symbol: {premium, asof, nav_lag, missing, stale}}。

    missing = 无 raw/nav 缓存(规则对该标的静默失效);
    nav_lag = 单位净值落后行情数据多少个交易日 —— 溢价率的口径是"场内价 / 当日
              单位净值",净值当晚才公布。净值一旦落后,算出的溢价就是旧值。
              实测 513100 曾落后 3 个交易日(溢价读数 13.93% 实为 3 日前的数),
              而只按"距今 7 天内"判定会认为它很新鲜,规则就在过期数据上静默决策;
    stale   = nav_lag > max_lag_days,或最新溢价距今超 max_age_days 天(缓存久未刷新)。
    """
    today = pd.Timestamp.today().normalize()
    out = {}
    for sym in symbols:
        prem = fetch_premium(sym)
        raw_f = DATA_DIR / f"{sym}.raw.csv"
        nav_f = DATA_DIR / f"{sym}.nav.csv"
        raw_dates = nav_last = None
        if raw_f.exists():
            raw_dates = pd.read_csv(raw_f, usecols=["date"], parse_dates=["date"])["date"]
        if nav_f.exists():
            nav_last = pd.read_csv(nav_f, usecols=["date"], parse_dates=["date"])["date"].max()
        lag = int((raw_dates > nav_last).sum()) if raw_dates is not None and nav_last is not None else None
        if prem.empty:
            out[sym] = {"premium": None, "asof": None, "nav_lag": lag,
                        "missing": True, "stale": True}
            continue
        asof = prem.index[-1]
        stale = (lag is not None and lag > max_lag_days) or (today - asof).days > max_age_days
        out[sym] = {
            "premium": round(float(prem.iloc[-1]), 4),
            "asof": str(asof.date()),
            "nav_lag": lag,
            "missing": False,
            "stale": stale,
        }
    return out


def fetch_today_opens(symbols: list[str], retries: int = 15, wait_s: float = 60.0) -> dict[str, float]:
    """tushare rt_k 实时日线 bar 的开盘价(9:25 集合竞价后可用),供 cli execute 成交虚盘计划。

    盘初实时 bar 可能延迟生成,以 trade_time 是否为今日判定;带轮询重试,
    重试耗尽仍无(非交易日/停牌)则抛错,由调用方按"非交易日"跳过。
    开盘价全天不变,晚几分钟取数不影响成交语义。
    """
    import time

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    out: dict[str, float] = {}
    last_err = "无数据"
    for attempt in range(retries):
        for sym in [s for s in symbols if s not in out]:
            try:
                df = _ts_pro().rt_k(ts_code=_ts_code(sym),
                                    fields="ts_code,open,trade_time")
                row = df.iloc[0]
                open_px = float(row["open"])
                # open≤0/NaN:trade_time 已滚到今日但竞价未撮合(9:15–9:25 时隙),
                # 不能把 0 当成成交价写虚盘
                if not str(row["trade_time"]).startswith(today):
                    last_err = f"{sym} 今日 bar 尚未生成(最新 {row['trade_time']})"
                    continue
                if not open_px > 0:
                    last_err = f"{sym} 今日开盘价无效({row['open']}),竞价或数据源未就绪"
                    continue
                out[sym] = open_px
            except Exception as e:
                last_err = f"{sym} {e}"
            time.sleep(0.3)
        if len(out) == len(symbols):
            return out
        if attempt < retries - 1:
            time.sleep(wait_s)
    raise RuntimeError(f"{last_err}——非交易日或行情未开始,跳过(已重试 {retries} 次)")


def update_premium_caches(symbols: list[str], start: str = "2019-01-01") -> None:
    """刷新溢价数据源缓存:{sym}.raw.csv(tushare 未复权收盘,重叠增量)+ {sym}.nav.csv(东财单位净值,全量)。

    nav 全量重拉而非增量:净值历史可能被数据源修订(尤其 QDII),全量最干净;
    raw 收盘价不会被修订,且中转站扛不住全历史单拉,走增量。
    东财接口偶发限流,带重试与间隔。
    """
    import time

    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    DATA_DIR.mkdir(exist_ok=True)
    for sym in symbols:
        # 未复权收盘(tushare fund_daily,交易所原始精度;失败直接抛错,不写可疑数据)。
        # 中转站上游池扛不住全历史单拉(2026-10-13 实测 ~1900 行必 pool_exhausted,
        # 小窗口正常):有缓存走 9 天重叠增量(交易所收盘价不会被修订,增量安全),
        # 无缓存按年分块重建。
        raw_f = DATA_DIR / f"{sym}.raw.csv"
        if raw_f.exists():
            old = pd.read_csv(raw_f)
            win_start = (pd.Timestamp(old["date"].max()) - pd.Timedelta(days=9)).strftime("%Y%m%d")
            inc = _fetch_ts(sym, win_start, end.replace("-", ""), "")
            if inc.empty:
                warnings.warn(f"{sym} tushare 未复权收盘增量为空,raw 缓存停留旧值。", stacklevel=2)
                raw = old
            else:
                new = pd.DataFrame({
                    "date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in inc["日期"]],
                    "close": inc["收盘"].astype(float),
                })
                raw = pd.concat([old, new]).drop_duplicates("date").sort_values("date")
        else:
            parts = []
            for y in range(int(start[:4]), int(end[:4]) + 1):
                chunk = _fetch_ts(sym, f"{y}0101",
                                  end.replace("-", "") if y == int(end[:4]) else f"{y}1231", "")
                if not chunk.empty:
                    parts.append(pd.DataFrame({
                        "date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in chunk["日期"]],
                        "close": chunk["收盘"].astype(float),
                    }))
                time.sleep(1)
            if not parts:
                raise RuntimeError(f"{sym} tushare 未复权收盘返回空")
            raw = pd.concat(parts).drop_duplicates("date").sort_values("date")
        raw.to_csv(raw_f, index=False)

        # 单位净值(东财,带重试)
        last_err = None
        for attempt in range(4):
            try:
                nav = ak.fund_etf_fund_info_em(fund=sym, start_date=start.replace("-", ""),
                                               end_date=end.replace("-", ""))
                break
            except Exception as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"{sym} 净值抓取失败:{last_err}")
        nav = nav.rename(columns={"净值日期": "date", "单位净值": "nav"})[["date", "nav"]]
        nav.to_csv(DATA_DIR / f"{sym}.nav.csv", index=False)
        time.sleep(2)
