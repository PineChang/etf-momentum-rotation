"""行情数据获取与本地缓存(akshare 东方财富 ETF 日线,前复权)。

缓存为 CSV:data/{symbol}.csv。首次全量下载,之后增量更新。
"""

from __future__ import annotations

from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_COL_MAP = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}


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


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_COL_MAP)[list(_COL_MAP.values())]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_etf(symbol: str, start: str = "20190101", end: str | None = None,
              use_cache: bool = True, adjust: str = "qfq") -> pd.DataFrame:
    """获取单只 ETF 复权日线。返回 date/open/high/low/close/volume 列。

    adjust: "qfq" 前复权(默认,价格序列连续便于看图,但除息后历史价格会被改写,
            含轻微未来信息);"hfq" 后复权(锚定上市首日,历史价格不被改写,
            无未来信息,严谨的因子计算应以此验证)。
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / (f"{symbol}.csv" if adjust == "qfq" else f"{symbol}.{adjust}.csv")
    end = end or pd.Timestamp.today().strftime("%Y%m%d")

    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        latest = df["date"].max()
        today = pd.Timestamp.today().normalize()
        if latest < today - pd.Timedelta(days=4):  # 容忍周末/小长假
            # 带 5 个交易日重叠增量拉取:重叠区用于校验复权锚点。
            # 前复权(qfq)在分红后整条历史价格会被数据源改写,若重叠区价格
            # 与缓存不一致,说明锚点已变——直接追加会在除息日产生假跳空,
            # 必须丢弃缓存全量重拉(后复权锚定上市日,不受影响)。
            ov_start = (latest - pd.Timedelta(days=9)).strftime("%Y%m%d")
            inc = _fetch_em(symbol, ov_start, end, adjust)
            if not inc.empty:
                inc = _normalize(inc)
                overlap = inc[inc["date"] <= latest].set_index("date")["close"]
                cached = df[df["date"].isin(overlap.index)].set_index("date")["close"]
                reanchored = (not overlap.empty
                              and not np.allclose(overlap.sort_index(), cached.sort_index(), atol=1e-4))
                if reanchored:
                    df = _normalize(_fetch_em(symbol, start, end, adjust))
                else:
                    df = pd.concat([df, inc]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
                df.to_csv(cache, index=False)
        return df

    df = _normalize(_fetch_em(symbol, start, end, adjust))
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

    依赖缓存 data/{symbol}.raw.csv(腾讯未复权日线)与 {symbol}.nav.csv(东财单位净值),
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
    无缓存数据的标的自动放行,不会因数据缺失而误杀。
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
