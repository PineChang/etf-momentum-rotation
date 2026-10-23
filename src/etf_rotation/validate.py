"""数据校验核准 —— 缓存行情/权益账/交叉源一致性的质量门。

数据是回测与信号的基础。本模块提供两级校验:

- **离线校验**(每次 plan 前必过,纯本地、零网络):
  OHLC 结构(重复日期/非法四价/异常涨跌)、交易日历完整性、四标的日期对齐、
  权益账一致性(total = cash + market_value、最后一笔成交后的行可按当前持仓复算)。
- **在线交叉核准**(``cli validate --cross``,每日晚间一次):
  未复权收盘价与东财源逐分比对(交易所原始价,两源必须一致)。

问题分两级:
- ``critical``:数据损坏,``cmd_plan`` 拒绝生成计划;
- ``warning``:可继续(策略已有 data_lag 等降级路径),但会进入 Bark 通知。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNIVERSE

CRITICAL = "critical"
WARNING = "warning"

# ETF 单日涨跌幅硬界:主板 ETF ±10%、创业板/科创 ETF ±20%,留余量取 25%
MAX_DAILY_MOVE = 0.25
# 交叉核准容差:未复权收盘为交易所价,两源应逐分一致(最小价位 0.001)
CROSS_TOL = 5e-4


@dataclass(frozen=True)
class Issue:
    level: str            # CRITICAL | WARNING
    symbol: str           # 标代码;"-" 表示跨标的/账户级
    check: str            # 检查名
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.symbol} {self.check}: {self.message}"


def validate_ohlc(df: pd.DataFrame, symbol: str) -> list[Issue]:
    """单标的结构校验:重复/乱序日期、NaN、四价关系、价格非正、异常涨跌。"""
    issues: list[Issue] = []
    if df.empty:
        return [Issue(CRITICAL, symbol, "ohlc", "数据为空")]
    dates = df["date"]
    if dates.duplicated().any():
        dup = dates[dates.duplicated()].dt.date.astype(str).tolist()
        issues.append(Issue(CRITICAL, symbol, "ohlc", f"重复日期 {dup[:3]}"))
    if not dates.is_monotonic_increasing:
        issues.append(Issue(CRITICAL, symbol, "ohlc", "日期非单调递增"))
    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any(axis=None):
        bad = df[ohlc.isna().any(axis=1)]["date"].dt.date.astype(str).tolist()
        issues.append(Issue(CRITICAL, symbol, "ohlc", f"四价含缺失值 {bad[:3]}"))
    clean = ohlc.dropna()
    illegal = clean[(clean["low"] > clean[["open", "close"]].min(axis=1))
                    | (clean["high"] < clean[["open", "close"]].max(axis=1))]
    if not illegal.empty:
        bad = df.loc[illegal.index, "date"].dt.date.astype(str).tolist()
        issues.append(Issue(CRITICAL, symbol, "ohlc",
                            f"四价关系非法(low>开/收 或 high<开/收) {bad[:3]}"))
    if (clean <= 0).any(axis=None):
        issues.append(Issue(CRITICAL, symbol, "ohlc", "存在非正价格"))
    moves = df["close"].pct_change().abs()
    extreme = moves[moves > MAX_DAILY_MOVE]
    if not extreme.empty:
        bad = [f"{df.loc[i, 'date'].date()} {moves[i]:+.1%}" for i in extreme.index[:3]]
        issues.append(Issue(CRITICAL, symbol, "ohlc",
                            f"单日涨跌超 ±{MAX_DAILY_MOVE:.0%}(疑似坏点/除权未复权) {bad}"))
    return issues


def validate_calendar(df: pd.DataFrame, symbol: str,
                      trading_days: set[pd.Timestamp] | None) -> list[Issue]:
    """交易日历完整性:缺交易日 bar / 非交易日有 bar。

    trading_days 为 None(日历不可用)时跳过——宁可不查,不可误报。
    缺 bar 记 warning:plan 的 data_lag 机制已定义降级;停复牌也表现为缺 bar,
    不应与数据损坏同级。非交易日有 bar 记 critical:缓存混入了假日期。
    """
    if trading_days is None or df.empty:
        return []
    have = set(df["date"])
    start, end = df["date"].iloc[0], df["date"].iloc[-1]
    expected = {d for d in trading_days if start <= d <= end}
    missing = sorted(expected - have)
    extra = sorted(d for d in have - expected if d.weekday() < 5)  # 周内非交易日=假期混入
    issues: list[Issue] = []
    if missing:
        issues.append(Issue(WARNING, symbol, "calendar",
                            f"缺 {len(missing)} 个交易日 bar: "
                            f"{[str(d.date()) for d in missing[:3]]}"))
    if extra:
        issues.append(Issue(CRITICAL, symbol, "calendar",
                            f"非交易日出现 bar: {[str(d.date()) for d in extra[:3]]}"))
    return issues


def validate_alignment(data: dict[str, pd.DataFrame]) -> list[Issue]:
    """四标的最新交易日应一致;不一致即 data_lag,plan 有降级,记 warning。"""
    latest = {s: df["date"].max() for s, df in data.items() if not df.empty}
    if len(set(latest.values())) <= 1:
        return []
    detail = ", ".join(f"{s} {d.date()}" for s, d in sorted(latest.items()))
    return [Issue(WARNING, "-", "alignment", f"标的最新交易日不一致: {detail}")]


def validate_equity(store, state, data: dict[str, pd.DataFrame]) -> list[Issue]:
    """权益账一致性。

    - 每行 total 必须等于 cash + market_value(记账错误 → critical);
    - 日期必须单调不重复(critical);
    - 最后一笔成交之后的行用当前持仓复算(heal_equity_tail 同口径),
      不一致 → warning(仪表盘生成时会自愈重写;此处负责发现并留痕)。
    """
    f = store.dir / "equity.csv"
    if not f.exists():
        return []
    df = pd.read_csv(f, parse_dates=["date"])
    if df.empty:
        return []
    issues: list[Issue] = []
    if df["date"].duplicated().any() or not df["date"].is_monotonic_increasing:
        issues.append(Issue(CRITICAL, "-", "equity", "equity.csv 日期重复或乱序"))
    bad_sum = df[np.abs(df["total"] - df["cash"] - df["market_value"]) > 0.01]
    if not bad_sum.empty:
        days = bad_sum["date"].dt.date.astype(str).tolist()
        issues.append(Issue(CRITICAL, "-", "equity", f"total ≠ cash+market_value: {days[:3]}"))

    if not state.holdings:
        return issues
    last_fill_plan = None
    if store.log_file.exists():
        for line in store.log_file.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "fill" and rec.get("plan_date"):
                d = pd.Timestamp(rec["plan_date"])
                if last_fill_plan is None or d > last_fill_plan:
                    last_fill_plan = d
    closes = {}
    for symbol in state.holdings:
        df_s = data.get(symbol)
        if df_s is None or df_s.empty:
            return issues
        closes[symbol] = df_s.set_index("date")["close"].sort_index()
    today = pd.Timestamp.today().normalize()
    stale: list[str] = []
    for _, row in df.iloc[::-1].iterrows():
        d = row["date"]
        if d >= today:
            continue
        if last_fill_plan is not None and d <= last_fill_plan:
            break
        mv = 0.0
        for symbol, h in state.holdings.items():
            s = closes[symbol]
            s = s[s.index <= d]
            if s.empty or s.index[-1] < d:
                mv = None
                break
            mv += h.shares * float(s.iloc[-1])
        if mv is not None and abs(state.cash + mv - row["total"]) > 0.01:
            stale.append(str(d.date()))
    if stale:
        issues.append(Issue(WARNING, "-", "equity",
                            f"历史权益与最终收盘不符(盘中价冻结): {sorted(stale)}"
                            f"(下次生成仪表盘时自动重写)"))
    return issues


def crosscheck_with_em(symbols: list[str], days: int = 10) -> list[Issue]:
    """在线交叉核准:本地未复权收盘 vs 东财源,逐分比对最近 days 个 bar。

    未复权价是交易所原始价,两源必须一致(容差 CROSS_TOL,最小价位的一半)。
    网络失败记 warning(离线日不应阻断流程),价格不一致记 critical。
    """
    from .data import DATA_DIR, _fetch_em

    issues: list[Issue] = []
    for symbol in symbols:
        raw_file = DATA_DIR / f"{symbol}.raw.csv"
        if not raw_file.exists():
            issues.append(Issue(WARNING, symbol, "crosscheck", "无 raw.csv 缓存,跳过比对"))
            continue
        local = pd.read_csv(raw_file, parse_dates=["date"]).tail(days)
        try:
            em = _fetch_em(symbol, local["date"].iloc[0].strftime("%Y%m%d"),
                           local["date"].iloc[-1].strftime("%Y%m%d"), adjust="")
        except Exception as e:
            issues.append(Issue(WARNING, symbol, "crosscheck", f"东财源取数失败({e}),无法比对"))
            continue
        em = em.rename(columns={"日期": "date", "收盘": "close"})
        em["date"] = pd.to_datetime(em["date"])
        merged = local.merge(em[["date", "close"]], on="date", suffixes=("_local", "_em"))
        if merged.empty:
            issues.append(Issue(WARNING, symbol, "crosscheck", "两源日期无重叠,无法比对"))
            continue
        diff = merged[np.abs(merged["close_local"] - merged["close_em"]) > CROSS_TOL]
        if not diff.empty:
            bad = [f"{r.date.date()} 本地{r.close_local} vs 东财{r.close_em}"
                   for r in diff.itertuples()][:3]
            issues.append(Issue(CRITICAL, symbol, "crosscheck",
                                f"未复权收盘两源不一致(>{CROSS_TOL:.0e}): {bad}"))
    return issues


def trading_days_from_calendar(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp] | None:
    """从 data_source 交易日历取区间交易日;不可用返回 None(检查跳过)。"""
    from .market_calendar import _load_data_source

    ds = _load_data_source()
    if ds is None:
        return None
    try:
        cal = ds.calendar(str(start.date()), str(end.date()), open_only=True)
        if cal.empty:
            return None
        col = cal.columns[0]
        return set(pd.to_datetime(cal[col]))
    except Exception:
        return None


def validate_all(data: dict[str, pd.DataFrame], store=None, state=None) -> list[Issue]:
    """离线全量校验:结构 + 日历 + 对齐 + (可选)权益账。"""
    issues: list[Issue] = []
    starts = [df["date"].iloc[0] for df in data.values() if not df.empty]
    ends = [df["date"].iloc[-1] for df in data.values() if not df.empty]
    days = trading_days_from_calendar(min(starts), max(ends)) if starts else None
    for symbol, df in data.items():
        issues.extend(validate_ohlc(df, symbol))
        issues.extend(validate_calendar(df, symbol, days))
    issues.extend(validate_alignment(data))
    if store is not None and state is not None:
        issues.extend(validate_equity(store, state, data))
    return issues


def has_critical(issues: list[Issue]) -> bool:
    return any(i.level == CRITICAL for i in issues)


def format_report(issues: list[Issue]) -> str:
    """打印报告。措辞对齐 notify.summarize_log 的捕获词(critical→'Error', warning→'警告')。"""
    if not issues:
        return "数据校验: 全部通过"
    lines = []
    for i in issues:
        tag = "Error: 数据校验" if i.level == CRITICAL else "数据校验警告"
        lines.append(f"{tag} [{i.symbol}] {i.check}: {i.message}")
    n_crit = sum(1 for i in issues if i.level == CRITICAL)
    lines.append(f"数据校验: {len(issues)} 个问题(critical {n_crit}, warning {len(issues) - n_crit})")
    return "\n".join(lines)


def load_cached_universe(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """读本地缓存(不联网);缺文件的标的以空 df 占位,由结构校验报 critical。"""
    from .data import DATA_DIR

    root = data_dir or DATA_DIR
    data: dict[str, pd.DataFrame] = {}
    for symbol in UNIVERSE:
        f = root / f"{symbol}.csv"
        data[symbol] = (pd.read_csv(f, parse_dates=["date"])
                        if f.exists() else pd.DataFrame(columns=["date", "open", "high",
                                                                 "low", "close", "volume"]))
    return data
