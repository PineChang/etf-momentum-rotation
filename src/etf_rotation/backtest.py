"""回测引擎:评分驱动轮动,信号日收盘评分,次日开盘成交(避免前视偏差)。

- 每个交易日收盘后计算三因子 → Z-Score → 加权评分;
- 按阈值机制决定目标持仓(scoring.select_targets,n_hold 只等权);
- 目标变化 → 次日开盘调仓,佣金双边收取;
- 全程复利:卖出回款(含浮动盈亏)全额再投,对应文章
  “每次调仓后把之前持仓盈亏相加,实现复利效应”。

风控扩展(默认关闭/默认 1,即忠实原文,见 StrategyConfig):
- n_hold:持仓只数,1=原文全仓单标的;2=双标的等权分散(建仓时等权,
  持仓期间不 rebalance,权重随涨跌漂移);
- min_momentum:绝对动量过滤(原始收益,非 Z-Score),全池无趋势时空仓观望;
- ma_filter_n:候选标的跌破自身均线时不建仓/离场;
- stop_loss:固定止损(对成本价);trailing_stop:移动止损(对持仓期间最高价,保护浮盈);
  触发后进入 stop_cooldown_days 禁买期;
- confirm_days:全空仓状态下,标的须连续 N 天评分第一才允许建仓(防震荡期假信号)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import StrategyConfig
from .factors import compute_factor_scores, dynamic_lookback_window
from .scoring import combined_score, select_targets

_UNSET = object()  # 今日无信号(区别于信号为空集 = 目标空仓)


@dataclass
class Trade:
    date: pd.Timestamp
    action: str          # BUY / SELL
    symbol: str
    price: float
    shares: float
    commission: float


@dataclass
class BacktestResult:
    equity: pd.Series            # 每日收盘净值(初始=1)
    trades: pd.DataFrame
    scores: pd.DataFrame         # 每日各 ETF 综合评分(诊断用)
    holdings: pd.Series          # 每日收盘持仓(None=空仓,多持仓为 "+" 连接)
    config: StrategyConfig = field(default=None)


def _align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """按交易日内连接对齐,保证同日可比。"""
    common = None
    for df in data.values():
        dates = set(df["date"])
        common = dates if common is None else common & dates
    common = sorted(common)
    return {s: df[df["date"].isin(common)].reset_index(drop=True) for s, df in data.items()}


def _passes_risk_filters(symbol: str, data: dict[str, pd.DataFrame], t: int, cfg: StrategyConfig) -> bool:
    """绝对动量过滤 + 均线过滤(原始收益,非 Z-Score——横截面标准化后
    总有一个相对最高的标的,无法表达"绝对无趋势")。"""
    close = data[symbol]["close"].iloc[: t + 1]
    if cfg.min_momentum is not None and len(close) >= cfg.slope_n:
        if close.iloc[-1] / close.iloc[-cfg.slope_n] - 1 <= cfg.min_momentum:
            return False
    if cfg.ma_filter_n is not None and len(close) >= cfg.ma_filter_n:
        if close.iloc[-1] < close.rolling(cfg.ma_filter_n).mean().iloc[-1]:
            return False
    # 知乎改进③:近 5 日出现单日跌超 5%,或连跌且累计跌超 5% 的标的不买(防接飞刀)
    if cfg.drop_filter and len(close) >= 5:
        p = close.iloc[-5:].to_numpy(dtype=float)
        con1 = min(p[-1] / p[-2], p[-2] / p[-3], p[-3] / p[-4]) < 0.95
        con2 = (p[-1] < p[-2] < p[-3] < p[-4]) and (p[-1] / p[-4] < 0.95)
        con3 = (p[-2] < p[-3] < p[-4] < p[-5]) and (p[-2] / p[-5] < 0.95)
        if con1 or con2 or con3:
            return False
    # 知乎改进②:场内价对单位净值溢价过高(资金拥挤)不买。premium 列由
    # 实验脚本按 未复权收盘/单位净值-1 预计算注入;行情 df 无该列时视为无数据放行。
    if cfg.premium_max is not None and "premium" in data[symbol]:
        prem = data[symbol]["premium"].iloc[t]
        if pd.notna(prem) and prem > cfg.premium_max:
            return False
    return True


def compute_target_set(
    data: dict[str, pd.DataFrame],
    t: int,
    cfg: StrategyConfig,
    positions: dict[str, float],
    entries: dict[str, float],
    peaks: dict[str, float],
    stopped: dict[str, int],
    top_streak: dict[str, int],
) -> tuple[frozenset[str], pd.Series]:
    """T 日收盘信号:回测引擎与实盘模块(live.py)共用的唯一实现。

    输入 T 日及之前的行情(data 已截断或传入 t 索引)、当前持仓与风控状态,
    输出目标持仓集合与当日综合评分。会就地更新 peaks / stopped / top_streak
    (调用方负责持久化),持有份额 positions 不被修改。

    回测对每个交易日调用本函数;live.py 只对最新交易日调用——
    两者因此保证信号逻辑逐行一致,不会出现"回测一套代码、实盘另一套"。
    """
    symbols = list(data)
    window = {s: data[s].iloc[: t + 1] for s in symbols}
    lookbacks = None
    if cfg.dynamic_lookback:
        lookbacks = {
            s: dynamic_lookback_window(
                window[s], cfg.lb_min, cfg.lb_max, cfg.lb_ratio_cap, cfg.vol_short_n, cfg.vol_long_n
            )
            for s in symbols
        }
        lookbacks = {s: n for s, n in lookbacks.items() if n is not None} or None
    factor_df = compute_factor_scores(
        window, cfg.bias_ma_n, cfg.momentum_day, cfg.slope_n, cfg.efficiency_n, lookbacks
    )
    scores = combined_score(factor_df, cfg.w_bias, cfg.w_slope, cfg.w_efficiency)

    banned = [s for s, ts in stopped.items() if t - ts < cfg.stop_cooldown_days]
    candidates = scores.drop(index=banned, errors="ignore")
    target_set = select_targets(candidates, set(positions), cfg.rebalance_threshold, cfg.n_hold)
    target_set = {s for s in target_set if _passes_risk_filters(s, data, t, cfg)}

    # 入场确认:全空仓时,第一名须连续 confirm_days 天居首才允许建仓
    top: str | None = None
    if not candidates.empty:
        top = candidates.idxmax()
        top_streak[top] = top_streak.get(top, 0) + 1
        for s in list(top_streak):
            if s != top:
                top_streak[s] = 0
    if not positions and target_set and top_streak.get(top, 0) < cfg.confirm_days:
        target_set = set()

    # 固定止损(对成本价)+ 移动止损(对持仓最高价)+ 固定止盈(对成本价),优先级最高
    for sym in list(positions):
        close_t = float(data[sym]["close"].iloc[t])
        peaks[sym] = max(peaks[sym], close_t)
        hit_fixed = cfg.stop_loss is not None and close_t < entries[sym] * (1 - cfg.stop_loss)
        hit_trailing = cfg.trailing_stop is not None and close_t < peaks[sym] * (1 - cfg.trailing_stop)
        hit_tp = cfg.take_profit is not None and close_t > entries[sym] * (1 + cfg.take_profit)
        if hit_fixed or hit_trailing or hit_tp:
            stopped[sym] = t
            target_set.discard(sym)
    return frozenset(target_set), scores


def run_backtest(data: dict[str, pd.DataFrame], cfg: StrategyConfig | None = None) -> BacktestResult:
    cfg = cfg or StrategyConfig()
    data = _align(data)
    symbols = list(data)
    calendar = data[symbols[0]]["date"]

    cash = cfg.initial_cash
    positions: dict[str, float] = {}     # symbol -> shares
    entries: dict[str, float] = {}       # symbol -> 成本价(固定止损)
    peaks: dict[str, float] = {}         # symbol -> 持仓期间最高收盘价(移动止损)
    stopped: dict[str, int] = {}         # 止损禁买名单:symbol -> 触发日 t
    top_streak: dict[str, int] = {}      # 连续评分第一天数(入场确认)
    pending = _UNSET
    trades: list[Trade] = []
    equity, holdings_rec, scores_rec = {}, {}, {}

    for t in range(len(calendar)):
        date = calendar.iloc[t]

        # 1) 开盘:执行昨日信号(pending 为空集表示目标空仓)
        if pending is not _UNSET:
            target_set: set[str] = pending
            for sym in list(positions):              # 先卖出不在目标中的持仓
                if sym not in target_set:
                    price = float(data[sym]["open"].iloc[t])
                    proceeds = positions[sym] * price
                    commission = proceeds * cfg.commission
                    trades.append(Trade(date, "SELL", sym, price, positions[sym], commission))
                    cash += proceeds - commission
                    del positions[sym], entries[sym], peaks[sym]
            total = cash + sum(sh * float(data[s]["open"].iloc[t]) for s, sh in positions.items())
            slot = total / cfg.n_hold                # 每个槽位等权
            for sym in sorted(target_set):           # 再买入空槽位
                if sym not in positions:
                    budget = min(slot, cash)
                    price = float(data[sym]["open"].iloc[t])
                    positions[sym] = budget * (1 - cfg.commission) / price
                    entries[sym] = peaks[sym] = price
                    trades.append(Trade(date, "BUY", sym, price, positions[sym], budget * cfg.commission))
                    cash -= budget
        pending = _UNSET

        # 2) 收盘:记账 + 计算评分生成次日信号(需足够 warmup)
        equity[date] = cash + sum(sh * float(data[s]["close"].iloc[t]) for s, sh in positions.items())
        holdings_rec[date] = "+".join(sorted(positions)) if positions else None

        if t + 1 >= cfg.warmup_bars:
            target_set, scores = compute_target_set(
                data, t, cfg, positions, entries, peaks, stopped, top_streak
            )
            scores_rec[date] = scores
            pending = frozenset(target_set)

    trades_df = pd.DataFrame([vars(x) for x in trades])
    return BacktestResult(
        equity=pd.Series(equity, name="equity") / cfg.initial_cash,
        trades=trades_df,
        scores=pd.DataFrame(scores_rec).T,
        holdings=pd.Series(holdings_rec, name="holding"),
        config=cfg,
    )
