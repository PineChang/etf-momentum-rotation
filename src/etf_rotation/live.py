"""实盘信号模块 —— 与回测共用同一信号引擎(backtest.compute_target_set)。

复刻文章第 4 节 QMT 实盘的四机制,并完整实现 V3 风控(双持仓/移动止损/
动量过滤/止盈/禁买期/入场确认):

1. 时间管理:盘前(8:00)生成当日交易计划提前查看,开盘(9:30)执行。
   调度由外部(crontab / QMT 定时器)触发,本模块提供 make_plan / execute_plan 两个入口。
2. 持久化:持仓、现金、风控状态(禁买名单/评分连击)、交易日志落地本地 JSON,
   程序重启后恢复。
3. 限价单:买入价 = 现价 * 1.01,卖出价 = 现价 * 0.99(现价取最新收盘;
   开盘跳空超过 1% 可能不成交,与回测"开盘价必成交"假设存在差异,实盘需留意)。
4. 复利盈亏:卖出回款(扣佣金)全额并入现金,调仓时按槽位等权再投。

信号一致性保证:目标持仓的计算调用 backtest.compute_target_set——与回测引擎
逐行相同的实现,不会出现"回测一套代码、实盘另一套"。

本模块不含任何券商接口;对接 QMT/Ptrade 时只需把 execute_plan 里的
成交价来源替换为券商回报,计划生成与状态管理保持不变。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd

from .backtest import _align, compute_target_set
from .config import StrategyConfig

BUY_PRICE_FACTOR = 1.01    # 文章:买入限价 = 现价 * 1.01
SELL_PRICE_FACTOR = 0.99   # 文章:卖出限价 = 现价 * 0.99


@dataclass
class Holding:
    symbol: str
    shares: float
    cost: float    # 开仓价(固定止损/止盈锚点)
    peak: float    # 持仓期间最高收盘价(移动止损锚点)


@dataclass
class LiveState:
    """实盘账户与风控状态(全部持久化,重启可恢复)。"""

    cash: float = 0.0
    holdings: dict[str, Holding] = field(default_factory=dict)
    realized_pnl: float = 0.0                 # 历史累计已实现盈亏(复利记账,信息用)
    stopped: dict[str, str] = field(default_factory=dict)  # symbol -> 止损/止盈触发日(禁买期锚点)
    top_streak: dict[str, int] = field(default_factory=dict)  # 连续评分第一天数(入场确认)
    last_signal: str = ""                     # 最近一次已计入连击的信号日(data_asof)

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "holdings": {s: asdict(h) for s, h in self.holdings.items()},
            "realized_pnl": self.realized_pnl,
            "stopped": dict(self.stopped),
            "top_streak": dict(self.top_streak),
            "last_signal": self.last_signal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiveState":
        holdings = {s: Holding(**h) for s, h in d.get("holdings", {}).items()}
        return cls(cash=d.get("cash", 0.0), holdings=holdings,
                   realized_pnl=d.get("realized_pnl", 0.0),
                   stopped=d.get("stopped", {}), top_streak=d.get("top_streak", {}),
                   last_signal=d.get("last_signal", ""))


class StateStore:
    """持仓与交易日志的本地持久化(JSON),支持程序重启后恢复。"""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.log_file = self.dir / "trade_log.jsonl"
        self.params_file = self.dir / "params.json"

    def save_state(self, state: LiveState) -> None:
        self.state_file.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))

    def load_state(self, initial_cash: float = 0.0) -> LiveState:
        """无存档时返回初始状态(空仓 + 全部现金)。"""
        if not self.state_file.exists():
            return LiveState(cash=initial_cash)
        return LiveState.from_dict(json.loads(self.state_file.read_text()))

    def append_log(self, record: dict) -> None:
        with self.log_file.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_params(self, cfg: StrategyConfig) -> None:
        self.params_file.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    def latest_plan(self) -> dict | None:
        """最近一份盘前计划(trade_log 中最后一条 type=plan 记录),无则 None。"""
        if not self.log_file.exists():
            return None
        for line in reversed(self.log_file.read_text().splitlines()):
            rec = json.loads(line)
            if rec.get("type") == "plan":
                return rec
        return None

    def plan_executed(self, plan: dict) -> bool:
        """该计划是否已有成交记录(按 plan_date + data_asof 匹配,防重复执行)。"""
        if not self.log_file.exists():
            return False
        for line in self.log_file.read_text().splitlines():
            rec = json.loads(line)
            if (rec.get("type") == "fill"
                    and rec.get("plan_date") == plan.get("plan_date")
                    and rec.get("data_asof") == plan.get("data_asof")):
                return True
        return False


def make_plan(
    data: dict[str, pd.DataFrame],
    state: LiveState,
    cfg: StrategyConfig | None = None,
) -> tuple[dict, LiveState]:
    """盘前计划(对应文章"8点打印当日交易计划")。

    用截至最新交易日的收盘数据,经与回测共用的信号引擎计算目标持仓,
    给出 SELL/BUY 限价单。同时返回更新风控状态后的 LiveState(锚点、
    禁买名单、评分连击已推进到最新交易日)——调用方应立即持久化,
    即使当日订单未成交。

    返回 (plan, new_state)。
    """
    cfg = cfg or StrategyConfig()
    data = _align(data)
    symbols = list(data)
    calendar = data[symbols[0]]["date"]
    t = len(calendar) - 1
    date_t = calendar.iloc[t]

    # 止损日期 -> bar 序号(compute_target_set 以 bar 距离计算禁买期)
    stopped_idx: dict[str, int] = {}
    for sym, d in state.stopped.items():
        ts = pd.Timestamp(d)
        if ts in set(calendar):
            stopped_idx[sym] = int(calendar[calendar == ts].index[0])

    # compute_target_set 会就地推进 peaks(持仓期最高收盘价)与 top_streak(评分连击)。
    # 这两个入参必须是**随后用来落盘的那一份**,否则推进结果被丢弃——
    # 实盘移动止损会退化为"锚定开仓价的固定止损",confirm_days 也永不生效。
    peaks = {s: h.peak for s, h in state.holdings.items()}
    streak = dict(state.top_streak)
    signal_day = str(date_t.date())
    if state.last_signal == signal_day:
        # 同一信号日重复 plan(换参数重跑/手动补跑):前一次已把当日连击 +1,
        # 先回退一格再交给引擎,使推进一次后恰好落在正确值——否则 confirm_days≥2
        # 的入场确认会被虚假满足,连击 1 天被数成 2 天并生成建仓单。
        # (peaks/stopped 天然幂等,无需处理。)
        streak = {s: max(0, n - 1) for s, n in streak.items()}
    target, scores = compute_target_set(
        data, t, cfg,
        positions={s: h.shares for s, h in state.holdings.items()},
        entries={s: h.cost for s, h in state.holdings.items()},
        peaks=peaks,
        stopped=stopped_idx,
        top_streak=streak,
    )

    # 推进后的风控状态(份额/现金不变,成交由 execute_plan 处理)
    new_state = LiveState(
        cash=state.cash,
        holdings={s: replace(h, peak=peaks.get(s, h.peak)) for s, h in state.holdings.items()},
        realized_pnl=state.realized_pnl,
        stopped={s: str(calendar.iloc[i].date()) for s, i in stopped_idx.items()},
        top_streak=streak,
        last_signal=signal_day,
    )

    closes = {s: float(data[s]["close"].iloc[t]) for s in symbols}

    orders: list[dict] = []
    held = set(state.holdings)
    # 估值:卖出按收盘价折算(实际以开盘成交价为准)
    total_est = state.cash + sum(h.shares * closes[s] for s, h in state.holdings.items())
    slot = total_est / cfg.n_hold
    for sym in sorted(held - target, key=lambda s: -state.holdings[s].shares):
        orders.append({"action": "SELL", "symbol": sym,
                       "limit_price": round(closes[sym] * SELL_PRICE_FACTOR, 3),
                       "est_budget": round(state.holdings[sym].shares * closes[sym], 2),
                       "reason": f"不在目标持仓(评分 {scores.get(sym, float('nan')):.3f})"})
    for sym in sorted(target - held):
        orders.append({"action": "BUY", "symbol": sym,
                       "limit_price": round(closes[sym] * BUY_PRICE_FACTOR, 3),
                       "est_budget": round(min(slot, total_est), 2),
                       "reason": f"进入目标持仓(评分 {scores.get(sym, float('nan')):.3f})"})

    plan = {
        "plan_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "data_asof": str(date_t.date()),
        "config_note": (f"n_hold={cfg.n_hold}, trailing_stop={cfg.trailing_stop}, "
                        f"min_momentum={cfg.min_momentum}, take_profit={cfg.take_profit}"),
        "current": sorted(held),
        "target": sorted(target),
        "cash": round(state.cash, 2),
        "realized_pnl": round(state.realized_pnl, 2),
        "scores": scores.round(4).to_dict(),
        "reference_prices": {s: closes[s] for s in symbols},
        "orders": orders,
    }
    return plan, new_state


def check_fillable_at_open(plan: dict, open_prices: dict[str, float]) -> list[str]:
    """限价单在开盘价能否成交:买入需 open ≤ 限价,卖出需 open ≥ 限价。

    返回未成交原因列表(空列表 = 全部可成交)。调用方应采用**整批执行或整批放弃**:
    限价单开盘没成交不代表日内不成交,稍后重跑即可;部分执行会让买单用到
    卖单尚未释放的现金,污染仓位结构。
    """
    problems = []
    for o in plan["orders"]:
        sym = o["symbol"]
        open_px = open_prices.get(sym)
        if open_px is None or open_px <= 0:
            problems.append(f"{sym}: 无今日开盘价(行情未开始或数据源缺失)")
            continue
        if o["action"] == "BUY" and open_px > o["limit_price"]:
            problems.append(f"{sym}: 开盘 {open_px} > 买入限价 {o['limit_price']}(高开超 1%,暂未成交)")
        elif o["action"] == "SELL" and open_px < o["limit_price"]:
            problems.append(f"{sym}: 开盘 {open_px} < 卖出限价 {o['limit_price']}(低开超 1%,暂未成交)")
    return problems


def execute_plan(
    plan: dict,
    state: LiveState,
    fill_prices: dict[str, float],
    cfg: StrategyConfig,
) -> LiveState:
    """开盘执行(对应文章"9点半实际交易")。

    fill_prices: {symbol: 实际成交价} —— 对接实盘时由券商回报提供,
    回测/模拟时可用开盘价;未成交的持仓按计划参考价估值。
    返回成交后的 LiveState(先卖后买,槽位等权,复利全额再投)。

    本函数**不幂等**:同一份 plan 只能执行一次,重放会二次买入。卖出不在持仓中的
    标的时直接抛错(而不是静默跳过),以免重放把账户状态改坏。
    """
    if not plan["orders"]:
        return state
    cash = state.cash
    holdings = dict(state.holdings)
    realized = state.realized_pnl
    ref = plan.get("reference_prices", {})
    price_of = lambda s: fill_prices.get(s, ref.get(s, 0.0))

    for o in plan["orders"]:                  # 订单已按先卖后买排序
        sym, price = o["symbol"], price_of(o["symbol"])
        if o["action"] == "SELL":
            if sym not in holdings:
                raise ValueError(
                    f"卖出 {sym} 失败:当前并不持有该标的。计划可能已被执行过"
                    f"(execute_plan 非幂等),或计划与账户状态不匹配。"
                )
            h = holdings.pop(sym)
            proceeds = h.shares * price
            comm = proceeds * cfg.commission
            cash += proceeds - comm
            realized += proceeds - comm - h.shares * h.cost   # 盈亏累加 → 复利记账
        elif o["action"] == "BUY":
            total = cash + sum(h.shares * price_of(s) for s, h in holdings.items())
            budget = min(total / cfg.n_hold, cash)            # 槽位等权,不透支
            holdings[sym] = Holding(sym, budget * (1 - cfg.commission) / price, price, price)
            cash -= budget

    return replace(state, cash=cash, holdings=holdings, realized_pnl=realized)
