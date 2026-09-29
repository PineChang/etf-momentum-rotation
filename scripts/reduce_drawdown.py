#!/usr/bin/env python
"""降回撤实验矩阵:单因子风控实验 + 组合实验,全周期 + 最近一年双窗口评估。

运行:PYTHONPATH=src python scripts/reduce_drawdown.py
结果输出到 output/drawdown_experiments.csv
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import fetch_universe
from etf_rotation.metrics import perf_metrics

WINDOW_START = pd.Timestamp("2025-09-28")
OUT = Path(__file__).resolve().parents[1] / "output" / "drawdown_experiments.csv"

EXPERIMENTS: dict[str, dict] = {
    # ---- 基线(忠实原文) ----
    "baseline(原文)": {},
    # ---- 调仓阈值 ----
    "thr=2.0": {"rebalance_threshold": 2.0},
    "thr=2.5": {"rebalance_threshold": 2.5},
    "thr=3.0": {"rebalance_threshold": 3.0},
    # ---- 绝对动量过滤 ----
    "min_mom=0": {"min_momentum": 0.0},
    # ---- 均线过滤 ----
    "ma10": {"ma_filter_n": 10},
    "ma20": {"ma_filter_n": 20},
    "ma60": {"ma_filter_n": 60},
    # ---- 止损 ----
    "stop6%": {"stop_loss": 0.06},
    "stop8%": {"stop_loss": 0.08},
    "stop10%": {"stop_loss": 0.10},
    "stop12%": {"stop_loss": 0.12},
    # ---- 两两组合 ----
    "thr2+min_mom": {"rebalance_threshold": 2.0, "min_momentum": 0.0},
    "min_mom+ma20": {"min_momentum": 0.0, "ma_filter_n": 20},
    "min_mom+stop8": {"min_momentum": 0.0, "stop_loss": 0.08},
    "ma20+stop8": {"ma_filter_n": 20, "stop_loss": 0.08},
    "ma60+min_mom": {"ma_filter_n": 60, "min_momentum": 0.0},
    # ---- 三件套 ----
    "thr2+min_mom+stop8": {"rebalance_threshold": 2.0, "min_momentum": 0.0, "stop_loss": 0.08},
    "min_mom+ma20+stop8": {"min_momentum": 0.0, "ma_filter_n": 20, "stop_loss": 0.08},
    "thr2+min_mom+ma20": {"rebalance_threshold": 2.0, "min_momentum": 0.0, "ma_filter_n": 20},
    "thr2+min_mom+ma20+stop8": {"rebalance_threshold": 2.0, "min_momentum": 0.0, "ma_filter_n": 20, "stop_loss": 0.08},
}


def window_metrics(res) -> dict:
    eq = res.equity[res.equity.index >= WINDOW_START]
    if eq.empty:
        return {}
    eq = eq / eq.iloc[0]
    m = perf_metrics(eq)
    return {"y1_ret": m["total_return"], "y1_dd": m["max_drawdown"]}


def main() -> None:
    data = fetch_universe(list(UNIVERSE))
    base = StrategyConfig()
    rows = []
    for name, overrides in EXPERIMENTS.items():
        cfg = replace(base, **overrides)
        res = run_backtest(data, cfg)
        m = perf_metrics(res.equity, res.trades)
        cash_ratio = (res.holdings.isna()).mean()
        rows.append({
            "experiment": name,
            "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
            "calmar": m["calmar"], "trades": m.get("n_round_trips", 0),
            "cash%": cash_ratio, **window_metrics(res),
        })
        print(f"done: {name}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 160)
    pct = lambda v: f"{v:.1%}"
    fmt = {"cagr": pct, "sharpe": lambda v: f"{v:.2f}", "max_dd": pct,
           "calmar": lambda v: f"{v:.2f}", "cash%": lambda v: f"{v:.0%}",
           "y1_ret": pct, "y1_dd": pct}
    print("\n=== 按最大回撤升序(回撤越小越靠前) ===")
    print(df.sort_values("max_dd", ascending=False).to_string(index=False, formatters=fmt))
    print(f"\n完整结果: {OUT}")


if __name__ == "__main__":
    main()
