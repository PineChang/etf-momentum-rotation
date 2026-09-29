"""参数遍历:因子权重 × 调仓阈值网格回测(对应文章提到的聚宽参数遍历分析)。

注意(文章原话):参数最优是双刃剑,存在过拟合风险,不要太执着于调参。
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import pandas as pd

from .backtest import run_backtest
from .config import StrategyConfig
from .metrics import perf_metrics


def weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    """生成单纯形上的权重组合:w_bias + w_slope + w_efficiency = 1。"""
    n = int(round(1 / step))
    return [
        (i * step, j * step, (n - i - j) * step)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    ]


def sweep(
    data: dict[str, pd.DataFrame],
    thresholds: list[float] | None = None,
    step: float = 0.1,
    base: StrategyConfig | None = None,
) -> pd.DataFrame:
    """权重 × 阈值全网格回测,返回按夏普降序的结果表。"""
    base = base or StrategyConfig()
    thresholds = thresholds or [1.0, 1.2, 1.5, 2.0]
    rows = []
    for (wb, ws, we), th in product(weight_grid(step), thresholds):
        cfg = replace(base, w_bias=wb, w_slope=ws, w_efficiency=we, rebalance_threshold=th)
        res = run_backtest(data, cfg)
        m = perf_metrics(res.equity, res.trades)
        rows.append({
            "w_bias": wb, "w_slope": ws, "w_efficiency": we, "threshold": th,
            "cagr": m["cagr"], "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"], "n_trades": m.get("n_round_trips", 0),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
