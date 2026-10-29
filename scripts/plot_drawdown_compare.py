#!/usr/bin/env python
"""重绘 README 的"降回撤三阶段"对比图(output/drawdown_compare.png)。

三条曲线 = 三阶段:基础版策略 → 单持仓+风控 → 双持仓推荐(V3.1)。
图例数字由 perf_metrics 从当次回测现算,永远不会与曲线脱节。

运行:NO_PROXY='*' PYTHONPATH=src python scripts/plot_drawdown_compare.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from etf_rotation.backtest import run_backtest
from etf_rotation.cli import PRESETS
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import attach_premium, fetch_universe
from etf_rotation.metrics import perf_metrics

OUT = Path(__file__).resolve().parents[1] / "output" / "drawdown_compare.png"

# macOS 中文字体回退链(与 report.py 一致)
plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

STAGES = [
    ("基础版策略", {}, "#7f7f7f"),
    # trail8+stop6+thr2+min_mom+confirm3(README 三档方案表的"单持仓+风控"行)
    ("单持仓+风控", dict(trailing_stop=0.08, stop_loss=0.06, rebalance_threshold=2.0,
                         min_momentum=0.0, confirm_days=3), "#2980b9"),
    ("双持仓推荐V3.1", PRESETS["v3.1"], "#c0392b"),
]


def main() -> None:
    data = fetch_universe(list(UNIVERSE))
    curves = []
    for name, params, color in STAGES:
        cfg = StrategyConfig(**params)
        d = attach_premium(data) if cfg.premium_max is not None else data
        res = run_backtest(d, cfg)
        m = perf_metrics(res.equity, res.trades)
        curves.append((name, res.equity, m, color))
        print(f"{name}: 年化 {m['cagr']:.2%} 回撤 {m['max_drawdown']:.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for name, equity, m, color in curves:
        label = f"{name} (年化{m['cagr']:.1%} 回撤{m['max_drawdown']:.1%})"
        ax1.plot(equity.index, equity.values, lw=1.4, color=color, label=label)
        dd = equity / equity.cummax() - 1
        ax2.plot(dd.index, dd.values, lw=1.2, color=color, label=name)

    ax1.set_yscale("log")
    ax1.set_title("降回撤三阶段:净值对比(对数轴)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axhline(-0.20, ls="--", lw=1, color="gray", alpha=0.7)
    ax2.text(dd.index[5], -0.195, "-20% 目标线", fontsize=9, color="gray")
    ax2.set_title("回撤对比")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"图表: {OUT}")


if __name__ == "__main__":
    main()
