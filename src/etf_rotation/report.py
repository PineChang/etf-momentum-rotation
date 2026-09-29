"""回测报告:净值曲线 + 回撤 + 基准对比图,交易明细导出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from .config import UNIVERSE

# macOS 中文字体回退链
plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_report(res: BacktestResult, data: dict[str, pd.DataFrame], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # 策略净值
    ax1.plot(res.equity.index, res.equity.values, lw=1.8, color="#c0392b", label="三因子动量轮动策略")

    # 各 ETF 基准净值(对齐到策略净值起点)
    start = res.equity.index[0]
    for symbol, name in UNIVERSE.items():
        df = data[symbol]
        df = df[df["date"] >= start]
        if df.empty:
            continue
        norm = df["close"] / df["close"].iloc[0]
        ax1.plot(df["date"], norm.values, lw=0.8, alpha=0.55, label=name)

    ax1.set_yscale("log")
    ax1.set_title("ETF 三因子动量轮动策略 — 净值曲线(对数轴)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # 回撤
    dd = res.equity / res.equity.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, color="#2c3e50", alpha=0.6)
    ax2.set_title("回撤")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def export_trades(res: BacktestResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out_path, index=False)
    return out_path
