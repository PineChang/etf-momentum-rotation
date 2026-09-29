#!/usr/bin/env python
"""随机 10 个一年期窗口的 V3 策略批量图文报告。

随机种子固定(42),可复现。窗口起点范围:2019-06-01 ~ 2025-09-28
(保证窗口完整落在数据区间内,且 warmup 有历史数据)。

运行:NO_PROXY='*' PYTHONPATH=src python scripts/report_random10.py
产出:output/rand10/winNN_YYYY-MM-DD/ 下各一份图文报告 + output/rand10/summary.csv
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_last_year import CFG, OUT, generate_report

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE
from etf_rotation.data import fetch_universe

N_WINDOWS = 10
SEED = 42
EARLIEST_START = pd.Timestamp("2019-06-01")
LATEST_START = pd.Timestamp("2025-09-28")   # 保证 start+1年 ≤ 数据末尾


def main() -> None:
    random.seed(SEED)
    span_days = (LATEST_START - EARLIEST_START).days
    starts = sorted(EARLIEST_START + pd.Timedelta(days=random.randint(0, span_days))
                    for _ in range(N_WINDOWS))

    data = fetch_universe(list(UNIVERSE))
    res = run_backtest(data, CFG)   # 全周期只跑一次,各窗口切片

    root = OUT / "rand10"
    summaries = []
    for i, start in enumerate(starts, 1):
        out_dir = root / f"win{i:02d}_{start.date()}"
        s = generate_report(start, out_dir, res)
        summaries.append(s)
        print(f"[{i:2d}/{N_WINDOWS}] {s['window']}  收益 {s['total_return']:+.1%}  "
              f"回撤 {s['max_drawdown']:.1%}  夏普 {s['sharpe']:.2f}  胜率 {s['win_rate']:.1%}")

    df = pd.DataFrame(summaries)
    df.to_csv(root / "summary.csv", index=False)

    pct = lambda v: f"{v:+.1%}"
    print("\n=== 10 个随机一年期窗口汇总(V3)===")
    print(df[["window", "total_return", "max_drawdown", "sharpe", "calmar", "n_trips", "win_rate"]]
          .to_string(index=False, formatters={
              "total_return": pct, "max_drawdown": lambda v: f"{v:.1%}",
              "sharpe": lambda v: f"{v:.2f}", "calmar": lambda v: f"{v:.2f}",
              "win_rate": lambda v: f"{v:.1%}"}))
    print(f"\n收益: 均值 {df['total_return'].mean():+.1%} | 中位 {df['total_return'].median():+.1%} | "
          f"最差 {df['total_return'].min():+.1%} | 最好 {df['total_return'].max():+.1%}")
    print(f"回撤: 最差 {df['max_drawdown'].min():.1%} | 均值 {df['max_drawdown'].mean():.1%}")
    print(f"盈利窗口占比: {(df['total_return'] > 0).mean():.0%}")
    print(f"\n汇总: {root}/summary.csv | 各窗口报告: {root}/winNN_*/report.md")


if __name__ == "__main__":
    main()
