#!/usr/bin/env python
"""随机 20 个一年期窗口的 V3 批量回测(含溢价率过滤的当前官方配置)。

配置取 cli.PRESETS["v3.1"](与 cli backtest --preset v3.1 一致,单一来源),
溢价数据注入 data.attach_premium(PIT)。随机种子固定(2026),可复现。
窗口起点范围:2019-06-01 ~ 2025-09-28(窗口完整落数据区间内,warmup 有历史)。

运行:NO_PROXY='*' PYTHONPATH=src python scripts/report_random20.py
产出:output/rand20/summary.csv(README 表格数据来源)
"""

import random
from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.cli import PRESETS
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import attach_premium, fetch_universe
from etf_rotation.metrics import perf_metrics

N_WINDOWS = 20
SEED = 2026
EARLIEST_START = pd.Timestamp("2019-06-01")
LATEST_START = pd.Timestamp("2025-09-28")   # 保证 start+1年 ≤ 数据末尾


def main() -> None:
    cfg = StrategyConfig(**PRESETS["v3.1"])
    print(f"配置: {PRESETS['v3.1']}")

    random.seed(SEED)
    span_days = (LATEST_START - EARLIEST_START).days
    starts = sorted(EARLIEST_START + pd.Timedelta(days=random.randint(0, span_days))
                    for _ in range(N_WINDOWS))

    data = attach_premium(fetch_universe(list(UNIVERSE)))
    res = run_backtest(data, cfg)   # 全周期只跑一次,各窗口切片

    rows = []
    for i, start in enumerate(starts, 1):
        end = start + pd.Timedelta(days=365)
        eq = res.equity[(res.equity.index >= start) & (res.equity.index <= end)]
        m = perf_metrics(eq)
        trips = res.trades[(res.trades["date"] >= start) & (res.trades["date"] <= end)]
        # 窗口内往返配对(按标的 FIFO,含双边佣金)
        pnl = []
        open_buy = {}
        for _, tr in trips.sort_values("date").iterrows():
            if tr["action"] == "BUY":
                open_buy[tr["symbol"]] = tr
            elif tr["action"] == "SELL" and tr["symbol"] in open_buy:
                b = open_buy.pop(tr["symbol"])
                pnl.append(tr["price"] * tr["shares"] - b["price"] * b["shares"]
                           - tr["commission"] - b["commission"])
        rows.append({
            "window": f"{start.date()} ~ {end.date()}",
            "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "calmar": m["calmar"],
            "n_trips": len(pnl),
            "win_rate": (sum(p > 0 for p in pnl) / len(pnl)) if pnl else float("nan"),
        })

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "output" / "rand20"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    print(f"\n=== {N_WINDOWS} 个随机一年期窗口汇总(V3 当前配置)===")
    print(df.to_string(index=False, formatters={
        "total_return": lambda v: f"{v:+.1%}",
        "max_drawdown": lambda v: f"{v:.1%}",
        "sharpe": lambda v: f"{v:.2f}",
        "calmar": lambda v: f"{v:.2f}",
        "win_rate": lambda v: f"{v:.0%}"}))
    print(f"\n收益: 均值 {df['total_return'].mean():+.1%} | 中位 {df['total_return'].median():+.1%} | "
          f"最差 {df['total_return'].min():+.1%} | 最好 {df['total_return'].max():+.1%}")
    print(f"回撤: 最差 {df['max_drawdown'].min():.1%} | 均值 {df['max_drawdown'].mean():.1%}")
    print(f"夏普: 均值 {df['sharpe'].mean():.2f} | 最差 {df['sharpe'].min():.2f}")
    print(f"盈利窗口占比: {(df['total_return'] > 0).mean():.0%}({(df['total_return'] > 0).sum()}/{len(df)})")
    print(f"\n汇总: {out}/summary.csv")


if __name__ == "__main__":
    main()
