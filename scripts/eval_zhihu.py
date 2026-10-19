#!/usr/bin/env python
"""知乎三点优化对照实验(2026-09-29):动态回溯窗口 / 溢价率过滤 / 急跌过滤。

来源:https://zhuanlan.zhihu.com/p/2008662097983211095
对照纪律:全周期 + IS(2019-2022,与 V3 选参同期)/ OOS(2023-2026,选参不可见)双口径,
防止在 IS 上再做过拟合。V3 基线 = n_hold=2 + trail8 + min_mom0 + tp25。

溢价率 = 未复权收盘 / 单位净值 - 1(data/{sym}.raw.csv + {sym}.nav.csv,腾讯+东财)。

运行:PYTHONPATH=src python scripts/eval_zhihu.py
"""

from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import attach_premium, fetch_universe
from etf_rotation.metrics import perf_metrics

V3 = dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25)
SPLIT = pd.Timestamp("2023-01-01")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_with_premium() -> dict[str, pd.DataFrame]:
    """qfq 行情 + 注入 premium 列。

    走 canonical 实现 `data.attach_premium`(未复权收盘/单位净值-1,按信号日前
    最新已公布净值注入,保证 PIT)。本脚本早期版本在这里自行 merge、没有时移,等价于在信号日就用到
    当晚才公布的净值 —— 那一版的对照表含 1 日前视,数字已被替换(见
    docs/audit-report.md 第八节的修订说明)。
    """
    return attach_premium(fetch_universe(list(UNIVERSE)))


def seg_metrics(res, start=None, end=None) -> dict:
    eq = res.equity
    if start is not None:
        eq = eq[eq.index >= start]
    if end is not None:
        eq = eq[eq.index <= end]
    m = perf_metrics(eq)
    m["n_trades"] = len(res.trades)
    m["cash_pct"] = float(res.holdings.isna().mean())
    return m


def row(name: str, res) -> dict:
    full = seg_metrics(res)
    is_ = seg_metrics(res, end=SPLIT - pd.Timedelta(days=1))
    oos = seg_metrics(res, start=SPLIT)
    return {
        "变体": name,
        "年化": full["cagr"], "回撤": full["max_drawdown"], "夏普": full["sharpe"], "卡玛": full["calmar"],
        "IS年化": is_["cagr"], "IS卡玛": is_["calmar"],
        "OOS年化": oos["cagr"], "OOS回撤": oos["max_drawdown"], "OOS夏普": oos["sharpe"], "OOS卡玛": oos["calmar"],
        "调仓次数": full["n_trades"], "空仓占比": full["cash_pct"],
    }


def main() -> None:
    data = load_with_premium()

    variants = [
        ("V3 基线", StrategyConfig(**V3)),
        ("①动窗口(10-60)", StrategyConfig(**V3, dynamic_lookback=True)),
        ("①动窗口(15-40)", StrategyConfig(**V3, dynamic_lookback=True, lb_min=15, lb_max=40)),
        ("③急跌过滤", StrategyConfig(**V3, drop_filter=True)),
        ("②溢价≤1%", StrategyConfig(**V3, premium_max=0.01)),
        ("②溢价≤0.5%", StrategyConfig(**V3, premium_max=0.005)),
        ("②溢价≤2%", StrategyConfig(**V3, premium_max=0.02)),
        ("①+③", StrategyConfig(**V3, dynamic_lookback=True, drop_filter=True)),
        ("①+③+②(1%)", StrategyConfig(**V3, dynamic_lookback=True, drop_filter=True, premium_max=0.01)),
    ]
    rows = [row(name, run_backtest(data, cfg)) for name, cfg in variants]
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    fmt = out.copy()
    for c in out.columns:
        if c not in ("变体", "调仓次数"):
            fmt[c] = out[c].map(lambda x: f"{x:.2%}" if "回撤" in c or "占比" in c or "年化" in c else f"{x:.2f}")
    print("\n=== 知乎三点优化对照(全周期 2019-01 → 2026-09;IS=2019-2022,OOS=2023-2026)===\n")
    print(fmt.to_string(index=False))
    out.to_csv(DATA_DIR.parent / "output" / "zhihu_optimizations.csv", index=False)

    # 溢价率本身长什么样(判断过滤是否有实际触发空间)
    print("\n=== 各标的溢价率分布(2019-2026)===")
    for sym, name in UNIVERSE.items():
        p = data[sym]["premium"].dropna()
        print(f"  {name:<10} 中位 {p.median():+.2%}  P90 {p.quantile(0.9):+.2%}  最大 {p.max():+.2%}  >1% 天数占比 {(p > 0.01).mean():.1%}")


if __name__ == "__main__":
    main()
