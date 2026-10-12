#!/usr/bin/env python
"""溢价入场安检规则对照实验(2026-10-12):固定 7% 门槛 vs 相对均值门槛。

背景:QDII 额度紧张使 513100 溢价自 2026-06-12 起连续 78 个交易日 >7%,
固定的 premium_max=0.07 在当前 regime 下等价于无限期禁买纳指。
新规则 premium_window=15:当日溢价 > 前 15 个交易日溢价均值(不含当日)才拦截,
即只拦"溢价异常抬升日",不拦"溢价高位企稳日"。

对照纪律同 eval_zhihu.py:全周期 + IS(2019-2022)/ OOS(2023-2026)双口径。

运行:PYTHONPATH=src python scripts/eval_premium_rule.py
"""

from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import attach_premium, fetch_universe
from etf_rotation.metrics import perf_metrics

V3 = dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25)
SPLIT = pd.Timestamp("2023-01-01")
OUT_DIR = Path(__file__).resolve().parents[1] / "output"


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
    data = attach_premium(fetch_universe(list(UNIVERSE)))

    variants = [
        ("无溢价规则(对照)", StrategyConfig(**V3)),
        ("现行:溢价≤7%(实盘)", StrategyConfig(**V3, premium_max=0.07)),
        ("新规则:>15日均值拦截", StrategyConfig(**V3, premium_window=15)),
        ("双闸门:≤7% 且 ≤15日均值", StrategyConfig(**V3, premium_max=0.07, premium_window=15)),
        ("敏感性:>10日均值", StrategyConfig(**V3, premium_window=10)),
        ("敏感性:>20日均值", StrategyConfig(**V3, premium_window=20)),
    ]
    rows = [row(name, run_backtest(data, cfg)) for name, cfg in variants]
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fmt = out.copy()
    for c in out.columns:
        if c not in ("变体", "调仓次数"):
            fmt[c] = out[c].map(lambda x: f"{x:.2%}" if "回撤" in c or "占比" in c or "年化" in c else f"{x:.2f}")
    print("\n=== 溢价安检规则对照(全周期 2019-01 → 今;IS=2019-2022,OOS=2023-2026)===\n")
    print(fmt.to_string(index=False))
    out.to_csv(OUT_DIR / "premium_rule_variants.csv", index=False)

    # 新规则下 513100 的可买入日占比,验证"解禁"效果
    print("\n=== 513100 各规则下的入场放行率(信号日占比)===")
    p = data["513100"].set_index("date")["premium"].dropna()
    base15 = p.rolling(15).mean().shift(1)
    for label, allow in [
        ("现行 ≤7%", p <= 0.07),
        ("新规则 ≤15日均值", p <= base15),
        ("双闸门", (p <= 0.07) & (p <= base15)),
    ]:
        recent = allow[allow.index >= "2026-01-01"]
        print(f"  {label:<16} 全周期 {allow.mean():.1%}  | 2026 以来 {recent.mean():.1%}")


if __name__ == "__main__":
    main()
