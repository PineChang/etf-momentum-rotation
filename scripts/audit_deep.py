#!/usr/bin/env python
"""第二轮深检脚本(2026-09-29):代码/数据法证 + 多重检验量化 + 收益来源拆解。

对应 docs/audit-report.md 第七节。四组检验:
  1. 逐笔交易取证 + 数据法证(OHLC/涨停/跳空分布)
  2. 未来数据扰动实验(截断点后数据换成随机游走垃圾,之前的结果必须不变)
  3. Deflated Sharpe(重现 408 组调参试验,量化多重检验;耗时约 10 分钟,可用 --fast 跳过)
  4. 躺平对照(同期买入持有,收益来源归因)

运行:PYTHONPATH=src python scripts/audit_deep.py [--fast]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import fetch_etf
from etf_rotation.metrics import perf_metrics

V3 = StrategyConfig(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25)
SPLIT = pd.Timestamp("2023-01-01")
CUT = pd.Timestamp("2022-06-30")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def is_sharpe(data, cfg) -> float:
    r = run_backtest(data, cfg)
    eq = r.equity[r.equity.index <= SPLIT]
    ret = (eq / eq.iloc[0]).pct_change().dropna()
    return float(ret.mean() / ret.std(ddof=0) * np.sqrt(252))


def main() -> None:
    fast = "--fast" in sys.argv
    data = {s: fetch_etf(s) for s in UNIVERSE}
    res = run_backtest(data, V3)
    trades = res.trades

    # ---------- 1. 交易取证 + 数据法证 ----------
    section("1. 逐笔交易取证 + 数据法证")
    bad_price = 0
    for _, tr in trades.iterrows():
        raw = data[tr["symbol"]]
        row = raw[raw["date"] == tr["date"]]
        if row.empty or abs(float(row["open"].iloc[0]) - tr["price"]) > 1e-9:
            bad_price += 1
    cash, pos = V3.initial_cash, {}
    for _, tr in trades.sort_values("date").iterrows():
        if tr["action"] == "BUY":
            cash -= tr["price"] * tr["shares"] + tr["commission"]
            pos[tr["symbol"]] = tr["shares"]
        else:
            cash += tr["price"] * tr["shares"] - tr["commission"]
            pos.pop(tr["symbol"], None)
    last_date = res.equity.index[-1]
    mv = sum(sh * float(data[s][data[s]["date"] == last_date]["close"].iloc[0])
             for s, sh in pos.items())
    eq_final = (cash + mv) / V3.initial_cash
    print(f"  成交笔数 {len(trades)},成交价与原始开盘价不符: {bad_price} 笔")
    print(f"  独立复算期末净值: {eq_final:.6f} vs 引擎 {res.equity.iloc[-1]:.6f}"
          f"  一致={abs(eq_final - res.equity.iloc[-1]) < 1e-9}")

    for s, df in data.items():
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        ohlc_bad = int(((o > h) | (o < l) | (c > h) | (c < l)).sum())
        gap = o / c.shift(1) - 1
        limit_up = gap[gap > 0.095]
        hits = 0
        for d in limit_up.index:
            if len(trades[(trades["date"] == d) & (trades["symbol"] == s)
                          & (trades["action"] == "BUY")]):
                hits += 1
        print(f"  {s}: OHLC越界 {ohlc_bad} | 涨停开盘 {len(limit_up)} 天,V3 撞单 {hits} 笔 | 最大跳空 {gap.max():+.1%}")
    buy_gaps = []
    for _, tr in trades.iterrows():
        raw = data[tr["symbol"]].set_index("date")
        i = raw.index.get_loc(tr["date"])
        buy_gaps.append(tr["price"] / raw["close"].iloc[i - 1] - 1)
    bg = pd.Series(buy_gaps)
    print(f"  成交日跳空: |gap|>1% 占 {(bg.abs() > 0.01).mean():.0%},>2% 占 {(bg.abs() > 0.02).mean():.0%}")

    # ---------- 2. 未来数据扰动实验 ----------
    section(f"2. 未来数据扰动实验({CUT.date()} 后换成随机游走垃圾)")
    rng = np.random.default_rng(0)
    corrupt = {}
    for s, df in data.items():
        d = df.copy()
        m = d["date"] > CUT
        base = d.loc[~m, "close"].iloc[-1]
        junk = base * np.exp(np.cumsum(rng.normal(0, 0.02, int(m.sum()))))
        d.loc[m, "close"] = junk
        d.loc[m, "open"] = junk
        d.loc[m, "high"] = junk * 1.01
        d.loc[m, "low"] = junk * 0.99
        corrupt[s] = d
    r2 = run_backtest(corrupt, V3)
    e_ok = np.allclose(res.equity.loc[:CUT].to_numpy(), r2.equity.loc[:CUT].to_numpy(), atol=1e-12)
    t1 = res.trades[res.trades["date"] <= CUT]
    t2 = r2.trades[r2.trades["date"] <= CUT]
    print(f"  扰动前净值一致: {e_ok} | 交易笔数一致: {len(t1) == len(t2)}({len(t1)} 笔)")
    print(f"  ==> {'PASS 无未来信息' if e_ok and len(t1) == len(t2) else 'FAIL!'}")

    # ---------- 3. Deflated Sharpe ----------
    section("3. 多重检验量化(Deflated Sharpe)")
    if fast:
        print("  (--fast 跳过 408 组参数扫描;完整跑约 10 分钟)")
    else:
        trial = []
        w_grid = [(i * 0.1, j * 0.1, (10 - i - j) * 0.1) for i in range(11) for j in range(11 - i)]
        for wb, ws, we in w_grid:
            for th in (1.0, 1.2, 1.5, 2.0):
                trial.append(is_sharpe(data, StrategyConfig(
                    w_bias=wb, w_slope=ws, w_efficiency=we, rebalance_threshold=th)))
        for trail in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15):
            for tp in (None, 0.15, 0.20, 0.25, 0.30, 0.40):
                for nh in (1, 2, 3):
                    trial.append(is_sharpe(data, StrategyConfig(
                        n_hold=nh, trailing_stop=trail, min_momentum=0.0, take_profit=tp)))
        ts = pd.Series(trial)
        v3_is = is_sharpe(data, V3)
        e_max = ts.mean() + ts.std(ddof=1) * np.sqrt(2 * np.log(len(ts)))
        print(f"  试验数 N = {len(ts)};IS 夏普 均值 {ts.mean():.2f} / 最大 {ts.max():.2f}")
        print(f"  V3 IS 夏普 {v3_is:.2f}(排名 {int((ts > v3_is).sum()) + 1}/{len(ts)})")
        print(f"  试 {len(ts)} 次的期望最大虚假夏普 ≈ {e_max:.2f}")
        print(f"  ==> V3 IS 夏普 {'仍高于' if v3_is > e_max else '不高于'}纯数据挖掘期望上限")
    r = run_backtest(data, V3)
    eq = r.equity[r.equity.index >= SPLIT]
    ret = (eq / eq.iloc[0]).pct_change().dropna()
    print(f"  V3 OOS 夏普(2023-2026,选参不可见): {ret.mean() / ret.std(ddof=0) * np.sqrt(252):.2f}")

    # ---------- 4. 躺平对照 ----------
    section("4. 躺平对照(同期买入持有,收益来源归因)")
    m = perf_metrics(res.equity, res.trades)
    print(f"  {'V3 策略':<12} 年化 {m['cagr']:+.1%}  回撤 {m['max_drawdown']:.1%}  卡玛 {m['calmar']:.2f}")
    eqs = {}
    for s, name in UNIVERSE.items():
        df = data[s]
        eq = df["close"] / df["close"].iloc[0]
        dd = (eq / eq.cummax() - 1).min()
        yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365
        cagr = (1 + eq.iloc[-1] - 1) ** (1 / yrs) - 1
        print(f"  {name:<12} 年化 {cagr:+.1%}  回撤 {dd:.1%}  卡玛 {cagr / abs(dd):.2f}  总收益 {eq.iloc[-1] - 1:+.0%}")
        eqs[s] = pd.Series(eq.values, index=df["date"].values)
    common = sorted(set.intersection(*[set(e.index) for e in eqs.values()]))
    bench = sum(e.loc[common] for e in eqs.values()) / 4
    dd = (bench / bench.cummax() - 1).min()
    yrs = (common[-1] - common[0]).days / 365
    cagr = bench.iloc[-1] ** (1 / yrs) - 1
    print(f"  {'等权躺平':<12} 年化 {cagr:+.1%}  回撤 {dd:.1%}  卡玛 {cagr / abs(dd):.2f}  总收益 {bench.iloc[-1] - 1:+.0%}")


if __name__ == "__main__":
    main()
