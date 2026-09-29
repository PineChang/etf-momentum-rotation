#!/usr/bin/env python
"""独立审计脚本(检查官视角,不信任项目自带验证,全部从缓存数据重算)。

审计项:
  A. 未来函数截断实验:在全数据回测 vs 截断数据回测,截断日之前的交易/净值必须完全一致。
  B. 复权一致性:qfq 缓存与 hfq 缓存的日收益序列应一致(检验 qfq 增量更新是否造成除息断点)。
  C. rand10 复算:从缓存数据重跑 V3 全周期回测,复算 10 个随机窗口指标,与 summary.csv 对账。
  D. 样本内/外检验:V3 止盈参数(0.25)在 IS(2019-2022)与 OOS(2023-2026)是否都有效;
     止盈参数平台;README 声称的"三个分时段止盈全优于不止盈"复验。
  E. 调仓阈值负分退化统计:阈值比较 scores[challenger] > scores[weakest]*1.5 在负分区间的占比。
  F. 夏普口径:cagr 口径 vs 均值收益口径。

运行:PYTHONPATH=src python scripts/audit_independent.py
"""

from dataclasses import replace
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from etf_rotation import backtest as bt_mod
from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.metrics import perf_metrics
from etf_rotation.scoring import combined_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
V3 = StrategyConfig(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25)
V2 = StrategyConfig(n_hold=2, trailing_stop=0.08, min_momentum=0.0)
BASE = StrategyConfig()
SPLIT = pd.Timestamp("2023-01-01")


def load_cached(adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    """直接读缓存 CSV,绕过网络,保证审计可复现。"""
    suffix = "" if adjust == "qfq" else f".{adjust}"
    out = {}
    for s in UNIVERSE:
        df = pd.read_csv(DATA_DIR / f"{s}{suffix}.csv", parse_dates=["date"])
        out[s] = df.sort_values("date").reset_index(drop=True)
    return out


def wm(res, a=None, b=None) -> dict:
    """窗口指标:切片→归一→perf_metrics。"""
    eq = res.equity
    if a is not None:
        eq = eq[eq.index >= a]
    if b is not None:
        eq = eq[eq.index <= b]
    eq = eq / eq.iloc[0]
    return perf_metrics(eq)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    data = load_cached("qfq")

    # ---------- A. 未来函数截断实验 ----------
    section("A. 未来函数截断实验(全量 vs 截断,截断日前必须逐笔一致)")
    full = run_backtest(data, V3)
    rng = np.random.default_rng(7)
    dates = full.equity.index
    cut_pts = sorted(rng.choice(len(dates) // 2, size=8, replace=False) + 100)
    all_ok = True
    for idx in cut_pts:
        cut = dates[idx]
        trunc = {s: df[df["date"] <= cut].reset_index(drop=True) for s, df in data.items()}
        r = run_backtest(trunc, V3)
        t_full = full.trades[full.trades["date"] <= cut].reset_index(drop=True)
        t_trunc = r.trades.reset_index(drop=True)
        eq_ok = np.allclose(full.equity.loc[:cut].to_numpy(), r.equity.to_numpy(), rtol=0, atol=1e-12)
        tr_ok = len(t_full) == len(t_trunc) and (
            t_full.empty or (t_full[["date", "action", "symbol", "price"]].equals(
                t_trunc[["date", "action", "symbol", "price"]]))
        )
        all_ok &= eq_ok and tr_ok
        print(f"  截断 {cut.date()}: 净值前缀一致={eq_ok}  交易逐笔一致={tr_ok}  ({len(t_trunc)} 笔)")
    print(f"  ==> 截断实验结论: {'PASS 无未来函数' if all_ok else 'FAIL 存在前视偏差!'}")

    # ---------- B. 复权一致性 ----------
    section("B. qfq 缓存 vs hfq 缓存 日收益一致性(检测除息断点/锚点错位)")
    for adjust, suffix in (("hfq(东财)", ".hfq"), ("hfq(腾讯)", ".tx_hfq")):
        print(f"  对照源: {adjust}")
        for s in UNIVERSE:
            p = DATA_DIR / f"{s}{suffix}.csv"
            if not p.exists():
                print(f"    {s}: 无缓存文件,跳过")
                continue
            hfq = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
            q = data[s].set_index("date")["close"]
            h = hfq.set_index("date")["close"]
            common = q.index.intersection(h.index)
            rq = q.loc[common].pct_change()
            rh = h.loc[common].pct_change()
            diff = (rq - rh).abs().dropna()
            bad = diff[diff > 0.005]
            print(f"    {s}: 共同交易日 {len(common)}, 收益最大偏差 {diff.max():.2e}, >0.5% 偏差天数 {len(bad)}"
                  + (f" 日期: {[str(d.date()) for d in bad.index[:5]]}" if len(bad) else ""))

    # ---------- C. rand10 复算 ----------
    section("C. rand10 复算(固定种子 42 重新生成窗口,与 summary.csv 对账)")
    random.seed(42)
    EARLIEST, LATEST = pd.Timestamp("2019-06-01"), pd.Timestamp("2025-09-28")
    span = (LATEST - EARLIEST).days
    starts = sorted(EARLIEST + pd.Timedelta(days=random.randint(0, span)) for _ in range(10))
    res = run_backtest(data, V3)
    reported = pd.read_csv(Path(__file__).resolve().parents[1] / "output/rand10/summary.csv")
    max_err = 0.0
    for i, st in enumerate(starts):
        en = st + pd.DateOffset(months=12)
        m = wm(res, a=st, b=en - pd.Timedelta(days=1))
        rep = reported.iloc[i]
        err = abs(m["total_return"] - rep["total_return"])
        max_err = max(max_err, err)
        print(f"  win{i + 1:02d} {st.date()}: 复算收益 {m['total_return']:+.2%} vs 报告 {rep['total_return']:+.2%}"
              f"  偏差 {err:.2e}  复算回撤 {m['max_drawdown']:.1%} vs {rep['max_drawdown']:.1%}")
    print(f"  ==> 10 窗口收益最大对账偏差: {max_err:.2e}  ({'一致' if max_err < 1e-9 else '不一致,需排查'})")
    # 窗口重叠度
    wins = [(st, st + pd.DateOffset(months=12)) for st in starts]
    overlaps = sum(1 for i in range(10) for j in range(i + 1, 10)
                   if wins[i][1] > wins[j][0] and wins[j][1] > wins[i][0])
    print(f"  ==> 10 个窗口中日历重叠的窗口对数: {overlaps}/45(重叠则非独立样本)")

    # ---------- D. 样本内/外:止盈参数审计 ----------
    section("D. 样本内/外检验(V3 止盈参数是全样本第 7 轮挑的,重点审计)")
    rows = []
    for tp in (None, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        cfg = replace(V2, take_profit=tp) if tp is not None else V2
        r = run_backtest(data, cfg)
        fullm, ism, oosm = wm(r), wm(r, b=SPLIT), wm(r, a=SPLIT)
        rows.append({"take_profit": "无" if tp is None else f"{tp:.0%}",
                     "全样本卡玛": fullm["calmar"], "全样本年化": fullm["cagr"], "全样本回撤": fullm["max_drawdown"],
                     "IS卡玛": ism["calmar"], "IS年化": ism["cagr"],
                     "OOS卡玛": oosm["calmar"], "OOS年化": oosm["cagr"], "OOS回撤": oosm["max_drawdown"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={
        **{k: lambda v: f"{v:.2f}" for k in ("全样本卡玛", "IS卡玛", "OOS卡玛")},
        **{k: lambda v: f"{v:+.1%}" for k in ("全样本年化", "IS年化", "OOS年化")},
        **{k: lambda v: f"{v:.1%}" for k in ("全样本回撤", "OOS回撤")}}))

    # README 声称:三个分时段止盈全优于不止盈 → 复验
    print("\n  README '三分时段止盈全优于不止盈' 复验:")
    for a, b in (("2019-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31"), ("2025-01-01", "2026-09-28")):
        m2 = wm(run_backtest(data, V2), a=pd.Timestamp(a), b=pd.Timestamp(b))
        m3 = wm(run_backtest(data, V3), a=pd.Timestamp(a), b=pd.Timestamp(b))
        print(f"    {a[:7]}~{b[:7]}: 无止盈卡玛 {m2['calmar']:.2f}(年化{m2['cagr']:+.1%}) vs "
              f"止盈卡玛 {m3['calmar']:.2f}(年化{m3['cagr']:+.1%})  止盈更优={m3['calmar'] > m2['calmar']}")

    # ---------- E. 调仓阈值负分退化统计 ----------
    section("E. 调仓阈值 'x1.5' 比较在负分区的退化统计(逻辑缺陷审计)")
    from etf_rotation import scoring as sc_mod
    log: list[dict] = []

    def logging_select(scores, current, threshold, n):
        scores = scores.dropna()
        if scores.empty:
            return set(current)
        ranked = scores.sort_values(ascending=False)
        target = {s for s in current if s in scores.index}
        for sym in ranked.index:
            if len(target) >= n:
                break
            target.add(sym)
        for _ in range(n):
            held = [s for s in target if s in current]
            challengers = [s for s in ranked.index if s not in target]
            if not held or not challengers:
                break
            weakest = min(held, key=lambda s: scores[s])
            log.append({"weakest_score": scores[weakest], "challenger_score": scores[challengers[0]],
                        "switched": bool(sc_mod._beats(scores[weakest], scores[challengers[0]], threshold))})
            if log[-1]["switched"]:
                target.discard(weakest)
                target.add(challengers[0])
            else:
                break
        return target

    orig = bt_mod.select_targets
    bt_mod.select_targets = logging_select
    run_backtest(data, V3)
    bt_mod.select_targets = orig
    lg = pd.DataFrame(log)
    neg = lg[lg["weakest_score"] < 0]
    inversion = neg[(neg["switched"]) & (neg["challenger_score"] < neg["weakest_score"])]
    print(f"  阈值比较总次数: {len(lg)}")
    print(f"  最弱在持者得分为负的比较次数: {len(neg)} ({len(neg) / max(len(lg), 1):.0%})")
    print(f"  其中'换到比在持者分数更低标的'的真逻辑反转次数: {len(inversion)}"
          f"(修复前旧公式为 54 次,应为 0)")
    print(f"  负分区比较中实际发生切换的比例: {neg['switched'].mean():.0%}"
          f"(修复后负分区同样要求显著超出,不再必然放行)")

    # ---------- F. 夏普口径 ----------
    section("F. 夏普口径审计(cagr 口径 vs 均值收益口径)")
    r = run_backtest(data, V3)
    ret = r.equity.pct_change().dropna()
    sharpe_mean = ret.mean() / ret.std(ddof=0) * np.sqrt(252)
    m = perf_metrics(r.equity)
    print(f"  V3 全样本: cagr口径夏普 {m['sharpe']:.2f}  vs  均值口径夏普 {sharpe_mean:.2f}"
          f"  (cagr>均值收益,前者略高,口径问题非 bug)")
    print(f"  V3 全样本年化 {m['cagr']:+.1%}  回撤 {m['max_drawdown']:.1%}  卡玛 {m['calmar']:.2f}")


if __name__ == "__main__":
    main()
