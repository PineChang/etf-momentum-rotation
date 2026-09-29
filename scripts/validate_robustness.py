#!/usr/bin/env python
"""推荐方案(双持仓+移动止损+动量过滤)有效性验证套件。

回答两个问题:
1. 有没有未来函数?(复权方式审计 + 信号/成交时点审计)
2. 会不会过拟合?(样本外 walk-forward + 参数平台 + 股票池替换)

运行:NO_PROXY='*' PYTHONPATH=src python scripts/validate_robustness.py
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import DATA_DIR, fetch_universe
from etf_rotation.metrics import perf_metrics

RECOMMENDED = StrategyConfig(n_hold=2, trailing_stop=0.08, min_momentum=0.0)
BASELINE = StrategyConfig()

SPLIT = pd.Timestamp("2023-01-01")  # 样本内/外分界
PCT = lambda v: f"{v:.1%}"


def fetch_tx(symbol: str, adjust: str) -> pd.DataFrame:
    """腾讯行情接口(东财限流时的备用源),分页拉取全历史。

    动量因子均为尺度不变量(归一化/对数收益),与东财复权基准不同不影响结论。
    """
    import json
    import time
    import urllib.request

    cache = DATA_DIR / f"{symbol}.tx_{adjust}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    prefix = ("sh" if symbol[0] == "5" else "sz") + symbol
    key = f"{adjust}day"
    frames = []
    for end in ("2020-06-01", "2022-06-01", "2024-06-01", "2026-09-28"):
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
               f"param={prefix},day,2018-06-01,{end},640,{adjust}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    payload = json.loads(r.read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        node = payload["data"][prefix]
        rows = node.get(key) or node.get("day") or []
        frames.append(pd.DataFrame(rows))
        time.sleep(1)

    df = pd.concat(frames).drop_duplicates(0)
    df.columns = ["date", "open", "close", "high", "low", "volume"][: len(df.columns)]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def fetch_universe_tx(symbols: list[str], adjust: str) -> dict[str, pd.DataFrame]:
    return {s: fetch_tx(s, adjust) for s in symbols}


def metrics_in_window(res, a=None, b=None):
    eq = res.equity
    if a:
        eq = eq[eq.index >= a]
    if b:
        eq = eq[eq.index <= b]
    eq = eq / eq.iloc[0]
    m = perf_metrics(eq)
    return m


def row(tag, m):
    return {"case": tag, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"], "calmar": m["calmar"]}


def show(df, title):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, formatters={
        "cagr": PCT, "max_dd": PCT,
        "sharpe": lambda v: f"{v:.2f}", "calmar": lambda v: f"{v:.2f}"}))


def main():
    rows = []

    # ---- 1. 未来函数:前复权(qfq,除息改写历史=轻微未来信息) vs 后复权(hfq,无) ----
    # hfq 用腾讯源(东财 hfq 接口限流);因子尺度不变,跨源比较仍有效
    for adjust, fetcher in (("qfq(东财)", lambda syms, adj: fetch_universe(syms)),
                            ("hfq(腾讯)", fetch_universe_tx)):
        adj = adjust[:3]
        data = fetcher(list(UNIVERSE), adj)
        for name, cfg in [("原文策略", BASELINE), ("推荐方案", RECOMMENDED)]:
            m = perf_metrics(run_backtest(data, cfg).equity)
            rows.append(row(f"{adjust}/{name}", m))
    show(pd.DataFrame(rows), "检验1 复权方式(qfq 含轻微未来信息,hfq 无)——结论应无实质差异")

    data = fetch_universe(list(UNIVERSE))  # 后续用东财 qfq 缓存

    # ---- 2. 样本外 walk-forward:样本内选参,样本外验证 ----
    # 用一个小型候选池模拟"当时选参"的过程
    candidates = [
        ("原文", BASELINE),
        ("top2裸", replace(BASELINE, n_hold=2)),
        ("推荐(trail8)", RECOMMENDED),
        ("trail6", replace(RECOMMENDED, trailing_stop=0.06)),
        ("trail10", replace(RECOMMENDED, trailing_stop=0.10)),
        ("trail12", replace(RECOMMENDED, trailing_stop=0.12)),
    ]
    rows = []
    for name, cfg in candidates:
        res = run_backtest(data, cfg)
        is_m = metrics_in_window(res, b=SPLIT)          # 2019-2022 样本内
        oos_m = metrics_in_window(res, a=SPLIT)         # 2023-2026 样本外(选参时不可见)
        rows.append({"case": name,
                     "IS_cagr": is_m["cagr"], "IS_dd": is_m["max_drawdown"], "IS_calmar": is_m["calmar"],
                     "OOS_cagr": oos_m["cagr"], "OOS_dd": oos_m["max_drawdown"], "OOS_calmar": oos_m["calmar"]})
    df = pd.DataFrame(rows)
    print("\n=== 检验2 Walk-Forward:样本内(2019-2022) vs 样本外(2023-2026) ===")
    print(df.to_string(index=False, formatters={
        "IS_cagr": PCT, "IS_dd": PCT, "OOS_cagr": PCT, "OOS_dd": PCT,
        "IS_calmar": lambda v: f"{v:.2f}", "OOS_calmar": lambda v: f"{v:.2f}"}))
    is_best = df.loc[df["IS_calmar"].idxmax(), "case"]
    print(f"若仅用样本内数据选参,最优为 [{is_best}];其样本外表现见上表。")

    # ---- 3. 参数平台:移动止损邻域(过拟合策略在邻域上应剧烈波动) ----
    rows = []
    for trail in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15):
        cfg = replace(BASELINE, n_hold=2, trailing_stop=trail, min_momentum=0.0)
        res = run_backtest(data, cfg)
        m = perf_metrics(res.equity)
        oos = metrics_in_window(res, a=SPLIT)
        rows.append({"case": f"trail={trail:.0%}", "cagr": m["cagr"], "max_dd": m["max_drawdown"],
                     "calmar": m["calmar"], "OOS_cagr": oos["cagr"], "OOS_dd": oos["max_drawdown"]})
    df = pd.DataFrame(rows)
    print("\n=== 检验3 参数平台:trailing_stop 邻域(全周期 + 样本外) ===")
    print(df.to_string(index=False, formatters={
        "cagr": PCT, "max_dd": PCT, "OOS_cagr": PCT, "OOS_dd": PCT,
        "calmar": lambda v: f"{v:.2f}"}))

    # ---- 4. 股票池替换:原文池是作者事后选出的,换同类标的结论是否仍成立 ----
    pools = {
        "原文池(红利低波/创业板50/纳指/黄金)": ["512890", "159949", "513100", "518880"],
        "同类替换(红利/创业板/标普500/黄金)": ["510880", "159915", "513500", "518880"],
        "全新池(沪深300/中证500/国债/黄金)": ["510300", "510500", "511010", "518880"],
    }
    rows = []
    for pname, symbols in pools.items():
        d = fetch_universe_tx(symbols, "qfq")   # 统一腾讯源,池内口径一致
        for name, cfg in [("原文策略", BASELINE), ("推荐方案", RECOMMENDED)]:
            m = perf_metrics(run_backtest(d, cfg).equity)
            rows.append(row(f"{pname[:12]}../{name}", m))
    show(pd.DataFrame(rows), "检验4 股票池替换(排除'池子选得好'的幸存者偏差)")


if __name__ == "__main__":
    main()
