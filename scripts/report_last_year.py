#!/usr/bin/env python
"""V3 改良策略一年期图文报告生成器(可复用)。

单次窗口:NO_PROXY='*' PYTHONPATH=src python scripts/report_last_year.py
多窗口批量:见 scripts/report_random10.py

产出(指定目录下):
- report_daily.csv       每个交易日:净值、当日盈亏(元/%)、累计盈亏、回撤、持仓
- report_trades.csv      交易清单
- report_round_trips.csv 往返交易(按标的 FIFO 配对)
- report_daily.png       三联图:净值(带买卖点)/每日盈亏/回撤
- report.md              汇总报告(胜率等指标 + 月度统计)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import UNIVERSE, StrategyConfig
from etf_rotation.data import fetch_universe
from etf_rotation.metrics import format_metrics, perf_metrics

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

WINDOW_START = pd.Timestamp("2025-09-28")
CFG = StrategyConfig(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25)  # V3
OUT = Path(__file__).resolve().parents[1] / "output"


def plot_daily(daily: pd.DataFrame, trades: pd.DataFrame, out: Path, title: str) -> Path:
    """三联图:净值(带买卖点)/ 每日盈亏红绿柱 / 回撤。"""
    dates = pd.to_datetime(daily["日期"])
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]}
    )

    # 1) 净值曲线 + 买卖点
    ax1.plot(dates, daily["净值"], lw=1.6, color="#c0392b")
    ax1.axhline(1.0, lw=0.6, color="gray", alpha=0.5)
    t = trades[trades["date"] >= dates.iloc[0]]
    nav_map = dict(zip(dates, daily["净值"]))
    for _, r in t.iterrows():
        y = nav_map.get(r["date"])
        if y is None:
            continue
        is_buy = r["action"] == "BUY"
        ax1.scatter(r["date"], y, marker="^" if is_buy else "v",
                    color="#2ecc71" if is_buy else "#e74c3c", s=28, zorder=5, alpha=0.85)
    ax1.set_title(f"{title}:净值曲线(▲买入 ▼卖出)")
    ax1.set_ylabel("净值(期初=1)")
    ax1.legend(["净值"], loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # 2) 每日盈亏红绿柱(A股配色:红涨绿跌)
    pnl = daily["当日盈亏(元)"].fillna(0)
    colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in pnl]
    ax2.bar(dates, pnl, color=colors, width=1.0, alpha=0.8)
    ax2.axhline(0, lw=0.6, color="black", alpha=0.4)
    ax2.set_title("每日浮动盈亏(元)")
    ax2.grid(alpha=0.3)

    # 3) 回撤
    ax3.fill_between(dates, daily["回撤"], 0, color="#2c3e50", alpha=0.6)
    ax3.axhline(-0.20, ls="--", lw=0.8, color="black", alpha=0.5)
    ax3.set_title("回撤(目标线 -20%)")
    ax3.grid(alpha=0.3)
    ax3.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pair_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """按标的 FIFO 配对 BUY→SELL,计算每笔往返盈亏。"""
    trips, open_buys = [], {}
    for _, t in trades.sort_values("date").iterrows():
        if t["action"] == "BUY":
            open_buys[t["symbol"]] = t
        elif t["action"] == "SELL" and t["symbol"] in open_buys:
            b = open_buys.pop(t["symbol"])
            pnl = t["price"] * t["shares"] - b["price"] * b["shares"]
            trips.append({
                "买入日": b["date"].date(), "卖出日": t["date"].date(),
                "标的": f'{UNIVERSE[t["symbol"]]}({t["symbol"]})',
                "买入价": b["price"], "卖出价": t["price"],
                "份额": t["shares"], "盈亏(元)": round(pnl, 2),
                "收益率": pnl / (b["price"] * b["shares"]),
                "持有天数": (t["date"] - b["date"]).days,
            })
    return pd.DataFrame(trips)


def generate_report(
    window_start: pd.Timestamp,
    out_dir: Path,
    res,
    cfg: StrategyConfig = CFG,
    window_months: int = 12,
) -> dict:
    """对回测结果 res 截取 [window_start, window_start+window_months) 窗口,生成图文报告。

    返回汇总指标 dict(供多窗口汇总)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    window_end = window_start + pd.DateOffset(months=window_months)
    label = f"{window_start.date()} → {(window_end - pd.Timedelta(days=1)).date()}"

    # ---------- 每日浮动盈亏(窗口内) ----------
    eq = res.equity[(res.equity.index >= window_start) & (res.equity.index < window_end)] * cfg.initial_cash
    hold = res.holdings[(res.holdings.index >= window_start) & (res.holdings.index < window_end)]
    nav = eq / eq.iloc[0]                                # 净值(窗口首日=1)
    value = eq
    daily = pd.DataFrame({
        "日期": nav.index.date,
        "市值(元)": value.round(2),
        "当日盈亏(元)": value.diff().round(2),
        "当日收益率": value.pct_change(),
        "净值": nav.round(4),
        "累计收益率": nav - 1,
        "回撤": (nav / nav.cummax() - 1),
        "持仓": hold.map(lambda h: "+".join(UNIVERSE[s] for s in h.split("+")) if isinstance(h, str) else "空仓"),
    })
    daily.to_csv(out_dir / "report_daily.csv", index=False)

    # ---------- 交易清单与往返配对 ----------
    # 交易清单按窗口过滤(开仓回看 200 天,保证窗口内卖出的 SELL 能对上它的 BUY)
    trades = res.trades.copy()
    trades = trades[(trades["date"] >= window_start - pd.Timedelta(days=200))
                    & (trades["date"] < window_end)]
    trades["名称"] = trades["symbol"].map(UNIVERSE)
    trades.to_csv(out_dir / "report_trades.csv", index=False)
    trips = pair_round_trips(trades)
    trips = trips[(trips["卖出日"] >= window_start.date()) & (trips["卖出日"] < window_end.date())]
    trips.to_csv(out_dir / "report_round_trips.csv", index=False)

    # ---------- 汇总指标 ----------
    m = perf_metrics(nav)                                # 净值类指标
    win_rate = (trips["盈亏(元)"] > 0).mean() if len(trips) else float("nan")

    monthly = nav.resample("ME").last()
    monthly_ret = monthly.pct_change()
    monthly_ret.iloc[0] = monthly.iloc[0] - 1
    monthly_df = pd.DataFrame({"月份": monthly_ret.index.strftime("%Y-%m"), "月收益": monthly_ret.values})

    plot_daily(daily, trades, out_dir / "report_daily.png", title=f"V3 策略 {label}")

    hold_days = (hold.notna()).sum()
    trip_lines = (
        [
            f"- 往返交易(窗口内完成):{len(trips)} 笔,胜率 **{win_rate:.1%}**,"
            f"平均持有 {trips['持有天数'].mean():.1f} 天",
            f"- 单笔最大盈利:{trips['盈亏(元)'].max():+,.0f} 元 | 单笔最大亏损:{trips['盈亏(元)'].min():+,.0f} 元",
        ]
        if len(trips) else ["- 窗口内无完成往返"]
    )
    lines = [
        f"# V3 改良策略一年期报告({label})",
        "",
        "参数:V3 = `n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25`(其余同基础版默认)",
        f"初始资金:{cfg.initial_cash:,.0f} 元 | 佣金:双边 {cfg.commission:.4%} | 数据:akshare 前复权",
        "",
        "![每日明细](report_daily.png)",
        "",
        "## 核心指标",
        "",
        "```",
        format_metrics(m),
        "```",
        "",
        f"- 期末市值:**{value.iloc[-1]:,.0f} 元**(期初 {value.iloc[0]:,.0f} 元)",
        f"- 持仓天数占比:{hold_days / len(hold):.1%}(其余为空仓/半仓)",
        *trip_lines,
        "",
        "## 月度收益",
        "",
        monthly_df.to_markdown(index=False, floatfmt="+.2%"),
        "",
        "## 往返交易明细",
        "",
        trips.to_markdown(index=False, floatfmt=(".3f", ".3f", ".1f", "+,.0f", "+.2%", "d")) if len(trips) else "(无)",
        "",
        f"每日浮动盈亏明细见 `report_daily.csv`(共 {len(daily)} 个交易日);"
        f"窗口相关委托流水(开仓回看 200 天)见 `report_trades.csv`。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "window": label,
        "total_return": m["total_return"], "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"], "calmar": m["calmar"],
        "n_trips": len(trips), "win_rate": win_rate,
        "out_dir": str(out_dir),
    }


def main() -> None:
    data = fetch_universe(list(UNIVERSE))
    res = run_backtest(data, CFG)
    s = generate_report(WINDOW_START, OUT, res)
    print(f"收益 {s['total_return']:+.1%} | 回撤 {s['max_drawdown']:.1%} | 夏普 {s['sharpe']:.2f} | "
          f"往返 {s['n_trips']} 笔 | 胜率 {s['win_rate']:.1%}")
    print(f"报告: {s['out_dir']}/report.md")


if __name__ == "__main__":
    main()
