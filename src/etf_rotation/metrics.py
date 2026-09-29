"""回测绩效指标:年化收益、夏普、最大回撤、卡玛、换手与胜率。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def perf_metrics(equity: pd.Series, trades: pd.DataFrame | None = None, rf: float = 0.0) -> dict:
    """equity 为净值序列(初始 1)。rf 为年化无风险利率,默认 0。

    口径说明:
    - 年化收益(CAGR)按实际收益天数 len(equity)-1 折算(n 个净值点只有 n-1 个交易日);
    - 夏普用日均值收益年化(算术口径)而非 CAGR——CAGR 含波动拖累,会低估
      高波动策略的夏普、高估低波动策略,均值口径才是标准定义;
    - 往返交易盈亏扣除双边佣金。
    """
    ret = equity.pct_change().dropna()
    n_days = len(equity) - 1
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / n_days) - 1
    ann_vol = ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    ann_ret = ret.mean() * TRADING_DAYS
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    out = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "days": int(len(equity)),
    }

    if trades is not None and not trades.empty:
        # 按标的分别配对:BUY 开仓,同标的 SELL 与其最近未平仓 BUY 配对。
        # (1) 不能用位置对齐——窗口截取时首笔可能是 SELL(开仓在窗口外);
        # (2) 多持仓时不同标的的买卖交错,必须按标的各自配对。
        trips = []
        open_buys: dict = {}
        for _, t in trades.sort_values("date").iterrows():
            if t["action"] == "BUY":
                open_buys[t["symbol"]] = t
            elif t["action"] == "SELL" and t["symbol"] in open_buys:
                b = open_buys.pop(t["symbol"])
                pnl = (t["price"] * t["shares"] - b["price"] * b["shares"]
                       - float(t["commission"]) - float(b["commission"]))
                trips.append({
                    "buy_date": b["date"],
                    "sell_date": t["date"],
                    "pnl": pnl,
                })
        out["n_round_trips"] = len(trips)
        if trips:
            out["win_rate"] = float(np.mean([x["pnl"] > 0 for x in trips]))
            out["avg_hold_days"] = float(np.mean([(x["sell_date"] - x["buy_date"]).days for x in trips]))
    return out


def format_metrics(m: dict) -> str:
    lines = [
        f"总收益率     : {m['total_return']:.2%}",
        f"年化收益     : {m['cagr']:.2%}",
        f"年化波动     : {m['ann_vol']:.2%}",
        f"夏普比率     : {m['sharpe']:.2f}",
        f"最大回撤     : {m['max_drawdown']:.2%}",
        f"卡玛比率     : {m['calmar']:.2f}",
        f"回测天数     : {m['days']}",
    ]
    if "n_round_trips" in m:
        lines += [
            f"完整调仓次数 : {m['n_round_trips']}",
            f"单笔胜率     : {m['win_rate']:.2%}",
            f"平均持有天数 : {m['avg_hold_days']:.1f}",
        ]
    return "\n".join(lines)
