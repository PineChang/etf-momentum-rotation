"""metrics 往返交易配对逻辑测试(回归:窗口截取导致的首笔 SELL 错位)。"""

import pandas as pd
import pytest

from etf_rotation.metrics import perf_metrics


def trade(date, action, price, shares=100.0):
    return {"date": pd.Timestamp(date), "action": action, "symbol": "X",
            "price": price, "shares": shares, "commission": 0.0}


def equity(n=60):
    return pd.Series(range(1, n + 1), dtype=float)


class TestRoundTripPairing:
    def test_normal_alternating(self):
        trades = pd.DataFrame([
            trade("2025-01-01", "BUY", 10.0),
            trade("2025-01-11", "SELL", 12.0),   # +20%, 持有10天
            trade("2025-02-01", "BUY", 12.0),
            trade("2025-02-11", "SELL", 11.0),   # 亏损, 持有10天
        ])
        m = perf_metrics(equity(), trades)
        assert m["n_round_trips"] == 2
        assert m["win_rate"] == pytest.approx(0.5)
        assert m["avg_hold_days"] == pytest.approx(10.0)

    def test_window_starts_with_sell(self):
        """窗口截取场景:窗口内首笔是 SELL,其 BUY 在流水更前面——
        旧的位置配对会把 SELL 错配给窗口内第一笔 BUY,产生虚假胜率。"""
        trades = pd.DataFrame([
            trade("2025-09-01", "BUY", 10.0),    # 窗口外开仓
            trade("2025-10-10", "SELL", 9.0),    # 窗口内首笔:亏损 -10%
            trade("2025-10-10", "BUY", 20.0),
            trade("2025-10-29", "SELL", 19.0),   # 亏损 -5%
        ])
        m = perf_metrics(equity(), trades)
        assert m["n_round_trips"] == 2
        assert m["win_rate"] == pytest.approx(0.0)   # 两笔皆亏,不是虚假的 100%
        # 第一笔持有 39 天(跨窗口),第二笔 19 天
        assert m["avg_hold_days"] == pytest.approx(29.0)

    def test_unclosed_buy_not_counted(self):
        """末笔 BUY 未平仓:不计入往返次数。"""
        trades = pd.DataFrame([
            trade("2025-01-01", "BUY", 10.0),
            trade("2025-01-11", "SELL", 12.0),
            trade("2025-02-01", "BUY", 12.0),
        ])
        m = perf_metrics(equity(), trades)
        assert m["n_round_trips"] == 1
        assert m["win_rate"] == pytest.approx(1.0)

    def test_orphan_sell_ignored(self):
        """流水的 BUY 部分被截掉、只剩 SELL:无法配对则跳过,不造假数据。"""
        trades = pd.DataFrame([
            trade("2025-10-10", "SELL", 9.0),
            trade("2025-10-10", "BUY", 20.0),
            trade("2025-10-29", "SELL", 22.0),
        ])
        m = perf_metrics(equity(), trades)
        assert m["n_round_trips"] == 1
        assert m["win_rate"] == pytest.approx(1.0)
        assert m["avg_hold_days"] == pytest.approx(19.0)
