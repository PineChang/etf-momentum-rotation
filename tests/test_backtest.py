"""回测引擎集成测试:合成数据验证轮动行为与指标正确性。"""

import numpy as np
import pandas as pd
import pytest

from etf_rotation.backtest import run_backtest
from etf_rotation.config import StrategyConfig
from etf_rotation.metrics import perf_metrics


def synth_ohlc(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame({"date": dates, "open": s, "high": s, "low": s, "close": s, "volume": 1e6})


@pytest.fixture
def two_assets():
    up = synth_ohlc([10 + 0.1 * i for i in range(80)])
    flat = synth_ohlc([10.0] * 80)
    return {"UP": up, "FLAT": flat}


class TestBacktestEngine:
    def test_picks_uptrend_and_grows(self, two_assets):
        cfg = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                             commission=0.0, initial_cash=100_000)
        res = run_backtest(two_assets, cfg)
        assert not res.trades.empty
        first_buy = res.trades.iloc[0]
        assert first_buy["action"] == "BUY" and first_buy["symbol"] == "UP"
        assert res.equity.iloc[-1] > 1.0
        # 强势资产得分恒高于横盘资产,阈值机制下不应换仓
        assert (res.trades["symbol"] == "UP").all()
        assert len(res.trades[res.trades["action"] == "SELL"]) == 0

    def test_no_lookahead(self, two_assets):
        cfg = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10, commission=0.0)
        res = run_backtest(two_assets, cfg)
        first_buy_date = res.trades.iloc[0]["date"]
        # 首笔成交必须发生在 warmup 完成之后(信号日次日开盘)
        assert first_buy_date >= two_assets["UP"]["date"].iloc[cfg.warmup_bars]

    def test_compounding(self, two_assets):
        """无佣金、严格上涨资产:净值应等于 close_t/close_buy,体现复利全额再投。"""
        cfg = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                             commission=0.0, initial_cash=100_000)
        res = run_backtest(two_assets, cfg)
        buy_price = res.trades.iloc[0]["price"]
        final_close = two_assets["UP"]["close"].iloc[-1]
        assert res.equity.iloc[-1] == pytest.approx(final_close / buy_price, rel=1e-9)

    def test_commission_reduces_return(self, two_assets):
        base = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10, commission=0.0)
        taxed = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10, commission=0.001)
        assert run_backtest(two_assets, taxed).equity.iloc[-1] < run_backtest(two_assets, base).equity.iloc[-1]


class TestMetrics:
    def test_known_curve(self):
        """250 天(251 个净值点)从 1 线性涨到 2:总收益 100%,
        年化 = 2^(252/250) - 1(按实际收益天数 n-1 折算),回撤=0。"""
        equity = pd.Series(np.linspace(1, 2, 251))
        m = perf_metrics(equity)
        assert m["total_return"] == pytest.approx(1.0, rel=1e-9)
        assert m["cagr"] == pytest.approx(2.0 ** (252 / 250) - 1, rel=1e-9)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)

    def test_max_drawdown(self):
        equity = pd.Series([1.0, 1.5, 0.75, 1.2])
        m = perf_metrics(equity)
        assert m["max_drawdown"] == pytest.approx(-0.5, rel=1e-9)

    def test_round_trip_stats(self, two_assets):
        cfg = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10, commission=0.0)
        res = run_backtest(two_assets, cfg)
        m = perf_metrics(res.equity, res.trades)
        assert m["days"] == 80
