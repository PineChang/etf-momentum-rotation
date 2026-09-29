"""三因子单元测试:期望值由独立方法(np.polyfit / 手算)验证。"""

import numpy as np
import pandas as pd
import pytest

from etf_rotation.factors import (
    bias_momentum_score,
    dynamic_lookback_window,
    efficiency_momentum_score,
    slope_momentum_score,
)


def make_close(values):
    return pd.Series(values, dtype=float)


class TestBiasMomentum:
    def test_uptrend_positive(self):
        """价格加速上涨、均线滞后加剧,bias 持续走高 → 斜率为正。"""
        ma_n, mom = 3, 5
        close = make_close([10, 10, 10, 11, 12.5, 14.5, 17, 20])
        score = bias_momentum_score(close, ma_n, mom)
        assert score > 0

    def test_matches_polyfit(self):
        """与 np.polyfit 独立计算结果一致。"""
        ma_n, mom = 3, 6
        close = make_close([10, 10.5, 10.2, 11, 11.5, 10.8, 12, 12.5, 13, 12.8])
        bias = close / close.rolling(ma_n, min_periods=1).mean()
        bias_recent = bias.iloc[-mom:]
        y = (bias_recent / bias_recent.iloc[0]).to_numpy()
        x = np.arange(mom)
        expected = np.polyfit(x, y, 1)[0] * 10000
        assert bias_momentum_score(close, ma_n, mom) == pytest.approx(expected, rel=1e-9)

    def test_downtrend_negative(self):
        close = make_close([15, 14, 13, 12, 11, 10, 9, 8])
        assert bias_momentum_score(close, 3, 5) < 0

    def test_insufficient_data_nan(self):
        assert np.isnan(bias_momentum_score(make_close([1, 2, 3]), 5, 5))


class TestSlopeMomentum:
    def test_perfect_line(self):
        """close=[1,2,3,4,5]:归一化后 y=x,斜率=1,R²=1 → score=10000。"""
        close = make_close([1, 2, 3, 4, 5])
        assert slope_momentum_score(close, 5) == pytest.approx(10000.0, rel=1e-9)

    def test_flat_zero(self):
        """完全横盘:斜率 0 → score 0。"""
        close = make_close([5, 5, 5, 5, 5])
        assert slope_momentum_score(close, 5) == pytest.approx(0.0, abs=1e-12)

    def test_noisy_uptrend_less_than_clean(self):
        """同样涨幅,震荡路径的 R² 更低,得分应低于线性路径。"""
        clean = make_close([10, 10.5, 11, 11.5, 12])
        noisy = make_close([10, 11.4, 10.6, 11.9, 12])
        assert 0 < slope_momentum_score(noisy, 5) < slope_momentum_score(clean, 5)

    def test_downtrend_negative(self):
        close = make_close([5, 4, 3, 2, 1])
        assert slope_momentum_score(close, 5) < 0


class TestEfficiencyMomentum:
    def ohlc_from_close(self, closes):
        """用 close 构造 O=H=L=C 的最简 OHLC。"""
        s = pd.Series(closes, dtype=float)
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s})

    def test_monotonic_uptrend_efficiency_one(self):
        """单调上涨:总波动=净位移,效率系数=1,score=100*log(末/初)。"""
        closes = [1.0, 1.1, 1.21, 1.331]
        df = self.ohlc_from_close(closes)
        expected = 100 * np.log(closes[-1] / closes[0])
        assert efficiency_momentum_score(df, 4) == pytest.approx(expected, rel=1e-9)

    def test_round_trip_zero_momentum(self):
        """价格回到原点:动量为 0 → score 0(即使有大量波动)。"""
        closes = [1.0, 2.0, 1.0]
        df = self.ohlc_from_close(closes)
        assert efficiency_momentum_score(df, 3) == pytest.approx(0.0, abs=1e-12)

    def test_choppy_uptrend_lower_than_smooth(self):
        """同样净涨幅,震荡路径效率系数 <1,得分低于平滑路径。"""
        smooth = self.ohlc_from_close([1.0, 1.2, 1.44])
        choppy = self.ohlc_from_close([1.0, 1.8, 1.44])
        assert 0 < efficiency_momentum_score(choppy, 3) < efficiency_momentum_score(smooth, 3)

    def test_downtrend_negative(self):
        df = self.ohlc_from_close([2.0, 1.5, 1.0])
        assert efficiency_momentum_score(df, 3) < 0


class TestDynamicLookbackWindow:
    """知乎改进①:波动率自适应回溯窗口。"""

    @staticmethod
    def make_df(closes, hi_lo_spread=0.0):
        closes = np.asarray(closes, dtype=float)
        return pd.DataFrame({
            "open": closes,
            "high": closes * (1 + hi_lo_spread),
            "low": closes * (1 - hi_lo_spread),
            "close": closes,
        })

    def test_short_window_when_vol_spikes(self):
        """近 10 日波动骤升(相对 60 日)→ 窗口收缩到公式下限
        lb_min + (lb_max-lb_min)*(1-ratio_cap) = 10 + 50*0.1 = 15。"""
        calm = list(np.linspace(1.0, 1.2, 50))          # 长期平稳
        spike = [1.2, 1.35, 1.15, 1.38, 1.1, 1.4, 1.12, 1.42, 1.08, 1.45, 1.2]
        df = self.make_df(calm + spike, hi_lo_spread=0.001)
        n = dynamic_lookback_window(df, lb_min=10, lb_max=60, ratio_cap=0.9, vol_short_n=10, vol_long_n=60)
        assert n == 15

    def test_long_window_when_calm(self):
        """近期波动远低于长期 → 窗口拉长到接近 lb_max。"""
        spike = list(1.0 + 0.15 * np.sin(np.arange(50)))  # 前期高波动
        calm = list(np.linspace(1.5, 1.6, 11))             # 近期平稳
        df = self.make_df(spike + calm, hi_lo_spread=0.001)
        n = dynamic_lookback_window(df, lb_min=10, lb_max=60, ratio_cap=0.9, vol_short_n=10, vol_long_n=60)
        assert n > 40

    def test_insufficient_history_returns_none(self):
        df = self.make_df([1.0] * 30)
        assert dynamic_lookback_window(df, 10, 60, 0.9, 10, 60) is None
