"""评分与轮动选择逻辑测试。"""

import pandas as pd
import pytest

from etf_rotation.scoring import combined_score, select_target, zscore


class TestZScore:
    def test_mean_zero_std_one(self):
        df = pd.DataFrame({"bias": [1.0, 2.0, 3.0, 4.0], "slope": [10.0, 20.0, 30.0, 40.0]})
        z = zscore(df)
        assert z["bias"].mean() == pytest.approx(0.0, abs=1e-12)
        assert z["bias"].std(ddof=0) == pytest.approx(1.0, rel=1e-9)
        assert z["slope"].std(ddof=0) == pytest.approx(1.0, rel=1e-9)

    def test_scale_invariance(self):
        """标准化消除量纲:整体放大 1000 倍,标准化结果不变。"""
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        big = pd.DataFrame({"f": [1000.0, 2000.0, 3000.0, 4000.0]})
        pd.testing.assert_frame_equal(zscore(df), zscore(big))

    def test_constant_column_zero(self):
        df = pd.DataFrame({"f": [5.0, 5.0, 5.0]})
        assert (zscore(df)["f"] == 0).all()


class TestCombinedScore:
    def test_weighted_sum(self):
        df = pd.DataFrame(
            {"bias": [1.0, 3.0], "slope": [3.0, 1.0], "efficiency": [2.0, 2.0]},
            index=["A", "B"],
        )
        z = zscore(df)
        scores = combined_score(df, 0.3, 0.3, 0.4)
        expected_a = 0.3 * z.loc["A", "bias"] + 0.3 * z.loc["A", "slope"] + 0.4 * z.loc["A", "efficiency"]
        assert scores["A"] == pytest.approx(expected_a, rel=1e-12)


class TestSelectTarget:
    def scores(self):
        return pd.Series({"A": 1.0, "B": 1.2, "C": 0.5})

    def test_cash_buys_best(self):
        assert select_target(self.scores(), None, 1.5) == "B"

    def test_hold_when_already_best(self):
        assert select_target(self.scores(), "B", 1.5) == "B"

    def test_no_switch_below_threshold(self):
        """B 得分 1.2 < A 得分 1.0 * 1.5 → 不切换。"""
        assert select_target(self.scores(), "A", 1.5) == "A"

    def test_switch_above_threshold(self):
        """挑战者 1.6 > 持仓 1.0 * 1.5 → 切换。"""
        scores = pd.Series({"A": 1.0, "B": 1.6})
        assert select_target(scores, "A", 1.5) == "B"

    def test_exact_boundary_no_switch(self):
        """恰好等于 1.5 倍:文章语义为“超过”,边界不切换。"""
        scores = pd.Series({"A": 1.0, "B": 1.5})
        assert select_target(scores, "A", 1.5) == "A"

    def test_all_nan_keeps_current(self):
        scores = pd.Series({"A": float("nan"), "B": float("nan")})
        assert select_target(scores, "A", 1.5) == "A"
        assert select_target(scores, None, 1.5) is None
