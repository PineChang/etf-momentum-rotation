"""评分与轮动选择逻辑测试。"""

import pandas as pd
import pytest

from etf_rotation.scoring import _beats, combined_score, select_targets, zscore


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
        """三列都取到互不相同的 z 值,权重必须真的参与计算。

        (旧夹具让 z_efficiency ≡ 0 且 z_bias = -z_slope,硬编码 0.5/0.5/0.0
        也能过,等于没锁住权重。)
        """
        df = pd.DataFrame({"bias": [1.0, 3.0, 6.0], "slope": [3.0, 1.0, 2.0],
                           "efficiency": [2.0, 9.0, 2.0]},
                          index=["A", "B", "C"])
        z = zscore(df)
        assert z["efficiency"].abs().sum() > 0 and (z["bias"] + z["slope"]).abs().sum() > 0
        scores = combined_score(df, 0.3, 0.3, 0.4)
        for sym in ("A", "B", "C"):
            expected = (0.3 * z.loc[sym, "bias"] + 0.3 * z.loc[sym, "slope"]
                        + 0.4 * z.loc[sym, "efficiency"])
            assert scores[sym] == pytest.approx(expected, rel=1e-12)
        # 换一组权重必须得到不同结果(否则等于忽略入参)
        assert not scores.equals(combined_score(df, 0.5, 0.5, 0.0))


class TestBeats:
    """调仓阈值语义(审计 R1:负分区不得退化)。"""

    def test_positive_region_equivalent_to_article_multiple(self):
        """正分区必须与文章的 "挑战者 > 在持者 × 阈值" 完全等价。"""
        for held in (0.5, 1.0, 2.0):
            for thr in (1.2, 1.5, 2.0):
                assert _beats(held, held * thr, thr) is False        # 恰好达界不算超过
                assert _beats(held, held * thr + 1e-9, thr) is True

    def test_negative_region_does_not_degenerate(self):
        """负分区回归:旧公式 `挑战者 > 在持者 × 1.5` 在在持者为负时反向放行
        (`-0.9 > -1.5` 成立),会把仓位换到更差的标的上。"""
        assert _beats(-1.0, -0.9, 1.5) is False    # 只是"没那么差" -> 不换
        assert _beats(-1.0, -0.4, 1.5) is True     # 显著更强 -> 换
        assert (-0.9 > -1.0 * 1.5) is True         # 旧公式确实会放行这两种情况


class TestSelectTargets:
    """多持仓选择(生产路径)。单标的版 select_target 是死代码,已删除。"""

    def scores(self):
        return pd.Series({"A": 2.0, "B": 1.0, "C": 0.5, "D": -3.0})

    def test_empty_portfolio_fills_top_n(self):
        assert select_targets(self.scores(), set(), 1.5, 2) == {"A", "B"}

    def test_keeps_holdings_when_challenger_below_threshold(self):
        assert select_targets(self.scores(), {"A", "B"}, 1.5, 2) == {"A", "B"}

    def test_no_switch_below_threshold(self):
        s = pd.Series({"A": 5.0, "B": 1.0, "C": 1.4, "D": -1.0})
        assert select_targets(s, {"A", "B"}, 1.5, 2) == {"A", "B"}   # 1.4 < 1.0×1.5

    def test_swaps_weakest_when_challenger_beats_threshold(self):
        s = pd.Series({"A": 5.0, "B": 1.0, "C": 2.0, "D": -1.0})
        assert select_targets(s, {"A", "B"}, 1.5, 2) == {"A", "C"}   # 2.0 > 1.0×1.5

    def test_held_missing_from_scores_is_dropped(self):
        s = pd.Series({"A": 1.0, "B": 0.5})
        assert select_targets(s, {"A", "GONE"}, 1.5, 2) == {"A", "B"}

    def test_all_nan_returns_current(self):
        s = pd.Series({"A": float("nan")})
        assert select_targets(s, {"A"}, 1.5, 2) == {"A"}
