"""参数遍历网格的语义测试。"""

import pytest

from etf_rotation.sweep import weight_grid


class TestWeightGrid:
    def test_weights_sum_to_one_for_any_step(self):
        """回归:step 不整除 1 时(0.15 / 0.3)权重和曾偏离 1(1.05 / 0.9),
        违背 docstring 声称的单纯形约束。"""
        for step in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5):
            for w in weight_grid(step):
                assert sum(w) == pytest.approx(1.0, abs=1e-12), f"step={step} 权重和 {sum(w)}"

    def test_grid_sizes(self):
        assert len(weight_grid(0.1)) == 66     # C(12,2)
        assert len(weight_grid(0.2)) == 21

    def test_covers_corners_and_center(self):
        g = weight_grid(0.2)
        assert (1.0, 0.0, 0.0) in g            # 纯乖离
        assert (0.0, 0.0, 1.0) in g            # 纯效率
        assert (0.4, 0.2, 0.4) in g            # 包含既定权重附近
