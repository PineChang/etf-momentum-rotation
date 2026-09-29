"""配置与"文章披露参数"的锁测试。

这些数字是忠实复现文章的基准(文章明确给出)。改动它们等于不再忠实原文,
必须显式改测试而不是让默认值悄悄漂移 —— 此前没有任何测试锁定它们,
把 momentum_day 从 25 改成 10、把权重改成任意值,整套测试都照样通过。
"""

from etf_rotation.config import UNIVERSE, StrategyConfig

# 文章明确给出的参数
ARTICLE = {
    "momentum_day": 25,
    "w_bias": 0.3,
    "w_slope": 0.3,
    "w_efficiency": 0.4,
    "rebalance_threshold": 1.5,
}


class TestArticleDisclosedParameters:
    def test_defaults_match_article(self):
        cfg = StrategyConfig()
        for k, v in ARTICLE.items():
            assert getattr(cfg, k) == v, f"{k} 已偏离文章披露值"

    def test_undisclosed_assumptions_are_as_documented(self):
        """文章未披露、项目自设的假设(config.py 已标注)。"""
        cfg = StrategyConfig()
        assert (cfg.bias_ma_n, cfg.slope_n, cfg.efficiency_n) == (20, 25, 25)
        assert cfg.commission == 0.0001

    def test_risk_extensions_default_off(self):
        """风控扩展默认必须全部关闭 = 忠实原文。"""
        cfg = StrategyConfig()
        assert cfg.n_hold == 1
        assert cfg.min_momentum is None and cfg.ma_filter_n is None
        assert cfg.stop_loss is None and cfg.trailing_stop is None
        assert cfg.take_profit is None and cfg.premium_max is None
        assert cfg.dynamic_lookback is False and cfg.drop_filter is False


class TestUniverse:
    def test_four_etfs(self):
        assert set(UNIVERSE) == {"512890", "159949", "513100", "518880"}


class TestWarmupBars:
    def test_固定窗口覆盖全部因子所需历史(self):
        cfg = StrategyConfig()
        assert cfg.warmup_bars == max(cfg.bias_ma_n + cfg.momentum_day,
                                      cfg.slope_n, cfg.efficiency_n)
        assert cfg.warmup_bars == 45

    def test_动态窗口分支足够长(self):
        """dynamic_lookback=True 时 warmup 必须覆盖最长回溯窗口与长窗 ATR。"""
        cfg = StrategyConfig(dynamic_lookback=True)
        assert cfg.warmup_bars >= cfg.bias_ma_n + cfg.lb_max
        assert cfg.warmup_bars >= cfg.vol_long_n + 1
