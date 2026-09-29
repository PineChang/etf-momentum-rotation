"""无前视(未来函数)确定性检验。

为什么单独一个文件:项目原有的两套招牌检验——截断实验(`scripts/audit_independent.py` §A)
与未来数据扰动实验(`scripts/audit_deep.py` §7.1)——只有"边界那一根 bar"的敏感度。
把窗口偷看一根 bar(`iloc[:t+1]` 写成 `iloc[:t+2]`)注入后,两者都全绿地放过
(实测:截断实验 0/3 发现,扰动实验报"未发现")。原因是截断版在最后一根 bar 之外
同样能拿到 t+1,所以两边算出的信号逐笔相同。

这里改成直接断言因果方向:T 日信号必须与 T+1 日及以后的行情**完全无关**。
"""

import numpy as np
import pandas as pd
import pytest

from etf_rotation.backtest import compute_target_set, run_backtest
from etf_rotation.config import StrategyConfig

BASE = dict(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10, commission=0.0)


def _rng_data(n=70, seed=0):
    """三只强弱不同的标的,且 open != close。

    open != close 是关键:项目原有夹具全部令 open == close,于是任何断言都
    区分不了"按 T 日收盘成交"与"按 T+1 开盘成交",真正的成交时点无人把关。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for i, sym in enumerate(("A", "B", "C")):
        close = 10 * np.exp(np.cumsum(rng.normal(0.003 * (3 - i), 0.02, n)))
        gap = rng.normal(0, 0.006, n)
        out[sym] = pd.DataFrame({
            "date": dates,
            "open": close * (1 + gap),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1e6,
        })
    return out


class TestNoLookahead:
    def test_signal_invariant_to_future_bars(self):
        """把 t+1 及以后的行情整体放缩 3 倍,T 日信号必须一字不变。

        这是对 off-by-one 窗口偷看最直接的检验:任何多取一根 bar 的实现都会
        让得分序列发生变化而失败。
        """
        cfg = StrategyConfig(**BASE)
        for t in (20, 35, 50):
            data = _rng_data()
            tampered = {s: df.copy() for s, df in data.items()}
            for df in tampered.values():
                df.loc[t + 1:, ["open", "high", "low", "close"]] *= 3.0

            base_target, base_scores = compute_target_set(data, t, cfg, {}, {}, {}, {}, {})
            fut_target, fut_scores = compute_target_set(tampered, t, cfg, {}, {}, {}, {}, {})

            assert base_target == fut_target, f"t={t}: 目标持仓随未来行情改变 —— 存在前视"
            pd.testing.assert_series_equal(base_scores, fut_scores)

    def test_signal_invariant_to_future_bars_with_risk_filters_on(self):
        """风控过滤链同样只能看 T 日及之前(旧版曾在 _passes_risk_filters 里另取一次窗口)。"""
        cfg = StrategyConfig(**BASE, min_momentum=0.0, ma_filter_n=10, trailing_stop=0.08,
                             take_profit=0.25, n_hold=2)
        t = 40
        data = _rng_data()
        tampered = {s: df.copy() for s, df in data.items()}
        for df in tampered.values():
            df.loc[t + 1:, ["open", "high", "low", "close"]] *= 0.3

        pos = {"A": 100.0}
        ent = {"A": float(data["A"]["close"].iloc[t - 5])}
        pk = {"A": ent["A"]}
        base = compute_target_set(data, t, cfg, dict(pos), dict(ent), dict(pk), {}, {})
        fut = compute_target_set(tampered, t, cfg, dict(pos), dict(ent), dict(pk), {}, {})
        assert base[0] == fut[0]
        pd.testing.assert_series_equal(base[1], fut[1])

    def test_trades_fill_at_open_never_at_signal_bar_close(self):
        """成交价必须是成交日的开盘价。

        夹具里 open != close,所以"用信号日收盘价成交"这种典型前视会被抓出来。
        """
        data = _rng_data()
        cfg = StrategyConfig(**BASE, n_hold=2)
        res = run_backtest(data, cfg)
        assert not res.trades.empty, "夹具应产生交易"
        for _, tr in res.trades.iterrows():
            df = data[tr["symbol"]]
            open_px = float(df.loc[df["date"] == tr["date"], "open"].iloc[0])
            close_px = float(df.loc[df["date"] == tr["date"], "close"].iloc[0])
            assert tr["price"] == pytest.approx(open_px), "成交价不等于当日开盘价"
            assert abs(open_px - close_px) > 1e-9, "夹具需保持 open != close"

    def test_first_trade_waits_for_warmup(self):
        """首笔成交必须晚于 warmup 完成日(信号日收盘 -> 次日开盘)。"""
        data = _rng_data()
        cfg = StrategyConfig(**BASE)
        res = run_backtest(data, cfg)
        first = res.trades.iloc[0]["date"]
        assert first > data["A"]["date"].iloc[cfg.warmup_bars - 1]
