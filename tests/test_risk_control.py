"""风控扩展测试:空仓转换、绝对动量过滤、均线过滤、止损(含禁买期)。"""

import numpy as np
import pandas as pd
import pytest

from etf_rotation.backtest import run_backtest
from etf_rotation.config import StrategyConfig


def synth_ohlc(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame({"date": dates, "open": s, "high": s, "low": s, "close": s, "volume": 1e6})


def geom(start, rate, n):
    """等比价格序列(恒正,避免对数因子遇到负价格)。"""
    return [start * rate**i for i in range(n)]


BASE = dict(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
            commission=0.0, initial_cash=100_000)


class TestCashTransition:
    def test_min_momentum_filter_goes_to_cash(self):
        """两只标的全程下跌:开 min_momentum=0 应全程空仓;不开则被迫持仓亏损。"""
        data = {"D1": synth_ohlc(geom(20, 0.995, 60)), "D2": synth_ohlc(geom(20, 0.997, 60))}

        unfiltered = run_backtest(data, StrategyConfig(**BASE))
        assert unfiltered.equity.iloc[-1] < 1.0     # 被迫持仓亏损

        filtered = run_backtest(data, StrategyConfig(**BASE, min_momentum=0.0))
        assert filtered.equity.iloc[-1] == pytest.approx(1.0)  # 全程空仓
        assert filtered.trades.empty

    def test_ma_filter_blocks_downtrend(self):
        """价格持续低于 MA10:ma_filter_n=10 不应建仓。"""
        data = {"A": synth_ohlc(geom(20, 0.995, 60)), "B": synth_ohlc(geom(20, 0.998, 60))}
        res = run_backtest(data, StrategyConfig(**BASE, ma_filter_n=10))
        assert res.trades.empty
        assert res.equity.iloc[-1] == pytest.approx(1.0)

    def test_filters_do_not_block_uptrend(self):
        """上涨行情中开过滤:不误伤正常建仓。"""
        data = {"UP": synth_ohlc(geom(10, 1.01, 60)), "FLAT": synth_ohlc([10.0] * 60)}
        for kw in (dict(min_momentum=0.0), dict(ma_filter_n=10)):
            res = run_backtest(data, StrategyConfig(**BASE, **kw))
            assert not res.trades.empty
            assert res.equity.iloc[-1] > 1.0


class TestStopLoss:
    UP_DAYS, DOWN_DAYS = 30, 24

    def crash_data(self):
        """单标的池:先 30 天 +1%/天,再 24 天 -3%/天。

        用单标的是因为双标的池里 Z-Score 恒有正负之分,1.5 倍阈值本身
        就会触发换仓,无法隔离止损的独立作用;单标的池评分恒为 0,
        基线策略买入后永不卖出,止损机制的效果才能被单独验证。
        """
        up = geom(10, 1.01, self.UP_DAYS)
        down = geom(up[-1] * 0.97, 0.97, self.DOWN_DAYS)
        return {"ONE": synth_ohlc(up + down)}

    def test_stop_loss_exits(self):
        """跌破止损线后应出现 SELL(基线策略在该池永不卖出)。"""
        res = run_backtest(self.crash_data(), StrategyConfig(**BASE, stop_loss=0.08))
        sells = res.trades[res.trades["action"] == "SELL"]
        assert not sells.empty
        # 止损卖出应发生在下跌段(价格低于峰值)
        peak = 10 * 1.01 ** (self.UP_DAYS - 1)
        assert sells.iloc[0]["price"] < peak * 0.95

    def test_stop_loss_reduces_drawdown(self):
        """止损(带禁买期)的回撤应显著小于不止损。"""
        data = self.crash_data()
        with_stop = run_backtest(data, StrategyConfig(**BASE, stop_loss=0.08))
        without = run_backtest(data, StrategyConfig(**BASE))
        dd = lambda r: (r.equity / r.equity.cummax() - 1).min()
        assert dd(with_stop) > dd(without) + 0.05   # 至少改善 5 个百分点

    def test_cooldown_prevents_immediate_reentry(self):
        """止损后禁买期内不应立即买回同一标的。"""
        res = run_backtest(self.crash_data(), StrategyConfig(**BASE, stop_loss=0.08, stop_cooldown_days=10))
        trades = res.trades.reset_index(drop=True)
        # 找到止损 SELL,其后 10 个交易日内不应有同标的 BUY
        sell = trades[trades["action"] == "SELL"].iloc[0]
        later_buys = trades[(trades["action"] == "BUY") & (trades["date"] > sell["date"])]
        for _, b in later_buys.iterrows():
            gap = len(res.equity.loc[sell["date"]: b["date"]]) - 1
            assert gap >= 10   # 信号隔日成交,买回最早为卖出后第 10 个交易日


class TestTrailingStop:
    """移动止损:锚定持仓期间最高收盘价,保护浮盈(固定止损锚定成本价做不到)。"""

    def spike_data(self):
        """单标的:30 天缓涨 +1%/天 → 3 天急涨 +15%/天 → 10 天阴跌 -3%/天。

        急涨把成本价远远甩开,回落时不触发固定止损(价格仍高于成本),
        但触发移动止损(自峰值回落)。
        """
        up = geom(10, 1.01, 30)
        spike = geom(up[-1] * 1.15, 1.15, 3)
        down = geom(spike[-1] * 0.97, 0.97, 10)
        return {"ONE": synth_ohlc(up + spike + down)}

    def test_trailing_stop_exits_on_pullback(self):
        """自峰值回落超 8% 应离场;仅固定止损(锚成本)则不会。"""
        data = self.spike_data()
        trail = run_backtest(data, StrategyConfig(**BASE, trailing_stop=0.08))
        fixed = run_backtest(data, StrategyConfig(**BASE, stop_loss=0.08))
        trail_sell = trail.trades[trail.trades["action"] == "SELL"]
        fixed_sell = fixed.trades[fixed.trades["action"] == "SELL"]
        assert not trail_sell.empty          # 移动止损触发
        assert fixed_sell.empty              # 固定止损不触发(价格仍高于成本)

    def test_trailing_stop_protects_open_profit(self):
        """移动止损的卖出价应显著高于成本价(锁住浮盈)。"""
        res = run_backtest(self.spike_data(), StrategyConfig(**BASE, trailing_stop=0.08))
        buy_price = res.trades.iloc[0]["price"]
        sell = res.trades[res.trades["action"] == "SELL"].iloc[0]
        assert sell["price"] > buy_price * 1.15   # 急涨 3 天 +52%,回落 8% 离场仍大赚

    def test_trailing_dd_smaller_than_fixed(self):
        data = self.spike_data()
        trail = run_backtest(data, StrategyConfig(**BASE, trailing_stop=0.08))
        fixed = run_backtest(data, StrategyConfig(**BASE, stop_loss=0.08))
        dd = lambda r: (r.equity / r.equity.cummax() - 1).min()
        assert dd(trail) > dd(fixed)


class TestMultiHolding:
    """n_hold=2 双标的等权分散。"""

    def test_holds_two_positions_equal_weight(self):
        data = {"UP": synth_ohlc(geom(10, 1.01, 60)), "MID": synth_ohlc(geom(10, 1.005, 60)),
                "FLAT": synth_ohlc([10.0] * 60)}
        res = run_backtest(data, StrategyConfig(**BASE, n_hold=2))
        last = res.holdings.iloc[-1]
        assert "+" in last                      # 期末持有两只
        # 首两日应各有一笔 BUY,且金额相近(等权)
        buys = res.trades[res.trades["action"] == "BUY"].head(2)
        v = (buys["price"] * buys["shares"]).values
        assert abs(v[0] / v[1] - 1) < 0.05

    def test_diversification_cuts_drawdown(self):
        """一只暴涨后暴跌、一只横盘:双持仓的回撤应小于单持仓。"""
        up = geom(10, 1.01, 30)
        spike = geom(up[-1] * 1.2, 1.2, 3)
        down = geom(spike[-1] * 0.90, 0.90, 15)
        data = {"VOL": synth_ohlc(up + spike + down),
                "CALM": synth_ohlc([10.0] * 48), "MID": synth_ohlc([10.5] * 48)}
        dd = lambda r: (r.equity / r.equity.cummax() - 1).min()
        one = run_backtest(data, StrategyConfig(**BASE, n_hold=1))
        two = run_backtest(data, StrategyConfig(**BASE, n_hold=2))
        assert dd(two) > dd(one)


class TestTakeProfit:
    def test_take_profit_exits_and_bans(self):
        """单标的持续上涨:止盈 +20% 触发离场,且禁买期内不买回。"""
        data = {"ONE": synth_ohlc(geom(10, 1.01, 120))}
        res = run_backtest(data, StrategyConfig(**BASE, take_profit=0.20, stop_cooldown_days=10))
        sells = res.trades[res.trades["action"] == "SELL"]
        assert not sells.empty
        buy_price = res.trades.iloc[0]["price"]
        # 止盈卖出价应在成本价 1.2 倍附近(信号日收盘触发,次日开盘成交)
        assert sells.iloc[0]["price"] > buy_price * 1.18


class TestZhihuOptimizations:
    """知乎三点优化:急跌过滤与溢价率过滤(默认关闭,不影响 V3)。"""

    def test_drop_filter_blocks_recent_crash(self):
        """近 5 日单日跌超 5% 且连跌累计超 5% → drop_filter 拒绝;关闭时放行。"""
        from etf_rotation.backtest import _passes_risk_filters
        closes = geom(20, 1.01, 40) + [21.0, 19.8, 18.5, 17.3, 16.2]  # 末段单日 -6%
        data = {"B": synth_ohlc(closes)}
        t = len(closes) - 1
        assert _passes_risk_filters("B", data, t, StrategyConfig(**BASE, drop_filter=True)) is False
        assert _passes_risk_filters("B", data, t, StrategyConfig(**BASE)) is True

    def test_drop_filter_passes_smooth_uptrend(self):
        """平稳上行不触发急跌过滤。"""
        from etf_rotation.backtest import _passes_risk_filters
        data = {"B": synth_ohlc(geom(20, 1.005, 45))}
        assert _passes_risk_filters("B", data, 44, StrategyConfig(**BASE, drop_filter=True)) is True

    def test_premium_filter_blocks_high_premium(self):
        """premium 列超阈值时禁止买入;关闭时不拦。"""
        n = 45
        data = {"A": synth_ohlc(geom(20, 1.005, n)), "B": synth_ohlc(geom(20, 1.008, n))}
        data["B"]["premium"] = 0.02    # 恒定 2% 溢价
        data["A"]["premium"] = 0.0

        on = run_backtest(data, StrategyConfig(**BASE, n_hold=2, premium_max=0.01))
        assert on.trades[on.trades["symbol"] == "B"].empty

        off = run_backtest(data, StrategyConfig(**BASE, n_hold=2))
        assert not off.trades[off.trades["symbol"] == "B"].empty

    def test_screened_out_candidate_does_not_leave_empty_slot(self):
        """回归:评分第一的标的被风控筛掉时,名额必须由下一个合格标的递补。

        旧版先 select_targets 再过滤,被剔掉的标的只是让槽位空着 —— V3.1 的溢价
        规则会把常居第一的 513100 反复剔掉,组合长期只持 1 只,违反 n_hold=2 的意图。
        """
        n = 45
        # 评分排序:A(强) > B(中) > C(弱)
        data = {"A": synth_ohlc(geom(20, 1.010, n)),
                "B": synth_ohlc(geom(20, 1.006, n)),
                "C": synth_ohlc(geom(20, 1.002, n))}
        data["A"]["premium"] = 0.50          # 评分第一但溢价爆表 -> 被筛掉
        for s in ("B", "C"):
            data[s]["premium"] = 0.0

        res = run_backtest(data, StrategyConfig(**BASE, n_hold=2, premium_max=0.01))
        assert "A" not in res.holdings.iloc[-1]          # 被筛掉的确实不持有
        assert res.holdings.iloc[-1] == "B+C"           # 两个槽位都被合格标的填满
        assert res.trades[res.trades["symbol"] == "A"].empty   # 被筛掉的标的从未买入

    def test_premium_does_not_exit_existing_holding(self):
        """第五轮口径:溢价门槛是**开仓前安检**,已持有的标的溢价爆表不因此被卖出
        (持有期离场归移动止损/止盈/评分轮动管)。"""
        n = 45
        data = {"A": synth_ohlc(geom(20, 1.010, n)), "B": synth_ohlc(geom(20, 1.006, n))}
        data["A"]["premium"] = 0.0
        data["B"]["premium"] = 0.0
        data["A"].loc[data["A"].index >= 30, "premium"] = 0.50   # 后半段 A 溢价爆表

        res = run_backtest(data, StrategyConfig(**BASE, n_hold=1, premium_max=0.01))
        buys_a = res.trades[(res.trades["symbol"] == "A") & (res.trades["action"] == "BUY")]
        sells_a = res.trades[(res.trades["symbol"] == "A") & (res.trades["action"] == "SELL")]
        assert not buys_a.empty                  # 溢价正常时买入
        assert sells_a.empty                     # 溢价爆表后不被强制卖出
        assert res.holdings.iloc[-1] == "A"

    def test_blocked_top_pick_does_not_trigger_phantom_rotation(self):
        """回归(第五轮):评分第一被入场门槛拦截时,不能把在持标的换出给它腾位置
        (必须先验票再选股;先换血后验票会出现 A 进不来、C 也回不来的空槽位)。"""
        n = 45
        data = {"A": synth_ohlc(geom(20, 1.010, n)),
                "B": synth_ohlc(geom(20, 1.006, n)),
                "C": synth_ohlc(geom(20, 1.002, n))}
        data["A"]["premium"] = 0.50              # 全程爆表,始终不可买入
        for s in ("B", "C"):
            data[s]["premium"] = 0.0

        res = run_backtest(data, StrategyConfig(**BASE, n_hold=2, premium_max=0.01))
        assert len(res.trades) == 2                      # 只有最初两笔建仓 BUY
        assert (res.trades["action"] == "BUY").all()     # 之后全程无调仓
        assert res.holdings.iloc[-1] == "B+C"

    def test_all_filtered_exits_to_cash_while_holding(self):
        """回归(第五轮):全池被 regime 过滤(min_momentum)筛掉时,持仓也必须离场
        (空仓观望)——不能撞上 select_targets 的空序列守卫变成继续持有死扛。"""
        up = geom(10, 1.01, 40)                  # 涨段:A 动量为正,建仓
        down = geom(up[-1] * 0.99, 0.99, 30)     # 跌段:全池动量转负
        data = {"A": synth_ohlc(up + down),
                "B": synth_ohlc([10.0] * 40 + geom(10.0, 0.99, 30)[1:])}
        res = run_backtest(data, StrategyConfig(**BASE, min_momentum=0.0))
        sells = res.trades[res.trades["action"] == "SELL"]
        assert not sells.empty                   # 全池无趋势时必须离场
        assert pd.isna(res.holdings.iloc[-1])    # 期末空仓
        # 离场后净值走平(空仓观望,不再随下跌段亏损)
        eq = res.equity
        assert eq.iloc[-1] == pytest.approx(eq.loc[sells.iloc[0]["date"]:].max(), rel=1e-9)


class TestPremiumRule:
    """溢价率 7% 购买规则(2026-09-29 写入 V3.1 预设)。"""

    def test_v31_preset_contains_premium_rule(self):
        from etf_rotation.cli import PRESETS
        assert PRESETS["v3.1"]["premium_max"] == 0.07
        assert "premium_max" not in PRESETS["v3"]   # V3 保持无溢价过滤原版

    def test_fetch_premium_keeps_valuation_dates(self, tmp_path, monkeypatch):
        """premium 序列按净值估值日保存;是否可被信号日使用由 attach_premium 决定。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        raw = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                            "close": [1.0, 1.1, 1.2]})
        nav = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                            "nav": [1.0, 1.0, 1.0]})
        raw.to_csv(tmp_path / "X.raw.csv", index=False)
        nav.to_csv(tmp_path / "X.nav.csv", index=False)
        prem = data_mod.fetch_premium("X")
        assert list(prem.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        assert prem.tolist() == pytest.approx([0.0, 0.1, 0.2])
        assert data_mod.fetch_premium("NOPE").empty

    def test_attach_premium_uses_latest_known_nav_when_nav_lags(self, tmp_path, monkeypatch):
        """回归:净值滞后时,信号日应取最新已公布净值,不能把最后一行再 shift 掉。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                      "close": [1.1, 1.2]}).to_csv(tmp_path / "X.raw.csv", index=False)
        pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                      "nav": [1.0, 1.0]}).to_csv(tmp_path / "X.nav.csv", index=False)
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        out = data_mod.attach_premium({"X": market})["X"]
        assert pd.isna(out["premium"].iloc[0])       # 01-01 之前无已知净值
        assert out["premium"].iloc[1] == pytest.approx(0.1)   # 01-02 用 01-01
        assert out["premium"].iloc[2] == pytest.approx(0.2)   # 净值滞后:01-03 用 01-02
        assert out["premium"].iloc[3] == pytest.approx(0.2)

    def test_attach_premium_dedupes_dates_drops_nan_and_sorts_input(self, tmp_path, monkeypatch):
        """缓存边界:重复日期取最后一行,NaN 净值不沿用旧值,乱序行情仍按日期升序输出。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                      "close": [1.0, 1.1, 1.2]}).to_csv(tmp_path / "X.raw.csv", index=False)
        pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                      "nav": [1.0, None]}).to_csv(tmp_path / "X.nav.csv", index=False)
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"]),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        out = data_mod.attach_premium({"X": market})["X"]
        assert list(out["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        assert pd.isna(out["premium"].iloc[0])
        assert out["premium"].iloc[1] == pytest.approx(0.1)   # 重复 01-01 取最后 close=1.1
        assert out["premium"].iloc[2] == pytest.approx(0.1)   # 01-02 净值 NaN,不伪装成新日期

    def test_attach_premium_missing_cache_passes_through(self, tmp_path, monkeypatch):
        """无 raw/nav 缓存的标的:原样返回(无 premium 列,过滤放行)。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        df = synth_ohlc(geom(20, 1.01, 30))
        out = data_mod.attach_premium({"A": df})
        assert "premium" not in out["A"].columns

    def test_premium_status_flags_nav_lag(self, tmp_path, monkeypatch):
        """回归:单位净值落后行情数据时必须报 stale。

        旧版只按"距今 7 天内"判定新鲜度,513100 净值落后 3 个交易日仍报
        stale=False,实盘就在一个 3 天前的溢价上做买入决策。
        """
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        bars = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=5)
        pd.DataFrame({"date": bars, "close": 1.0}).to_csv(tmp_path / "LAG.raw.csv", index=False)
        pd.DataFrame({"date": bars[:-3], "nav": 1.0}).to_csv(tmp_path / "LAG.nav.csv", index=False)
        st = data_mod.premium_status(["LAG"])
        assert st["LAG"]["missing"] is False
        assert st["LAG"]["nav_lag"] == 3
        assert st["LAG"]["stale"] is True

        # 净值与行情同步(只落后当日净值,属正常)-> 不算 stale
        pd.DataFrame({"date": bars, "close": 1.0}).to_csv(tmp_path / "OK.raw.csv", index=False)
        pd.DataFrame({"date": bars, "nav": 1.0}).to_csv(tmp_path / "OK.nav.csv", index=False)
        ok = data_mod.premium_status(["OK"])
        assert ok["OK"]["nav_lag"] == 0
        assert ok["OK"]["stale"] is False

    def test_premium_status_allows_normal_qdii_t_plus_2_lag(self, tmp_path, monkeypatch):
        """513100 美股 QDII 周一落后 2 个行情日属正常披露节奏,不升级 stale。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        bars = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=4)
        pd.DataFrame({"date": bars, "close": 2.0}).to_csv(tmp_path / "513100.raw.csv", index=False)
        pd.DataFrame({"date": bars[:-2], "nav": 1.9}).to_csv(tmp_path / "513100.nav.csv", index=False)
        st = data_mod.premium_status(["513100"])["513100"]
        assert st["nav_lag"] == 2
        assert st["stale"] is False
        assert st["asof"] == str(bars[-3].date())
        assert st["premium"] == pytest.approx(round(2.0 / 1.9 - 1, 4))

    def test_premium_status_missing_and_stale(self, tmp_path, monkeypatch):
        """premium_status:无缓存 missing=True;旧数据 stale=True。"""
        from etf_rotation import data as data_mod
        monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
        st = data_mod.premium_status(["NOPE"])
        assert st["NOPE"]["missing"] is True and st["NOPE"]["stale"] is True

        raw = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "close": [1.0, 1.0]})
        nav = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "nav": [1.0, 1.0]})
        raw.to_csv(tmp_path / "OLD.raw.csv", index=False)
        nav.to_csv(tmp_path / "OLD.nav.csv", index=False)
        st = data_mod.premium_status(["OLD"])
        assert st["OLD"]["missing"] is False
        assert st["OLD"]["stale"] is True          # 2020 年的数据必然过期
        assert st["OLD"]["asof"] == "2020-01-02"
