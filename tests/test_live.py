"""实盘模块测试:计划生成、限价单、V3 风控、状态持久化与重启恢复、复利盈亏。"""

import pandas as pd
import pytest

from etf_rotation.config import StrategyConfig
from etf_rotation.live import (
    BUY_PRICE_FACTOR,
    SELL_PRICE_FACTOR,
    Holding,
    LiveState,
    StateStore,
    execute_plan,
    make_plan,
)


def synth(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame({"date": dates, "open": s, "high": s, "low": s, "close": s, "volume": 1e6})


def geom(start, rate, n):
    return [start * rate**i for i in range(n)]


@pytest.fixture
def data():
    return {
        "UP": synth([10 + 0.1 * i for i in range(60)]),
        "DOWN": synth([20 - 0.1 * i for i in range(60)]),
    }


@pytest.fixture
def cfg():
    return StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                          commission=0.0001, initial_cash=100_000)


class TestMakePlan:
    def test_cash_plan_buys_best_with_limit_up(self, data, cfg):
        plan, _ = make_plan(data, LiveState(cash=100_000), cfg)
        assert plan["target"] == ["UP"]
        assert len(plan["orders"]) == 1
        order = plan["orders"][0]
        last_close = data["UP"]["close"].iloc[-1]
        assert order["action"] == "BUY"
        assert order["limit_price"] == pytest.approx(round(last_close * BUY_PRICE_FACTOR, 3))
        assert order["est_budget"] > 0   # 回归:空仓首日建仓预算不为 0(旧版 bug 为 0)

    def test_hold_when_threshold_not_crossed(self, data, cfg):
        state = LiveState(cash=0.0, holdings={"UP": Holding("UP", 100, 10.0, 10.0)})
        plan, _ = make_plan(data, state, cfg)
        assert plan["target"] == ["UP"]
        assert plan["orders"] == []

    def test_switch_generates_two_orders(self, data, cfg):
        state = LiveState(cash=0.0, holdings={"DOWN": Holding("DOWN", 100, 20.0, 20.0)})
        plan, _ = make_plan(data, state, cfg)
        # UP 全面强于 DOWN(标准化后 UP 得分 > 0 > DOWN 得分),阈值必然触发
        assert plan["target"] == ["UP"]
        actions = [o["action"] for o in plan["orders"]]
        assert actions == ["SELL", "BUY"]
        sell = plan["orders"][0]
        last_close = data["DOWN"]["close"].iloc[-1]
        assert sell["limit_price"] == pytest.approx(round(last_close * SELL_PRICE_FACTOR, 3))

    def test_v3_n_hold_two_targets(self, cfg):
        """V3 双持仓:live 模块必须支持多持仓(旧版只有单持仓,漏掉全部 V3 风控)。"""
        data = {"UP": synth(geom(10, 1.01, 60)), "MID": synth(geom(10, 1.005, 60)),
                "FLAT": synth([10.0] * 60)}
        cfg2 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, n_hold=2)
        plan, _ = make_plan(data, LiveState(cash=100_000), cfg2)
        assert len(plan["target"]) == 2
        buys = [o for o in plan["orders"] if o["action"] == "BUY"]
        assert len(buys) == 2
        for o in buys:                     # 槽位等权:各半仓
            assert o["est_budget"] == pytest.approx(50_000, rel=1e-3)

    def test_trailing_stop_in_live(self, cfg):
        """V3 移动止损:持仓自峰值回落超阈值,plan 必须给出 SELL 并记入禁买名单。"""
        up = geom(10, 1.01, 30)
        spike = geom(up[-1] * 1.15, 1.15, 3)
        down = geom(spike[-1] * 0.97, 0.97, 10)
        data = {"ONE": synth(up + spike + down)}
        cfg3 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, trailing_stop=0.08)
        peak = spike[-1]
        state = LiveState(cash=0.0, holdings={"ONE": Holding("ONE", 1000, up[0], peak)})
        plan, new_state = make_plan(data, state, cfg3)
        sells = [o for o in plan["orders"] if o["action"] == "SELL"]
        assert sells and sells[0]["symbol"] == "ONE"
        last_date = str(data["ONE"]["date"].iloc[-1].date())
        assert new_state.stopped.get("ONE") == last_date   # 禁买状态已持久化


class TestExecutePlan:
    def test_first_buy_uses_cash_regression(self, data, cfg):
        """回归:空仓首日建仓必须真金白银买入(旧版 budget=0 买入 0 份)。"""
        plan, ns = make_plan(data, LiveState(cash=cfg.initial_cash), cfg)
        fills = {"UP": float(data["UP"]["close"].iloc[-1])}
        final = execute_plan(plan, ns, fills, cfg)
        assert final.holdings["UP"].shares > 0
        assert final.holdings["UP"].shares == pytest.approx(
            cfg.initial_cash * (1 - cfg.commission) / fills["UP"], rel=1e-9)
        assert final.cash == pytest.approx(0.0, abs=1e-9)

    def test_round_trip_realizes_pnl_and_compounds(self, data, cfg):
        state = LiveState(cash=0.0, holdings={"DOWN": Holding("DOWN", 1000.0, 15.0, 15.0)},
                          realized_pnl=2_000.0)
        plan, ns = make_plan(data, state, cfg)
        fills = {"DOWN": 16.0, "UP": 15.0}
        final = execute_plan(plan, ns, fills, cfg)
        # 卖出:1000*16*(1-c) - 1000*15 = 已实现盈亏,累加到历史 2000 上
        expected_pnl = 1000 * 16.0 * (1 - cfg.commission) - 15_000.0
        assert final.realized_pnl == pytest.approx(2_000.0 + expected_pnl, rel=1e-9)
        # 买入:卖出回款扣佣金后全额投入(复利)
        expected_cash = 1000 * 16.0 * (1 - cfg.commission)
        assert final.holdings["UP"].shares == pytest.approx(
            expected_cash * (1 - cfg.commission) / 15.0, rel=1e-9)
        assert final.holdings["UP"].cost == 15.0

    def test_empty_plan_keeps_state(self, data, cfg):
        state = LiveState(cash=0.0, holdings={"UP": Holding("UP", 100, 10.0, 12.0)})
        plan = {"orders": [], "reference_prices": {}}
        assert execute_plan(plan, state, {}, cfg) is state


class TestStateStore:
    def test_save_load_recover_after_restart(self, tmp_path):
        store = StateStore(tmp_path)
        state = LiveState(cash=1_234.5,
                          holdings={"518880": Holding("518880", 500.0, 12.0, 13.5)},
                          realized_pnl=888.0,
                          stopped={"513100": "2026-09-01"},
                          top_streak={"518880": 3})
        store.save_state(state)
        # 模拟程序重启:新建 StateStore 实例读取同一目录
        recovered = StateStore(tmp_path).load_state()
        assert recovered == state

    def test_empty_state_returns_default(self, tmp_path):
        assert StateStore(tmp_path).load_state(initial_cash=100_000) == LiveState(cash=100_000)

    def test_trade_log_appends(self, tmp_path):
        store = StateStore(tmp_path)
        store.append_log({"action": "BUY", "symbol": "UP"})
        store.append_log({"action": "SELL", "symbol": "UP"})
        lines = (tmp_path / "trade_log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
