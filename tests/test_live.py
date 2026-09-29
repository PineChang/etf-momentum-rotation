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
    check_fillable_at_open,
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
        """V3 移动止损:持仓自峰值回落超阈值,plan 必须给出 SELL 并记入禁买名单。

        注意:peak 由 make_plan 自己推进(不手工注入)——手工注入会掩盖
        "实盘从不更新峰值锚点"的真实缺陷。
        """
        closes = self.trail_series()
        data = {"ONE": synth(closes)}
        cfg3 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, trailing_stop=0.08)
        state = LiveState(cash=0.0, holdings={"ONE": Holding("ONE", 1000, 10.0, max(closes))})
        plan, new_state = make_plan(data, state, cfg3)
        sells = [o for o in plan["orders"] if o["action"] == "SELL"]
        assert sells and sells[0]["symbol"] == "ONE"
        last_date = str(data["ONE"]["date"].iloc[-1].date())
        assert new_state.stopped.get("ONE") == last_date   # 禁买状态已持久化

    @staticmethod
    def trail_series():
        """缓涨 → 急涨 → 回落:回落终点仍高于开仓价 20%+。

        只有锚定**持仓期峰值**的移动止损能离场;若峰值锚点不更新
        (退化为"锚定开仓价的固定止损"= 开仓价×0.92),则永不触发。
        """
        up = geom(10, 1.015, 25)
        spike = geom(up[-1] * 1.08, 1.08, 5)
        fall = geom(spike[-1] * 0.985, 0.985, 20)
        return up + spike + fall

    @staticmethod
    def _run_days(closes, cfg, start_i=15):
        """逐日推进实盘:make_plan → execute_plan(以当日收盘价成交)。

        返回 (末态, 每日 peak 列表, 全部动作序列, 每日订单数列表)。
        全程走生产路径,不做任何手工注入状态。
        """
        dates = pd.bdate_range("2024-01-01", periods=len(closes))
        state = LiveState(cash=100_000.0)
        peaks, actions, per_day = [], [], []
        for i in range(start_i, len(closes)):
            sub = {"ONE": synth(closes[: i + 1], start=str(dates[0].date()))}
            plan, state = make_plan(sub, state, cfg)
            actions += [o["action"] for o in plan["orders"]]
            per_day.append(len(plan["orders"]))
            if plan["orders"]:
                state = execute_plan(plan, state, {o["symbol"]: closes[i] for o in plan["orders"]}, cfg)
            if state.holdings:
                peaks.append(state.holdings["ONE"].peak)
        return state, peaks, actions, per_day

    def test_peak_never_regresses_and_tracks_holding_period_max(self):
        """回归:peak 必须逐日单调不减,且最终等于持仓期最高收盘价。

        旧版 make_plan 丢弃了 compute_target_set 就地推进的 peaks,落盘的
        peak 永远等于开仓价 —— 实盘 trailing_stop 形同虚设。
        """
        closes = self.trail_series()
        cfg3 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, trailing_stop=0.99)   # 阈值极松:只观察锚点,不触发离场
        _, peaks, _, _ = self._run_days(closes, cfg3)
        assert peaks, "应已建仓"
        assert peaks == sorted(peaks), f"peak 出现回退: {peaks}"
        assert max(closes) == pytest.approx(max(peaks))
        assert peaks[-1] > max(closes) * 0.99   # 锚点确实跟到了急涨高点,而非停在开仓价

    def test_trailing_stop_fires_via_production_state(self):
        """端到端回归:走完整 make_plan/execute_plan 链路时移动止损必须真的触发。

        旧版峰值锚点从不更新 -> 阈值退化为"开仓价×0.92",而本序列回落终点仍
        高于开仓价 20%+,故旧版一次都不会卖出。
        """
        closes = self.trail_series()
        cfg_stop = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                                  commission=0.0, trailing_stop=0.08)
        cfg_off = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                                 commission=0.0)
        _, _, actions_on, _ = self._run_days(closes, cfg_stop)
        _, _, actions_off, _ = self._run_days(closes, cfg_off)
        assert actions_off == ["BUY"], f"对照组只应建仓一次,实际 {actions_off}"
        assert "SELL" in actions_on, f"移动止损应触发卖出,实际 {actions_on}"

    def test_confirm_days_streak_persists_across_runs(self):
        """回归:空仓时 confirm_days=2 须连续两天评分第一才建仓。

        旧版丢弃了 top_streak 的推进(每次调用都从 0 重来),实盘会永久空仓。
        """
        closes = geom(10, 1.01, 60)
        cfg3 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, confirm_days=2)
        _, _, _, per_day = self._run_days(closes, cfg3, start_i=15)
        assert per_day[0] == 0, "第一天不该建仓(评分连击仅 1 天)"
        assert any(n > 0 for n in per_day[1:]), "连续第二天起应能建仓(旧版永远不建仓)"


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

    def test_n_hold_two_splits_capital_into_equal_slots(self):
        """回归:execute_plan 必须按 n_hold 等分槽位并全额投入。

        (原有 execute_plan 测试全是 n_hold=1 的夹具,槽位逻辑无人把关:
        把 total/n_hold 改成 total 也能全绿地通过。)
        """
        data = {"UP": synth(geom(10, 1.01, 60)), "MID": synth(geom(10, 1.005, 60)),
                "FLAT": synth([10.0] * 60)}
        cfg2 = StrategyConfig(bias_ma_n=5, momentum_day=10, slope_n=10, efficiency_n=10,
                              commission=0.0, n_hold=2)
        plan, ns = make_plan(data, LiveState(cash=100_000), cfg2)
        assert len(plan["target"]) == 2
        fills = dict.fromkeys(plan["target"], 10.0)
        final = execute_plan(plan, ns, fills, cfg2)
        values = sorted(h.shares * 10.0 for h in final.holdings.values())
        assert len(values) == 2
        assert sum(values) == pytest.approx(100_000, rel=1e-9)    # 全额投入,不留现金
        assert values[1] == pytest.approx(values[0], rel=1e-9)    # 两个槽位等权
        assert final.cash == pytest.approx(0.0, abs=1e-9)

    def test_replaying_a_sell_plan_raises_instead_of_corrupting(self, data, cfg):
        """回归:重放已执行过的计划必须显式报错,而不是静默改坏账户。"""
        state = LiveState(cash=0.0, holdings={"DOWN": Holding("DOWN", 1000.0, 15.0, 15.0)})
        plan, ns = make_plan(data, state, cfg)
        fills = {"DOWN": 16.0, "UP": 15.0}
        done = execute_plan(plan, ns, fills, cfg)
        with pytest.raises(ValueError, match="已被执行过"):
            execute_plan(plan, done, fills, cfg)


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


class TestCheckFillableAtOpen:
    """限价单开盘成交判定(cli execute 的整批执行/整批放弃依据)。"""

    PLAN = {"orders": [
        {"action": "SELL", "symbol": "A", "limit_price": 9.90},   # 昨收 10 × 0.99
        {"action": "BUY", "symbol": "B", "limit_price": 10.10},   # 昨收 10 × 1.01
    ]}

    def test_all_fill_when_open_within_limits(self):
        assert check_fillable_at_open(self.PLAN, {"A": 10.0, "B": 10.0}) == []

    def test_gap_up_blocks_buy(self):
        problems = check_fillable_at_open(self.PLAN, {"A": 10.0, "B": 10.20})
        assert len(problems) == 1 and "B" in problems[0]

    def test_gap_down_blocks_sell(self):
        problems = check_fillable_at_open(self.PLAN, {"A": 9.80, "B": 10.0})
        assert len(problems) == 1 and "A" in problems[0]

    def test_missing_open_price_is_a_problem(self):
        problems = check_fillable_at_open(self.PLAN, {"A": 10.0})
        assert len(problems) == 1 and "B" in problems[0]


class TestPlanExecutionTracking:
    """execute 的幂等保护:计划是否已执行,按 plan_date + data_asof 匹配。"""

    PLAN = {"plan_date": "2026-09-29", "data_asof": "2026-09-28", "orders": []}

    def test_latest_plan_and_executed_roundtrip(self, tmp_path):
        store = StateStore(tmp_path)
        assert store.latest_plan() is None
        store.append_log({"type": "plan", **self.PLAN, "orders": [{"action": "BUY"}]})
        plan = store.latest_plan()
        assert plan["plan_date"] == "2026-09-29"
        assert store.plan_executed(plan) is False
        store.append_log({"type": "fill", "plan_date": "2026-09-29",
                          "data_asof": "2026-09-28", "fills": []})
        assert store.plan_executed(plan) is True

    def test_fill_of_older_plan_does_not_mark_newer(self, tmp_path):
        store = StateStore(tmp_path)
        store.append_log({"type": "fill", "plan_date": "2026-09-26",
                          "data_asof": "2026-09-25", "fills": []})
        assert store.plan_executed(self.PLAN) is False


class TestGmAdapterMapping:
    """掘金适配器的纯映射逻辑(不连掘金,gm 库惰性导入所以可加载)。"""

    @staticmethod
    def _load():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "gm_paper", Path(__file__).resolve().parents[1] / "scripts" / "gm_paper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_symbol_prefix_and_order_shape(self):
        mod = self._load()
        plan = {"orders": [
            {"action": "SELL", "symbol": "518880", "limit_price": 8.5, "est_budget": 100.0},
            {"action": "BUY", "symbol": "513100", "limit_price": 2.4, "est_budget": 50000.0},
            {"action": "BUY", "symbol": "159949", "limit_price": 1.5, "est_budget": 50000.0},
        ]}
        orders = mod.plan_to_gm_orders(plan)
        assert orders[0] == {"action": "close_all", "symbol": "SHSE.518880", "reason": ""}
        assert orders[1]["action"] == "buy_value" and orders[1]["symbol"] == "SHSE.513100"
        assert orders[1]["value"] == 50000.0 and orders[1]["price"] == 2.4
        assert orders[2]["symbol"] == "SZSE.159949"     # 深市前缀
        assert [o["action"] for o in orders] == ["close_all", "buy_value", "buy_value"]  # 先卖后买
