"""cli 层测试:plan 幂等守卫(_plan_unchanged)与 status 展示。"""

import json

import pytest

import etf_rotation.cli as cli_mod
from etf_rotation.cli import _plan_unchanged, cmd_status
from etf_rotation.config import StrategyConfig
from etf_rotation.live import Holding, LiveState, StateStore

PLAN = {"plan_date": "2026-10-07", "data_asof": "2026-09-30", "orders": []}


class TestPlanUnchanged:
    def test_same_signal_and_params_skips(self, tmp_path):
        cfg = StrategyConfig()
        store = StateStore(tmp_path)
        store.save_params(cfg)
        store.append_log({"type": "plan", **PLAN})
        assert _plan_unchanged(store, {"data_asof": "2026-09-30"}, cfg) is True

    def test_new_signal_day_does_not_skip(self, tmp_path):
        cfg = StrategyConfig()
        store = StateStore(tmp_path)
        store.save_params(cfg)
        store.append_log({"type": "plan", **PLAN})
        assert _plan_unchanged(store, {"data_asof": "2026-10-08"}, cfg) is False

    def test_changed_params_does_not_skip(self, tmp_path):
        cfg = StrategyConfig()
        store = StateStore(tmp_path)
        store.save_params(cfg)
        store.append_log({"type": "plan", **PLAN})
        assert _plan_unchanged(store, {"data_asof": "2026-09-30"},
                               StrategyConfig(n_hold=2)) is False

    def test_no_prior_plan_does_not_skip(self, tmp_path):
        assert _plan_unchanged(StateStore(tmp_path), {"data_asof": "2026-09-30"},
                               StrategyConfig()) is False


class TestCmdStatus:
    def test_prints_holdings_and_pnl(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli_mod, "STATE_DIR", tmp_path)
        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=50_000.0,
                                   holdings={"518880": Holding("518880", 5000.0, 8.0, 8.5)},
                                   realized_pnl=123.0))
        cmd_status(None)
        out = capsys.readouterr().out
        assert "518880" in out and "浮盈亏" in out and "总权益" in out
        assert "空仓" not in out.split("持仓")[0]
