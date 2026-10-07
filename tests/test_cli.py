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


class TestDashboard:
    def test_render_contains_key_sections(self, tmp_path):
        import pandas as pd
        from etf_rotation.dashboard import record_equity, render_dashboard

        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=49_000.0,
                                   holdings={"518880": Holding("518880", 5000.0, 8.0, 8.5)},
                                   realized_pnl=123.0))
        df = pd.DataFrame({"date": pd.bdate_range("2026-09-28", periods=3),
                           "open": 8.6, "high": 8.7, "low": 8.5, "close": 8.6, "volume": 1e6})
        out = render_dashboard(store, {"518880": df}, tmp_path / "dashboard.html")
        html = out.read_text()
        assert "518880" in html and "总权益" in html and "成交历史" in html
        assert "+7.50%" in html          # 8.6/8.0 - 1 浮盈
        # 权益记录按日期去重:同一天重复渲染不追加重复行
        record_equity(store, 49_000.0, 43_000.0)
        lines = (tmp_path / "equity.csv").read_text().splitlines()
        assert len(lines) == 2           # header + 今日一行
