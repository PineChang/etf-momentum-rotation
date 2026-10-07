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


class TestDualTrack:
    """双轨对照:compare_fills 配对与价差,gm 快照渲染,未配置占位。"""

    LOCAL_FILLS = [{"plan_date": "2026-10-07",
                    "fills": [{"action": "BUY", "symbol": "512890", "fill_price": 1.190,
                               "limit_price": 1.208}]}]
    GM_ORDERS = [{"plan_date": "2026-10-07", "symbol": "512890", "side": "BUY",
                  "limit_price": 1.208, "est_budget": 50000.0}]
    GM_EXECS = [{"symbol": "512890", "side": "BUY", "price": 1.195, "volume": 40000,
                 "created_at": "2026-10-08 09:31:05"},
                {"symbol": "512890", "side": "BUY", "price": 1.185, "volume": 1300,
                 "created_at": "2026-10-08 09:31:05"}]

    def test_compare_fills_vwap_and_bp(self):
        from etf_rotation.dashboard import compare_fills
        rows = compare_fills(self.LOCAL_FILLS, self.GM_ORDERS, self.GM_EXECS)
        assert len(rows) == 1
        r = rows[0]
        # B 轨成交均价 = (1.195*40000 + 1.185*1300) / 41300 ≈ 1.19469
        assert r["b_price"] == pytest.approx(1.19469, rel=1e-4)
        assert r["b_volume"] == 41300
        assert r["diff_bp"] == pytest.approx((1.19469 / 1.190 - 1) * 1e4, abs=0.2)

    def test_compare_fills_without_gm_data(self):
        from etf_rotation.dashboard import compare_fills
        rows = compare_fills(self.LOCAL_FILLS, [], [])
        assert rows[0]["b_price"] is None and rows[0]["diff_bp"] is None

    def test_render_with_gm_snapshot(self, tmp_path):
        import pandas as pd
        from etf_rotation.dashboard import render_dashboard
        from scripts.gm_paper import write_gm_snapshot  # noqa: 惰性 gm 导入,可加载

        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=100_000.0))
        store.append_log({"type": "fill", **self.LOCAL_FILLS[0],
                          "data_asof": "2026-09-30"})
        (tmp_path / "gm_orders.jsonl").write_text(
            json.dumps(self.GM_ORDERS[0], ensure_ascii=False) + "\n")
        write_gm_snapshot(tmp_path, {"nav": 100_500.0, "cash_available": 50_500.0},
                          [{"symbol": "512890", "volume": 41300, "vwap": 1.1947,
                            "market_value": 49_400.0}], self.GM_EXECS)
        df = pd.DataFrame({"date": pd.bdate_range("2026-09-28", periods=3),
                           "open": 1.2, "high": 1.2, "low": 1.2, "close": 1.2, "volume": 1e6})
        html = render_dashboard(store, {"512890": df}, tmp_path / "dashboard.html").read_text()
        assert "B 轨 · 掘金仿真净值" in html and "100,500" in html
        assert "双轨逐笔对照" in html and "+39.4" in html   # diff_bp ≈ +39.4

    def test_render_without_gm_shows_placeholder(self, tmp_path):
        import pandas as pd
        from etf_rotation.dashboard import render_dashboard
        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=100_000.0))
        df = pd.DataFrame({"date": pd.bdate_range("2026-09-28", periods=3),
                           "open": 1.2, "high": 1.2, "low": 1.2, "close": 1.2, "volume": 1e6})
        html = render_dashboard(store, {"512890": df}, tmp_path / "dashboard.html").read_text()
        assert "未配置" in html
