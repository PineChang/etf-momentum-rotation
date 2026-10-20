"""cli 层测试:plan 幂等守卫(_plan_unchanged)、status/仪表盘、双轨适配,
以及 cmd_plan / cmd_execute 的日常自动化端到端路径。"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import etf_rotation.cli as cli_mod
from etf_rotation.cli import _plan_unchanged, cmd_execute, cmd_plan, cmd_status
from etf_rotation.config import StrategyConfig
from etf_rotation.live import Holding, LiveState, StateStore

PLAN = {"plan_date": "2026-10-07", "data_asof": "2026-09-30", "orders": []}
TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")


def _synth(closes):
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame({"date": dates, "open": s, "high": s, "low": s, "close": s, "volume": 1e6})


def _log(store: StateStore, kind: str) -> list[dict]:
    if not store.log_file.exists():
        return []
    return [r for r in (json.loads(x) for x in store.log_file.read_text().splitlines())
            if r.get("type") == kind]


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

    def test_premium_change_invalidates_same_signal_plan(self, tmp_path):
        """净值晚些时候补发时,同一信号日也必须重新落计划(溢价门槛可能改变)。"""
        cfg = StrategyConfig(premium_max=0.07)
        store = StateStore(tmp_path)
        store.save_params(cfg)
        old = {"A": {"premium": 0.06, "gate_premium": 0.06, "asof": "2026-10-15"}}
        new = {"A": {"premium": 0.08, "gate_premium": 0.08, "asof": "2026-10-16"}}
        store.append_log({"type": "plan", "data_asof": "2026-09-30", "premium_check": old})
        assert _plan_unchanged(store, {"data_asof": "2026-09-30"}, cfg, old) is True
        assert _plan_unchanged(store, {"data_asof": "2026-09-30"}, cfg, new) is False

    def test_stale_display_flag_does_not_duplicate_same_plan(self, tmp_path):
        """stale 是随 wall-clock 变化的展示字段,不应破坏同一信号日幂等。"""
        cfg = StrategyConfig(premium_max=0.07)
        store = StateStore(tmp_path)
        store.save_params(cfg)
        old = {"A": {"premium": 0.06, "gate_premium": 0.06, "asof": "2026-10-15",
                     "nav_lag": 1, "missing": False, "stale": False}}
        new = {**old, "A": {**old["A"], "stale": True}}
        store.append_log({"type": "plan", "data_asof": "2026-09-30", "premium_check": old})
        assert _plan_unchanged(store, {"data_asof": "2026-09-30"}, cfg, new) is True


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

    def test_status_tolerates_malformed_log_line(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli_mod, "STATE_DIR", tmp_path)
        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=100_000.0))
        store.log_file.write_text("not-json\n")
        cmd_status(None)
        assert "总权益" in capsys.readouterr().out


class TestCmdPlan:
    """日常自动化第一段:cli plan 落盘 + 二次运行幂等(此前端到端零覆盖)。"""

    @pytest.fixture
    def cli_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "STATE_DIR", tmp_path)
        return StateStore(tmp_path)

    @pytest.fixture
    def up_data(self):
        return {"512890": _synth([10 + 0.1 * i for i in range(60)]),
                "159949": _synth([10.0] * 60)}

    def test_plan_written_then_skipped_on_rerun(self, up_data, cli_state, monkeypatch, capsys):
        monkeypatch.setattr(cli_mod, "fetch_universe", lambda *a, **k: up_data)
        args = SimpleNamespace(preset="original")
        cmd_plan(args)
        plans = _log(cli_state, "plan")
        assert len(plans) == 1
        assert plans[0]["data_asof"] == str(up_data["512890"]["date"].iloc[-1].date())
        assert plans[0]["data_lag"] == {}          # 数据完整:无回退告警
        assert cli_state.state_file.exists() and cli_state.params_file.exists()
        cmd_plan(args)                              # 同一信号日 + 同一参数 → 幂等跳过
        assert len(_log(cli_state, "plan")) == 1
        assert "已生成过且参数未变" in capsys.readouterr().out

    def test_missing_symbol_bar_warns_and_plan_day_falls_back(self, cli_state, monkeypatch, capsys):
        """回归:单只标的缺当日 bar 时,信号日被交集回退,必须告警而不是静默。"""
        full = _synth([10 + 0.1 * i for i in range(60)])
        monkeypatch.setattr(cli_mod, "fetch_universe",
                            lambda *a, **k: {"512890": full, "159949": full.iloc[:-1]})
        cmd_plan(SimpleNamespace(preset="original"))
        err = capsys.readouterr().err
        assert "信号日" in err and "512890" in err
        plan = _log(cli_state, "plan")[0]
        assert plan["data_lag"] == {"512890": str(full["date"].iloc[-1].date())}
        assert plan["data_asof"] == str(full["date"].iloc[-2].date())


class TestCmdExecute:
    """日常自动化第二段:按开盘价成交最近计划(整批成交/整批放弃/幂等/守时)。"""

    @pytest.fixture
    def cli_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "STATE_DIR", tmp_path)
        return StateStore(tmp_path)

    @staticmethod
    def _seed(store: StateStore, plan_date: str, orders: list[dict], cash: float = 100_000.0) -> None:
        store.save_state(LiveState(cash=cash))
        store.save_params(StrategyConfig(commission=0.0, initial_cash=cash))
        store.append_log({"type": "plan", "plan_date": plan_date, "data_asof": "2024-02-01",
                          "orders": orders,
                          "reference_prices": {o["symbol"]: o["limit_price"] for o in orders}})

    BUY = {"action": "BUY", "symbol": "512890", "limit_price": 1.21, "est_budget": 50_000.0}

    def test_fills_at_open_and_records_signal_age(self, cli_state, monkeypatch):
        self._seed(cli_state, TODAY, [self.BUY])
        monkeypatch.setattr(cli_mod, "fetch_today_opens", lambda syms, **k: {"512890": 1.20})
        cmd_execute(None)
        state = cli_state.load_state()
        assert state.holdings["512890"].shares > 0
        fills = _log(cli_state, "fill")
        assert len(fills) == 1
        assert fills[0]["fills"][0]["fill_price"] == 1.20
        assert fills[0]["signal_age_days"] == 0

    def test_gap_blocks_whole_batch_and_leaves_trace(self, cli_state, monkeypatch):
        self._seed(cli_state, TODAY, [self.BUY])
        monkeypatch.setattr(cli_mod, "fetch_today_opens", lambda syms, **k: {"512890": 1.25})
        cmd_execute(None)
        assert cli_state.load_state().holdings == {}          # 整批放弃,不半执行
        skips = _log(cli_state, "skip")                       # 未成交必须留痕(否则无人知道)
        assert len(skips) == 1 and "高开" in skips[0]["problems"][0]
        assert _log(cli_state, "fill") == []

    def test_second_run_refuses_replay(self, cli_state, monkeypatch, capsys):
        self._seed(cli_state, TODAY, [self.BUY])
        monkeypatch.setattr(cli_mod, "fetch_today_opens", lambda syms, **k: {"512890": 1.20})
        cmd_execute(None)
        cmd_execute(None)
        assert len(_log(cli_state, "fill")) == 1              # 只成交一次
        assert "拒绝重复执行" in capsys.readouterr().out

    def test_stale_plan_refused_before_any_data_fetch(self, cli_state, monkeypatch, capsys):
        old = (pd.Timestamp.today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        self._seed(cli_state, old, [self.BUY])
        monkeypatch.setattr(cli_mod, "fetch_today_opens",
                            lambda *a, **k: pytest.fail("陈旧计划不应拉取行情"))
        cmd_execute(None)
        assert "信号可能过时" in capsys.readouterr().out
        assert _log(cli_state, "fill") == []

    def test_source_failure_skips_cleanly(self, cli_state, monkeypatch, capsys):
        """非交易日/行情源故障:打印原因后静默退出,不写任何成交或 skip 记录。"""
        self._seed(cli_state, TODAY, [self.BUY])

        def boom(syms, **k):
            raise RuntimeError("今日 bar 尚未生成——非交易日或行情未开始")

        monkeypatch.setattr(cli_mod, "fetch_today_opens", boom)
        cmd_execute(None)
        assert "今日无法执行" in capsys.readouterr().out
        assert _log(cli_state, "fill") == [] and _log(cli_state, "skip") == []
        assert cli_state.load_state().holdings == {}

    def test_exchange_holiday_skips_before_open_price_fetch(self, cli_state, monkeypatch, capsys):
        """工作日但交易所休市:日历确认后直接跳过,不触发 15 分钟开盘价重试。"""
        import etf_rotation.market_calendar as market_calendar

        self._seed(cli_state, TODAY, [self.BUY])
        monkeypatch.setattr(market_calendar, "is_trading_day", lambda: False)
        monkeypatch.setattr(cli_mod, "fetch_today_opens",
                            lambda *a, **k: pytest.fail("休市日不应拉取开盘价"))
        cmd_execute(None)
        assert "今日非交易日" in capsys.readouterr().out
        assert _log(cli_state, "fill") == [] and _log(cli_state, "skip") == []

    def test_empty_order_plan_is_noop(self, cli_state, monkeypatch, capsys):
        self._seed(cli_state, TODAY, [])
        monkeypatch.setattr(cli_mod, "fetch_today_opens",
                            lambda *a, **k: pytest.fail("无订单不应拉取行情"))
        cmd_execute(None)
        assert "无订单" in capsys.readouterr().out
        assert _log(cli_state, "fill") == []


class TestDashboard:
    def test_render_contains_key_sections(self, tmp_path):
        import pandas as pd
        from etf_rotation.dashboard import record_equity, render_dashboard

        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=49_000.0,
                                   holdings={"518880": Holding("518880", 5000.0, 8.0, 8.5)},
                                   realized_pnl=123.0))
        store.log_file.write_text("not-json\n")
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
        rows = compare_fills(self.LOCAL_FILLS, self.GM_EXECS)
        assert len(rows) == 1
        r = rows[0]
        # B 轨成交均价 = (1.195*40000 + 1.185*1300) / 41300 ≈ 1.19469
        assert r["b_price"] == pytest.approx(1.19469, rel=1e-4)
        assert r["b_volume"] == 41300
        assert r["diff_bp"] == pytest.approx((1.19469 / 1.190 - 1) * 1e4, abs=0.2)

    def test_compare_fills_without_gm_data(self):
        from etf_rotation.dashboard import compare_fills
        rows = compare_fills(self.LOCAL_FILLS, [])
        assert rows[0]["b_price"] is None and rows[0]["diff_bp"] is None

    def test_render_with_gm_snapshot(self, tmp_path):
        import pandas as pd
        from etf_rotation.dashboard import render_dashboard
        from scripts.gm_paper import write_gm_snapshot  # noqa: 惰性 gm 导入,可加载

        store = StateStore(tmp_path)
        store.save_state(LiveState(cash=100_000.0))
        store.append_log({"type": "fill", **self.LOCAL_FILLS[0],
                          "data_asof": "2026-09-30"})
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


class TestGmCsvImport:
    """掘金仿真 CSV 下载 → gm_snapshot.json 导入(Mac 手工路径)。"""

    def test_import_cash_position_exec(self, tmp_path, monkeypatch):
        import importlib.util
        from pathlib import Path

        import pandas as pd

        spec = importlib.util.spec_from_file_location(
            "gm_import_csv", Path(__file__).resolve().parents[1] / "scripts" / "gm_import_csv.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        src = tmp_path / "dl"
        src.mkdir()
        pd.DataFrame({"symbol": ["SHSE.512890"], "nav": [100500.0],
                      "available": [50500.0], "frozen": [0.0]}).to_csv(src / "cash.csv", index=False)
        pd.DataFrame({"symbol": ["SHSE.512890"], "volume": [41300],
                      "vwap": [1.1947], "market_value": [49400.0]}).to_csv(src / "position.csv", index=False)
        pd.DataFrame({"symbol": ["SHSE.512890"], "side": [1], "price": [1.195],
                      "volume": [41300], "created_at": ["2026-10-08 09:31:05"]}
                     ).to_csv(src / "execution.csv", index=False)

        state = tmp_path / "state"
        state.mkdir()
        tables = mod._read_tables(src)
        assert set(tables) == {"cash", "position", "exec"}
        from gm_paper import write_gm_snapshot  # noqa: scripts/ 命名空间可导入(gm 惰性)
        account = {"nav": 100500.0, "cash_available": 50500.0, "frozen": 0.0}
        snap = write_gm_snapshot(state, account,
                                 [{"symbol": "512890", "volume": 41300, "vwap": 1.1947,
                                   "market_value": 49400.0}],
                                 [{"symbol": "512890", "side": "BUY", "price": 1.195,
                                   "volume": 41300, "created_at": "2026-10-08 09:31:05"}])
        data = json.loads(snap.read_text())
        assert data["account"]["nav"] == 100500.0
        assert data["executions"][0]["side"] == "BUY"
        # 中文表头也应识别(映射多候选)
        pd.DataFrame({"代码": ["SHSE.512890"], "方向": ["买入"], "成交价": [1.19],
                      "成交量": [41300], "成交时间": ["2026-10-08 09:31:05"]}
                     ).to_csv(src / "execution_cn.csv", index=False)
        df = pd.read_csv(src / "execution_cn.csv")
        assert mod._pick(df, "symbol") == "代码" and mod._pick(df, "price") == "成交价"
