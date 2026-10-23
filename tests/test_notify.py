"""Bark 例行任务通知:消息摘要、密钥读取与 HTTP 请求格式。"""

import json

import pandas as pd
import pytest

import etf_rotation.notify as notify
from etf_rotation.config import StrategyConfig
from etf_rotation.live import Holding, LiveState, StateStore


class _Response:
    def __init__(self, payload: bytes = b'{"code": 200}'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


@pytest.fixture
def paper_fixture(tmp_path):
    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = StateStore(state_dir)
    shares = 50_000 * (1 - 0.0001) / 1.197
    store.save_state(LiveState(
        cash=50_000.0,
        holdings={"512890": Holding("512890", shares, 1.197, 1.231)},
        realized_pnl=0.0,
    ))
    store.save_params(StrategyConfig(
        initial_cash=100_000.0,
        n_hold=2,
        trailing_stop=0.08,
        premium_max=0.07,
    ))
    store.append_log({
        "type": "plan",
        "plan_date": "2026-10-16",
        "data_asof": "2026-10-16",
        "orders": [],
        "scores": {
            "512890": 0.476,
            "159949": -1.5226,
            "513100": 1.1641,
            "518880": -0.1175,
        },
        "premium_check": {
            "513100": {"premium": 0.1955, "asof": "2026-10-15", "nav_lag": 1, "stale": False},
        },
    })

    pd.DataFrame({
        "date": [pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-16")],
        "open": [1.228, 1.214],
        "high": [1.230, 1.226],
        "low": [1.205, 1.211],
        "close": [1.212, 1.219],
        "volume": [1.0, 1.0],
    }).to_csv(data_dir / "512890.csv", index=False)

    log_file = tmp_path / "paper.log"
    log_file.write_text(
        "=== 2026-10-19 13:18:03 fetch + plan + execute ===\n"
        "512890 红利低波ETF: 1874 bars, 2019-01-18 → 2026-10-16\n"
        "溢价数据(raw+nav)已刷新\n"
        "仪表盘: state/dashboard.html\n"
    )
    return state_dir, data_dir, log_file


def test_load_bark_key_from_project_env(tmp_path, monkeypatch):
    monkeypatch.delenv("BARK_KEY", raising=False)
    (tmp_path / ".env").write_text("TUSHARE_RELAY_TOKEN=secret\nBARK_KEY=bark-key\n")
    assert notify.load_bark_key(tmp_path) == "bark-key"


def test_load_bark_key_prefers_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("BARK_KEY=file-key\n")
    monkeypatch.setenv("BARK_KEY", "env-key")
    assert notify.load_bark_key(tmp_path) == "env-key"


def test_send_daily_notification_requires_key(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "load_bark_key", lambda: None)
    with pytest.raises(RuntimeError, match="BARK_KEY"):
        notify.send_daily_notification(tmp_path, tmp_path, tmp_path / "paper.log")


def test_log_summary_success_and_failure():
    status, issues = notify.summarize_log(["=== run ===", "数据已刷新", "仪表盘: ok"])
    assert status == "正常" and issues == []

    status, issues = notify.summarize_log([
        "=== run ===",
        "Traceback (most recent call last):",
        "requests.exceptions.ConnectionError: network down",
    ])
    assert status == "异常"
    assert issues and "ConnectionError" in issues[-1]


def test_log_summary_marks_unfilled_order_as_warning():
    status, issues = notify.summarize_log([
        "=== run ===",
        "以下限价单在开盘价未成交,本次整批不执行(可稍候重跑):",
    ])
    assert status == "有警告"
    assert "未成交" in issues[-1] and "不执行" in issues[-1]


def test_log_summary_traceback_without_error_line_uses_last_meaningful_line():
    status, issues = notify.summarize_log([
        "=== run ===",
        "Traceback (most recent call last):",
        "KeyboardInterrupt",
    ])
    assert status == "异常"
    assert issues == ["KeyboardInterrupt"]


def test_build_notification_contains_plan_scores_and_account(paper_fixture):
    state_dir, data_dir, log_file = paper_fixture
    title, body = notify.build_notification(state_dir, data_dir, log_file, "morning")

    assert title == "✅ ETF轮动｜早间正常"
    assert "状态：正常｜行情 2026-10-16" in body
    assert "【计划】\n计划 2026-10-16｜信号 2026-10-16\n无订单（持有不变/空仓）" in body
    assert "【评分 2026-10-16】\n513100 +1.1641\n512890 +0.4760\n518880 -0.1175\n159949 -1.5226" in body
    assert "【收盘 2026-10-16】\n512890 红利低波ETF 1.219" in body
    assert "【账户】\n权益 100,913.87\n收益 +913.87（+0.91%）" in body
    assert "【持仓】\n512890 红利低波ETF\n41,767份｜1.197→1.219\n浮盈 +1.84%｜+919元\n止损线 1.133" in body
    assert "【提示】\n513100 门槛溢价 19.6%｜禁新开仓" in body


def test_send_bark_uses_post_json(monkeypatch):
    called = {}

    def fake_urlopen(request, timeout):
        called["request"] = request
        called["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    notify.send_bark("test key", "ETF 标题", "第一行\n第二行", group="测试", ttl=60)

    request = called["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert called["timeout"] == 10
    assert request.full_url == "https://api.day.app/push"
    assert request.get_method() == "POST"
    assert request.headers["Content-type"] == "application/json; charset=utf-8"
    assert payload == {
        "device_key": "test key",
        "title": "ETF 标题",
        "body": "第一行\n第二行",
        "group": "测试",
        "ttl": 60,
    }


def test_send_bark_rejects_api_failure(monkeypatch):
    monkeypatch.setattr(notify.urllib.request, "urlopen",
                        lambda *a, **k: _Response(b'{"code": 400, "message": "bad key"}'))
    with pytest.raises(RuntimeError, match="bad key"):
        notify.send_bark("bad", "title", "body")


def test_load_bark_key_accepts_export_whitespace_and_comment(tmp_path, monkeypatch):
    monkeypatch.delenv("BARK_KEY", raising=False)
    (tmp_path / ".env").write_text("export BARK_KEY = \"quoted-key\"  # local\n")
    assert notify.load_bark_key(tmp_path) == "quoted-key"


def test_load_bark_key_preserves_hash_inside_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("BARK_KEY", raising=False)
    (tmp_path / ".env").write_text('BARK_KEY="ab#cd"\n')
    assert notify.load_bark_key(tmp_path) == "ab#cd"


def test_send_bark_rejects_plain_http():
    with pytest.raises(RuntimeError, match="https"):
        notify.send_bark("key", "title", "body", server="http://example.com")


def test_log_summary_strips_local_path_and_suppresses_holiday_noise():
    block = [
        "=== run ===",
        "/Users/apple/Desktop/demo/src/data.py:205: UserWarning: 513100 取数失败(timeout),回退东财。",
    ]
    status, issues = notify.summarize_log(block)
    assert status == "有警告"
    assert issues == ["513100 取数失败(timeout),回退东财。"]

    holiday = ["=== run ===", "以下限价单在开盘价未成交,本次整批不执行(可稍候重跑):"]
    assert notify.summarize_log(holiday, trading_day=False) == ("正常", [])

    frame = '  File "/Users/apple/Desktop/demo project/src/data.py", line 205, in fetch_etf'
    assert notify._clean_issue(frame) == "Traceback: fetch_etf (line 205)"
    spaced_warning = "/Users/apple/Desktop/demo project/src/data.py:205: UserWarning: 513100 取数失败"
    assert notify._clean_issue(spaced_warning) == "513100 取数失败"


def test_premium_lines_include_blocked_and_stale():
    plan = {"premium_check": {
        "A": {"premium": 0.09, "gate_premium": 0.08, "stale": False},
        "B": {"premium": 0.01, "gate_premium": 0.01, "stale": True},
    }}
    lines = notify._premium_lines(plan, StrategyConfig(premium_max=0.07))
    assert lines == ["【提示】", "A 门槛溢价 8.0%｜禁新开仓（最新估值 9.0%）", "B 溢价数据滞后"]


def test_plan_lines_pending_and_executed_with_malformed_log_line(tmp_path):
    store = StateStore(tmp_path)
    plan = {"type": "plan", "plan_date": "2026-10-19", "data_asof": "2026-10-16",
            "orders": [{"action": "BUY", "symbol": "512890", "limit_price": 1.208,
                        "est_budget": 50_000.0}]}
    store.append_log(plan)
    lines, _ = notify._plan_lines(store)
    assert "待执行" in lines and "限价 1.208｜约 50,000元" in lines

    with store.log_file.open("a") as f:
        f.write("not-json\n")
    store.append_log({"type": "fill", "plan_date": "2026-10-19", "data_asof": "2026-10-16",
                      "fills": [{"action": "BUY", "symbol": "512890", "fill_price": 1.197}]})
    lines, _ = notify._plan_lines(store)
    assert "已成交" in lines and "BUY 512890｜成交价 1.197" in lines


def test_record_notify_status(tmp_path):
    notify.record_notify_status(tmp_path, False, "api failed")
    payload = json.loads((tmp_path / "notify_status.json").read_text())
    assert payload["ok"] is False and payload["message"] == "api failed" and payload["at"]


def test_latest_log_block_uses_last_task_header(tmp_path):
    log = tmp_path / "paper.log"
    log.write_text("=== old ===\nold warning\n=== new ===\nnew line\n")
    assert notify.latest_log_block(log) == ["=== new ===", "new line"]
