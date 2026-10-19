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
    assert "【评分】\n513100 +1.1641\n512890 +0.4760\n518880 -0.1175\n159949 -1.5226" in body
    assert "【账户】\n权益 100,913.87\n收益 +913.87（+0.91%）" in body
    assert "【持仓】\n512890 红利低波ETF\n41,767份｜1.197→1.219\n浮盈 +1.84%｜+919元\n止损线 1.133" in body
    assert "【提示】\n513100 溢价 19.6%｜禁新开仓" in body


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
