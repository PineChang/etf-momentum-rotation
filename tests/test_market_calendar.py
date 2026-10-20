"""交易日历接入测试:data_source 可用时识别交易日,不可用时安全回退。"""

import pandas as pd

import etf_rotation.market_calendar as mc


class _FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def calendar(self, start, end, open_only=True):
        days = [pd.Timestamp(day) for day, is_open in self.rows.items()
                if pd.Timestamp(start) <= pd.Timestamp(day) <= pd.Timestamp(end)
                and (is_open or not open_only)]
        return pd.Series(days)


def test_is_trading_day_true_and_false(monkeypatch):
    monkeypatch.setattr(mc, "_load_data_source",
                        lambda: _FakeDS({"2026-10-19": True, "2026-10-18": False}))
    assert mc.is_trading_day("2026-10-19") is True
    assert mc.is_trading_day("2026-10-18") is False


def test_is_trading_day_missing_date_returns_none(monkeypatch):
    """日历没有当天记录时无法确认,不能让执行层误判成休市。"""
    monkeypatch.setattr(mc, "_load_data_source", lambda: _FakeDS({"2026-10-19": True}))
    assert mc.is_trading_day("2027-01-04") is None


def test_is_trading_day_unavailable_returns_none(monkeypatch):
    monkeypatch.setattr(mc, "_load_data_source", lambda: None)
    assert mc.is_trading_day("2026-10-19") is None
