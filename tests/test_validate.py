"""validate.py 数据校验核准的测试。"""

import pandas as pd
import pytest

from etf_rotation.live import Holding, LiveState, StateStore
from etf_rotation.validate import (CRITICAL, WARNING, crosscheck_with_em, format_report,
                                   has_critical, validate_alignment, validate_calendar,
                                   validate_equity, validate_ohlc)


def _bars(closes, start="2026-10-19"):
    """构造合法 OHLC df:close 按给定序列,open/high/low 围绕 close 合法。"""
    n = len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": [1e6] * n,
    })


class TestOhlc:
    def test_clean_passes(self):
        assert validate_ohlc(_bars([1.20, 1.21, 1.19]), "512890") == []

    def test_empty_is_critical(self):
        df = _bars([]).head(0)
        issues = validate_ohlc(df, "512890")
        assert issues[0].level == CRITICAL

    def test_duplicate_date(self):
        df = _bars([1.20, 1.21])
        df.loc[3] = df.loc[1]                      # 追加重复日期(保持单调以便单独触发)
        issues = validate_ohlc(df, "512890")
        assert any(i.level == CRITICAL and "重复日期" in i.message for i in issues)

    def test_illegal_ohlc_relation(self):
        df = _bars([1.20, 1.21, 1.19])
        df.loc[1, "low"] = 1.30                    # low > open/close
        issues = validate_ohlc(df, "512890")
        assert any(i.level == CRITICAL and "四价关系非法" in i.message for i in issues)

    def test_extreme_move(self):
        df = _bars([1.20, 1.60])                   # +33% 单日,ETF 不可能
        issues = validate_ohlc(df, "512890")
        assert any(i.level == CRITICAL and "单日涨跌" in i.message for i in issues)

    def test_na_close(self):
        df = _bars([1.20, 1.21, 1.19])
        df.loc[1, "close"] = None
        issues = validate_ohlc(df, "512890")
        assert any(i.level == CRITICAL and "缺失值" in i.message for i in issues)


class TestCalendar:
    def test_missing_trading_day_warns(self):
        df = _bars([1.20, 1.21, 1.19])             # 10-19/20/21
        days = set(df["date"])
        df = df[df["date"] != pd.Timestamp("2026-10-20")]  # 挖掉中间一天
        issues = validate_calendar(df, "512890", days)
        assert any(i.level == WARNING and "缺 1 个交易日" in i.message for i in issues)

    def test_holiday_bar_is_critical(self):
        df = _bars([1.20, 1.21, 1.19])
        days = set(df["date"]) - {pd.Timestamp("2026-10-20")}  # 20 日实际是假期
        issues = validate_calendar(df, "512890", days)
        assert any(i.level == CRITICAL and "非交易日" in i.message for i in issues)

    def test_calendar_unavailable_skips(self):
        assert validate_calendar(_bars([1.20]), "512890", None) == []


class TestAlignment:
    def test_mismatch_warns(self):
        data = {"512890": _bars([1.2, 1.21]), "159949": _bars([1.2])}
        # 159949 少最后一天 → 最新日期不一致
        issues = validate_alignment(data)
        assert len(issues) == 1 and issues[0].level == WARNING

    def test_aligned_passes(self):
        assert validate_alignment({"512890": _bars([1.2]), "159949": _bars([1.3])}) == []


class TestEquity:
    def _store(self, tmp_path, total_rows):
        store = StateStore(tmp_path)
        store.save_state(LiveState(
            cash=50_000.0,
            holdings={"512890": Holding("512890", 1000.0, 1.197, 1.236)},
            realized_pnl=0.0))
        store.append_log({"type": "fill", "plan_date": "2026-10-07",
                          "data_asof": "2026-10-07",
                          "fills": [{"action": "BUY", "symbol": "512890", "fill_price": 1.197}]})
        (tmp_path / "equity.csv").write_text("date,cash,market_value,total\n" + total_rows)
        return store

    def test_total_mismatch_critical(self, tmp_path):
        store = self._store(tmp_path, "2026-10-08,50000.00,1210.00,99999.00\n")
        issues = validate_equity(store, store.load_state(), {"512890": _bars([1.21], "2026-10-08")})
        assert any(i.level == CRITICAL and "total ≠" in i.message for i in issues)

    def test_stale_row_warns_and_boundary_respected(self, tmp_path):
        store = self._store(tmp_path,
                            "2026-10-06,100000.00,0.00,100000.00\n"   # 成交边界前,不复算
                            "2026-10-08,50000.00,1200.00,51200.00\n")  # 盘中价冻结(应收 1.21)
        df = pd.DataFrame({"date": pd.to_datetime(["2026-10-06", "2026-10-08"]),
                           "open": 1.2, "high": 1.2, "low": 1.2,
                           "close": [1.19, 1.21], "volume": 1e6})
        issues = validate_equity(store, store.load_state(), {"512890": df})
        assert any(i.level == WARNING and "盘中价冻结" in i.message and "2026-10-08" in i.message
                   for i in issues)
        assert not any("2026-10-06" in i.message for i in issues)

    def test_clean_equity_passes(self, tmp_path):
        store = self._store(tmp_path, "2026-10-08,50000.00,1210.00,51210.00\n")
        df = pd.DataFrame({"date": pd.to_datetime(["2026-10-08"]),
                           "open": 1.2, "high": 1.2, "low": 1.2, "close": [1.21], "volume": 1e6})
        assert validate_equity(store, store.load_state(), {"512890": df}) == []


class TestCrosscheck:
    def test_mismatch_critical(self, tmp_path, monkeypatch):
        # crosscheck 内部在调用时 from .data import DATA_DIR/_fetch_em,打补丁到 data 模块即可
        monkeypatch.setattr("etf_rotation.data.DATA_DIR", tmp_path)
        (tmp_path / "512890.raw.csv").write_text(
            "date,close\n2026-10-21,1.213\n2026-10-22,1.202\n")
        em = pd.DataFrame({"日期": ["2026-10-21", "2026-10-22"], "收盘": [1.213, 1.198]})
        monkeypatch.setattr("etf_rotation.data._fetch_em", lambda *a, **k: em)
        issues = crosscheck_with_em(["512890"], days=2)
        assert any(i.level == CRITICAL and "两源不一致" in i.message for i in issues)

    def test_match_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("etf_rotation.data.DATA_DIR", tmp_path)
        (tmp_path / "512890.raw.csv").write_text(
            "date,close\n2026-10-21,1.213\n2026-10-22,1.202\n")
        em = pd.DataFrame({"日期": ["2026-10-21", "2026-10-22"], "收盘": [1.213, 1.202]})
        monkeypatch.setattr("etf_rotation.data._fetch_em", lambda *a, **k: em)
        assert crosscheck_with_em(["512890"], days=2) == []

    def test_network_failure_warns_not_critical(self, tmp_path, monkeypatch):
        monkeypatch.setattr("etf_rotation.data.DATA_DIR", tmp_path)
        (tmp_path / "512890.raw.csv").write_text("date,close\n2026-10-22,1.202\n")

        def _boom(*a, **k):
            raise RuntimeError("限流")
        monkeypatch.setattr("etf_rotation.data._fetch_em", _boom)
        issues = crosscheck_with_em(["512890"], days=1)
        assert issues[0].level == WARNING and not has_critical(issues)


class TestReport:
    def test_format_and_critical_detection(self):
        assert format_report([]) == "数据校验: 全部通过"
        issues = [c for c in [
            *validate_ohlc(_bars([1.2, 1.6]), "512890"),
            *validate_alignment({"a": _bars([1.2]), "b": _bars([1.3, 1.31])}),
        ]]
        report = format_report(issues)
        assert "Error: 数据校验" in report and "数据校验警告" in report
        assert has_critical(issues)
