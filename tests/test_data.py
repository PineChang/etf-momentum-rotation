"""数据层测试:缓存新鲜度(_needs_refresh)、盘中半根 bar 防护、
腾讯兜底未复权守卫、fetch_today_opens 盘初重试。"""

import pandas as pd
import pytest

import etf_rotation.data as d
from etf_rotation.data import _needs_refresh, fetch_etf, fetch_today_opens


def _now(day: str, hour: int) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hour:02d}:30")


class TestNeedsRefresh:
    """2026-10 审计修复的回归:旧条件(落后>4天才刷)使每日增量更新从不发生,
    晚间 plan 信号永远晚一个交易日。"""

    def test_lacking_yesterday_refreshes_anytime(self):
        # 休眠唤醒/隔夜补跑:缺昨日 bar(已定型),盘中也必须刷
        assert _needs_refresh(pd.Timestamp("2026-10-07"), _now("2026-10-09", 9)) is True

    def test_only_missing_today_requires_after_close(self):
        # 正常晚间运行:只缺今日 bar,20:30 应刷
        assert _needs_refresh(pd.Timestamp("2026-10-08"), _now("2026-10-09", 20)) is True

    def test_only_missing_today_intraday_no_refresh(self):
        # 盘中(如 9:31 execute 前手动跑 fetch):不刷,防半根 bar 入缓存
        assert _needs_refresh(pd.Timestamp("2026-10-08"), _now("2026-10-09", 9)) is False

    def test_up_to_date_never_refreshes(self):
        assert _needs_refresh(pd.Timestamp("2026-10-09"), _now("2026-10-09", 20)) is False

    def test_old_staleness_still_triggers(self):
        # 旧行为保留:假期后大缺口,任何时刻都刷
        assert _needs_refresh(pd.Timestamp("2026-09-30"), _now("2026-10-08", 10)) is True


class TestFetchEtfRefresh:
    """fetch_etf 集成行为(全部 mock 数据源,不联网)。"""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(d, "DATA_DIR", tmp_path)
        self.cache_dir = tmp_path
        yield

    def _write_cache(self, dates, closes):
        pd.DataFrame({"date": [pd.Timestamp(x) for x in dates],
                      "open": closes, "high": closes, "low": closes,
                      "close": closes, "volume": [1e6] * len(dates)}
                     ).to_csv(self.cache_dir / "512890.csv", index=False)

    @staticmethod
    def _src_df(dates, closes):
        return pd.DataFrame({"日期": dates, "开盘": closes, "收盘": closes,
                             "最高": closes, "最低": closes, "成交量": [1e6] * len(dates)})

    def _freeze_time(self, monkeypatch, day, hour):
        monkeypatch.setattr(pd.Timestamp, "now",
                            classmethod(lambda cls, *a, **k: _now(day, hour)))

    def test_evening_fetch_appends_today(self, monkeypatch):
        # 核心回归:晚间运行,数据源有今日 bar,缓存必须更新到今日
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 20)
        src = self._src_df(["2026-10-07", "2026-10-08"], [1.0, 1.1])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: (src, False))
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-08")

    def test_intraday_fetch_drops_partial_today(self, monkeypatch):
        # 盘中运行:数据源返回了今日未定型 bar,不得落缓存、不得出现在返回值
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 10)
        src = self._src_df(["2026-10-07", "2026-10-08"], [1.0, 1.1])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: (src, False))
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-07")
        cached = pd.read_csv(self.cache_dir / "512890.csv", parse_dates=["date"])
        assert cached["date"].max() == pd.Timestamp("2026-10-07")

    def test_wakeup_catchup_appends_yesterday(self, monkeypatch):
        # 休眠唤醒(漏了昨晚任务):盘中补跑,昨日 bar 已定型应入缓存,今日半根 bar 丢弃
        self._write_cache(["2026-10-05", "2026-10-06"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 9)
        src = self._src_df(["2026-10-06", "2026-10-07", "2026-10-08"], [1.0, 1.0, 1.2])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: (src, False))
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-07")

    def test_reanchor_with_unadjusted_fallback_raises(self, monkeypatch):
        # 锚点校验不一致且全量重拉只有未复权序列:拒绝改写缓存(审计#5 防污染守卫)
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 20)
        inc = self._src_df(["2026-10-07", "2026-10-08"], [0.9, 0.9])   # 与缓存不一致 → 判定锚点变化
        full = self._src_df(["2026-01-02", "2026-10-08"], [0.9, 0.9])
        monkeypatch.setattr(d, "_fetch_daily",
                            lambda *a, **k: (inc, False) if a[1] != "20190101" else (full, True))
        with pytest.raises(RuntimeError, match="未复权"):
            fetch_etf("512890")
        # 缓存未被破坏
        cached = pd.read_csv(self.cache_dir / "512890.csv")
        assert list(cached["close"]) == [1.0, 1.0]

    def test_incremental_unadjusted_fallback_warns(self, monkeypatch):
        """增量只能用未复权序列时必须告警:重叠区是"未复权 vs 未复权",锚点校验
        恒等、看不见除息,期间除息会凭空插进一根假阴线(旧实现完全静默)。
        """
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 20)
        src = self._src_df(["2026-10-07", "2026-10-08"], [1.0, 1.0])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: (src, True))
        with pytest.warns(UserWarning, match="未复权序列"):
            df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-08")   # 告警但照常更新

    def test_first_fetch_with_unadjusted_fallback_raises(self, monkeypatch):
        self._freeze_time(monkeypatch, "2026-10-08", 20)
        src = self._src_df(["2026-10-08"], [1.1])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: (src, True))
        with pytest.raises(RuntimeError, match="未复权"):
            fetch_etf("512890")
        assert not (self.cache_dir / "512890.csv").exists()


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class TestFetchTodayOpens:
    """盘初腾讯日 bar 延迟生成 → 轮询重试(2026-10-08 实测 9:31 尚无 bar)。"""

    @pytest.fixture(autouse=True)
    def _freeze_today(self, monkeypatch):
        # "今日 bar" 的判定用 pd.Timestamp.today(),必须冻结,否则测试只在工作日当天成立
        monkeypatch.setattr(pd.Timestamp, "today",
                            classmethod(lambda cls, *a, **k: pd.Timestamp("2026-10-08")))

    @staticmethod
    def _payload(today: bool, open_px=1.23):
        day = ([["2026-10-07", "1.20", "1.21", "1.22", "1.19", "1000"]]
               + ([["2026-10-08", str(open_px), "1.24", "1.25", "1.22", "2000"]] if today else []))
        return {"data": {"sz159949": {"day": day}}}

    def test_retries_until_bar_appears(self, monkeypatch):
        seq = [_Resp(self._payload(False))] * 2 + [_Resp(self._payload(True))]
        monkeypatch.setattr("requests.get", lambda *a, **k: seq.pop(0))
        out = fetch_today_opens(["159949"], retries=3, wait_s=0)
        assert out == {"159949": pytest.approx(1.23)}

    def test_raises_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(self._payload(False)))
        with pytest.raises(RuntimeError, match="尚未生成"):
            fetch_today_opens(["159949"], retries=2, wait_s=0)

    def test_network_error_is_retried_not_fatal(self, monkeypatch):
        def flaky(*a, **k):
            if flaky.n < 1:
                flaky.n += 1
                raise ConnectionError("boom")
            return _Resp(self._payload(True))
        flaky.n = 0
        monkeypatch.setattr("requests.get", flaky)
        out = fetch_today_opens(["159949"], retries=2, wait_s=0)
        assert out == {"159949": pytest.approx(1.23)}
