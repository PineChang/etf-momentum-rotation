"""数据层测试:缓存新鲜度(_needs_refresh)、盘中半根 bar 防护、圆整 bar 守卫、
tushare 复权计算、fetch_today_opens 盘初重试。"""

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
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-08")

    def test_intraday_fetch_drops_partial_today(self, monkeypatch):
        # 盘中运行:数据源返回了今日未定型 bar,不得落缓存、不得出现在返回值
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 10)
        src = self._src_df(["2026-10-07", "2026-10-08"], [1.0, 1.1])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-07")
        cached = pd.read_csv(self.cache_dir / "512890.csv", parse_dates=["date"])
        assert cached["date"].max() == pd.Timestamp("2026-10-07")

    def test_wakeup_catchup_appends_yesterday(self, monkeypatch):
        # 休眠唤醒(漏了昨晚任务):盘中补跑,昨日 bar 已定型应入缓存,今日半根 bar 丢弃
        self._write_cache(["2026-10-05", "2026-10-06"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 9)
        src = self._src_df(["2026-10-06", "2026-10-07", "2026-10-08"], [1.0, 1.0, 1.2])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-07")

    def test_reanchor_triggers_full_refetch(self, monkeypatch):
        # 锚点校验不一致(除息致前复权锚点变化):丢弃缓存全量重拉(审计#5 防污染守卫)
        self._write_cache(["2026-10-06", "2026-10-07"], [1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-08", 20)
        inc = self._src_df(["2026-10-07", "2026-10-08"], [0.9, 0.9])   # 与缓存不一致 → 判定锚点变化
        full = self._src_df(["2026-01-02", "2026-10-08"], [0.9, 0.9])
        monkeypatch.setattr(d, "_fetch_daily",
                            lambda *a, **k: inc if a[1] != "20190101" else full)
        df = fetch_etf("512890")
        # 缓存被全量重拉后的新序列替换
        assert list(df["close"]) == [0.9, 0.9]
        assert df["date"].min() == pd.Timestamp("2026-01-02")

    def test_rounded_latest_bar_dropped(self, monkeypatch):
        """腾讯兜底偶发把最新 bar 圆整到 2 位小数(2026-10-09 实证:512890 真实
        收 1.208 被写成 1.21,518880 真实收 8.617 被写成 8.62)。历史有 3 位小数
        精度而最新行 4 价全为 0.01 整数倍 → 判定圆整,丢弃该行并告警,
        缓存停留前一日,后续运行自愈。"""
        self._write_cache(["2026-10-06", "2026-10-07", "2026-10-08"],
                          [1.197, 1.212, 1.208])
        self._freeze_time(monkeypatch, "2026-10-09", 20)
        src = self._src_df(["2026-10-08", "2026-10-09"], [1.208, 1.21])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        with pytest.warns(UserWarning, match="圆整"):
            df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-08")
        cached = pd.read_csv(self.cache_dir / "512890.csv", parse_dates=["date"])
        assert cached["date"].max() == pd.Timestamp("2026-10-08")

    def test_precise_latest_bar_kept(self, monkeypatch):
        # 真实 3 位小数的最新 bar 不得误杀(真实收 1.208 这类)
        self._write_cache(["2026-10-06", "2026-10-07", "2026-10-08"],
                          [1.197, 1.212, 1.208])
        self._freeze_time(monkeypatch, "2026-10-09", 20)
        src = self._src_df(["2026-10-08", "2026-10-09"], [1.208, 1.217])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-09")

    def test_all_round_history_not_flagged(self, monkeypatch):
        # 整段历史本就是 2 位精度(如低价债券 ETF):无 3 位小数对照,不误判
        self._write_cache(["2026-10-06", "2026-10-07", "2026-10-08"], [1.0, 1.0, 1.0])
        self._freeze_time(monkeypatch, "2026-10-09", 20)
        src = self._src_df(["2026-10-08", "2026-10-09"], [1.0, 1.1])
        monkeypatch.setattr(d, "_fetch_daily", lambda *a, **k: src)
        df = fetch_etf("512890")
        assert df["date"].max() == pd.Timestamp("2026-10-09")


class TestFetchTs:
    """tushare 中转站首选数据源:fund_daily 原始价 + fund_adj 因子自行复权。"""

    class _FakePro:
        """fund_daily 返回交易所原始价;fund_adj 返回复权因子(512890 在
        2026-10-08 除息,因子 2.0 → 2.0462)。"""
        daily = pd.DataFrame({"trade_date": ["20261009", "20261007"],
                              "open": [1.21, 1.19], "high": [1.217, 1.20],
                              "low": [1.208, 1.18], "close": [1.208, 1.185],
                              "vol": [100.0, 90.0]})
        adj = pd.DataFrame({"trade_date": ["20261009", "20261007"],
                            "adj_factor": [2.0462, 2.0]})

        def fund_daily(self, **k):
            return self.daily

        def fund_adj(self, **k):
            return self.adj

    def test_qfq_anchored_at_latest_factor(self, monkeypatch):
        monkeypatch.setattr(d, "_ts_pro", lambda: self._FakePro())
        df = d._fetch_ts("512890", "20261007", "20261009", "qfq")
        df = df.sort_values("日期").reset_index(drop=True)
        # 最新日因子 = 最新锚点 → 最新日 qfq = 原始价(与交易所逐分一致)
        assert df["收盘"].iloc[-1] == 1.208
        # 除息前 qfq = 原始价 × 2.0 / 2.0462,保留 3 位小数(东财口径)
        assert df["收盘"].iloc[0] == round(1.185 * 2.0 / 2.0462, 3)

    def test_hfq_multiplies_factor(self, monkeypatch):
        monkeypatch.setattr(d, "_ts_pro", lambda: self._FakePro())
        df = d._fetch_ts("512890", "20261007", "20261009", "hfq")
        df = df.sort_values("日期").reset_index(drop=True)
        assert df["收盘"].iloc[0] == round(1.185 * 2.0, 3)

    def test_missing_adj_factor_raises(self, monkeypatch):
        class BadPro(self._FakePro):
            adj = pd.DataFrame({"trade_date": ["20261009"], "adj_factor": [2.0462]})

        monkeypatch.setattr(d, "_ts_pro", lambda: BadPro())
        with pytest.raises(RuntimeError, match="复权因子"):
            d._fetch_ts("512890", "20261007", "20261009", "qfq")

    def test_fetch_daily_prefers_ts_then_falls_back(self, monkeypatch):
        # tushare 失败 → 告警并回退东财
        def boom(*a, **k):
            raise ConnectionError("relay down")

        monkeypatch.setattr(d, "_fetch_ts", boom)
        monkeypatch.setattr(d, "_fetch_em",
                            lambda *a, **k: TestFetchEtfRefresh._src_df(["2026-10-09"], [1.208]))
        with pytest.warns(UserWarning, match="tushare"):
            df = d._fetch_daily("512890", "20261009", "20261009", "qfq")
        assert df["收盘"].iloc[0] == 1.208

    def test_fetch_daily_empty_ts_falls_back(self, monkeypatch):
        # tushare 返回空 df(不抛异常)也必须回退东财——空结果静默穿透会让缓存停更
        monkeypatch.setattr(d, "_fetch_ts",
                            lambda *a, **k: pd.DataFrame(columns=list(d._COL_MAP)))
        monkeypatch.setattr(d, "_fetch_em",
                            lambda *a, **k: TestFetchEtfRefresh._src_df(["2026-10-09"], [1.208]))
        with pytest.warns(UserWarning, match="返回空"):
            df = d._fetch_daily("512890", "20261009", "20261009", "qfq")
        assert df["收盘"].iloc[0] == 1.208


class TestFetchTodayOpens:
    """盘初实时 bar 延迟生成 → 轮询重试(rt_k 的 trade_time 非今日即尚未生成)。"""

    @pytest.fixture(autouse=True)
    def _freeze_today(self, monkeypatch):
        # "今日 bar" 的判定用 pd.Timestamp.today(),必须冻结,否则测试只在工作日当天成立
        monkeypatch.setattr(pd.Timestamp, "today",
                            classmethod(lambda cls, *a, **k: pd.Timestamp("2026-10-08")))

    @staticmethod
    def _pro(trade_time: str, open_px=1.23):
        class _Pro:
            def rt_k(self, **k):
                return pd.DataFrame({"ts_code": ["159949.SZ"], "open": [open_px],
                                     "trade_time": [trade_time]})
        return _Pro()

    def test_retries_until_bar_appears(self, monkeypatch):
        seq = ([self._pro("2026-10-07 15:34:58")] * 2
               + [self._pro("2026-10-08 09:31:00")])
        monkeypatch.setattr(d, "_ts_pro", lambda: seq.pop(0))
        out = fetch_today_opens(["159949"], retries=3, wait_s=0)
        assert out == {"159949": pytest.approx(1.23)}

    def test_raises_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr(d, "_ts_pro", lambda: self._pro("2026-10-07 15:34:58"))
        with pytest.raises(RuntimeError, match="尚未生成"):
            fetch_today_opens(["159949"], retries=2, wait_s=0)

    def test_network_error_is_retried_not_fatal(self, monkeypatch):
        def flaky():
            if flaky.n < 1:
                flaky.n += 1
                raise ConnectionError("boom")
            return self._pro("2026-10-08 09:31:00")
        flaky.n = 0
        monkeypatch.setattr(d, "_ts_pro", flaky)
        out = fetch_today_opens(["159949"], retries=2, wait_s=0)
        assert out == {"159949": pytest.approx(1.23)}

    def test_zero_open_not_accepted(self, monkeypatch):
        # trade_time 已到今日但竞价未撮合(open=0):继续重试,直到拿到有效开盘价
        seq = [self._pro("2026-10-08 09:16:00", open_px=0.0),
               self._pro("2026-10-08 09:31:00")]
        monkeypatch.setattr(d, "_ts_pro", lambda: seq.pop(0))
        out = fetch_today_opens(["159949"], retries=2, wait_s=0)
        assert out == {"159949": pytest.approx(1.23)}
