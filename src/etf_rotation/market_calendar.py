"""交易日历(可选接入 ETF 工作区的 data_source 项目)。

data_source 与本项目在同一父目录时自动读取其本地 DuckDB/Parquet 日历;
未安装或数据不可用时返回 None,调用方按原逻辑继续,避免把日历服务变成硬依赖。
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE_SRC = ROOT.parent / "data_source" / "src"


def _load_data_source():
    """加载 sibling data_source 包;不可用返回 None。"""
    try:
        return import_module("data_source")
    except ModuleNotFoundError:
        if not DATA_SOURCE_SRC.exists():
            return None
    sys.path.insert(0, str(DATA_SOURCE_SRC))
    try:
        return import_module("data_source")
    except Exception:
        return None
    finally:
        sys.path.remove(str(DATA_SOURCE_SRC))


def is_trading_day(day: str | pd.Timestamp | None = None) -> bool | None:
    """判断某天是否交易所开市;无法确认返回 None(调用方走原逻辑)。"""
    ds = _load_data_source()
    if ds is None:
        return None
    date = pd.Timestamp(day) if day is not None else pd.Timestamp.today()
    day_text = date.strftime("%Y-%m-%d")
    try:
        known = ds.calendar(day_text, day_text, open_only=False)
        if known.empty:
            return None               # 日历缺当天记录:无法确认,回退开盘价探测
        return not ds.calendar(day_text, day_text, open_only=True).empty
    except Exception:
        return None
