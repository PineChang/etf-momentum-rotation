"""三因子动量 —— 三因子轮动策略的公式。

原始公式(伪代码)→ 本实现:

1. 乖离动量因子 (bias_momentum_score)
   bias = close / close.rolling(BIAS_N, min_periods=1).mean()
   bias_recent = bias.iloc[-MOMENTUM_DAY:]
   y = bias_recent / bias_recent.iloc[0], x = np.arange(MOMENTUM_DAY)
   线性回归 → bias_score = slope * 10000

2. 斜率动量因子 (slope_momentum_score)
   normalized = close / close.iloc[0]  (最近 SLOPE_N 天)
   x = np.arange(1, SLOPE_N + 1)
   线性回归 → score = 10000 * slope * r_squared

3. 效率动量因子 (efficiency_momentum_score)
   pivot = (open + high + low + close) / 4
   momentum = 100 * log(pivot[-1] / pivot[0])
   efficiency_ratio = |log(pivot[-1]) - log(pivot[0])| / sum(|diff(log(pivot))|)
   score = momentum * efficiency_ratio

示例代码用 sklearn.linear_model.LinearRegression;这里用 numpy 最小二乘,
数学上等价(普通最小二乘闭式解),减少一个重依赖。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """一元 OLS:返回 (slope, r_squared),与 sklearn LinearRegression 等价。"""
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
    ss_xx = float(((x - x_mean) ** 2).sum())
    if ss_xx == 0.0:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r_squared


def bias_momentum_score(close: pd.Series, ma_n: int, momentum_day: int) -> float:
    """乖离动量:价格相对长期均线的偏离趋势,放大 10000 倍。

    向上偏离且偏离加强 → 强势上涨;向下偏离且加强 → 弱势下跌。
    """
    if len(close) < ma_n + momentum_day:
        return float("nan")
    bias = close / close.rolling(window=ma_n, min_periods=1).mean()
    bias_recent = bias.iloc[-momentum_day:]
    y = (bias_recent / bias_recent.iloc[0]).to_numpy(dtype=float)
    x = np.arange(momentum_day, dtype=float)
    slope, _ = _ols_slope_r2(x, y)
    return slope * 10000.0


def slope_momentum_score(close: pd.Series, slope_n: int) -> float:
    """斜率动量:10000 * 回归斜率 * R²,兼顾趋势强度与趋势质量。"""
    if len(close) < slope_n:
        return float("nan")
    window = close.iloc[-slope_n:]
    y = (window / window.iloc[0]).to_numpy(dtype=float)
    x = np.arange(1, slope_n + 1, dtype=float)
    slope, r_squared = _ols_slope_r2(x, y)
    return 10000.0 * slope * r_squared


def efficiency_momentum_score(ohlc: pd.DataFrame, n: int) -> float:
    """效率动量:价格中枢对数动量 * 效率系数(净位移/总波动)。

    效率系数越大,走势越流畅(上涨不拖泥带水)。
    ohlc 需含 open/high/low/close 四列。
    """
    if len(ohlc) < n:
        return float("nan")
    df = ohlc.iloc[-n:]
    pivot = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).to_numpy(dtype=float)
    log_pivot = np.log(pivot)
    momentum = 100.0 * (log_pivot[-1] - log_pivot[0])
    direction = abs(log_pivot[-1] - log_pivot[0])
    volatility = float(np.abs(np.diff(log_pivot)).sum())
    efficiency_ratio = direction / volatility if volatility > 0 else 0.0
    return momentum * efficiency_ratio


def _atr_mean(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> float:
    """近 n 日真实波幅(TR)均值,即 ATR(未除以价格,作波动率比较用)。"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return float(tr[-n:].mean())


def dynamic_lookback_window(
    df: pd.DataFrame,
    lb_min: int,
    lb_max: int,
    ratio_cap: float,
    vol_short_n: int,
    vol_long_n: int,
) -> int | None:
    """波动率动量回溯窗口(知乎改进①)。

    lookback = lb_min + (lb_max - lb_min) * (1 - min(ratio_cap, vol_ratio))
    其中 vol_ratio = 短窗 ATR / 长窗 ATR:近期波动骤升 → 窗口缩短(提高敏感度),
    近期平稳 → 窗口拉长(捕捉长期趋势)。历史不足时返回 None(回退固定窗口)。
    """
    if len(df) < vol_long_n + 1:
        return None
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    vol_short = _atr_mean(high, low, close, vol_short_n)
    vol_long = _atr_mean(high, low, close, vol_long_n)
    if vol_long <= 0:
        return None
    vol_ratio = vol_short / vol_long
    return int(round(lb_min + (lb_max - lb_min) * (1.0 - min(ratio_cap, vol_ratio))))


def compute_factor_scores(
    data: dict[str, pd.DataFrame],
    ma_n: int,
    momentum_day: int,
    slope_n: int,
    efficiency_n: int,
    lookbacks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """对股票池中每只 ETF 计算三因子原始分。

    data: {symbol: 含 open/high/low/close 列的 DataFrame}(已截断到信号日)
    lookbacks: 每标的动量回溯窗口(知乎改进①);None 的标的用固定窗口。
    返回 index=symbol, columns=[bias, slope, efficiency] 的 DataFrame。
    """
    rows = {}
    for symbol, df in data.items():
        close = df["close"]
        n = lookbacks.get(symbol) if lookbacks else None
        md, sn, en = (n, n, n) if n else (momentum_day, slope_n, efficiency_n)
        rows[symbol] = {
            "bias": bias_momentum_score(close, ma_n, md),
            "slope": slope_momentum_score(close, sn),
            "efficiency": efficiency_momentum_score(df, en),
        }
    return pd.DataFrame(rows).T
