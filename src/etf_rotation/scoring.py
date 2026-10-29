"""Z-Score 标准化 + 加权综合评分 + 1.5 倍调仓阈值轮动选择。

本模块负责:
- 三因子量纲不同,先各自做 Z-Score 标准化(均值 0、标准差 1),再加权,
  防止某个因子因数值过大主导最终得分。
- 调仓阈值:挑战者得分必须超过当前持仓得分的 threshold 倍(默认 1.5)才调仓,
  避免震荡期分数接近时频繁出手。
"""

from __future__ import annotations

import pandas as pd


def zscore(factor_df: pd.DataFrame) -> pd.DataFrame:
    """对每个因子列在股票池横截面上做 Z-Score 标准化。

    股票池只有 4 只 ETF,用样本标准差(ddof=1)与总体标准差差异不大,
    这里取总体标准差(ddof=0),单一标的时返回 0。
    """
    mean = factor_df.mean(axis=0)
    std = factor_df.std(axis=0, ddof=0)
    std = std.where(std > 0, other=1.0)  # 防除零:全部相同则标准化后全 0
    return (factor_df - mean) / std


def combined_score(factor_df: pd.DataFrame, w_bias: float, w_slope: float, w_efficiency: float) -> pd.Series:
    """标准化后按既定权重加权:默认 乖离0.3 / 斜率0.3 / 效率0.4。"""
    z = zscore(factor_df)
    return w_bias * z["bias"] + w_slope * z["slope"] + w_efficiency * z["efficiency"]


def _beats(score_held: float, score_challenger: float, threshold: float) -> bool:
    """阈值比较:挑战者须超出在持者 |score|×(threshold-1)。

    得分为正时等价于 challenger > held * threshold;
    得分为负时原乘法形式会反向放行(×1.5 让不等式更易满足),
    这里改为对称语义——负分区同样要求显著超出,而非必然切换。
    恰好达界不算"超过",不切换。
    """
    return score_challenger > score_held + (threshold - 1.0) * abs(score_held)


def select_targets(scores: pd.Series, current: set[str], threshold: float, n: int) -> set[str]:
    """多持仓版本:选出 top-n 目标集合,阈值机制的自然推广。

    - 空仓:直接选前 n 名;
    - 空槽位:由排名靠前者直接补上(无需过阈值);
    - 替换在持标的:挑战者得分 > 在持最弱者得分 * threshold 才换。
    """
    scores = scores.dropna()
    if scores.empty:
        return set(current)
    ranked = scores.sort_values(ascending=False)
    target = {s for s in current if s in scores.index}
    for sym in ranked.index:                # 先按排名填满 n 个槽位
        if len(target) >= n:
            break
        target.add(sym)
    for _ in range(n):                      # 阈值换血:挑战者须显著强于最弱在持者(见 _beats)
        held = [s for s in target if s in current]
        challengers = [s for s in ranked.index if s not in target]
        if not held or not challengers:
            break
        weakest = min(held, key=lambda s: scores[s])
        if _beats(scores[weakest], scores[challengers[0]], threshold):
            target.discard(weakest)
            target.add(challengers[0])
        else:
            break
    return target
