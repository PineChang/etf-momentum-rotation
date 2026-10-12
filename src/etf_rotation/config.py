"""策略配置:文章披露的参数 + 文章未披露参数的显式假设(见 docs/article-notes.md)。"""

from __future__ import annotations

from dataclasses import dataclass, field

# 文章指定的四只 ETF 股票池:覆盖 A 股红利/成长、美股、黄金,分散风格
UNIVERSE: dict[str, str] = {
    "512890": "红利低波ETF",
    "159949": "创业板50ETF",
    "513100": "纳指ETF",
    "518880": "黄金ETF",
}


@dataclass(frozen=True)
class StrategyConfig:
    # ---- 文章明确给出的参数 ----
    momentum_day: int = 25          # 乖离动量回归窗口 MOMENTUM_DAY
    w_bias: float = 0.3             # 乖离动量权重
    w_slope: float = 0.3            # 斜率动量权重
    w_efficiency: float = 0.4       # 效率动量权重
    rebalance_threshold: float = 1.5  # 调仓阈值:挑战者须超过持仓得分 1.5 倍

    # ---- 文章未披露数值、按常见惯例取的假设(可调参) ----
    bias_ma_n: int = 20             # 乖离率均线窗口 BIAS_N(常用 MA20)
    slope_n: int = 25               # 斜率动量窗口 SLOPE_N(与动量窗口对齐)
    efficiency_n: int = 25          # 效率动量窗口(与动量窗口对齐)

    # ---- 回测工程参数(文章未涉及,自设) ----
    initial_cash: float = 100_000.0
    commission: float = 0.0001      # ETF 佣金约万1,双边收取,无印花税

    # ---- 风控扩展(本项目改良,文章没有;默认 None = 关闭 = 忠实原文) ----
    # regime 过滤(离场语义:不合格的在持标的离场,全池不合格则空仓观望)
    min_momentum: float | None = None   # 绝对动量过滤:候选标的近 slope_n 天收益 ≤ 该值则离场/不建仓
    ma_filter_n: int | None = None      # 均线过滤:候选标的收盘价低于 MA(n) 则离场/不建仓
    stop_loss: float | None = None      # 止损:持仓浮亏超过该比例(如 0.08)强制清仓
    stop_cooldown_days: int = 10        # 止损后同标的禁买期(交易日),防止损-追高循环
    trailing_stop: float | None = None  # 移动止损:收盘价自持仓期间最高点回落超该比例即离场(保护浮盈)
    take_profit: float | None = None    # 固定止盈:对成本价浮盈超该比例即离场(触发后同样进禁买期)
    confirm_days: int = 1               # 入场确认:空仓状态下,标的须连续 N 天评分第一才可买入(防假信号)
    n_hold: int = 1                     # 持仓只数:1=原文全仓单标的;2=双标的等权分散(降低单标的事件风险)

    # ---- 知乎三点优化实验(默认关闭 = 忠实 V3;见 scripts/eval_zhihu.py)----
    dynamic_lookback: bool = False      # ① 波动率动量回溯窗口(短窗 ATR/长窗 ATR 比例自适应)
    lb_min: int = 10                    #    回溯窗口下界(文章示例值)
    lb_max: int = 60                    #    回溯窗口上界
    lb_ratio_cap: float = 0.9           #    波动率比例封顶
    vol_short_n: int = 10               #    短窗 ATR 天数
    vol_long_n: int = 60                #    长窗 ATR 天数
    # 入场门槛(只拦新开仓,不踢已有持仓;持有期离场归止损/止盈/轮动)
    drop_filter: bool = False           # ③ 急跌过滤:近 5 日出现单日-5%或连跌且累计-5%的标的不买
    premium_max: float | None = None    # ② 溢价率上限:开仓前安检,未复权收盘/单位净值 - 1 超过该值不买
                                        #    (需行情 df 带 premium 列,实验脚本注入;None = 关闭)
    premium_window: int | None = None   # ②b 相对溢价安检:当日溢价 > 前 N 个交易日溢价均值则不买
                                        #    (均值不含当日,2026-10 起替代固定 7% 门槛的实验规则;None = 关闭)

    # warmup 所需最少历史 bar 数
    @property
    def warmup_bars(self) -> int:
        if self.dynamic_lookback:
            return max(self.bias_ma_n + self.lb_max, self.lb_max, self.vol_long_n + 1)
        return max(self.bias_ma_n + self.momentum_day, self.slope_n, self.efficiency_n)
