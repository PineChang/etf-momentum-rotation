"""命令行入口:fetch / backtest / sweep / plan。

用法(PYTHONPATH=src 或 pip install -e . 后):
  python -m etf_rotation.cli fetch                     # 下载/更新数据
  python -m etf_rotation.cli backtest                  # 全量回测 + 图表 + 交易明细
  python -m etf_rotation.cli sweep [--step 0.2]        # 参数遍历
  python -m etf_rotation.cli plan                      # 生成今日交易计划(盘前)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import UNIVERSE, StrategyConfig
from .data import attach_premium, fetch_universe
from .live import StateStore, make_plan
from .metrics import format_metrics, perf_metrics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
STATE_DIR = ROOT / "state"


PRESETS: dict[str, dict] = {
    # V3 = 项目验证过的推荐配置(README/audit-report);original = 忠实原文。
    # premium_max=0.07(2026-09-29):场内价对单位净值溢价 >7% 时禁止买入——
    # 513100 常年暂停申购,高溢价期买入存在溢价坍塌风险(见 audit-report 第八节)。
    "v3": dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25, premium_max=0.07),
    "v2": dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0),
    "original": {},
}


def _cfg_from_args(args) -> StrategyConfig:
    cfg = StrategyConfig(**PRESETS[getattr(args, "preset", "original")])
    for k in ("bias_ma_n", "momentum_day", "slope_n", "efficiency_n",
              "w_bias", "w_slope", "w_efficiency", "rebalance_threshold", "commission",
              "min_momentum", "ma_filter_n", "stop_loss", "stop_cooldown_days",
              "trailing_stop", "confirm_days", "n_hold", "take_profit", "premium_max"):
        v = getattr(args, k, None)
        if v is not None:
            cfg = replace(cfg, **{k: v})
    return cfg


def _add_param_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bias-ma-n", type=int)
    p.add_argument("--momentum-day", type=int)
    p.add_argument("--slope-n", type=int)
    p.add_argument("--efficiency-n", type=int)
    p.add_argument("--w-bias", type=float)
    p.add_argument("--w-slope", type=float)
    p.add_argument("--w-efficiency", type=float)
    p.add_argument("--rebalance-threshold", type=float)
    p.add_argument("--commission", type=float)
    # 风控扩展(默认关闭 = 忠实原文)
    p.add_argument("--min-momentum", type=float, help="绝对动量过滤:近slope_n天收益≤该值则空仓")
    p.add_argument("--ma-filter-n", type=int, help="均线过滤窗口")
    p.add_argument("--stop-loss", type=float, help="止损比例,如 0.06")
    p.add_argument("--stop-cooldown-days", type=int, help="止损后禁买期(交易日)")
    p.add_argument("--trailing-stop", type=float, help="移动止损:自持仓期间最高价回落比例,如 0.08")
    p.add_argument("--take-profit", type=float, help="固定止盈:对成本价浮盈比例,如 0.25")
    p.add_argument("--confirm-days", type=int, help="入场确认天数:空仓时须连续N天评分第一才建仓")
    p.add_argument("--n-hold", type=int, help="持仓只数:1=原文全仓单标的,2=双标的等权分散")
    p.add_argument("--premium-max", type=float, help="溢价率过滤:场内价对单位净值溢价超过该值禁止买入(如 0.07);v3 预设已含")


def cmd_fetch(args) -> None:
    data = fetch_universe(list(UNIVERSE))
    for s, df in data.items():
        print(f"{s} {UNIVERSE[s]}: {len(df)} bars, {df['date'].min().date()} → {df['date'].max().date()}")


def cmd_backtest(args) -> None:
    cfg = _cfg_from_args(args)
    data = fetch_universe(list(UNIVERSE))
    if cfg.premium_max is not None:
        data = attach_premium(data)
    res = run_backtest(data, cfg)
    print(format_metrics(perf_metrics(res.equity, res.trades)))
    print(f"\n净值终值: {res.equity.iloc[-1]:.2f}")

    from .report import export_trades, plot_report

    fig = plot_report(res, data, OUTPUT_DIR / "equity_curve.png")
    trades_csv = export_trades(res, OUTPUT_DIR / "trades.csv")
    print(f"图表: {fig}\n交易明细: {trades_csv}")


def cmd_sweep(args) -> None:
    from .sweep import sweep

    data = fetch_universe(list(UNIVERSE))
    df = sweep(data, step=args.step)
    out = OUTPUT_DIR / "sweep_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.head(15).to_string(index=False))
    print(f"\n完整结果: {out}  (共 {len(df)} 组)")
    print("提示(文章原话):参数最优存在过拟合风险,不要太执着于调参。")


def cmd_plan(args) -> None:
    cfg = _cfg_from_args(args)
    data = fetch_universe(list(UNIVERSE))
    if cfg.premium_max is not None:
        data = attach_premium(data)
    store = StateStore(STATE_DIR)
    state = store.load_state(cfg.initial_cash)
    plan, new_state = make_plan(data, state, cfg)
    if cfg.premium_max is not None:
        plan["premium_check"] = {
            s: (None if "premium" not in data[s] or pd.isna(data[s]["premium"].iloc[-1])
                else round(float(data[s]["premium"].iloc[-1]), 4))
            for s in data
        }
        plan["premium_rule"] = f"溢价 > {cfg.premium_max:.0%} 禁止买入(当日已自动排除)"
    store.save_state(new_state)      # 风控状态(锚点/禁买/连击)先落地,再谈成交
    store.save_params(cfg)
    store.append_log({"type": "plan", **plan})
    print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="etf-rotation", description="ETF 三因子动量轮动策略")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="下载/更新行情数据")

    p_bt = sub.add_parser("backtest", help="运行回测")
    p_bt.add_argument("--preset", choices=list(PRESETS), default="original",
                      help="参数预设:v3=项目验证的推荐配置,original=忠实原文(默认)")
    _add_param_args(p_bt)

    p_sw = sub.add_parser("sweep", help="参数遍历")
    p_sw.add_argument("--step", type=float, default=0.2)

    p_plan = sub.add_parser("plan", help="生成今日交易计划(盘前)")
    p_plan.add_argument("--preset", choices=list(PRESETS), default="v3",
                        help="参数预设:v3=项目验证的推荐配置(默认),original=忠实原文")
    _add_param_args(p_plan)

    args = parser.parse_args()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "sweep": cmd_sweep, "plan": cmd_plan}[args.command](args)


if __name__ == "__main__":
    main()
