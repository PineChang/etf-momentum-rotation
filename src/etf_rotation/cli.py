"""命令行入口:fetch / backtest / sweep / plan。

用法(PYTHONPATH=src 或 pip install -e . 后):
  python -m etf_rotation.cli fetch                     # 下载/更新数据
  python -m etf_rotation.cli backtest                  # 全量回测 + 图表 + 交易明细
  python -m etf_rotation.cli sweep [--step 0.2]        # 参数遍历
  python -m etf_rotation.cli plan                      # 生成今日交易计划(盘前/前一晚)
  python -m etf_rotation.cli execute                   # 开盘按真实开盘价成交最近一份计划(虚盘)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import UNIVERSE, StrategyConfig
from .data import (DATA_DIR, attach_premium, fetch_today_opens, fetch_universe,
                   premium_status, update_benchmark_cache, update_premium_caches)
from .live import StateStore, check_fillable_at_open, execute_plan, make_plan
from .metrics import format_metrics, perf_metrics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
STATE_DIR = ROOT / "state"


PRESETS: dict[str, dict] = {
    # V3.1 = V3 + 溢价率入场门槛(当前推荐,2026-09-29 采纳,见 audit-report 8.1/十节):
    # 开仓前安检——场内价对单位净值溢价 >7% 不开新仓(只拦新买,不踢已有持仓);
    # 513100 常年暂停申购,高溢价期买入存在溢价坍塌风险。
    # V3 = 项目验证过的原推荐配置(无溢价过滤);original = 忠实原文。
    "v3.1": dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25, premium_max=0.07),
    "v3": dict(n_hold=2, trailing_stop=0.08, min_momentum=0.0, take_profit=0.25),
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
    p.add_argument("--premium-max", type=float, help="溢价率入场门槛:场内价对单位净值溢价超过该值不开新仓(如 0.07),不踢已有持仓;仅 v3.1 预设默认开启")


def cmd_fetch(args) -> None:
    data = fetch_universe(list(UNIVERSE))
    for s, df in data.items():
        print(f"{s} {UNIVERSE[s]}: {len(df)} bars, {df['date'].min().date()} → {df['date'].max().date()}")
    # 溢价率数据(V3.1 购买规则依赖;不刷会在过期数据上静默漂移)
    try:
        update_premium_caches(list(UNIVERSE))
        print("溢价数据(raw+nav)已刷新")
    except Exception as e:
        print(f"警告:溢价数据刷新失败({e}),V3.1 的 premium 规则将使用旧缓存", file=sys.stderr)
    # 业绩基准(沪深300,仪表盘"跑赢大盘"对比用)
    try:
        update_benchmark_cache()
        print("基准(沪深300)已刷新")
    except Exception as e:
        print(f"警告:基准刷新失败({e}),仪表盘对比将使用旧缓存", file=sys.stderr)


def _with_premium(cfg: StrategyConfig, data: dict) -> dict:
    """premium_max 生效时注入溢价列并做健康检查(缺失/过期即警告,绝不静默)。"""
    if cfg.premium_max is None:
        return data
    data = attach_premium(data)
    for sym, st in premium_status(list(data)).items():
        if st["missing"]:
            print(f"警告:{sym} 无溢价缓存,premium 规则对其静默失效!"
                  f"请先运行 cli fetch(会同时刷新 raw+nav 缓存)。", file=sys.stderr)
        elif st["stale"]:
            lag = st["nav_lag"]
            detail = f"单位净值落后行情 {lag} 个交易日" if lag else "数据陈旧"
            print(f"警告:{sym} 溢价数据截至 {st['asof']}({detail}),"
                  f"规则可能建立在旧值上;建议先运行 cli fetch 刷新。", file=sys.stderr)
    return data


def cmd_backtest(args) -> None:
    cfg = _cfg_from_args(args)
    data = _with_premium(cfg, fetch_universe(list(UNIVERSE)))
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


def _plan_unchanged(store: StateStore, plan: dict, cfg: StrategyConfig) -> bool:
    """同一信号日 + 同一参数 => 计划不变,不重复落盘。

    防止非交易日/手动重跑 plan 把同一信号重复写进日志、top_streak 被重复推进。
    """
    last = store.latest_plan()
    if last is None or last.get("data_asof") != plan["data_asof"]:
        return False
    if not store.params_file.exists():
        return False
    return json.loads(store.params_file.read_text()) == asdict(cfg)


def cmd_plan(args) -> None:
    cfg = _cfg_from_args(args)
    data = _with_premium(cfg, fetch_universe(list(UNIVERSE)))
    store = StateStore(STATE_DIR)
    state = store.load_state(cfg.initial_cash)
    plan, new_state = make_plan(data, state, cfg)
    if plan.get("data_lag"):
        # 交集对齐会因单只标的缺 bar 把信号日整体回退一天,而计划里只有 data_asof
        # 一个字段能看出来——必须显式告警,否则又是"信号晚一天"的静默路径。
        detail = "、".join(f"{s}(最新 {d})" for s, d in plan["data_lag"].items())
        print(f"警告:信号日 {plan['data_asof']} 落后于部分标的最新 bar——{detail};"
              f"这些标的当日缺 bar(停牌/数据源缺口),全池对齐后被回退。", file=sys.stderr)
    if _plan_unchanged(store, plan, cfg):
        print(f"信号日 {plan['data_asof']} 的计划已生成过且参数未变,跳过(避免连击重复推进)")
        return
    if cfg.premium_max is not None:
        status = premium_status(list(data))
        plan["premium_check"] = {
            s: {"premium": st["premium"], "asof": st["asof"],
                "nav_lag": st["nav_lag"], "stale": st["stale"]} for s, st in status.items()
        }
        plan["premium_rule"] = f"溢价 > {cfg.premium_max:.0%} 不开新仓(入场安检,已持有不受影响;当日计划已自动排除)"
    store.save_state(new_state)      # 风控状态(锚点/禁买/连击)先落地,再谈成交
    store.save_params(cfg)
    store.append_log({"type": "plan", **plan})
    print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))


def cmd_execute(args) -> None:
    """开盘执行最近一份计划(对应文章 9:30 实际交易;虚盘/实盘共用)。

    拉取今日开盘价(tushare rt_k),校验限价单可成交后调 execute_plan 落账。
    整批执行或整批放弃:任一限价单开盘未成交则本次不动,稍后重跑即可
    (限价单盘中仍可能成交)。同一份计划拒绝重复执行(execute_plan 非幂等)。
    """
    store = StateStore(STATE_DIR)
    plan = store.latest_plan()
    if plan is None:
        print("没有待执行的计划:请先运行 cli plan")
        return
    if not plan.get("orders"):
        print(f"计划 {plan['plan_date']}(信号数据 {plan['data_asof']})无订单,空仓日无需执行")
        return
    if store.plan_executed(plan):
        print(f"计划 {plan['plan_date']} 已执行过(成交记录见 trade_log.jsonl),拒绝重复执行")
        return
    age = (pd.Timestamp.today() - pd.Timestamp(plan["plan_date"])).days
    if age > 4:
        print(f"计划生成于 {plan['plan_date']},已过 {age} 天,信号可能过时——请先 cli plan 重新生成")
        return
    if store.params_file.exists():
        cfg = StrategyConfig(**json.loads(store.params_file.read_text()))
    else:                                   # 兼容:从未跑过 plan 直接 execute 的情况
        cfg = StrategyConfig(**PRESETS["v3.1"])

    try:
        # 只拉计划订单涉及的开盘价:无关标的停牌/行情延迟不应中止整批执行;
        # 持仓估值兜底用计划参考价(execute_plan 的 price_of 语义)。
        order_syms = sorted({o["symbol"] for o in plan["orders"]})
        open_prices = fetch_today_opens(order_syms)
    except Exception as e:
        print(f"今日无法执行({e})")
        return
    problems = check_fillable_at_open(plan, open_prices)
    if problems:
        print("以下限价单在开盘价未成交,本次整批不执行(可稍候重跑):")
        for p in problems:
            print(" -", p)
        store.append_log({"type": "skip", "plan_date": plan["plan_date"],
                          "data_asof": plan["data_asof"], "problems": problems})
        return

    state = store.load_state(cfg.initial_cash)
    new_state = execute_plan(plan, state, open_prices, cfg)
    store.save_state(new_state)
    fills = [{**o, "fill_price": open_prices[o["symbol"]]} for o in plan["orders"]]
    store.append_log({"type": "fill", "plan_date": plan["plan_date"],
                      "data_asof": plan["data_asof"], "signal_age_days": age, "fills": fills})
    print(f"已按今日开盘价成交 {len(fills)} 笔:")
    for f in fills:
        print(f"  {f['action']} {f['symbol']} @ {f['fill_price']}  (限价 {f['limit_price']})")
    if age > 0:
        print(f"注:该计划生成于 {age} 天前,限价由彼时信号日收盘推导,今日开盘价成交——"
              f"与真实限价单的持续挂单行为可能有偏差(已记录 signal_age_days)")
    print(f"现金 {new_state.cash:.2f} | 持仓 {sorted(new_state.holdings) or '空仓'} "
          f"| 累计已实现盈亏 {new_state.realized_pnl:.2f}")


def cmd_dashboard(args) -> None:
    """生成 HTML 虚盘仪表盘(自包含单文件,浏览器直接打开;纯离线)。"""
    from .dashboard import render_dashboard

    data = {s: pd.read_csv(DATA_DIR / f"{s}.csv", parse_dates=["date"]) for s in UNIVERSE
            if (DATA_DIR / f"{s}.csv").exists()}
    out = render_dashboard(StateStore(STATE_DIR), data, STATE_DIR / "dashboard.html")
    print(f"仪表盘: {out}")
    if args.open:
        import subprocess
        subprocess.run(["open", str(out)], check=False)


def cmd_status(args) -> None:
    """每日查看:现金/持仓(浮盈亏)/权益/风控状态/待执行计划/最近成交。

    直接读本地缓存(纯离线)——status 必须永远可用,不能因数据源限流而挂。
    """
    store = StateStore(STATE_DIR)
    state = store.load_state()
    data = {s: pd.read_csv(DATA_DIR / f"{s}.csv", parse_dates=["date"]) for s in UNIVERSE
            if (DATA_DIR / f"{s}.csv").exists()}
    closes = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    asof = str(data[list(data)[0]]["date"].iloc[-1].date()) if data else "无数据"

    mv = sum(h.shares * closes.get(s, 0.0) for s, h in state.holdings.items())
    print(f"=== 账户状态(行情截至 {asof})===")
    print(f"现金 {state.cash:,.2f} | 持仓市值 {mv:,.2f} | 总权益 {state.cash + mv:,.2f}")
    for s, h in sorted(state.holdings.items()):
        px = closes.get(s)
        if px:
            print(f"  {s} {UNIVERSE.get(s, '')}: {h.shares:,.0f} 份 | 成本 {h.cost:.3f} → 现价 {px:.3f} "
                  f"| 浮盈亏 {px / h.cost - 1:+.2%} ({h.shares * (px - h.cost):+,.0f} 元) "
                  f"| 移动止损锚点 {h.peak:.3f}")
    if not state.holdings:
        print("  (空仓)")
    print(f"累计已实现盈亏 {state.realized_pnl:+,.2f} | 初始本金 {state.cash + mv - state.realized_pnl:,.2f}(含浮动)")
    stopped = ", ".join(f"{k}(自 {v})" for k, v in state.stopped.items()) or "无"
    print(f"禁买名单: {stopped} | 评分连击: {state.top_streak or '无'}")

    plan = store.latest_plan()
    if plan and not store.plan_executed(plan):
        acts = ("; ".join(f"{o['action']} {o['symbol']} 限价 {o['limit_price']}"
                          for o in plan["orders"]) or "无订单(空仓)")
        print(f"待执行计划({plan['plan_date']} 生成,信号日 {plan['data_asof']}): {acts}")

    if store.log_file.exists():
        fills = [json.loads(x) for x in store.log_file.read_text().splitlines()
                 if json.loads(x).get("type") == "fill"]
        if fills:
            print("最近成交(新→旧,最多 8 条):")
            for f in reversed(fills[-8:]):
                for x in f["fills"]:
                    print(f"  {f['plan_date']} {x['action']} {x['symbol']} @ {x['fill_price']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="etf-rotation", description="ETF 三因子动量轮动策略")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="下载/更新行情数据")

    p_bt = sub.add_parser("backtest", help="运行回测")
    p_bt.add_argument("--preset", choices=list(PRESETS), default="original",
                      help="参数预设:original=忠实原文(默认,年化42%%/回撤-32%%);"
                           "v3.1=当前推荐(溢价入场门槛,年化30%%/回撤-13%%);v3/v2=历史版本")
    _add_param_args(p_bt)

    p_sw = sub.add_parser("sweep", help="参数遍历")
    p_sw.add_argument("--step", type=float, default=0.2)

    p_plan = sub.add_parser("plan", help="生成今日交易计划(盘前)")
    p_plan.add_argument("--preset", choices=list(PRESETS), default="v3.1",
                        help="参数预设:v3.1=当前推荐(V3+溢价过滤,默认),original=忠实原文")
    _add_param_args(p_plan)

    sub.add_parser("execute", help="开盘按真实开盘价成交最近一份计划(虚盘)")
    sub.add_parser("status", help="查看当前持仓/权益/风控状态/最近成交")

    p_dash = sub.add_parser("dashboard", help="生成 HTML 虚盘仪表盘(state/dashboard.html)")
    p_dash.add_argument("--open", action="store_true", help="生成后在浏览器打开")

    args = parser.parse_args()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "sweep": cmd_sweep,
     "plan": cmd_plan, "execute": cmd_execute, "status": cmd_status,
     "dashboard": cmd_dashboard}[args.command](args)


if __name__ == "__main__":
    main()
