#!/usr/bin/env python
"""掘金仿真交易适配器(B 轨):计划单送到掘金仿真账户 + 定时同步账户快照。

信号仍由本仓库引擎生成(canonical),掘金只做下单与撮合——A 轨(cli execute,本地
理想成交记账)与 B 轨(掘金真实撮合)双轨并行,仪表盘"双轨对照"区块直接比较
同一计划单两边的成交价差。

准备步骤(一次性):
  1. sim.myquant.cn 注册,创建股票仿真账户,获取 token;
  2. pip install gmtrade;
  3. 每晚照常:cron 的 daily_paper.sh evening(fetch + plan + dashboard)

⚠️ 平台限制(2026-10 实测):gmtrade 在 PyPI 上只有 **Windows / manylinux x86_64、
Python ≤3.10** 的二进制轮子,**macOS 无法原生安装**。Mac 用户两条路:
  - 手工路径(零安装):网页创建仿真账户 + 每日网页手工下限价单 + 定期"下载交易数据"
    CSV,用 scripts/gm_import_csv.py 导入,仪表盘双轨对照同样可用;
  - 自动化路径:Linux 环境(Docker / 云主机)里跑本文件。

运行(Linux/Windows,每个交易日挂着):
  PYTHONPATH=src python scripts/gm_paper.py --token <你的token>
  - 09:31 读最新计划 → 限价单送掘金仿真账户,送单记录写 state/gm_orders.jsonl;
  - 09:35 / 15:05 同步账户快照(净值/持仓/成交回报)→ state/gm_snapshot.json
    并重新生成 state/dashboard.html。

注意:本文件在本仓库环境未连掘金实测(gm 未安装),gm 字段名以 gm 3.x 文档为准
(_g() 已做 dict/属性双兼容);纯逻辑(write_gm_snapshot / plan_to_gm_orders)
有单元测试锁定。gm 的 run() 会重新加载本文件并调用模块级 init(),
所以 init/任务函数必须在模块级、gm 导入放在函数体内。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from etf_rotation.live import StateStore  # noqa: E402

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
ORDERS_LOG = STATE_DIR / "gm_orders.jsonl"
EXECUTE_TIME = "09:31"


def _g(obj, *names, default=None):
    """dict key / 对象属性双兼容取字段(gm SDK 返回类型以文档为准,留适配点)。"""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _gm_symbol(sym: str) -> str:
    """512890/518880 → SHSE.xxx;159949 → SZSE.xxx。"""
    return ("SHSE." if sym.startswith(("5", "6")) else "SZSE.") + sym


def _plain_symbol(gm_sym: str) -> str:
    return gm_sym.split(".")[-1]


def plan_to_gm_orders(plan: dict) -> list[dict]:
    """计划订单 → 掘金下单参数(保持先卖后买顺序)。

    SELL → action="close_all"(清仓该标的,对应 order_target_percent(symbol, 0));
    BUY  → action="buy_value"(按 est_budget 金额 + limit_price 限价买入)。
    """
    out = []
    for o in plan["orders"]:
        if o["action"] == "SELL":
            out.append({"action": "close_all", "symbol": _gm_symbol(o["symbol"]),
                        "reason": o.get("reason", "")})
        elif o["action"] == "BUY":
            out.append({"action": "buy_value", "symbol": _gm_symbol(o["symbol"]),
                        "value": o["est_budget"], "price": o["limit_price"],
                        "reason": o.get("reason", "")})
    return out


def write_gm_snapshot(store_dir: Path, account: dict, positions: list,
                      executions: list) -> Path:
    """掘金账户快照落盘 gm_snapshot.json(纯函数,与 gm SDK 解耦可单测)。"""
    store_dir = Path(store_dir)
    snap = {"updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "account": account, "positions": positions, "executions": executions}
    f = store_dir / "gm_snapshot.json"
    f.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    return f


def _log_sent_order(plan: dict, order: dict) -> None:
    rec = {"plan_date": plan["plan_date"], "symbol": _plain_symbol(order["symbol"]),
           "side": "SELL" if order["action"] == "close_all" else "BUY",
           "limit_price": order.get("price"), "est_budget": order.get("value"),
           "sent_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    with ORDERS_LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def init(context):
    from gm.api import schedule
    schedule(schedule_func=place_orders, date_rule="1d", time_rule=EXECUTE_TIME)
    schedule(schedule_func=sync_snapshot, date_rule="1d", time_rule="09:35")
    schedule(schedule_func=sync_snapshot, date_rule="1d", time_rule="15:05")


def place_orders(context):
    from gm.api import OrderSide_Buy, OrderType_Limit, order_target_percent, order_value

    store = StateStore(STATE_DIR)
    plan = store.latest_plan()
    if plan is None or not plan.get("orders"):
        print("今日计划无订单(空仓或无变化),不下单")
        return
    if store.plan_executed(plan):
        print(f"计划 {plan['plan_date']} 已执行过,跳过")
        return
    for o in plan_to_gm_orders(plan):
        if o["action"] == "close_all":
            order_target_percent(symbol=o["symbol"], percent=0.0, order_type=OrderType_Limit)
            print(f"SELL 清仓 {o['symbol']}  ({o['reason']})")
        else:
            order_value(symbol=o["symbol"], value=o["value"], side=OrderSide_Buy,
                        order_type=OrderType_Limit, price=o["price"])
            print(f"BUY {o['symbol']} 金额 {o['value']} 限价 {o['price']}  ({o['reason']})")
        _log_sent_order(plan, o)


def sync_snapshot(context):
    """拉掘金仿真账户状态 → gm_snapshot.json → 重新生成仪表盘。"""
    from gm.api import get_cash, get_position
    try:
        from gm.api import get_execution_reports
    except ImportError:                     # gm 版本差异:旧名 get_executions
        from gm.api import get_executions as get_execution_reports  # type: ignore

    cash = get_cash()
    account = {"nav": _g(cash, "nav", "total_value", default=0.0),
               "cash_available": _g(cash, "available", "cash", default=0.0),
               "frozen": _g(cash, "frozen", default=0.0)}
    positions = [{"symbol": _plain_symbol(_g(p, "symbol", default="")),
                  "volume": _g(p, "volume", default=0),
                  "vwap": _g(p, "vwap", "cost", default=0.0),
                  "market_value": _g(p, "market_value", "amount", default=0.0)}
                 for p in (get_position() or [])]
    executions = []
    for e in (get_execution_reports() or []):
        side = str(_g(e, "side", default=""))
        executions.append({"symbol": _plain_symbol(_g(e, "symbol", default="")),
                           "side": "BUY" if side in ("1", "BUY", "OrderSide_Buy") else "SELL",
                           "price": _g(e, "price", default=0.0),
                           "volume": _g(e, "volume", default=0),
                           "created_at": str(_g(e, "created_at", "createdAt", default=""))})
    out = write_gm_snapshot(STATE_DIR, account, positions, executions)
    print(f"快照已更新: {out}")

    from etf_rotation.config import UNIVERSE
    from etf_rotation.dashboard import render_dashboard
    from etf_rotation.data import DATA_DIR
    data = {s: pd.read_csv(DATA_DIR / f"{s}.csv", parse_dates=["date"])
            for s in UNIVERSE if (DATA_DIR / f"{s}.csv").exists()}
    render_dashboard(StateStore(STATE_DIR), data, STATE_DIR / "dashboard.html")


def main() -> None:
    global EXECUTE_TIME
    ap = argparse.ArgumentParser(description="掘金仿真(B 轨):按计划下限价单 + 同步账户快照")
    ap.add_argument("--token", required=True, help="掘金仿真账户 token")
    ap.add_argument("--time", default=EXECUTE_TIME, help="每日下单时间(默认 09:31)")
    args = ap.parse_args()
    EXECUTE_TIME = args.time

    from gm.api import MODE_LIVE, run, set_token  # noqa: 需要 pip install gm
    set_token(args.token)
    run(strategy_id="etf-rotation-paper", filename=str(Path(__file__).resolve()),
        mode=MODE_LIVE)


if __name__ == "__main__":
    main()
