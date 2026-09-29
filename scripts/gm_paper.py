#!/usr/bin/env python
"""掘金仿真交易适配器:把 cli plan 生成的计划单发送到掘金仿真账户。

信号仍由本仓库引擎生成(canonical),掘金只做下单通道——这样虚盘与回测/实盘计划
永远同源,不存在"平台策略代码漂移"问题。

准备步骤(一次性):
  1. sim.myquant.cn 注册,创建股票仿真账户,获取 token;
  2. pip install gm   (掘金 SDK,gm 3.x);
  3. 每晚照常:PYTHONPATH=src python -m etf_rotation.cli fetch && ... cli plan

运行(每个交易日上午挂着,09:31 自动读最新计划并下限价单):
  PYTHONPATH=src python scripts/gm_paper.py --token <你的token>

注意:本文件在本仓库环境未连掘金实测(gm 未安装),下单常量名以 gm 3.x 文档为准;
纯映射逻辑(plan_to_gm_orders)有单元测试锁定。gm 的 run() 会重新加载本文件并
调用模块级 init(),所以 init/place_orders 必须在模块级、gm 导入放在函数体内
(否则本仓库环境 import 本文件测试时会因缺 gm 而失败)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etf_rotation.live import StateStore  # noqa: E402

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
EXECUTE_TIME = "09:31"


def _gm_symbol(sym: str) -> str:
    """512890/518880 → SHSE.xxx;159949 → SZSE.xxx。"""
    return ("SHSE." if sym.startswith(("5", "6")) else "SZSE.") + sym


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


def init(context):
    from gm.api import schedule
    schedule(schedule_func=place_orders, date_rule="1d", time_rule=EXECUTE_TIME)


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
    # 掘金侧有成交回报;本仓库的成交记账以次日 cli execute 或手工对账为准


def main() -> None:
    global EXECUTE_TIME
    ap = argparse.ArgumentParser(description="掘金仿真:按计划下限价单")
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
