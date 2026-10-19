"""Bark 例行任务通知。

早晚定时任务结束后,把最近一次运行状态、最新计划、评分、账户权益与持仓
摘要推送到 iPhone。只依赖 Python 标准库;BARK_KEY 从环境变量或项目根
.env 读取,不写入代码和 Git。
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

from .config import UNIVERSE, StrategyConfig
from .live import StateStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = "https://api.day.app"
DEFAULT_GROUP = "ETF轮动"
DEFAULT_TTL = 3600
_TASK_LABELS = {"morning": "早间", "evening": "晚间", "manual": "手动"}
_STATUS_ICONS = {"正常": "✅", "有警告": "⚠️", "异常": "❌", "无日志": "ℹ️"}


def load_bark_key(root: Path | None = None) -> str | None:
    """读取 BARK_KEY:环境变量优先,其次项目根 .env;未配置返回 None。"""
    value = os.environ.get("BARK_KEY")
    if value:
        return value.strip()
    env_file = (root or ROOT) / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith("BARK_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def latest_log_block(log_file: Path) -> list[str]:
    """返回 paper.log 中最近一次例行任务块;无任务头时返回最后 200 行。"""
    if not log_file.exists():
        return []
    lines = log_file.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("===")]
    return lines[starts[-1]:] if starts else lines[-200:]


def summarize_log(block: list[str]) -> tuple[str, list[str]]:
    """把最近一次任务日志概括为 正常/有警告/异常,并提取最多 3 条问题行。"""
    if not block:
        return "无日志", []
    issues: list[str] = []
    traceback_count = 0
    meaningful: list[str] = []
    for raw in block:
        line = raw.strip()
        if not line or "it/s" in line:
            continue
        meaningful.append(line)
        if line == "Traceback (most recent call last):":
            traceback_count += 1
            continue
        if any(term in line for term in (
                "Error:", "取数失败", "刷新失败", "无法执行", "回退东财", "警告",
                "不执行", "未成交", "信号可能过时")):
            if line not in issues:
                issues.append(line)
    if traceback_count:
        if not issues and meaningful:
            issues.append(meaningful[-1])
        return "异常", issues[-3:]
    return ("有警告", issues[-3:]) if issues else ("正常", [])


def _load_params(store: StateStore) -> StrategyConfig:
    if not store.params_file.exists():
        return StrategyConfig()
    return StrategyConfig(**json.loads(store.params_file.read_text()))


def _latest_closes(data_dir: Path) -> tuple[dict[str, float], str]:
    closes: dict[str, float] = {}
    dates: list[pd.Timestamp] = []
    for symbol in UNIVERSE:
        path = data_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue
        closes[symbol] = float(df["close"].iloc[-1])
        dates.append(df["date"].iloc[-1])
    return closes, str(max(dates).date()) if dates else "无数据"


def _display_symbol(symbol: str) -> str:
    name = UNIVERSE.get(symbol, "")
    return f"{symbol} {name}" if name else symbol


def _latest_fill(store: StateStore, plan: dict) -> dict | None:
    if not store.log_file.exists():
        return None
    for line in reversed(store.log_file.read_text().splitlines()):
        rec = json.loads(line)
        if (rec.get("type") == "fill"
                and rec.get("plan_date") == plan.get("plan_date")
                and rec.get("data_asof") == plan.get("data_asof")):
            return rec
    return None


def _plan_lines(store: StateStore) -> tuple[list[str], dict | None]:
    plan = store.latest_plan()
    lines = ["【计划】"]
    if plan is None:
        return lines + ["无"], None
    lines.append(f"计划 {plan['plan_date']}｜信号 {plan['data_asof']}")
    orders = plan.get("orders") or []
    if not orders:
        lines.append("无订单（持有不变/空仓）")
        return lines, plan
    if store.plan_executed(plan):
        fill = _latest_fill(store, plan)
        lines.append("已成交")
        if fill:
            lines.extend(f"{x['action']} {x['symbol']}｜成交价 {x['fill_price']}"
                         for x in fill["fills"])
        return lines, plan
    lines.append("待执行")
    for order in orders:
        lines.append(f"{order['action']} {order['symbol']}")
        budget = order.get("est_budget")
        budget_text = f"｜约 {budget:,.0f}元" if isinstance(budget, int | float) else ""
        lines.append(f"限价 {order['limit_price']}{budget_text}")
    return lines, plan


def _scores_lines(plan: dict | None) -> list[str]:
    scores = (plan or {}).get("scores") or {}
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ["【评分】", *[f"{symbol} {score:+.4f}" for symbol, score in ranked]]


def _premium_lines(plan: dict | None, cfg: StrategyConfig) -> list[str]:
    checks = (plan or {}).get("premium_check") or {}
    if not checks or cfg.premium_max is None:
        return []
    blocked = []
    stale = []
    for symbol, status in checks.items():
        premium = status.get("premium")
        if premium is not None and premium > cfg.premium_max:
            blocked.append(f"{symbol} 溢价 {premium:.1%}｜禁新开仓")
        if status.get("stale"):
            stale.append(f"{symbol} 溢价数据滞后")
    lines = blocked or stale
    return ["【提示】", *lines] if lines else []


def _account_lines(store: StateStore, closes: dict[str, float], cfg: StrategyConfig) -> list[str]:
    state = store.load_state(cfg.initial_cash)
    market_value = sum(h.shares * closes.get(s, 0.0) for s, h in state.holdings.items())
    total = state.cash + market_value
    pnl = total - cfg.initial_cash
    lines = [
        "【账户】",
        f"权益 {total:,.2f}",
        f"收益 {pnl:+,.2f}（{pnl / cfg.initial_cash:+.2%}）",
        "",
        "【持仓】",
    ]
    if not state.holdings:
        lines.append(f"空仓｜现金 {state.cash:,.2f}")
        return lines
    for index, (symbol, holding) in enumerate(sorted(state.holdings.items())):
        if index:
            lines.append("")
        lines.append(_display_symbol(symbol))
        price = closes.get(symbol)
        if price is None:
            lines.append(f"{holding.shares:,.0f}份｜无最新收盘价")
            continue
        pnl_pct = price / holding.cost - 1
        pnl_yuan = holding.shares * (price - holding.cost)
        pnl_label = "浮盈" if pnl_yuan >= 0 else "浮亏"
        lines.append(f"{holding.shares:,.0f}份｜{holding.cost:.3f}→{price:.3f}")
        lines.append(f"{pnl_label} {pnl_pct:+.2%}｜{pnl_yuan:+,.0f}元")
        if cfg.trailing_stop is not None:
            lines.append(f"止损线 {holding.peak * (1 - cfg.trailing_stop):.3f}")
    return lines


def _short_issue(line: str, limit: int = 72) -> str:
    return line if len(line) <= limit else line[: limit - 1] + "…"


def build_notification(
    state_dir: Path,
    data_dir: Path,
    log_file: Path,
    task: str = "manual",
) -> tuple[str, str]:
    """生成 Bark 标题和正文(纯本地读取,不联网)。"""
    store = StateStore(state_dir)
    cfg = _load_params(store)
    closes, asof = _latest_closes(data_dir)
    status, issues = summarize_log(latest_log_block(log_file))
    label = _TASK_LABELS.get(task, task)
    icon = _STATUS_ICONS.get(status, "ℹ️")

    plan_lines, plan = _plan_lines(store)
    sections = [
        plan_lines,
        _scores_lines(plan),
        _premium_lines(plan, cfg),
        _account_lines(store, closes, cfg),
    ]
    body = [f"状态：{status}｜行情 {asof}"]
    for section in sections:
        if section:
            body.extend(["", *section])
    if issues:
        body.extend(["", "【日志】", *[_short_issue(x) for x in issues]])
    return f"{icon} ETF轮动｜{label}{status}", "\n".join(body)


def send_bark(
    key: str,
    title: str,
    body: str,
    *,
    server: str = DEFAULT_SERVER,
    group: str = DEFAULT_GROUP,
    ttl: int = DEFAULT_TTL,
) -> None:
    """通过 Bark POST JSON API 发送通知;接口返回非 200 时抛错。"""
    key = key.strip()
    if not key:
        raise RuntimeError("未配置 BARK_KEY")
    url = f"{server.rstrip('/')}/push"
    payload = {"device_key": key, "title": title, "body": body, "group": group, "ttl": ttl}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("code") != 200:
        raise RuntimeError(f"Bark 推送失败:{result.get('message', result)}")


def send_daily_notification(
    state_dir: Path,
    data_dir: Path,
    log_file: Path,
    task: str = "manual",
) -> None:
    """生成并发送例行任务摘要。"""
    key = load_bark_key()
    if key is None:
        raise RuntimeError("未配置 BARK_KEY(环境变量或项目根 .env)")
    title, body = build_notification(state_dir, data_dir, log_file, task)
    send_bark(key, title, body)
