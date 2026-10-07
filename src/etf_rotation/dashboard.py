"""虚盘仪表盘:生成自包含 HTML(state/dashboard.html),浏览器直接打开。

设计:不起 web 服务器——cron 每次任务后重新生成,文件即最新状态;
单文件内嵌 CSS 与 base64 净值图,无外部依赖。`<meta refresh>` 每 5 分钟自刷。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import UNIVERSE
from .live import StateStore

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

_CSS = """
:root { --up:#c0392b; --down:#27ae60; --bg:#0f1419; --card:#1a2129; --line:#2a3440; --txt:#e6e9ee; --dim:#8a94a0; }
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--txt); font: 14px/1.6 -apple-system, "PingFang SC", sans-serif; padding: 24px; max-width: 1080px; margin: 0 auto; }
h1 { font-size: 20px; margin-bottom: 4px; }
h2 { font-size: 15px; color: var(--dim); font-weight: 600; margin: 24px 0 8px; letter-spacing: .05em; }
.sub { color: var(--dim); font-size: 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card .k { color: var(--dim); font-size: 12px; }
.card .v { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--line); }
th { color: var(--dim); font-size: 12px; font-weight: 600; text-align: right; }
td:first-child, th:first-child { text-align: left; }
.up { color: var(--up); } .down { color: var(--down); } .dim { color: var(--dim); }
.scroll { max-height: 420px; overflow-y: auto; border-radius: 10px; }
.scroll table th { position: sticky; top: 0; background: var(--card); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: 12px; }
.badge.buy { background: rgba(192,57,43,.18); color: var(--up); }
.badge.sell { background: rgba(39,174,96,.18); color: var(--down); }
.plan { background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--up); border-radius: 8px; padding: 12px 16px; }
img.curve { width: 100%; border-radius: 10px; border: 1px solid var(--line); }
"""


def _cls(v: float) -> str:
    return "up" if v >= 0 else "down"     # A股配色:红涨绿跌


def record_equity(store: StateStore, cash: float, market_value: float) -> None:
    """把当日权益追加到 state/equity.csv(按日期去重,重跑覆盖当天)。"""
    f = store.dir / "equity.csv"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    if f.exists():
        rows = [x for x in f.read_text().splitlines()[1:] if not x.startswith(today)]
    rows.append(f"{today},{cash:.2f},{market_value:.2f},{cash + market_value:.2f}")
    f.write_text("date,cash,market_value,total\n" + "\n".join(rows) + "\n")


def _equity_chart(store: StateStore) -> str:
    """净值曲线 → base64 PNG;不足 2 个点时返回空串。"""
    f = store.dir / "equity.csv"
    if not f.exists() or len(f.read_text().splitlines()) < 3:
        return ""
    df = pd.read_csv(f, parse_dates=["date"])
    base = df["total"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(df["date"], df["total"] / base, lw=1.8, color="#c0392b")
    ax.axhline(1.0, lw=0.7, color="#8a94a0", alpha=0.6)
    ax.set_title(f"虚盘净值(起点 {df['date'].iloc[0].date()} = 1.0)")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="#1a2129")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_dashboard(store: StateStore, data: dict[str, pd.DataFrame], out_path: str | Path) -> Path:
    """渲染仪表盘 HTML。data = {symbol: 行情 df}(读缓存即可,纯离线)。"""
    out_path = Path(out_path)
    state = store.load_state()
    closes = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    asof = str(data[list(data)[0]]["date"].iloc[-1].date()) if data else "无数据"

    mv = sum(h.shares * closes.get(s, 0.0) for s, h in state.holdings.items())
    total = state.cash + mv
    record_equity(store, state.cash, mv)

    # ---- 持仓表 ----
    hold_rows = ""
    for s, h in sorted(state.holdings.items()):
        px = closes.get(s)
        if not px:
            continue
        pnl_pct = px / h.cost - 1
        pnl_yuan = h.shares * (px - h.cost)
        stop_line = h.peak * 0.92                       # trailing_stop=0.08
        hold_rows += (f"<tr><td>{s} {UNIVERSE.get(s, '')}</td><td>{h.shares:,.0f}</td>"
                      f"<td>{h.cost:.3f}</td><td>{px:.3f}</td>"
                      f"<td class='{_cls(pnl_pct)}'>{pnl_pct:+.2%}</td>"
                      f"<td class='{_cls(pnl_yuan)}'>{pnl_yuan:+,.0f}</td>"
                      f"<td class='dim'>{h.peak:.3f} / 止损线 {stop_line:.3f}</td></tr>")
    if not hold_rows:
        hold_rows = "<tr><td colspan='7' class='dim'>(空仓)</td></tr>"

    # ---- 待执行计划 + 评分 + 溢价 ----
    plan = store.latest_plan()
    plan_html = "<p class='dim'>无待执行计划</p>"
    scores_rows = prem_rows = ""
    if plan and not store.plan_executed(plan):
        orders = "".join(
            f"<div><span class='badge {o['action'].lower()}'>{o['action']}</span> "
            f"{o['symbol']} {UNIVERSE.get(o['symbol'], '')} 限价 {o['limit_price']} "
            f"<span class='dim'>约 {o.get('est_budget', 0):,.0f} 元 · {o.get('reason', '')}</span></div>"
            for o in plan["orders"]) or "<div class='dim'>无订单(空仓/持有不变)</div>"
        plan_html = (f"<div class='plan'><b>{plan['plan_date']} 生成</b>"
                     f"<span class='dim'>(信号日 {plan['data_asof']})</span>{orders}</div>")
        for s, v in sorted(plan.get("scores", {}).items(), key=lambda x: -x[1]):
            scores_rows += f"<tr><td>{s} {UNIVERSE.get(s, '')}</td><td>{v:+.4f}</td></tr>"
        for s, st in plan.get("premium_check", {}).items():
            prem_rows += (f"<tr><td>{s} {UNIVERSE.get(s, '')}</td>"
                          f"<td>{st['premium'] if st['premium'] is not None else '—'}</td>"
                          f"<td class='dim'>{st['asof']} / 滞后 {st['nav_lag']} 日"
                          f"{' / ⚠️ 陈旧' if st['stale'] else ''}</td></tr>")

    # ---- 成交历史 ----
    fill_rows = ""
    if store.log_file.exists():
        recs = [json.loads(x) for x in store.log_file.read_text().splitlines()]
        fills = [r for r in recs if r.get("type") == "fill"]
        for f in reversed(fills):
            for x in f["fills"]:
                fill_rows += (f"<tr><td>{f['plan_date']}</td>"
                              f"<td><span class='badge {x['action'].lower()}'>{x['action']}</span></td>"
                              f"<td>{x['symbol']} {UNIVERSE.get(x['symbol'], '')}</td>"
                              f"<td>{x['fill_price']}</td>"
                              f"<td class='dim'>限价 {x.get('limit_price', '—')}</td></tr>")
    if not fill_rows:
        fill_rows = "<tr><td colspan='5' class='dim'>尚无成交</td></tr>"

    chart = _equity_chart(store)
    chart_html = f"<img class='curve' src='data:image/png;base64,{chart}'>" if chart else \
        "<p class='dim'>净值数据积累中(明日开始有曲线)</p>"

    principal = total - state.realized_pnl
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<title>ETF 轮动虚盘</title><style>{_CSS}</style></head><body>
<h1>ETF 三因子轮动 · 虚盘仪表盘</h1>
<p class="sub">行情截至 {asof} · 页面每 5 分钟自动刷新 · 由 cron 每日任务生成</p>
<div class="cards">
  <div class="card"><div class="k">总权益</div><div class="v">{total:,.0f}</div></div>
  <div class="card"><div class="k">现金</div><div class="v">{state.cash:,.0f}</div></div>
  <div class="card"><div class="k">持仓市值</div><div class="v">{mv:,.0f}</div></div>
  <div class="card"><div class="k">本金(含浮动)</div><div class="v">{principal:,.0f}</div></div>
  <div class="card"><div class="k">累计盈亏(含浮动)</div>
    <div class="v {_cls(total - principal)}">{total - principal:+,.0f}</div></div>
  <div class="card"><div class="k">已实现盈亏</div>
    <div class="v {_cls(state.realized_pnl)}">{state.realized_pnl:+,.0f}</div></div>
</div>
<h2>待执行计划</h2>{plan_html}
<h2>当前持仓</h2>
<table><tr><th>标的</th><th>份额</th><th>成本</th><th>现价</th><th>浮盈亏%</th><th>浮盈亏(元)</th><th>峰值/止损线</th></tr>
{hold_rows}</table>
<h2>净值曲线</h2>{chart_html}
<h2>最新评分</h2>
<table><tr><th>标的</th><th>综合评分</th></tr>{scores_rows or "<tr><td colspan='2' class='dim'>—</td></tr>"}</table>
<h2>溢价安检(v3.1 规则:>7% 不开新仓)</h2>
<table><tr><th>标的</th><th>最新溢价</th><th>净值日期 / 滞后</th></tr>{prem_rows or "<tr><td colspan='3' class='dim'>—</td></tr>"}</table>
<h2>成交历史</h2>
<div class="scroll"><table><tr><th>日期</th><th>方向</th><th>标的</th><th>成交价</th><th>计划限价</th></tr>
{fill_rows}</table></div>
<p class="sub" style="margin-top:16px">禁买名单: {state.stopped or '无'} · 评分连击: {state.top_streak or '无'}</p>
</body></html>"""
    out_path.write_text(html)
    return out_path
