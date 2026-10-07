#!/usr/bin/env python
"""掘金仿真"下载交易数据"CSV → gm_snapshot.json 导入器(Mac 手工路径)。

背景:gmtrade SDK 仅 Win/Linux 二进制且无 macOS 轮子,Mac 上 B 轨走
"网页手工下单 + 定期下载 CSV 导入"路线:
  1. sim.myquant.cn 网页创建股票仿真账户;
  2. 每个有计划的交易日 9:31 后,在掘金网页手工下限价单(月均 ~3 次,空仓日不用动);
  3. 在账户页点"下载交易数据"(含 资金/持仓/历史委托/历史成交 的 CSV 或 zip),
     然后:PYTHONPATH=src python scripts/gm_import_csv.py <下载的文件或目录>
  4. 仪表盘"双轨对照"区块随即更新。

列名防御:掘金 CSV 实际列名以下载文件为准,下面 MAP 用多候选匹配;
若导入后字段全空,把 CSV 头一行发我/自行补一行映射即可。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                     # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))        # src/

import pandas as pd  # noqa: E402

STATE_DIR = Path(__file__).resolve().parents[1] / "state"

# 列名候选(英文 API 字段名 / 常见中文表头)
MAP = {
    "symbol": ["symbol", "sec_id", "代码", "证券代码", "标的"],
    "side": ["side", "direction", "方向", "买卖方向"],
    "price": ["price", "exec_price", "成交价", "成交价格", "price_or_avg"],
    "volume": ["volume", "qty", "exec_volume", "数量", "成交数量", "成交量"],
    "created_at": ["created_at", "exec_time", "time", "成交时间", "日期", "成交日期"],
    "vwap": ["vwap", "cost", "成本", "成本价", "持仓成本"],
    "market_value": ["market_value", "amount", "市值"],
    "nav": ["nav", "total_value", "total_asset", "净资产", "总资产", "权益"],
    "available": ["available", "cash", "可用资金", "可用"],
    "frozen": ["frozen", "冻结资金", "冻结"],
}
SIDE_BUY = {"1", "BUY", "OrderSide_Buy", "买", "买入"}


def _pick(df: pd.DataFrame, key: str) -> str | None:
    for c in MAP[key]:
        if c in df.columns:
            return c
    return None


def _read_tables(src: Path) -> dict[str, pd.DataFrame]:
    """从目录/zip/单文件里找出 资金/持仓/成交 三类 CSV。"""
    files: list[Path] = []
    if src.is_dir():
        files = list(src.glob("*.csv"))
    elif zipfile.is_zipfile(src):
        tmp = Path(tempfile.mkdtemp(prefix="gm_imp_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        files = list(tmp.rglob("*.csv"))
    elif src.suffix == ".csv":
        files = [src]
    out: dict[str, pd.DataFrame] = {}
    for f in files:
        name = f.name.lower()
        kind = None
        if any(k in name for k in ("cash", "资金", "account")):
            kind = "cash"
        elif any(k in name for k in ("position", "持仓")):
            kind = "position"
        elif any(k in name for k in ("exec", "trade", "fill", "成交")) and "委托" not in name:
            kind = "exec"
        if kind:
            out[kind] = pd.read_csv(f)
    if not out and files:      # 单文件混合导出:按列名特征猜
        df = pd.read_csv(files[0])
        if _pick(df, "nav"):
            out["cash"] = df
        elif _pick(df, "vwap") or _pick(df, "market_value"):
            out["position"] = df
        elif _pick(df, "price") and _pick(df, "volume"):
            out["exec"] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="掘金仿真 CSV 导入 → gm_snapshot.json")
    ap.add_argument("src", help="下载的 CSV / zip / 解压目录")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    args = ap.parse_args()

    from gm_paper import write_gm_snapshot  # noqa: 复用同一个快照写入器(scripts/ 已入 path)

    tables = _read_tables(Path(args.src).expanduser())
    if not tables:
        raise SystemExit("未识别到 资金/持仓/成交 CSV,请检查文件")

    account, positions, executions = {}, [], []
    if "cash" in tables:
        df = tables["cash"]
        r = df.iloc[-1]
        account = {"nav": float(r[_pick(df, "nav")] if _pick(df, "nav") else 0),
                   "cash_available": float(r[_pick(df, "available")] if _pick(df, "available") else 0),
                   "frozen": float(r[_pick(df, "frozen")] if _pick(df, "frozen") else 0)}
    if "position" in tables:
        df = tables["position"]
        cs, cv, cw, cm = (_pick(df, k) for k in ("symbol", "volume", "vwap", "market_value"))
        for _, r in df.iterrows():
            positions.append({"symbol": str(r[cs]).split(".")[-1], "volume": float(r[cv]),
                              "vwap": float(r[cw]) if cw else 0.0,
                              "market_value": float(r[cm]) if cm else 0.0})
    if "exec" in tables:
        df = tables["exec"]
        cs, cd, cp, cv, ct = (_pick(df, k) for k in ("symbol", "side", "price", "volume", "created_at"))
        for _, r in df.iterrows():
            executions.append({"symbol": str(r[cs]).split(".")[-1],
                               "side": "BUY" if str(r[cd]) in SIDE_BUY else "SELL",
                               "price": float(r[cp]), "volume": float(r[cv]),
                               "created_at": str(r[ct])})
    out = write_gm_snapshot(Path(args.state_dir), account, positions, executions)
    print(f"导入完成: {out}")
    print(f"  账户 {account or '—'}")
    print(f"  持仓 {len(positions)} 只 | 成交回报 {len(executions)} 笔")
    # 重新生成仪表盘
    from etf_rotation.config import UNIVERSE
    from etf_rotation.dashboard import render_dashboard
    from etf_rotation.data import DATA_DIR
    from etf_rotation.live import StateStore
    data = {s: pd.read_csv(DATA_DIR / f"{s}.csv", parse_dates=["date"])
            for s in UNIVERSE if (DATA_DIR / f"{s}.csv").exists()}
    render_dashboard(StateStore(args.state_dir), data, Path(args.state_dir) / "dashboard.html")
    print("仪表盘已更新")


if __name__ == "__main__":
    main()
