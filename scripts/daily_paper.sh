#!/bin/bash
# 虚盘每日自动任务(配合 crontab):
#   evening → cli fetch + cli plan   (每晚 20:30,净值公布后)
#   morning → cli execute            (每交易日 9:31,开盘成交昨日计划)
# 非交易日自动跳过:execute 检测"今日 bar 未生成"退出;plan 检测"信号日+参数未变"跳过。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONPATH=src NO_PROXY='*'
PY="${PYTHON:-/Users/apple/Desktop/AIExp/.venv/bin/python}"
LOG=logs/paper.log

case "${1:-}" in
  evening)
    echo "=== $(date '+%F %T') fetch + plan ===" >> "$LOG"
    "$PY" -m etf_rotation.cli fetch >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli plan  >> "$LOG" 2>&1
    ;;
  morning)
    echo "=== $(date '+%F %T') execute ===" >> "$LOG"
    "$PY" -m etf_rotation.cli execute >> "$LOG" 2>&1
    ;;
  *)
    echo "用法: $0 evening|morning"
    exit 1
    ;;
esac
tail -5 "$LOG"
