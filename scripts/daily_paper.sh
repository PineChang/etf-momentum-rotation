#!/bin/bash
# 虚盘每日自动任务(配合 crontab 补跑时隙,见 README/审计报告):
#   evening → cli fetch + cli plan + dashboard + Bark 通知   (每晚 20:30,净值公布后;21:30 补跑时隙)
#   morning → cli fetch + cli plan + cli execute + dashboard + Bark 通知
#     (每交易日 9:31;fetch+plan 是补跑——前一晚任务被睡眠错过时在这里追上,
#      缓存已新鲜则全部自动跳过,零副作用;10:31 为崩溃重试时隙)
# 幂等性保证补跑安全:execute 检测"计划已执行"拒绝重放;plan 检测"信号日+参数
# 未变"跳过;fetch 检测缓存新鲜度(_needs_refresh)盘中不刷、收盘后补昨日 bar。
# 注:macOS 上 launchd 的子进程无 ~/Desktop 读权限(TCC),cron 有——故用 cron。
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
    "$PY" -m etf_rotation.cli dashboard >> "$LOG" 2>&1
    # 数据校验核准:放 dashboard 后(权益自愈先执行),notify 前(问题进 Bark 摘要)
    "$PY" -m etf_rotation.cli validate --cross >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli notify --task evening >> "$LOG" 2>&1
    ;;
  morning)
    echo "=== $(date '+%F %T') fetch + plan + execute ===" >> "$LOG"
    "$PY" -m etf_rotation.cli fetch >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli plan  >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli execute >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli dashboard >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli validate >> "$LOG" 2>&1
    "$PY" -m etf_rotation.cli notify --task morning >> "$LOG" 2>&1
    ;;
  *)
    echo "用法: $0 evening|morning"
    exit 1
    ;;
esac
tail -5 "$LOG"
