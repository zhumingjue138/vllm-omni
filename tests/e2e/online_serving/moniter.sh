#!/bin/bash
#
# L5 长稳 GPU 显存监控（RFC）
# - 每 5s 采集显存利用率，数据持久化 CSV + latest.json
# - 对服务性能影响 <1%（单次 nvidia-smi 查询，无轮询占用）
#
# 运行环境：Linux + NVIDIA 显卡 + 已安装驱动（需 nvidia-smi）
# 依赖：bash, nvidia-smi（必须）, jq（可选，用于实时仪表板 latest.json）
#
# 用法：./moniter.sh [GPU_IDs] [间隔秒数]
#   GPU_IDs: 逗号分隔的 GPU 索引，或 all（默认 all）
#   间隔: 默认 5
# 示例：./moniter.sh all 5    ./moniter.sh 0,1 5
#
# 在目标机器上检查能否运行：nvidia-smi && command -v jq >/dev/null && echo "OK"
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
# Interval (seconds): env GPU_MONITOR_INTERVAL or second positional arg; default 5
INTERVAL="${2:-${GPU_MONITOR_INTERVAL:-5}}"
# GPU IDs: env GPU_MONITOR_DEVICES or first positional arg; "all" or "0,1,2"
GPU_IDS_RAW="${1:-${GPU_MONITOR_DEVICES:-all}}"

# 依赖检查（可设 SKIP_DEPS_CHECK=1 跳过）
if [[ -z "${SKIP_DEPS_CHECK:-}" ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found. Run this script on a Linux machine with NVIDIA drivers."
        exit 1
    fi
    if ! command -v jq &>/dev/null; then
        echo "Note: jq not installed; only CSV will be written (no latest.json, real-time dashboard unavailable)."
    fi
fi

# 当前运行 ID（用于持久化与仪表板）
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$DATA_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$DATA_ROOT/current_run_id"

# CSV 表头
CSV_FILE="$RUN_DIR/gpu_metrics.csv"
echo "timestamp_iso,timestamp_epoch,gpu_index,memory_used_mb,memory_total_mb,memory_util_pct" > "$CSV_FILE"

# 供仪表板使用的最近点数（约 200*5s ≈ 16 分钟）
HISTORY_SIZE=200
HISTORY_FILE="$RUN_DIR/history.jsonl"
LATEST_JSON="$RUN_DIR/latest.json"

# 捕获退出信号
trap 'echo "[$(date +%H:%M:%S)] Stopping; data saved to $RUN_DIR"; exit 0' SIGTERM SIGINT

validate_interval() {
    [[ "$INTERVAL" =~ ^[0-9]+$ ]] && [[ "$INTERVAL" -ge 1 ]] || {
        echo "Error: interval must be a positive integer (seconds)"
        echo "Usage: $0 [GPU_IDs|all] [interval_seconds]"
        exit 1
    }
}
validate_interval

# 构建 nvidia-smi -i 参数
NVSMI_QUERY="index,memory.used,memory.total"
if [[ "$GPU_IDS_RAW" == "all" ]]; then
    NVSMI_IDS=""
else
    NVSMI_IDS="-i $GPU_IDS_RAW"
fi

echo "========================================"
echo "L5 GPU memory monitor started"
echo "RUN_ID: $RUN_ID"
echo "Data dir: $RUN_DIR"
echo "Interval: ${INTERVAL}s | GPU: $GPU_IDS_RAW"
echo "Live dashboard: run ./serve_dashboard.sh in $SCRIPT_DIR, then open the URL (default http://127.0.0.1:8765/gpu_dashboard.html)"
echo "To stop: kill $$ or Ctrl+C"
echo "========================================"

# 主循环：采集并持久化（单次 nvidia-smi + 追加写，对服务影响 <1%）
while true; do
    TS_ISO=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')
    TS_EPOCH=$(date +%s)
    RAW=$(nvidia-smi --query-gpu="$NVSMI_QUERY" --format=csv,noheader,nounits $NVSMI_IDS 2>/dev/null) || true
    if [[ -z "$RAW" ]]; then
        sleep "$INTERVAL"
        continue
    fi

    # 解析每张卡并写 CSV，同时收集当前快照用于 latest.json
    GPUS_ARR=""
    while IFS= read -r line; do
        line=$(echo "$line" | tr -d ' ')
        [[ -z "$line" ]] && continue
        idx=$(echo "$line" | cut -d',' -f1)
        used=$(echo "$line" | cut -d',' -f2)
        total=$(echo "$line" | cut -d',' -f3)
        used=${used:-0}
        total=${total:-1}
        [[ "$total" -le 0 ]] && total=1
        pct=$((used * 100 / total))
        echo "${TS_ISO},${TS_EPOCH},${idx},${used},${total},${pct}" >> "$CSV_FILE"
        if [[ -n "$GPUS_ARR" ]]; then
            GPUS_ARR="$GPUS_ARR,{\"gpu_index\":$idx,\"memory_used_mb\":$used,\"memory_total_mb\":$total,\"memory_util_pct\":$pct}"
        else
            GPUS_ARR="{\"gpu_index\":$idx,\"memory_used_mb\":$used,\"memory_total_mb\":$total,\"memory_util_pct\":$pct}"
        fi
    done <<< "$RAW"
    CURR_JSON="[$GPUS_ARR]"
    ROW_JSON="{\"t\":$TS_EPOCH,\"gpus\":$CURR_JSON}"

    # 追加 history（仪表板用近期曲线，不在此处截断文件以减轻 IO）
    echo "$ROW_JSON" >> "$HISTORY_FILE"

    # 写入 latest.json（当前快照 + 最近 HISTORY_SIZE 条历史）
    if command -v jq &>/dev/null; then
        HIST_JSON=$(tail -n "$HISTORY_SIZE" "$HISTORY_FILE" 2>/dev/null | jq -s . 2>/dev/null) || HIST_JSON="[]"
        echo "{\"run_id\":\"$RUN_ID\",\"last_updated\":\"$TS_ISO\",\"last_updated_epoch\":$TS_EPOCH,\"current\":$CURR_JSON,\"history\":$HIST_JSON}" > "$LATEST_JSON"
    fi

    sleep "$INTERVAL"
done
