#!/bin/bash
#
# 长稳结束后调用：打包当前 run 的 CSV + 报告，供 CI 归档（upload-artifact 等）。
# 在环境被清理前执行本脚本，将输出目录上传为 CI 产物，即可在流水线结束后仍可下载查看。
#
# 用法：./finalize_monitor.sh [run_id]
#   run_id: 不传则使用 current_run_id（即最近一次 moniter.sh 的 run）
# 输出：打印 GPU_MONITOR_BUNDLE_DIR=<绝对路径>，CI 可解析该行并归档该目录。
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
RUN_ID="${1:-}"

if [[ -z "$RUN_ID" ]]; then
    if [[ -f "$DATA_ROOT/current_run_id" ]]; then
        RUN_ID=$(cat "$DATA_ROOT/current_run_id")
    else
        echo "Error: run_id not specified and $DATA_ROOT/current_run_id does not exist" >&2
        exit 1
    fi
fi

RUN_DIR="$DATA_ROOT/$RUN_ID"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: run dir does not exist: $RUN_DIR" >&2
    exit 1
fi

CSV_FILE="$RUN_DIR/gpu_metrics.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV not found: $CSV_FILE" >&2
    exit 1
fi

# 打包目录：与 run 同级的 bundle，便于 CI 只归档这一份
BUNDLE_DIR="$DATA_ROOT/gpu_monitor_bundle_${RUN_ID}"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# 复制原始数据
cp "$CSV_FILE" "$BUNDLE_DIR/gpu_metrics.csv"
if [[ -f "$RUN_DIR/history.jsonl" ]]; then
    cp "$RUN_DIR/history.jsonl" "$BUNDLE_DIR/" 2>/dev/null || true
fi

# 生成报告（需 Python3）
REPORT_HTML="$BUNDLE_DIR/report.html"
if command -v python3 &>/dev/null; then
    if python3 "$SCRIPT_DIR/generate_report.py" "$CSV_FILE" "$REPORT_HTML"; then
        echo "Report generated: $REPORT_HTML"
    else
        echo "Warning: report generation failed; only CSV archived" >&2
    fi
else
    echo "Warning: python3 not found; skipping report, only CSV archived" >&2
fi

# 简要说明，便于下载后查看
cat > "$BUNDLE_DIR/README.txt" << EOF
L5 GPU monitor bundle - ${RUN_ID}
- gpu_metrics.csv: raw samples (timestamp_iso, timestamp_epoch, gpu_index, memory_used_mb, memory_total_mb, memory_util_pct)
- report.html: report with charts and anomaly markers (open in browser)
Upload this dir as a CI artifact to view after the run.
EOF

# 供 CI 解析：归档此目录即可保留监控数据与报告
BUNDLE_ABS=$(cd "$BUNDLE_DIR" && pwd)
echo "GPU_MONITOR_BUNDLE_DIR=$BUNDLE_ABS"
echo "Archive path: $BUNDLE_ABS"
