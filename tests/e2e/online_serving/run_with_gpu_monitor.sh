#!/bin/bash
#
# CI/本地 带 GPU 显存监控的测试包装脚本（一步完成）：
# - 后台运行 moniter.sh 采集显存数据
# - 每 15s 将最新一行 GPU 数据打印到日志（CI 中可实时看日志）
# - 可选：设置 GPU_MONITOR_SERVE_DASHBOARD=1 时同时启动网页仪表盘（本地可浏览器实时看）
# - 测试结束后打包 CSV + report.html 并上传 Buildkite artifact（CI）或仅生成本地 bundle（本地）
#
# 用法：./run_with_gpu_monitor.sh -- <任意要执行的命令>
# 本地想边跑边看仪表盘：GPU_MONITOR_SERVE_DASHBOARD=1 ./run_with_gpu_monitor.sh -- pytest ...
# CI 中不设该变量即可，无需额外步骤。
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根目录（脚本在 tests/e2e/online_serving/ 下）
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
export GPU_MONITOR_DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
export SKIP_DEPS_CHECK="${SKIP_DEPS_CHECK:-1}"

# 解析 "--" 后的命令
CMD=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        CMD=("$@")
        break
    fi
    shift
done
if [[ ${#CMD[@]} -eq 0 ]]; then
    echo "Usage: $0 -- <command to run>" >&2
    exit 1
fi

MONITOR_PID=""
DASHBOARD_PID=""
LOG_REPORTER_PID=""
DASHBOARD_PORT="${GPU_MONITOR_DASHBOARD_PORT:-8765}"
cleanup() {
    if [[ -n "$LOG_REPORTER_PID" ]] && kill -0 "$LOG_REPORTER_PID" 2>/dev/null; then
        kill "$LOG_REPORTER_PID" 2>/dev/null || true
    fi
    if [[ -n "$DASHBOARD_PID" ]] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        kill "$DASHBOARD_PID" 2>/dev/null || true
    fi
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    # 有监控数据时始终打包（本地与 CI 都会生成 bundle）；仅在测试完全结束后执行
    if [[ -f "$GPU_MONITOR_DATA_ROOT/current_run_id" ]]; then
        echo "--- Finalizing: bundling GPU monitor data ---"
        TMPF=$(mktemp)
        ./finalize_monitor.sh 2>&1 | tee "$TMPF"
        BUNDLE_LINE=$(grep '^GPU_MONITOR_BUNDLE_DIR=' "$TMPF" || true)
        rm -f "$TMPF"
        if [[ -n "$BUNDLE_LINE" ]]; then
            eval "$BUNDLE_LINE"
            if [[ -d "$GPU_MONITOR_BUNDLE_DIR" ]]; then
                echo "--- GPU monitor bundle dir: $GPU_MONITOR_BUNDLE_DIR ---"
                echo "--- Line chart (memory utilization): open in browser: $GPU_MONITOR_BUNDLE_DIR/report.html ---"
                # 仅在 CI 中上传 artifact
                if command -v buildkite-agent &>/dev/null; then
                    echo "--- Uploading GPU monitor artifacts ---"
                    for f in "$GPU_MONITOR_BUNDLE_DIR"/*; do
                        [[ -e "$f" ]] && buildkite-agent artifact upload "$f"
                    done
                fi
            fi
        else
            echo "--- Bundle not created; check finalize_monitor.sh output above for errors ---"
        fi
    fi
    exit "${TEST_EXIT_CODE:-0}"
}
TEST_EXIT_CODE=0
trap 'cleanup' EXIT

# 启动监控（若无可用的 nvidia-smi 会静默跳过）
if command -v nvidia-smi &>/dev/null; then
    ./moniter.sh all 5 &
    MONITOR_PID=$!
    echo "[GPU Monitor] Started moniter.sh (PID $MONITOR_PID), sampling every 5s; log prints latest line every 15s."
    # 可选：同时启动网页仪表盘（本地设 GPU_MONITOR_SERVE_DASHBOARD=1；CI 不设则跳过）
    if [[ -n "${GPU_MONITOR_SERVE_DASHBOARD:-}" ]] && command -v python3 &>/dev/null; then
        sleep 2
        ./serve_dashboard.sh "$DASHBOARD_PORT" &
        DASHBOARD_PID=$!
        echo "[GPU Monitor] Dashboard started (PID $DASHBOARD_PID). Open in browser: http://127.0.0.1:$DASHBOARD_PORT/gpu_dashboard.html"
        echo "[GPU Monitor] On a remote host, run on your machine: ssh -L $DASHBOARD_PORT:127.0.0.1:$DASHBOARD_PORT <user>@<host> then open the URL above"
    fi
else
    echo "[GPU Monitor] nvidia-smi not found; skipping GPU monitor."
fi

# 后台：每 15s 将最新一行 CSV 打印到 stdout，便于在 CI 日志中实时查看
(
    sleep 10
    while true; do
        sleep 15
        RID_FILE="$GPU_MONITOR_DATA_ROOT/current_run_id"
        [[ -f "$RID_FILE" ]] || continue
        RUN_ID=$(cat "$RID_FILE" 2>/dev/null)
        CSV="$GPU_MONITOR_DATA_ROOT/$RUN_ID/gpu_metrics.csv"
        [[ -f "$CSV" ]] || continue
        LINE=$(tail -1 "$CSV" 2>/dev/null)
        [[ -n "$LINE" ]] && echo "[GPU] $LINE"
    done
) &
LOG_REPORTER_PID=$!

# 在仓库根目录执行用户命令，并记录退出码供 cleanup 使用
(cd "$REPO_ROOT" && "${CMD[@]}") || TEST_EXIT_CODE=$?

# 由 trap 调用 cleanup 后以测试退出码退出
exit $TEST_EXIT_CODE
