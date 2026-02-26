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
    echo "用法: $0 -- <要执行的命令>" >&2
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
        echo "--- 收尾：打包 GPU 监控数据 ---"
        TMPF=$(mktemp)
        ./finalize_monitor.sh 2>&1 | tee "$TMPF"
        BUNDLE_LINE=$(grep '^GPU_MONITOR_BUNDLE_DIR=' "$TMPF" || true)
        rm -f "$TMPF"
        if [[ -n "$BUNDLE_LINE" ]]; then
            eval "$BUNDLE_LINE"
            if [[ -d "$GPU_MONITOR_BUNDLE_DIR" ]]; then
                echo "--- GPU 监控产物目录: $GPU_MONITOR_BUNDLE_DIR ---"
                echo "--- 用浏览器打开其中的 report.html 查看折线图 ---"
                # 仅在 CI 中上传 artifact
                if command -v buildkite-agent &>/dev/null; then
                    echo "--- 上传 GPU 监控产物 ---"
                    for f in "$GPU_MONITOR_BUNDLE_DIR"/*; do
                        [[ -e "$f" ]] && buildkite-agent artifact upload "$f"
                    done
                fi
            fi
        else
            echo "--- 未生成 bundle，请查看上方 finalize_monitor.sh 的错误输出 ---"
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
    echo "[GPU Monitor] 已启动 moniter.sh (PID $MONITOR_PID)，每 5s 采集显存；日志每 15s 打印最新一行。"
    # 可选：同时启动网页仪表盘（本地设 GPU_MONITOR_SERVE_DASHBOARD=1；CI 不设则跳过）
    if [[ -n "${GPU_MONITOR_SERVE_DASHBOARD:-}" ]] && command -v python3 &>/dev/null; then
        sleep 2
        ./serve_dashboard.sh "$DASHBOARD_PORT" &
        DASHBOARD_PID=$!
        echo "[GPU Monitor] 已启动仪表盘 (PID $DASHBOARD_PID)，浏览器访问: http://127.0.0.1:$DASHBOARD_PORT/gpu_dashboard.html"
        echo "[GPU Monitor] 若为远程机器，请在本机执行: ssh -L $DASHBOARD_PORT:127.0.0.1:$DASHBOARD_PORT <用户>@<主机> 后访问上述 URL"
    fi
else
    echo "[GPU Monitor] 未检测到 nvidia-smi，跳过 GPU 监控。"
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
