#!/bin/bash
#
# CI 中带 GPU 显存监控的测试包装脚本：
# - 后台运行 moniter.sh 采集显存数据
# - 每 15s 将最新一行 GPU 数据打印到日志（Buildkite 中可实时看到）
# - 可选：若需网页仪表盘，可设置 GPU_MONITOR_TUNNEL=1 并配置隧道（如 ngrok）
# - 测试结束后打包 CSV + report.html（含折线图）并上传 Buildkite artifact
#
# 用法：./run_with_gpu_monitor.sh -- <任意要执行的命令>
# 示例：./run_with_gpu_monitor.sh -- pytest -v -s -m 'core_model and cpu' ...
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
LOG_REPORTER_PID=""
cleanup() {
    if [[ -n "$LOG_REPORTER_PID" ]] && kill -0 "$LOG_REPORTER_PID" 2>/dev/null; then
        kill "$LOG_REPORTER_PID" 2>/dev/null || true
    fi
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    # 打包并上传 artifact（仅当有 buildkite-agent 且在 CI 中）
    if command -v buildkite-agent &>/dev/null && [[ -f "$GPU_MONITOR_DATA_ROOT/current_run_id" ]]; then
        BUNDLE_LINE=$(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
        if [[ -n "$BUNDLE_LINE" ]]; then
            eval "$BUNDLE_LINE"
            if [[ -d "$GPU_MONITOR_BUNDLE_DIR" ]]; then
                echo "--- 上传 GPU 监控产物 ---"
                for f in "$GPU_MONITOR_BUNDLE_DIR"/*; do
                    [[ -e "$f" ]] && buildkite-agent artifact upload "$f"
                done
            fi
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
