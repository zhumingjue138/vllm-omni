# L5 GPU 监控在 CI 中的使用

长稳结束后 CI 环境会被清理，网页和 CSV 都会丢失。通过**在清理前打包并上传为 CI 产物**，流水线结束后仍可下载查看。

**脚本与数据目录**：`tests/e2e/online_serving/L5/`（moniter.sh、run_with_gpu_monitor.sh、finalize_monitor.sh、generate_report.py、serve_dashboard.sh、gpu_dashboard.html 等）。

## CI 清理后如何查看？（实时仪表盘 vs 静态报告）

- **实时仪表盘**（`gpu_dashboard.html` + `serve_dashboard.sh`）依赖本机 HTTP 服务和 `latest.json`，适合**本地长稳时**边跑边看。CI 跑完环境被回收，**无法在 CI 里提供可访问的网页**。
- **解决方式**：不在 CI 里依赖实时网页，改用 **静态报告 `report.html`**。`finalize_monitor.sh` 会从 CSV 调用 `generate_report.py` 生成单文件 HTML（图表、统计、异常表均内嵌），**无需任何服务器**。把打包目录上传为 CI artifact，流水线结束后**下载 artifact，在本地用浏览器打开其中的 `report.html`** 即可查看完整显存曲线与统计，不依赖当时的环境与网址。

因此：CI 中只做「监控 → 收尾打包 → 上传 artifact」；查看时从流水线下载 artifact，本地打开 `report.html` 即可。

## 本地与 CI 统一：一步完成

同一条命令在**本地**和 **CI** 都能用，无需分步。

- **CI**：不设环境变量，直接执行。会启动监控、跑测试、收尾打包并上传 artifact；实时看日志里的 `[GPU]` 行，结束后在 Artifacts 下载 `report.html`。
- **本地**：同上；若想**边跑边看网页仪表盘**，在命令前加 `GPU_MONITOR_SERVE_DASHBOARD=1`，脚本会同时拉起仪表盘服务，浏览器访问输出的 URL 即可（远程机器需 SSH 端口转发）。

示例（仓库根目录）：

```bash
# CI 或本地仅要日志 + 结束后 report.html
bash tests/e2e/online_serving/L5/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v

# 本地想边跑边看仪表盘（一步，无需另开终端）
GPU_MONITOR_SERVE_DASHBOARD=1 bash tests/e2e/online_serving/L5/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v
```

- 仪表盘 URL 会打印在终端，默认 `http://127.0.0.1:8765/gpu_dashboard.html`。远程机器上在本机执行 `ssh -L 8765:127.0.0.1:8765 用户@主机` 后访问该 URL。
- 打包目录：`tests/e2e/online_serving/L5/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`，内含 `gpu_metrics.csv`、`report.html`、`README.txt`。折线图在 `report.html` 中；日志结束时会打印路径（如 `Line chart (memory utilization): open in browser: .../report.html`）。

### 可选：用 conftest 集成（不经过 wrapper 脚本）

GPU 监控的 pytest 逻辑在 `tests/e2e/online_serving/L5/conftest.py`，由上层 `online_serving/conftest.py` 按路径加载。设置 `GPU_MONITOR=1` 后直接跑 pytest，会在 session 开始时启动 L5 下的 `moniter.sh`，session 结束时自动 finalize 并（在 CI 中）上传 artifact。环境变量 `GPU_MONITOR_INTERVAL`、`GPU_MONITOR_DEVICES`、`GPU_MONITOR_LOG_INTERVAL` 等同样生效；仪表盘需单独开 `L5/serve_dashboard.sh` 或继续用 wrapper 并设 `GPU_MONITOR_SERVE_DASHBOARD=1`。

```bash
GPU_MONITOR=1 GPU_MONITOR_INTERVAL=60 pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_sleep_001
```

## 流程概览

1. **启动监控**：后台运行 `./moniter.sh`（在 L5 目录下），整个长稳期间持续写 CSV。
2. **跑长稳**：执行你的长稳用例（如 `test_qwen_edit.sh`）。
3. **收尾**：长稳结束后、环境清理前，执行 `./finalize_monitor.sh`（在 L5 目录下），生成报告并打包。
4. **归档**：把 `finalize_monitor.sh` 输出的目录上传为 CI artifact。

之后在流水线页面下载该 artifact，即可得到 `gpu_metrics.csv` 和 `report.html`（含图表与异常标记），无需再访问当时的环境。

## 步骤说明

### 1. 启动监控（后台）

```bash
cd tests/e2e/online_serving/L5
./moniter.sh all 5 &
MONITOR_PID=$!
```

可选：把 `GPU_MONITOR_DATA_ROOT` 设到固定目录，便于与后续步骤一致。

### 2. 运行长稳测试

按你现有方式跑 24h/72h 长稳即可。

### 3. 收尾（必须放在「清理前」执行）

在 **finally** 或 **after script** 里执行（确保长稳失败也会跑）：

```bash
cd tests/e2e/online_serving/L5
# 可选：结束监控进程，避免重复写（不杀也会因 job 结束而停）
# kill $MONITOR_PID 2>/dev/null || true
# 打包当前 run 的 CSV + 报告，并输出要归档的目录
BUNDLE_LINE=$(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "归档目录: $GPU_MONITOR_BUNDLE_DIR"
```

### 4. 上传为 CI 产物

**GitHub Actions** 示例：

```yaml
- name: Finalize GPU monitor and upload
  if: always()
  run: |
    cd tests/e2e/online_serving/L5
    BUNDLE_LINE=$(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
    if [[ -n "$BUNDLE_LINE" ]]; then
      eval "$BUNDLE_LINE"
      echo "GPU_MONITOR_BUNDLE_DIR=$GPU_MONITOR_BUNDLE_DIR" >> $GITHUB_ENV
    fi
- name: Upload GPU monitor bundle
  if: env.GPU_MONITOR_BUNDLE_DIR != ''
  uses: actions/upload-artifact@v4
  with:
    name: gpu-monitor-${{ github.run_id }}
    path: ${{ env.GPU_MONITOR_BUNDLE_DIR }}
```

**Buildkite（推荐用 run_with_gpu_monitor.sh 一条龙）**

用包装脚本一次完成「启动监控 + 跑测试 + 收尾 + 上传 artifact」：

```yaml
commands:
  - bash tests/e2e/online_serving/L5/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/online_serving/test_foo.py ...
```

- **用到的脚本**：`tests/e2e/online_serving/L5/run_with_gpu_monitor.sh`（内部会调 L5 下的 `moniter.sh`、`finalize_monitor.sh`、`generate_report.py`）。
- **运行中「实时」看什么**：CI 没有单独的可访问网页。请打开 **Buildkite 该次构建的 Job 页面**，在**日志区域**里会每隔约 15 秒出现一行 `[GPU] ...`，即当前最新一次采样的显存数据。
- **结束后在哪里下载**：同一 Job 页面的 **Artifacts** 中可下载 `gpu_metrics.csv`、`report.html`、`README.txt`。本地用浏览器打开 `report.html` 即可。

## 产物内容

- `gpu_metrics.csv`：原始采样（时间戳、GPU 索引、显存占用、利用率）。
- `report.html`：单文件报告，含统计表、时序图、简单异常标记，浏览器打开即可。
- `README.txt`：简要说明。

长稳结束后在流水线里下载该 artifact，本地打开 `report.html` 即可查看，不依赖当时的环境与网址。
