# L5 GPU 监控在 CI 中的使用

长稳结束后 CI 环境会被清理，网页和 CSV 都会丢失。通过**在清理前打包并上传为 CI 产物**，流水线结束后仍可下载查看。

## CI 清理后如何查看？（实时仪表盘 vs 静态报告）

- **实时仪表盘**（`gpu_dashboard.html` + `serve_dashboard.sh`）依赖本机 HTTP 服务和 `latest.json`，适合**本地长稳时**边跑边看。CI 跑完环境被回收，**无法在 CI 里提供可访问的网页**。
- **解决方式**：不在 CI 里依赖实时网页，改用 **静态报告 `report.html`**。`finalize_monitor.sh` 会从 CSV 调用 `generate_report.py` 生成单文件 HTML（图表、统计、异常表均内嵌），**无需任何服务器**。把打包目录上传为 CI artifact，流水线结束后**下载 artifact，在本地用浏览器打开其中的 `report.html`** 即可查看完整显存曲线与统计，不依赖当时的环境与网址。

因此：CI 中只做「监控 → 收尾打包 → 上传 artifact」；查看时从流水线下载 artifact，本地打开 `report.html` 即可。

## 本地验证（不上 CI）

在 Linux 本机（有 NVIDIA 显卡和 `nvidia-smi`）可以完整走一遍流程，无需等 CI。

### 方式一：一条龙（推荐，和 CI 行为一致）

在仓库根目录执行（与 CI 里 Diffusion Model Test 相同命令）：

```bash
bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/offline_inference/test_t2i_model.py -m "core_model and diffusion" --run-level "core_model"
```

- 会后台启动 `moniter.sh`、跑 pytest，结束时自动执行 `finalize_monitor.sh`（本地没有 `buildkite-agent` 会跳过上传）。
- 打包目录在 `tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`，里面有 `gpu_metrics.csv`、`report.html`、`README.txt`。用浏览器打开 **report.html** 即可看折线图与统计。

想少跑一会儿可以用更快的用例，例如：

```bash
bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -v tests/engine/test_async_omni_engine_abort.py -x
```

### 方式二：边跑边看网页仪表盘

1. **终端 1**：启动监控  
   `cd tests/e2e/online_serving && ./moniter.sh all 5`（不要后台，方便结束时 Ctrl+C）

2. **终端 2**：启动仪表盘服务  
   `cd tests/e2e/online_serving && ./serve_dashboard.sh`  
   浏览器打开 **http://127.0.0.1:8765/gpu_dashboard.html**，即可实时看折线图。

3. **终端 3**（仓库根目录）：跑测试  
   `pytest -s -v tests/e2e/offline_inference/test_t2i_model.py -m "core_model and diffusion" --run-level "core_model"`（或任意短测试）

4. 测试结束后，在终端 1 按 **Ctrl+C** 停掉监控，然后执行收尾并打开报告：  
   `cd tests/e2e/online_serving && ./finalize_monitor.sh`  
   用脚本输出的 `GPU_MONITOR_BUNDLE_DIR` 里的 **report.html** 在浏览器打开，可核对与仪表盘一致的折线图。

## 流程概览

1. **启动监控**：后台运行 `./moniter.sh`，整个长稳期间持续写 CSV。
2. **跑长稳**：执行你的长稳用例（如 `test_qwen_edit.sh`）。
3. **收尾**：长稳结束后、环境清理前，执行 `./finalize_monitor.sh`，生成报告并打包。
4. **归档**：把 `finalize_monitor.sh` 输出的目录上传为 CI artifact。

之后在流水线页面下载该 artifact，即可得到 `gpu_metrics.csv` 和 `report.html`（含图表与异常标记），无需再访问当时的环境。

## 步骤说明

### 1. 启动监控（后台）

```bash
cd tests/e2e/online_serving
./moniter.sh all 5 &
MONITOR_PID=$!
```

可选：把 `GPU_MONITOR_DATA_ROOT` 设到固定目录，便于与后续步骤一致。

### 2. 运行长稳测试

按你现有方式跑 24h/72h 长稳即可。

### 3. 收尾（必须放在「清理前」执行）

在 **finally** 或 **after script** 里执行（确保长稳失败也会跑）：

```bash
cd tests/e2e/online_serving
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
    cd tests/e2e/online_serving
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

**GitLab CI** 示例（在 `job` 里）：

```yaml
after_script:
  - cd tests/e2e/online_serving
  - eval $(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
artifacts:
  when: always
  paths:
    - tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_*/
```

若路径不固定，可用脚本把 `GPU_MONITOR_BUNDLE_DIR` 拷到固定路径再在 `artifacts.paths` 里写该路径。

**Jenkins**：在 finally 里执行 `finalize_monitor.sh`，用 `archiveArtifacts` 归档 `gpu_monitor_data/gpu_monitor_bundle_*/**`。

### Buildkite（推荐用 run_with_gpu_monitor.sh 一条龙）

用包装脚本一次完成「启动监控 + 跑测试 + 收尾 + 上传 artifact」：

```yaml
commands:
  - bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/offline_inference/test_t2i_model.py ...
```

- **用到的脚本**：`run_with_gpu_monitor.sh`（内部会调 `moniter.sh`、`finalize_monitor.sh`、`generate_report.py`）。
- **运行中「实时」看什么**：CI 没有单独的可访问网页。请打开 **Buildkite 该次构建的 Job 页面**，在**日志区域**里会每隔约 15 秒出现一行 `[GPU] 时间戳,...,gpu_index,used_mb,total_mb,util_pct`，即当前最新一次采样的显存数据，相当于在日志里实时看仪表盘数据。
- **结束后在哪里下载 GPU 数据**：同一 Job 页面上方或侧边有 **Artifacts**，点进去会看到本步骤上传的文件（如 `gpu_metrics.csv`、`report.html`、`README.txt`）。下载后本地用浏览器打开 `report.html` 即可看到与仪表盘类似的折线图及统计表。

## 产物内容

- `gpu_metrics.csv`：原始采样（时间戳、GPU 索引、显存占用、利用率）。
- `report.html`：单文件报告，含统计表、时序图、简单异常标记，浏览器打开即可。
- `README.txt`：简要说明。

长稳结束后在流水线里下载该 artifact，本地打开 `report.html` 即可查看，不依赖当时的环境与网址。
