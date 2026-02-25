# L5 GPU 监控在 CI 中的使用

长稳结束后 CI 环境会被清理，网页和 CSV 都会丢失。通过**在清理前打包并上传为 CI 产物**，流水线结束后仍可下载查看。

## CI 清理后如何查看？（实时仪表盘 vs 静态报告）

- **实时仪表盘**（`gpu_dashboard.html` + `serve_dashboard.sh`）依赖本机 HTTP 服务和 `latest.json`，适合**本地长稳时**边跑边看。CI 跑完环境被回收，**无法在 CI 里提供可访问的网页**。
- **解决方式**：不在 CI 里依赖实时网页，改用 **静态报告 `report.html`**。`finalize_monitor.sh` 会从 CSV 调用 `generate_report.py` 生成单文件 HTML（图表、统计、异常表均内嵌），**无需任何服务器**。把打包目录上传为 CI artifact，流水线结束后**下载 artifact，在本地用浏览器打开其中的 `report.html`** 即可查看完整显存曲线与统计，不依赖当时的环境与网址。

因此：CI 中只做「监控 → 收尾打包 → 上传 artifact」；查看时从流水线下载 artifact，本地打开 `report.html` 即可。

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

## 产物内容

- `gpu_metrics.csv`：原始采样（时间戳、GPU 索引、显存占用、利用率）。
- `report.html`：单文件报告，含统计表、时序图、简单异常标记，浏览器打开即可。
- `README.txt`：简要说明。

长稳结束后在流水线里下载该 artifact，本地打开 `report.html` 即可查看，不依赖当时的环境与网址。
