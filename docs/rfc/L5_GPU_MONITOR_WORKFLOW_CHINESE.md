# \[RFC\]: L5 GPU 显存监控流程 & 邮件报告

## 1 概览

我们希望在 L5（长稳）用例运行时，对 GPU 显存进行采样监控，在 CI 环境中生成可归档的 HTML 报告，并可选地通过邮件分发给维护者。该 RFC 描述整体流程及其与现有 L4 nightly 性能流水线的关系。

### 1.1 动机

- L5 长稳用例通常运行数小时甚至 24h/72h。CI 任务结束后环境会被清理，临时网页、容器内文件和日志都不再可用。
- 维护者需要事后查看 GPU 显存随时间的趋势（是否接近 100%、是否有异常波动），以及每块卡的汇总统计（Min/Max/Avg、P50、P95），以辅助排查 OOM、负载不均等问题。
- 希望“一条命令”既能满足本地调试（实时仪表盘 + 最终报告），也能满足 CI 场景（产出报告 artifact + 可选邮件），与已有 L4 nightly 邮件机制风格统一。

### 1.2 目标

**功能目标**

- 在 L5 用例运行时，以固定间隔（默认 5s）通过 `nvidia-smi` 采集各 GPU 显存使用情况，持久化为 CSV，并在有 `jq` 时生成 `latest.json` 以支持实时仪表盘。
- 提供一个统一入口脚本，负责：
  - 启动显存监控脚本；
  - 执行用户指定的测试命令（如 pytest L5 用例）；
  - 测试结束后自动打包本次 run 的数据（CSV + HTML 报告）；
  - 在 CI 中上传这些文件为 Buildkite artifacts。
- HTML 报告 `report.html` 包含：
  - 统计表（每块 GPU 的 Min/Max/Avg、P50、P95、采样数）；
  - 显存利用率随时间变化的折线图（支持「所有 GPU 合并」与「按 GPU 拆分多图」两种视图）；
  - 简单异常列表（如 ≥95% 视为 high，≤5% 且>0 视为 low）。
- 邮件分发：
  - 邮件正文内嵌 Statistics 表格（方便快速预览）；
  - 邮件附件包含 `report.html`（完整图表）和可选的 `gpu_metrics.csv`；
  - 复用 L4 nightly 性能邮件的 SMTP 和收件人配置（`SMTP_*`、`DAILY_EMAIL_LIST` 等）。
- CI 集成：
  - 新增 L5 专用 Buildkite 流水线配置（如 `.buildkite/test-L5.yml`）：
    - Step1：在 Docker 中执行 L5 测试 + GPU 监控，并上传报告相关 artifacts；
    - Step2：下载这些 artifacts，运行 `send_gpu_monitor_email.py`，发送邮件。
  - 在主 `pipeline.yml` 中通过环境变量 `L5=1` 控制是否上传/执行 L5 流水线。

**不在范围内**

- 实时告警、阈值触发阻断 CI。
- 长期历史趋势存储（如 S3）和多 run 聚合分析（目前仅依赖 Buildkite artifacts 的保留期）。

**准确性**

- 完整性：所有通过 wrapper 运行的 L5 用例都会产出一份 bundle（CSV + HTML 报告）；在 CI 中均会上传为 artifacts。邮件发送仅在配置了 SMTP 的环境中启用。
- 一致性：统计表字段（Min/Max/Avg、P50、P95 等）、异常阈值、折线图配置在版本间保持稳定。
- 可追踪性：报告和邮件正文中包含 run_id、commit SHA、Buildkite build URL（在 CI 环境下）。

**性能**

- 采样开销：每个周期仅一次 `nvidia-smi` 查询，外加简单解析和追加写 CSV，目标对业务负载影响 <1%。
- 报告生成：在 run 结束后单次执行，目标耗时 <1 分钟。
- 邮件发送：与 L4 nightly 相同的超时与重试策略，满足日常通知场景。

---

## 2 设计

### 2.1 总览

整体分为三阶段：**采集（Sample）→ 打包（Bundle）→ 分发（Distribute）**。

- **采集**：`moniter.sh` 后台运行，周期性调用 `nvidia-smi` 采集各 GPU 显存并写入 CSV；若可用 `jq`，则生成 `latest.json` 和 `history.jsonl`，供实时仪表盘使用。
- **打包**：测试结束时（无论成功或失败），wrapper 调用 `finalize_monitor.sh`：
  - 将本次 run 的 CSV 拷贝到专用 bundle 目录；
  - 调用 `generate_report.py` 从 CSV 生成 `report.html`（统计表 + 折线图 + 异常表）；
  - 写入一个 `README.txt` 说明文件；
  - 打印 `GPU_MONITOR_BUNDLE_DIR=...` 供 CI 解析；在 CI 中，这个 bundle 下的文件会通过 Buildkite artifact 上传。
- **分发**：在 CI 第二步中：
  - 从 Buildkite artifacts 下载 `report.html`、`gpu_metrics.csv`、`README.txt` 到本地 bundle 目录；
  - 设置 `GPU_MONITOR_BUNDLE_DIR` 并调用 `send_gpu_monitor_email.py`：
    - 解析 CSV，计算统计数据；
    - 拼装 HTML 正文（Statistics 表格）；
    - 附加 `report.html` 和 `gpu_metrics.csv` 作为附件；
    - 使用与 L4 nightly 一致的 SMTP 配置发送邮件。

本地场景下只执行「采集 + 打包」，不强制启用邮件。

### 2.2 关键组件

- `tests/e2e/online_serving/moniter.sh`  
  - 功能：按间隔调用 `nvidia-smi`，输出 CSV、history.jsonl 和 latest.json（若有 jq）。
  - 参数：GPU 列表（all 或逗号分隔）、间隔秒数。
- `tests/e2e/online_serving/run_with_gpu_monitor.sh`  
  - 功能：统一入口：
    - 在后台启动 `moniter.sh`；
    - 执行用户传入的命令（如 pytest L5 用例）；
    - 在 EXIT trap 中调用 `finalize_monitor.sh` 打包当前 run；
    - 在 CI 中上传 bundle 中的文件为 Buildkite artifacts；
    - 本地可选设置 `GPU_MONITOR_SERVE_DASHBOARD=1` 同时启动仪表盘。
- `tests/e2e/online_serving/finalize_monitor.sh`  
  - 功能：基于 `current_run_id` 或指定 run_id：
    - 检查 CSV 是否存在；
    - 创建 `gpu_monitor_bundle_<run_id>` 目录；
    - 拷贝 CSV 和 history.jsonl；
    - 调用 `generate_report.py` 生成 `report.html`；
    - 写入 `README.txt`（说明内容与字段定义）；
    - 打印 `GPU_MONITOR_BUNDLE_DIR=<绝对路径>`。
- `tests/e2e/online_serving/generate_report.py`  
  - 功能：从 CSV 生成单文件 HTML 报告：
    - 计算每 GPU 的 Min/Max/Avg/P50/P95、采样数；
    - 构建时间序列数据，并使用 Chart.js 画折线图；
    - 支持「合并所有 GPU」和「按 GPU 拆分多张图」两种模式；
    - 生成异常点表（高/低阈值）；
    - X 轴使用真实时间戳（例如 `02-27 11:38:07`）。
- `tests/e2e/online_serving/serve_dashboard.sh` + `gpu_dashboard.html`  
  - 功能：本地实时仪表盘，提供 `/api/latest` 接口并渲染最近一段显存利用率曲线。
  - CI 环境不依赖此组件。
- `tools/L5/send_gpu_monitor_email.py`  
  - 功能：读取 `GPU_MONITOR_BUNDLE_DIR` 下的 `gpu_metrics.csv` 和 `report.html`：
    - 复用 `load_csv` + `compute_stats` 计算统计数据；
    - 生成 HTML 正文（包含 Statistics 表格）；
    - 附加 `report.html` 和 `gpu_metrics.csv`；
    - 通过 `SMTP_HOST/PORT/USERNAME/PASSWORD/DAILY_EMAIL_LIST` 等环境变量发送邮件；
    - 支持 `--dry-run` 模式本地调试（仅打印收件人/主题/正文预览，不真正发信）。
- `.buildkite/test-L5.yml`  
  - Step1（`l5-gpu-test`）：在 Docker 中执行 `run_with_gpu_monitor.sh -- pytest ...`，由 wrapper 自动上传 bundle 中的文件为 artifacts。
  - Step2（`l5-gpu-email`）：下载 Step1 上传的 `report.html`/`gpu_metrics.csv`/`README.txt`，设置 `GPU_MONITOR_BUNDLE_DIR`，执行 `send_gpu_monitor_email.py` 发送邮件。
- `.buildkite/pipeline.yml`  
  - 增加一个「Upload L5 Pipeline」步骤：在 `L5=1` 时执行 `buildkite-agent pipeline upload .buildkite/test-L5.yml`，从而挂载 L5 的两步到当前 Build。

---

## 3 实施计划

### P0（当前实现/本 RFC 覆盖）

- 完成 `moniter.sh` / `run_with_gpu_monitor.sh` / `finalize_monitor.sh` / `generate_report.py` / `serve_dashboard.sh` / `gpu_dashboard.html` 的实现与本地验证；
- 完成本地 `send_gpu_monitor_email.py --dry-run` 验证，确保：
  - 收件人列表解析正确；
  - 主题包含 L5 + run_id；
  - 正文中出现 Statistics 表格；
  - 附件列表中有 `report.html` 和 `gpu_metrics.csv`；
- 配置 `.buildkite/test-L5.yml` 与主 `pipeline.yml` 的 L5=1 触发逻辑；
- 在 Buildkite CI 上以 `--dry-run` 方式先验证 L5 流水线逻辑与 artifact 传递。

### P1（后续 / 依赖外部条件）

- 在 Buildkite Secrets 或环境变量中配置 SMTP_* 和 DAILY_EMAIL_LIST，使 CI 上可真正发出邮件；
- 根据需要扩展邮件模版，例如增加简短结论（显存是否接近上限、是否有 long tail 高负载段）；
- 若需要长期趋势分析，再考虑将 CSV/HTML 同步到 S3，并在 RFC 中补充历史聚合设计。

---

## 4 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 采样间隔大于预期（>5s） | 折线图上点间时间 ≈9–10s，而非理想 5s | 目前采用真实时间戳，`nvidia-smi + IO` 的耗时会计入周期，总体符合事实；如有需要，可以在报告中增加「逻辑步数」X 轴模式以便视觉对比。 |
| CI 环境缺少 `jq` | 无 `latest.json`，无法在 CI 上使用实时仪表盘 | CI 仅依赖 `report.html` 和 CSV；`moniter.sh` 中的依赖检查可通过环境变量跳过，不影响报告生成。 |
| SMTP 配置缺失或错误 | L5 邮件无法发出，但 L5 流水线本身仍可运行 | `send_gpu_monitor_email.py` 支持 `--dry-run` 模式用于预验证；CI 上邮件失败时 artifacts 仍可从 Buildkite 手工下载。 |
| artifact 路径/step key 配错 | 邮件 step 无法找到报告文件 | 在 `test-L5.yml` 中固定使用 `l5-gpu-test` 作为 key，并明确上传/下载路径；同时在 `GPU_MONITOR_CI.md` 中记录约定以便维护。 |

---

## 5 参考

- [RFC #1220](https://github.com/vllm-project/vllm-omni/issues/1220)：L4 tests workflow & daily email notifications（作为结构和动机参考）。
- `tests/e2e/online_serving/GPU_MONITOR_CI.md`：本地与 CI 使用文档。
