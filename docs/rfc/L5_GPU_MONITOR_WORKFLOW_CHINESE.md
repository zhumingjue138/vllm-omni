# [RFC]: L5 GPU 显存监控流程

## 1 概览

在 L5（长稳）用例运行时对 GPU 显存进行采样监控，并生成可归档的 HTML 报告。本 RFC 只覆盖「采集 → 打包 → 在 CI 中上传为 Buildkite artifact」；**邮件分发不在本 RFC 范围内，将放在后续 RFC 中**。

### 1.1 动机

- L5 长稳用例通常运行数小时甚至 24h/72h。CI 任务结束后环境会被清理，临时网页和日志不再可用。
- 维护者需要事后查看 GPU 显存随时间的趋势以及每块卡的汇总统计（Min/Max/Avg、P50、P95），以辅助排查 OOM、负载不均等问题。
- 希望一条命令同时支持本地调试（实时仪表盘 + 最终报告）和 CI（跑测试 + 上传报告为 artifact）。

### 1.2 目标

**功能目标**

- 在 L5 用例运行时，以固定间隔（默认 5s）通过 `nvidia-smi` 采集各 GPU 显存，持久化为 CSV，并在有 `jq` 时生成 `latest.json` 以支持实时仪表盘。
- 统一入口脚本：启动监控 → 执行用户测试命令 → 测试结束后打包（CSV + HTML 报告）→ 在 CI 中上传为 Buildkite artifacts；本地可选 `GPU_MONITOR_SERVE_DASHBOARD=1` 启动仪表盘。
- HTML 报告 `report.html`：统计表、显存利用率折线图（支持「合并所有 GPU」与「按 GPU 拆分」）、简单异常表（高/低阈值）。
- CI：当 `L5=1` 时执行 L5 步骤，跑带监控的测试并上传 bundle（report.html、gpu_metrics.csv、README.txt）为 artifact；维护者从流水线下载 artifact 后本地打开 report.html 查看。

**本 RFC 不包含**

- 邮件分发（正文 Statistics 表 + 附件 report.html）。→ 后续 RFC。
- 实时告警、阈值阻断 CI。
- 长期存储（如 S3）或多 run 聚合；目前仅依赖 Buildkite artifact 保留期。

**准确性**

- 完整性：通过 wrapper 运行的 L5 用例都会产出一份 bundle，在 CI 中上传为 artifacts。
- 一致性：报告格式（统计列、折线图、异常阈值）稳定；X 轴使用真实时间戳。
- 可追踪性：报告含 run_id；Buildkite 中 artifact 与 build 关联。

**性能**

- 采样：每周期一次 `nvidia-smi`，目标对业务影响 <1%。
- 报告生成：run 结束后单次执行，目标 <1 分钟。

---

## 2 设计

### 2.1 总览

两阶段：**采集（Sample）→ 打包（Bundle）**；在 CI 中由 wrapper 将 bundle 上传为 artifacts。

- **采集**：`moniter.sh` 后台运行，写 CSV；若有 `jq` 则写 `latest.json`/history.jsonl 供实时仪表盘使用。
- **打包**：测试结束时 `finalize_monitor.sh` 将当前 run 的 CSV 拷入 bundle 目录，调用 `generate_report.py` 生成 `report.html`，写 `README.txt`。在 CI 中 wrapper 将 bundle 内文件上传为 Buildkite artifacts，用户下载 artifact 后本地打开 report.html。

```
测试运行（本地或 CI）
       │
       ▼
run_with_gpu_monitor.sh ──► moniter.sh (CSV + latest.json)
       │                           │
       │                           └──► serve_dashboard.sh（可选，仅本地）
       │
       ▼ (退出时)
finalize_monitor.sh ──► generate_report.py ──► report.html + bundle
       │
       └── CI: buildkite-agent artifact upload (report.html, gpu_metrics.csv, README.txt)
```

### 2.2 关键组件

- `tests/e2e/online_serving/moniter.sh`：按间隔调用 `nvidia-smi`，输出 CSV、history.jsonl、latest.json（若有 jq）。
- `tests/e2e/online_serving/run_with_gpu_monitor.sh`：统一入口；后台启动 moniter、执行用户命令、退出时 finalize；在 CI 中上传 bundle 为 artifacts；本地可选启动仪表盘。
- `tests/e2e/online_serving/finalize_monitor.sh`：根据 current_run_id 拷贝 CSV、调用 generate_report.py、写 README.txt、打印 GPU_MONITOR_BUNDLE_DIR。
- `tests/e2e/online_serving/generate_report.py`：从 CSV 生成 report.html（统计表、折线图、异常表）；X 轴为真实时间戳（如 02-27 11:38:07）；支持「按 GPU 拆分」视图。
- `tests/e2e/online_serving/serve_dashboard.sh` + `gpu_dashboard.html`：本地实时仪表盘；CI 不依赖。
- `.buildkite/test-L5.yml`：L5 步骤：执行带监控的测试，由 wrapper 上传 report.html、gpu_metrics.csv、README.txt 为 artifacts。
- `.buildkite/pipeline.yml`：当 `L5=1` 时上传 test-L5.yml，从而执行 L5 步骤。

### 2.3 使用场景

**主要：CI L5 运行**

- 在 `L5=1` 时跑 L5 步骤；步骤内执行带 GPU 监控的测试并上传 bundle 为 artifacts。维护者从流水线下载 artifact，本地打开 report.html 查看折线图与统计。

**次要：本地长稳 + 仅报告**

- 在带 NVIDIA GPU 的 Linux 上执行 `run_with_gpu_monitor.sh -- pytest ...`。bundle 写在 `tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`，本地打开 report.html 即可。

**可选：本地实时仪表盘**

- 设置 `GPU_MONITOR_SERVE_DASHBOARD=1` 或单独运行 `serve_dashboard.sh`，浏览器打开 gpu_dashboard.html。CI 中不使用。

---

## 3 实施计划

### P0（本 RFC 覆盖）

- `moniter.sh`、`run_with_gpu_monitor.sh`、`finalize_monitor.sh`、`generate_report.py`、`serve_dashboard.sh`、`gpu_dashboard.html` 的实现与本地验证。
- `.buildkite/test-L5.yml` 与 `pipeline.yml` 中 `L5=1` 触发逻辑；CI 上验证「跑测试 + 上传 artifact」。
- 文档：`tests/e2e/online_serving/GPU_MONITOR_CI.md`（本地与 CI 用法、artifact 下载与查看）。

### P1（后续 / 其他 RFC）

- **邮件分发**：后续 RFC；正文内嵌 Statistics 表 + 附件 report.html，复用 L4 nightly 的 SMTP/收件人配置。
- S3 / 长期保留：若需要跨 run 趋势分析，再单独设计。

---

## 4 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 采样间隔大于 5s | 折线图点间隔约 9–10s | 使用真实时间戳，反映实际周期；可选在报告中增加「逻辑步数」X 轴。 |
| CI 无 jq | 无 latest.json，无实时仪表盘 | CI 仅依赖 report.html；SKIP_DEPS_CHECK 避免因缺 jq 失败。 |

---

## 5 参考

- [RFC #1220](https://github.com/vllm-project/vllm-omni/issues/1220)：L4 tests workflow & daily email notifications（结构与动机参考）。
- `tests/e2e/online_serving/GPU_MONITOR_CI.md`：本地与 CI 使用说明。
