# [RFC]: L5 GPU memory monitor workflow

## 1 Overview

We run L5 (long-run stability) tests with GPU memory monitoring and generate a static report (statistics + time-series chart). This RFC covers sampling, bundling, and uploading the report as a Buildkite artifact. **Email distribution is out of scope here and will be addressed in a follow-up RFC.**

### 1.1 Motivation

- L5 runs are long (e.g. 24h/72h) and run in CI or on shared machines. After the run, the environment is torn down and any live dashboard or logs are lost.
- Maintainers need to see GPU memory utilization over time (trends, peaks, anomalies) and a compact summary (min/max/avg, P50/P95) without re-running or accessing the original environment.
- A single command should support both local debugging (real-time dashboard + final report) and CI (run test with monitor, then upload report as artifact).

### 1.2 Target

**In scope**

- GPU memory sampling at a configurable interval (default 5s) via `nvidia-smi`, persisted to CSV and optional `latest.json` for a live dashboard.
- One-command wrapper for CI/local: start monitor → run test → finalize bundle (CSV + `report.html`) → upload Buildkite artifacts when in CI; optional live dashboard when `GPU_MONITOR_SERVE_DASHBOARD=1`.
- Static report (`report.html`): statistics table, memory utilization line chart (with optional “split per GPU” view), and simple anomaly table (high/low utilization thresholds).
- CI pipeline: when `L5=1`, run one L5 step that executes the test with the wrapper and uploads bundle contents (report.html, gpu_metrics.csv, README.txt) as artifacts. View the report by downloading the artifact and opening report.html locally.

**Out of scope (this RFC)**

- Email distribution (Statistics in body + report attachment). → Follow-up RFC.
- Real-time alerting or threshold-based blocking.
- Long-term storage (e.g. S3) or historical trend aggregation; retention is Buildkite artifact lifetime only unless extended later.

**Accuracy**

- Completeness: Every run that executes the wrapper produces a bundle (CSV + report) and, in CI, uploads artifacts.
- Consistency: Report format (statistics columns, chart axes, anomaly thresholds) is stable; X-axis uses real timestamps (actual sampling times).
- Traceability: Report includes run_id; in Buildkite, artifacts are tied to the build.

**Performance**

- Sampling overhead: Single `nvidia-smi` per cycle; target <1% impact on the workload.
- Report generation: One-off after the run; typically <1 min.

---

## 2 Design

### 2.1 Overview

Two phases: **Sample → Bundle** (and in CI, the wrapper uploads the bundle as artifacts).

- **Sample**: `moniter.sh` runs in the background (started by `run_with_gpu_monitor.sh`), writing CSV and, if `jq` is available, `latest.json` for the live dashboard.
- **Bundle**: On test exit, `finalize_monitor.sh` copies CSV into a bundle dir and runs `generate_report.py` to produce `report.html` (statistics, line chart, anomalies). In CI, the wrapper uploads bundle contents as Buildkite artifacts. Users download the artifact and open report.html locally.

```
Test run (local or CI)
       │
       ▼
run_with_gpu_monitor.sh ──► moniter.sh (CSV + latest.json)
       │                           │
       │                           └──► serve_dashboard.sh (optional, local only)
       │
       ▼ (on exit)
finalize_monitor.sh ──► generate_report.py ──► report.html + bundle
       │
       └── CI: buildkite-agent artifact upload (report.html, gpu_metrics.csv, README.txt)
```

### 2.2 Components

| Component | Role |
|-----------|------|
| `tests/e2e/online_serving/moniter.sh` | Background sampler: `nvidia-smi` at interval (default 5s), write CSV + history.jsonl + latest.json. |
| `tests/e2e/online_serving/run_with_gpu_monitor.sh` | Wrapper: start moniter (and optionally serve_dashboard), run user command, on exit run finalize and upload artifacts if buildkite-agent present. |
| `tests/e2e/online_serving/finalize_monitor.sh` | Copy current run CSV into bundle dir, call generate_report.py, write README.txt; print GPU_MONITOR_BUNDLE_DIR. |
| `tests/e2e/online_serving/generate_report.py` | Read CSV → statistics, time-series per GPU, anomalies → single HTML (Chart.js) with table, line chart, anomaly table. |
| `tests/e2e/online_serving/serve_dashboard.sh` | HTTP server for gpu_dashboard.html + /api/latest (for live view; local only). |
| `.buildkite/test-L5.yml` | One step (or more): run test with run_with_gpu_monitor.sh, which uploads report.html, gpu_metrics.csv, README.txt as artifacts. |
| `.buildkite/pipeline.yml` | When `L5=1`, upload test-L5.yml so the L5 step(s) run. |

### 2.3 Use Cases

**Primary: CI L5 run**

- Nightly or on-demand L5 run with `L5=1`. The L5 step runs the test with GPU monitoring and uploads the bundle as artifacts. Maintainers download the artifact and open report.html in a browser for the line chart and statistics.

**Secondary: Local long-run + report only**

- On a Linux host with NVIDIA GPUs, run `run_with_gpu_monitor.sh -- pytest ...`. Bundle is written under `tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`. Open report.html locally.

**Optional: Local real-time dashboard**

- Set `GPU_MONITOR_SERVE_DASHBOARD=1` when calling the wrapper, or run `serve_dashboard.sh` in a second terminal. Browser opens gpu_dashboard.html; data is read from latest.json. Not used in CI.

---

## 3 Implementation Plan

### P0 (Done / In tree)

| Item | Action | Notes |
|------|--------|------|
| Monitor script | `moniter.sh`: nvidia-smi loop, CSV + optional latest.json | SKIP_DEPS_CHECK for CI without jq |
| Wrapper | `run_with_gpu_monitor.sh`: start monitor, run command, finalize, upload artifacts | Same command for local and CI |
| Finalize | `finalize_monitor.sh`: bundle CSV, call generate_report.py, README | |
| Report generator | `generate_report.py`: stats, line chart (real timestamps), anomalies, “split per GPU” toggle | X-axis: e.g. MM-DD HH:MM:SS |
| Dashboard (local) | `serve_dashboard.sh` + gpu_dashboard.html | Optional; not used in CI |
| CI pipeline | `.buildkite/test-L5.yml` (run test + upload artifacts); `pipeline.yml` uploads it when `L5=1` | Artifacts: report.html, gpu_metrics.csv, README.txt |
| Docs | `tests/e2e/online_serving/GPU_MONITOR_CI.md` | Local vs CI, Buildkite artifact download |

### P1 (Later / other RFCs)

| Item | Notes |
|------|--------|
| Email distribution | Follow-up RFC: send report (e.g. Statistics in body + report.html attachment) via SMTP to DAILY_EMAIL_LIST. |
| S3 / long-term retention | If needed for trend across runs; out of scope for this RFC. |

---

## 4 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sampling interval longer than 5s | Points in the chart are 9–10s apart under load | Expected: nvidia-smi + IO add latency; chart uses real timestamps. Optional: add “logical step” X-axis mode in report. |
| No jq in CI image | No latest.json, no live dashboard in CI | By design; CI uses artifact report.html only. SKIP_DEPS_CHECK avoids hard failure. |

---

## 5 References

- [RFC #1220](https://github.com/vllm-project/vllm-omni/issues/1220): L4 tests workflow building & daily email notifications (design template and motivation).
- `tests/e2e/online_serving/GPU_MONITOR_CI.md`: Usage for CI and local runs.
