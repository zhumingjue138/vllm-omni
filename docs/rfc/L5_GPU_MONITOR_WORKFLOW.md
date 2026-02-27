# [RFC]: L5 GPU memory monitor workflow & report email

## 1 Overview

We run L5 (long-run stability) tests with GPU memory monitoring, generate a static report with statistics and time-series charts, and optionally send a daily digest email to maintainers. This RFC describes the workflow and how it fits alongside the existing L4 nightly performance pipeline.

### 1.1 Motivation

- L5 runs are long (e.g. 24h/72h) and run in CI or on shared machines. After the run, the environment is torn down and any live dashboard or logs are lost.
- Maintainers need to see GPU memory utilization over time (trends, peaks, anomalies) and a compact summary (min/max/avg, P50/P95) without re-running or accessing the original environment.
- A single pipeline should support both local debugging (real-time dashboard + final report) and CI (artifact upload + optional email with table preview and report attachment).

### 1.2 Target

**In scope**

- GPU memory sampling at a configurable interval (default 5s) via `nvidia-smi`, persisted to CSV and optional `latest.json` for a live dashboard.
- One-command wrapper for CI/local: start monitor → run test → finalize bundle (CSV + `report.html`) → upload Buildkite artifacts; optional live dashboard when `GPU_MONITOR_SERVE_DASHBOARD=1`.
- Static report (`report.html`): statistics table, memory utilization line chart (with optional “split per GPU” view), and simple anomaly table (high/low utilization thresholds).
- Email distribution: HTML body with Statistics table preview + attachment of `report.html` (and optionally `gpu_metrics.csv`), reusing the same SMTP/recipient configuration as the L4 nightly email.
- CI pipeline: dedicated L5 steps (e.g. `test-L5.yml`) triggered when `L5=1`; step 1 runs test with monitor and uploads artifacts; step 2 downloads artifacts and sends email.

**Out of scope**

- Real-time alerting or threshold-based blocking.
- Long-term storage (e.g. S3) or historical trend aggregation across runs; retention is Buildkite artifact lifetime only unless extended later.

**Accuracy**

- Completeness: Every run that executes the wrapper produces a bundle (CSV + report) and, in CI, uploads artifacts; email step runs only when SMTP is configured.
- Consistency: Report format (statistics columns, chart axes, anomaly thresholds) is stable; X-axis uses real timestamps (actual sampling times).
- Traceability: Report and email include run_id, commit SHA, and build URL when run in Buildkite.

**Performance**

- Sampling overhead: Single `nvidia-smi` per cycle; target &lt;1% impact on the workload.
- Report generation: One-off after the run; typically &lt;1 min.
- Email: Same constraints as L4 (e.g. attachment size, SMTP retries).

---

## 2 Design

### 2.1 Overview

Three phases: **Sample → Bundle → Distribute**.

- **Sample**: `moniter.sh` runs in the background (started by `run_with_gpu_monitor.sh`), writing CSV and, if `jq` is available, `latest.json` for the live dashboard.
- **Bundle**: On test exit, `finalize_monitor.sh` copies CSV into a bundle dir and runs `generate_report.py` to produce `report.html` (statistics, line chart, anomalies). In CI, the wrapper uploads bundle contents as Buildkite artifacts.
- **Distribute**: In CI, a second step downloads those artifacts and runs `send_gpu_monitor_email.py`, which builds an HTML email body (Statistics table), attaches `report.html` (and optionally CSV), and sends via the same SMTP/recipient env vars as L4.

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
       ├── CI: buildkite-agent artifact upload (report.html, gpu_metrics.csv, README.txt)
       │
       ▼
send_gpu_monitor_email.py (CI step 2)
       │
       ├── Body: Statistics table (HTML)
       ├── Attachments: report.html, gpu_metrics.csv
       └── SMTP (same as L4 nightly)
```

### 2.2 Components

| Component | Role |
|-----------|------|
| `tests/e2e/online_serving/moniter.sh` | Background sampler: `nvidia-smi` at interval (default 5s), write CSV + history.jsonl + latest.json. |
| `tests/e2e/online_serving/run_with_gpu_monitor.sh` | Wrapper: start moniter (and optionally serve_dashboard), run user command, on exit run finalize and upload artifacts if buildkite-agent present. |
| `tests/e2e/online_serving/finalize_monitor.sh` | Copy current run CSV into bundle dir, call generate_report.py, write README.txt; print GPU_MONITOR_BUNDLE_DIR. |
| `tests/e2e/online_serving/generate_report.py` | Read CSV → statistics, time-series per GPU, anomalies → single HTML (Chart.js) with table, line chart, anomaly table. |
| `tests/e2e/online_serving/serve_dashboard.sh` | HTTP server for gpu_dashboard.html + /api/latest (for live view; local only). |
| `tools/L5/send_gpu_monitor_email.py` | Load CSV, compute stats, build HTML body (Statistics table), attach report.html (+ CSV), send via SMTP (env: SMTP_*, DAILY_EMAIL_LIST, etc.). |
| `.buildkite/test-L5.yml` | Two steps: (1) run test with run_with_gpu_monitor.sh, upload artifacts; (2) download artifacts, run send_gpu_monitor_email.py. |
| `.buildkite/pipeline.yml` | When `L5=1`, upload test-L5.yml so the L5 steps run. |

### 2.3 Use Cases

**Primary: CI L5 run + email**

- Nightly or on-demand L5 run with `L5=1`. Step 1 runs the test with GPU monitoring and uploads report.html + CSV. Step 2 sends an email with Statistics in the body and report.html as attachment. Recipients get the table at a glance and open the attachment for the full chart.

**Secondary: Local long-run + report only**

- On a Linux host with NVIDIA GPUs, run `run_with_gpu_monitor.sh -- pytest ...`. No SMTP; bundle is written under `tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`. Open report.html locally for the line chart and statistics.

**Optional: Local real-time dashboard**

- Set `GPU_MONITOR_SERVE_DASHBOARD=1` when calling the wrapper, or run `serve_dashboard.sh` in a second terminal. Browser opens gpu_dashboard.html; data is read from latest.json. Not used in CI (environment is ephemeral).

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
| Email script | `tools/L5/send_gpu_monitor_email.py`: HTML body (Statistics), attach report.html + CSV, reuse L4 SMTP env | Dry-run for local validation |
| CI pipeline | `.buildkite/test-L5.yml` (test step + email step); `pipeline.yml` uploads it when `L5=1` | Artifacts: report.html, gpu_metrics.csv, README.txt |
| Docs | `tests/e2e/online_serving/GPU_MONITOR_CI.md` | Local vs CI, Buildkite artifact download |

### P1 (Optional / Later)

| Item | Dependency | Notes |
|------|-------------|-------|
| SMTP in Buildkite | Secrets / env for SMTP_*, DAILY_EMAIL_LIST | Same as L4 nightly; email step runs only if set |
| L5 trigger | Env `L5=1` or schedule | Pipeline upload step already conditional on L5=1 |
| S3 / long-term retention | If needed for trend across runs | Out of scope for this RFC |

---

## 4 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sampling interval longer than 5s | Points in the chart are 9–10s apart under load | Expected: nvidia-smi + IO add latency; chart uses real timestamps. Optional: add “logical step” X-axis mode in report. |
| No jq in CI image | No latest.json, no live dashboard in CI | By design; CI uses artifact report.html only. SKIP_DEPS_CHECK avoids hard failure. |
| Email step fails (SMTP/network) | No email; artifacts still available in Buildkite | Retry logic in send script; document that report is always in artifacts. |
| Artifact download in step 2 | Wrong step key or path | test-L5.yml uses explicit step key and artifact names; document in GPU_MONITOR_CI.md. |

---

## 5 References

- [RFC #1220](https://github.com/vllm-project/vllm-omni/issues/1220): L4 tests workflow building & daily email notifications (design template and motivation).
- `tests/e2e/online_serving/GPU_MONITOR_CI.md`: Usage for CI and local runs.
