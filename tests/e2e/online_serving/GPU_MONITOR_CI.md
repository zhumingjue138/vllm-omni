# L5 GPU monitor in CI

After a long run the CI environment is torn down and the web UI and CSV are lost. **Bundle and upload the output as a CI artifact before teardown** so you can download and view it after the pipeline.

## After CI teardown: how to view? (live dashboard vs static report)

- **Live dashboard** (`gpu_dashboard.html` + `serve_dashboard.sh`) needs a local HTTP server and `latest.json`; it is for **local** use while a run is in progress. In CI the environment is recycled, so **you cannot keep a live web UI in CI**.
- **Approach**: Do not rely on a live UI in CI; use the **static report `report.html`** instead. `finalize_monitor.sh` calls `generate_report.py` to produce a single HTML file (charts, stats, anomaly table embedded). **No server needed.** Upload the bundle as a CI artifact; after the run **download the artifact and open `report.html` in a browser** to see the full memory curve and stats.

So in CI: only “monitor → finalize bundle → upload artifact”; to view, download the artifact and open `report.html` locally.

## One step for local and CI

The same command works **locally** and in **CI**; no extra steps.

- **CI**: Do not set any env var. The script starts the monitor, runs the test, finalizes the bundle, and uploads artifacts. Watch the `[GPU]` lines in the log; after the run download artifacts and open `report.html` for the line chart.
- **Local**: Same. To **also** see the live dashboard in the browser, set `GPU_MONITOR_SERVE_DASHBOARD=1`; the script will start the dashboard server and print the URL (use SSH port forwarding if the machine is remote).

Example (from repo root):

```bash
# CI or local: log + report.html after the run
bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v

# Local with live dashboard (one command, no second terminal)
GPU_MONITOR_SERVE_DASHBOARD=1 bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v
```

- The dashboard URL is printed (default `http://127.0.0.1:8765/gpu_dashboard.html`). On a remote host run on your machine: `ssh -L 8765:127.0.0.1:8765 user@host` then open that URL.
- Bundle dir: `tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_<run_id>/` with `gpu_metrics.csv`, `report.html`, `README.txt`. **The line chart is in `report.html`**; the log prints the path at the end (e.g. `Line chart (memory utilization): open in browser: .../report.html`).

## Flow overview

1. **Start monitor**: run `./moniter.sh` in background for the whole run (writes CSV).
2. **Run long test**: run your test (e.g. long stability).
3. **Finalize**: before teardown run `./finalize_monitor.sh` to generate the report and bundle.
4. **Archive**: upload the dir printed by `finalize_monitor.sh` as a CI artifact.

Then download that artifact from the pipeline to get `gpu_metrics.csv` and `report.html` (with charts and anomaly markers).

## Step details

### 1. Start monitor (background)

```bash
cd tests/e2e/online_serving
./moniter.sh all 5 &
MONITOR_PID=$!
```

Optional: set `GPU_MONITOR_DATA_ROOT` to a fixed dir so later steps match.

### 2. Run long test

Run your 24h/72h or other long test as usual.

### 3. Finalize (must run before teardown)

In **finally** or **after script** (so it runs even if the test fails):

```bash
cd tests/e2e/online_serving
# Optional: stop monitor (or it will stop when the job ends)
# kill $MONITOR_PID 2>/dev/null || true
# Bundle current run and print dir to archive
BUNDLE_LINE=$(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "Archive dir: $GPU_MONITOR_BUNDLE_DIR"
```

### 4. Upload as CI artifact

**GitHub Actions** example:

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

**GitLab CI** (in the job):

```yaml
after_script:
  - cd tests/e2e/online_serving
  - eval $(./finalize_monitor.sh 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
artifacts:
  when: always
  paths:
    - tests/e2e/online_serving/gpu_monitor_data/gpu_monitor_bundle_*/
```

**Jenkins**: run `finalize_monitor.sh` in finally and use `archiveArtifacts` for `gpu_monitor_data/gpu_monitor_bundle_*/**`.

### Buildkite (use run_with_gpu_monitor.sh)

One command does “start monitor + run test + finalize + upload”:

```yaml
commands:
  - bash tests/e2e/online_serving/run_with_gpu_monitor.sh -- pytest -s -v tests/e2e/offline_inference/test_t2i_model.py ...
```

- **Scripts used**: `run_with_gpu_monitor.sh` (which runs `moniter.sh`, `finalize_monitor.sh`, `generate_report.py`).
- **Real-time in CI**: There is no separate web UI. Open the **Buildkite job log**; every ~15s you will see a line like `[GPU] timestamp,...,gpu_index,used_mb,total_mb,util_pct` (latest sample).
- **Where to get the line chart after the run**: On the same job page, open **Artifacts** and download the files (e.g. `gpu_metrics.csv`, `report.html`, `README.txt`). Open **report.html** in a browser to see the line chart and stats.

## Artifact contents

- `gpu_metrics.csv`: raw samples (timestamp, GPU index, memory used/total, util %).
- `report.html`: single-file report with stats table, time-series line chart, and simple anomaly markers; open in a browser.
- `README.txt`: short description.

After the run, download the artifact and open `report.html` locally; no need for the original environment or URL.
