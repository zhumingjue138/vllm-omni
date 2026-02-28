# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L5 GPU memory monitor pytest integration.

Set env GPU_MONITOR=1 to start gpu_monitor.sh start for the session, then
finalize and upload artifacts on exit. Loaded when running tests under
tests/e2e/online_serving/ (via parent conftest import).

  GPU_MONITOR=1 pytest tests/e2e/online_serving/test_foo.py -v
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

L5_DIR = Path(__file__).resolve().parent
# Only start monitor when explicitly requested (CI or local long-run)
GPU_MONITOR_ENABLED = os.environ.get("GPU_MONITOR") == "1"

_monitor_process = None
_log_reporter_stop = threading.Event()
_monitor_log_fp = None


def _gpu_monitor_env():
    env = os.environ.copy()
    env.setdefault("GPU_MONITOR_DATA_ROOT", str(L5_DIR / "gpu_monitor_data"))
    env.setdefault("SKIP_DEPS_CHECK", "1")
    env.setdefault("GPU_MONITOR_INTERVAL", "5")
    env.setdefault("GPU_MONITOR_DEVICES", "all")
    env.setdefault("GPU_MONITOR_LOG_INTERVAL", "15")
    return env


def _set_env_defaults(env: dict) -> None:
    """Ensure our defaults are visible to other helpers (log reporter, users)."""
    for k in (
        "GPU_MONITOR_DATA_ROOT",
        "SKIP_DEPS_CHECK",
        "GPU_MONITOR_INTERVAL",
        "GPU_MONITOR_DEVICES",
        "GPU_MONITOR_LOG_INTERVAL",
    ):
        v = env.get(k)
        if v is not None:
            os.environ.setdefault(k, str(v))


def _monitor_healthcheck_async(data_root: str, interval_s: int) -> None:
    """Warn (once) if no CSV data appears shortly after start."""
    try:
        # Give the sampler some time to write current_run_id and first sample.
        time.sleep(max(10, min(30, interval_s * 3)))
        run_id_file = os.path.join(data_root, "current_run_id")
        if not os.path.isfile(run_id_file):
            print(
                f"[GPU Monitor] No current_run_id under {data_root}. "
                "Sampler may have failed to start; check gpu_monitor_subprocess.log.",
                flush=True,
            )
            return
        with open(run_id_file) as f:
            run_id = f.read().strip()
        csv_path = os.path.join(data_root, run_id, "gpu_metrics.csv")
        if not os.path.isfile(csv_path):
            print(
                f"[GPU Monitor] Missing CSV: {csv_path}. "
                "Sampler may have exited early; check gpu_monitor_subprocess.log.",
                flush=True,
            )
            return
        try:
            with open(csv_path) as f:
                lines = f.readlines()
        except OSError:
            return
        # CSV has header line; require at least one sample line.
        if len(lines) <= 1:
            print(
                f"[GPU Monitor] CSV exists but has no samples yet: {csv_path}. "
                "If this persists, nvidia-smi may be failing in the test environment.",
                flush=True,
            )
    except Exception:
        # Best-effort only; never break tests.
        return


def _log_reporter_loop():
    """Print latest CSV line to stdout every GPU_MONITOR_LOG_INTERVAL seconds."""
    interval = int(os.environ.get("GPU_MONITOR_LOG_INTERVAL", "15"))
    data_root = os.environ.get("GPU_MONITOR_DATA_ROOT", str(L5_DIR / "gpu_monitor_data"))
    time.sleep(10)
    while not _log_reporter_stop.wait(timeout=interval):
        run_id_file = os.path.join(data_root, "current_run_id")
        if not os.path.isfile(run_id_file):
            continue
        try:
            with open(run_id_file) as f:
                run_id = f.read().strip()
        except OSError:
            continue
        csv_path = os.path.join(data_root, run_id, "gpu_metrics.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            with open(csv_path) as f:
                lines = f.readlines()
        except OSError:
            continue
        if lines:
            line = lines[-1].rstrip()
            if line:
                print(f"[GPU] {line}", flush=True)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Start GPU monitor subprocess when GPU_MONITOR=1 and nvidia-smi is available."""
    global _monitor_process, _monitor_log_fp
    if not GPU_MONITOR_ENABLED:
        return
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        session.config._gpu_monitor_skipped = True
        print(f"[GPU Monitor] Skip: nvidia-smi unavailable ({type(e).__name__}).", flush=True)
        return
    if probe.returncode != 0 or not (probe.stdout or b"").strip():
        session.config._gpu_monitor_skipped = True
        stderr = (probe.stderr or b"").decode("utf-8", errors="ignore").strip()
        msg = "nvidia-smi returned non-zero" if probe.returncode != 0 else "nvidia-smi returned empty output"
        print(f"[GPU Monitor] Skip: {msg}. stderr={stderr!r}", flush=True)
        return
    env = _gpu_monitor_env()
    _set_env_defaults(env)
    dev = env["GPU_MONITOR_DEVICES"]
    interval = env["GPU_MONITOR_INTERVAL"]
    data_root = env["GPU_MONITOR_DATA_ROOT"]
    gpu_monitor_sh = L5_DIR / "gpu_monitor.sh"
    os.makedirs(data_root, exist_ok=True)
    monitor_log = os.path.join(data_root, "gpu_monitor_subprocess.log")
    _monitor_log_fp = open(monitor_log, "ab")
    _monitor_process = subprocess.Popen(
        ["bash", str(gpu_monitor_sh), "start", dev, interval],
        cwd=str(L5_DIR),
        env=env,
        stdout=_monitor_log_fp,
        stderr=_monitor_log_fp,
    )
    log_interval = env["GPU_MONITOR_LOG_INTERVAL"]
    print(
        f"[GPU Monitor] Started gpu_monitor.sh start (PID {_monitor_process.pid}), "
        f"interval={interval}s, devices={dev}; log every {log_interval}s. "
        f"data_root={data_root}; subprocess_log={monitor_log}",
        flush=True,
    )
    t2 = threading.Thread(
        target=_monitor_healthcheck_async,
        args=(data_root, int(interval)),
        daemon=True,
    )
    t2.start()
    t = threading.Thread(target=_log_reporter_loop, daemon=True)
    t.start()


def _gpu_monitor_finalize_and_upload():
    """Stop monitor, run gpu_monitor.sh finalize, upload artifacts if in CI."""
    global _monitor_process, _monitor_log_fp
    if _monitor_process is None:
        return
    _log_reporter_stop.set()
    try:
        _monitor_process.terminate()
        _monitor_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _monitor_process.kill()
    except Exception:
        pass
    _monitor_process = None
    try:
        if _monitor_log_fp is not None:
            _monitor_log_fp.flush()
            _monitor_log_fp.close()
    except Exception:
        pass
    _monitor_log_fp = None

    env = _gpu_monitor_env()
    data_root = env["GPU_MONITOR_DATA_ROOT"]
    current_run_id = os.path.join(data_root, "current_run_id")
    if not os.path.isfile(current_run_id):
        return
    print("--- Finalizing: bundling GPU monitor data ---", flush=True)
    result = subprocess.run(
        ["bash", str(L5_DIR / "gpu_monitor.sh"), "finalize"],
        cwd=str(L5_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout or "gpu_monitor.sh finalize failed", flush=True)
        return
    print(result.stdout, flush=True)
    bundle_dir = None
    for line in (result.stdout or "").splitlines():
        if line.startswith("GPU_MONITOR_BUNDLE_DIR="):
            bundle_dir = line.split("=", 1)[1].strip().strip("'\"")
            break
    if bundle_dir and os.path.isdir(bundle_dir):
        print(f"--- GPU monitor bundle: {bundle_dir} ---", flush=True)
        print(f"--- Line chart: {bundle_dir}/report.html ---", flush=True)
        try:
            for name in os.listdir(bundle_dir):
                path = os.path.join(bundle_dir, name)
                if os.path.isfile(path):
                    subprocess.run(
                        ["buildkite-agent", "artifact", "upload", path],
                        check=False,
                        capture_output=True,
                        timeout=30,
                    )
        except FileNotFoundError:
            pass
        except Exception:
            pass


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Stop monitor and run finalize + optional artifact upload."""
    if not GPU_MONITOR_ENABLED:
        return
    _gpu_monitor_finalize_and_upload()


@pytest.fixture(scope="session")
def gpu_monitor_data_root():
    """Root directory for GPU monitor data (CSV, run_id). Set when GPU_MONITOR=1."""
    return os.environ.get(
        "GPU_MONITOR_DATA_ROOT",
        str(L5_DIR / "gpu_monitor_data"),
    )
