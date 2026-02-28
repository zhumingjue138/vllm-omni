# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L5 GPU memory monitor pytest integration.

Set env GPU_MONITOR=1 to start moniter.sh for the session, then finalize
and upload artifacts on exit. Loaded when running tests under
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


def _gpu_monitor_env():
    env = os.environ.copy()
    env.setdefault("GPU_MONITOR_DATA_ROOT", str(L5_DIR / "gpu_monitor_data"))
    env.setdefault("SKIP_DEPS_CHECK", "1")
    env.setdefault("GPU_MONITOR_INTERVAL", "5")
    env.setdefault("GPU_MONITOR_DEVICES", "all")
    env.setdefault("GPU_MONITOR_LOG_INTERVAL", "15")
    return env


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
    global _monitor_process
    if not GPU_MONITOR_ENABLED:
        return
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        session.config._gpu_monitor_skipped = True
        return
    env = _gpu_monitor_env()
    dev = env["GPU_MONITOR_DEVICES"]
    interval = env["GPU_MONITOR_INTERVAL"]
    _monitor_process = subprocess.Popen(
        [str(L5_DIR / "moniter.sh"), dev, interval],
        cwd=str(L5_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log_interval = env["GPU_MONITOR_LOG_INTERVAL"]
    print(
        f"[GPU Monitor] Started moniter.sh (PID {_monitor_process.pid}), "
        f"interval={interval}s, devices={dev}; log every {log_interval}s.",
        flush=True,
    )
    t = threading.Thread(target=_log_reporter_loop, daemon=True)
    t.start()


def _gpu_monitor_finalize_and_upload():
    """Stop monitor, run finalize_monitor.sh, upload artifacts if in CI."""
    global _monitor_process
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

    env = _gpu_monitor_env()
    data_root = env["GPU_MONITOR_DATA_ROOT"]
    current_run_id = os.path.join(data_root, "current_run_id")
    if not os.path.isfile(current_run_id):
        return
    print("--- Finalizing: bundling GPU monitor data ---", flush=True)
    result = subprocess.run(
        [str(L5_DIR / "finalize_monitor.sh")],
        cwd=str(L5_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout or "finalize_monitor.sh failed", flush=True)
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
