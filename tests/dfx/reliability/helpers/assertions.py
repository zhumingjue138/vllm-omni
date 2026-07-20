"""Post-fault assertions for reliability tests."""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Sequence
from typing import Any

import psutil
import pytest

from tests.dfx.reliability.helpers.http import get_health_raw

logger = logging.getLogger(__name__)


def assert_fault_exception(exc: Exception, error_keywords: tuple[str, ...]) -> None:
    text = str(exc).lower()
    assert any(key in text for key in error_keywords), f"unexpected error under fault injection: {exc}"


def assert_post_fault_health_terminal(host: str, port: int, *, scenario: str) -> None:
    deadline = time.monotonic() + 20.0
    last_observation = ""
    while time.monotonic() < deadline:
        try:
            status, body = get_health_raw(host, port, timeout_sec=5)
            last_observation = f"http={status}, body={body[:200]!r}"
            if status == 503:
                return
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
            return
        time.sleep(0.5)
    pytest.fail(f"[{scenario} health] no terminal post-fault health observed: {last_observation}")


def list_alive_pids(pids: Sequence[int]) -> list[int]:
    """Return PIDs from ``pids`` that still exist in the kernel and are **not** zombies.

    After SIGTERM/SIGKILL the serve child may exit before the test harness calls
    ``Popen.wait()``; until reaped, ``psutil.pid_exists`` stays true for the defunct
    slot. Those PIDs must not count as "residual server processes" for leak checks.
    """
    out: list[int] = []
    for pid in pids:
        pid_i = int(pid)
        if not psutil.pid_exists(pid_i):
            continue
        try:
            if psutil.Process(pid_i).status() == psutil.STATUS_ZOMBIE:
                continue
        except psutil.Error:
            continue
        out.append(pid_i)
    return out


def query_gpu_compute_pid_used_memory_mb() -> dict[int, int] | None:
    """Query NVIDIA compute-process memory map (pid -> used MB).

    Returns ``None`` when ``nvidia-smi`` is unavailable/unreadable on current host.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        logger.warning("[reliability][gpu] nvidia-smi query failed: %s", out.stderr.strip())
        return None

    pid_to_mem_mb: dict[int, int] = {}
    for line in out.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        pieces = [part.strip() for part in row.split(",", maxsplit=1)]
        if len(pieces) != 2:
            continue
        pid_raw, mem_raw = pieces
        if not pid_raw.isdigit():
            continue
        mem_digits = "".join(ch for ch in mem_raw if ch.isdigit())
        if not mem_digits:
            continue
        pid_to_mem_mb[int(pid_raw)] = int(mem_digits)
    return pid_to_mem_mb


def assert_no_server_tree_process_residual_and_gpu_release(
    server: Any,
    *,
    scenario: str,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.5,
) -> None:
    """Assert no live PIDs remain from the fault-time server tree and no GPU use by those PIDs.

    Uses ``reliability_fault_snapshot["tree_pids"]`` (root + all descendants captured at
    injection time), not the marker-filtered ``worker_pids`` subset — so helpers like
    ``multiprocessing.resource_tracker`` are included in the residual check.

    **Zombie** PIDs (exited children not yet reaped by the test ``Popen``) are excluded
    from the "alive" list; ``psutil.pid_exists`` alone would false-positive on them.

    Note: if a child is reparented to PID 1 while staying alive, its PID is unchanged
    and remains detected; if it **exits and a new unrelated process reuses the same
    numeric PID**, this check may false-positive (rare on short windows).
    """
    snapshot = getattr(server, "reliability_fault_snapshot", None)
    if not isinstance(snapshot, dict):
        pytest.fail(f"[{scenario}] missing reliability fault snapshot on server")
    tree_pids = [int(pid) for pid in snapshot.get("tree_pids", [])]
    if not tree_pids:
        pytest.skip(f"[{scenario}] no server process tree PIDs captured for this run")

    tree_pid_set = set(tree_pids)
    deadline = time.monotonic() + timeout_sec
    last_alive: list[int] = []
    last_gpu_leaks: dict[int, int] = {}
    while time.monotonic() < deadline:
        last_alive = list_alive_pids(tree_pids)
        gpu_map = query_gpu_compute_pid_used_memory_mb()
        if gpu_map is None:
            pytest.skip(f"[{scenario}] nvidia-smi unavailable; skip GPU release assertion")
        last_gpu_leaks = {pid: mem_mb for pid, mem_mb in gpu_map.items() if pid in tree_pid_set and mem_mb > 0}
        if not last_alive and not last_gpu_leaks:
            return
        time.sleep(poll_interval_sec)

    assert not last_alive, f"[{scenario}] residual server-tree processes remain alive: {last_alive}"
    assert not last_gpu_leaks, f"[{scenario}] server-tree PID GPU memory not released: {last_gpu_leaks}"


def assert_no_worker_residual_and_gpu_release(
    server: Any,
    *,
    scenario: str,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.5,
) -> None:
    """Deprecated: use :func:`assert_no_server_tree_process_residual_and_gpu_release`."""
    assert_no_server_tree_process_residual_and_gpu_release(
        server,
        scenario=scenario,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )
