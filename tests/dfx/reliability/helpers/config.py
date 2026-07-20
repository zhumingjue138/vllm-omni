"""Reliability test configuration helpers and constants."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PROCESS_KILL_ERROR_KEYWORDS: tuple[str, ...] = (
    "timeout",
    "did not complete within",
    "connection",
    "engine",
    "orchestrator",
    "dead",
    "internal",
    "500",
    "503",
)

# Substrings matched against ``psutil.Process.name`` + argv text for classifying GPU
# workers under the test server tree. Extend per-suite via
# ``server.reliability_worker_markers_extra`` or replace via
# ``server.reliability_worker_markers`` (see ``_resolve_runtime_worker_markers``).
DEFAULT_RUNTIME_WORKER_MARKERS: tuple[str, ...] = (
    "multiprocessing.spawn",
    # Covers ``VLLM::Worker``, ``VLLM::StageEngineCoreProc_*`` (Omni stage engines), etc.
    "VLLM::",
)


def supports_video_generation(model_name: str) -> bool:
    lower = model_name.lower()
    return any(key in lower for key in ("wan", "video", "i2v", "t2v"))


def _parse_stage_devices(stage_config_path: str) -> str:
    text = Path(stage_config_path).read_text(encoding="utf-8")
    raw_devices: list[str] = re.findall(r"^\s*devices:\s*\"?([0-9,\s]+)\"?\s*$", text, flags=re.MULTILINE)
    devices: set[int] = set()
    for item in raw_devices:
        for token in item.split(","):
            token = token.strip()
            if token:
                devices.add(int(token))
    if not devices:
        raise ValueError(f"No runtime.devices found in stage config: {stage_config_path}")
    return ",".join(str(x) for x in sorted(devices))


def resolve_oom_device_spec(config: dict[str, Any], stage_config_path: str | None) -> str:
    explicit = config.get("device")
    if explicit is not None:
        return str(explicit)
    if not stage_config_path:
        return "0"
    return _parse_stage_devices(stage_config_path)


def worker_residual_timeout_after_kill_signal(signal_name: str) -> float:
    """Wall-clock budget for fault-snapshot PIDs / GPU to clear after ``signal_name``."""
    if signal_name == "SIGKILL":
        return 30.0
    if signal_name == "SIGINT":
        return 120.0
    if signal_name == "SIGTERM":
        return 90.0
    return 30.0
