"""Shared reliability fault-injection helpers.

This package keeps fault injection callable from tests directly:
- GPU OOM (CUDA sidecar memory hog)
- process kill by pattern and signal
- post-ready hooks via ``fault_injector`` / ``omni_server_after_fault`` fixtures

Worker PID classification (``worker_pids`` in snapshots) is still used for logging /
markers; post-fault **process** cleanup assertions use the full captured
``tree_pids`` (serve root + descendants at fault time) — see
:func:`assert_no_server_tree_process_residual_and_gpu_release`.
"""

from tests.dfx.reliability.helpers.assertions import (
    assert_fault_exception,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_no_worker_residual_and_gpu_release,
    assert_post_fault_health_terminal,
    list_alive_pids,
    query_gpu_compute_pid_used_memory_mb,
)
from tests.dfx.reliability.helpers.config import (
    DEFAULT_RUNTIME_WORKER_MARKERS,
    PROCESS_KILL_ERROR_KEYWORDS,
    resolve_oom_device_spec,
    supports_video_generation,
    worker_residual_timeout_after_kill_signal,
)
from tests.dfx.reliability.helpers.http import (
    extract_openai_error_contract_from_bytes,
    extract_openai_error_contract_from_payload,
    get_health_raw,
    post_chat_completions_raw,
    post_json_raw,
    post_json_raw_http_client,
)
from tests.dfx.reliability.helpers.injection import (
    FaultInjector,
    OomHandle,
    inject_gpu_oom,
    inject_process_kill,
    list_remote_process_pids_by_pattern,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    run_fault_injection_with_rate_load,
    start_gpu_oom_hog,
    stop_gpu_oom_hog,
    stop_gpu_oom_hogs,
)

__all__ = [
    "DEFAULT_RUNTIME_WORKER_MARKERS",
    "FaultInjector",
    "OomHandle",
    "PROCESS_KILL_ERROR_KEYWORDS",
    "assert_fault_exception",
    "assert_no_server_tree_process_residual_and_gpu_release",
    "assert_no_worker_residual_and_gpu_release",
    "assert_post_fault_health_terminal",
    "extract_openai_error_contract_from_bytes",
    "extract_openai_error_contract_from_payload",
    "get_health_raw",
    "inject_gpu_oom",
    "inject_process_kill",
    "list_alive_pids",
    "list_remote_process_pids_by_pattern",
    "make_process_kill_fault_injector",
    "make_server_root_kill_fault_injector",
    "make_server_tree_kill_fault_injector",
    "post_chat_completions_raw",
    "post_json_raw",
    "post_json_raw_http_client",
    "query_gpu_compute_pid_used_memory_mb",
    "resolve_oom_device_spec",
    "run_fault_injection_with_rate_load",
    "start_gpu_oom_hog",
    "stop_gpu_oom_hog",
    "stop_gpu_oom_hogs",
    "supports_video_generation",
    "worker_residual_timeout_after_kill_signal",
]
