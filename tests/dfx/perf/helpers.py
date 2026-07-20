"""Shared perf benchmark helpers.

JSON dialect-specific loaders (omni vs diffusion) stay in ``tests/dfx/conftest.py``
and ``run_diffusion_benchmark.py`` respectively. Everything else shared by perf runners
lives here until a later package split.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from benchmarks.diffusion.backends import endpoint_filename_token, normalize_endpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
PERF_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PERF_RESULTS_DIR = Path(__file__).resolve().parent / "results"
OMNI_RAW_RESULT_TEMPLATE_PATH = PERF_SCRIPTS_DIR / "result_omni_template.json"
# Backward-compatible alias: raw vLLM bench metrics template used when subprocess omits JSON.
OMNI_RESULT_TEMPLATE_PATH = OMNI_RAW_RESULT_TEMPLATE_PATH
DIFFUSION_RESULT_TEMPLATE_PATH = PERF_SCRIPTS_DIR / "diffusion_result_template.json"
DEFAULT_DIFFUSION_BENCHMARK_SCRIPT = str(REPO_ROOT / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py")
DEFAULT_OMNI_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "600", "--init-timeout", "900"]

_BRANCHPOINT_COMMIT_SHA: str | None = None


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def resolve_baseline_value(
    baseline_raw: Any,
    *,
    sweep_index: int | None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> Any:
    """Pick the baseline threshold for this sweep step."""
    if baseline_raw is None:
        return 100000
    if isinstance(baseline_raw, dict):
        if max_concurrency is not None:
            for key in (max_concurrency, str(max_concurrency)):
                if key in baseline_raw:
                    return baseline_raw[key]
        if request_rate is not None:
            for key in (request_rate, str(request_rate)):
                if key in baseline_raw:
                    return baseline_raw[key]
        raise KeyError(
            f"baseline dict has no key for max_concurrency={max_concurrency!r} "
            f"or request_rate={request_rate!r}; keys={list(baseline_raw.keys())!r}"
        )
    if isinstance(baseline_raw, (list, tuple)):
        if sweep_index is None:
            raise ValueError("list baseline requires sweep_index")
        if not (0 <= sweep_index < len(baseline_raw)):
            raise IndexError(f"baseline list len={len(baseline_raw)} has no index {sweep_index}")
        return baseline_raw[sweep_index]
    return baseline_raw


def baseline_thresholds_for_step(
    baseline_data: dict[str, Any],
    *,
    sweep_index: int | None = None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> dict[str, Any]:
    """Resolve baseline config to one threshold per metric for this iteration."""
    return {
        metric_name: resolve_baseline_value(
            baseline_raw,
            sweep_index=sweep_index,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
        )
        for metric_name, baseline_raw in baseline_data.items()
    }


# ---------------------------------------------------------------------------
# Subprocess logging / result I/O
# ---------------------------------------------------------------------------


def safe_filename_token(value: Any | None, *, default: str = "na") -> str:
    """Make a single path segment safe for result filenames on common filesystems."""
    if value is None:
        return default
    s = str(value).strip()
    for bad in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        s = s.replace(bad, "_")
    return s if s else default


def run_subprocess_with_log(
    cmd: list[str],
    *,
    cwd: Path | str,
    log_file: Path | None = None,
    unbuffered: bool = False,
) -> int:
    """Run a benchmark subprocess, optionally teeing output to *log_file*."""
    if unbuffered and cmd:
        cmd = [cmd[0], "-u"] + cmd[1:]

    cwd_str = str(cwd)
    if log_file is None:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd_str,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Failed to capture benchmark process output streams")

        def _forward_stream(stream) -> None:
            try:
                for line in iter(stream.readline, ""):
                    print(line, end="")
            finally:
                stream.close()

        stdout_thread = threading.Thread(target=_forward_stream, args=(process.stdout,))
        stderr_thread = threading.Thread(target=_forward_stream, args=(process.stderr,))
        stdout_thread.start()
        stderr_thread.start()
        stdout_thread.join()
        stderr_thread.join()
        return process.wait()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"cmd: {' '.join(cmd)}\n\n")
        log_fh.flush()
        process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            cwd=cwd_str,
        )
        return_code = process.wait()

    with open(log_file, encoding="utf-8") as log_fh:
        print(log_fh.read(), end="")
    return return_code


def read_json_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_metrics_or_template(
    result_path: Path,
    template_path: Path,
    *,
    extract_template_metrics: Any | None = None,
) -> dict[str, Any]:
    """Load metrics JSON, falling back to *template_path* when missing."""
    if result_path.exists():
        return read_json_file(result_path)

    template_payload = read_json_file(template_path) if template_path.suffix == ".json" else {}
    if extract_template_metrics is not None:
        metrics = extract_template_metrics(template_payload)
    else:
        metrics = template_payload

    write_json_file(result_path, metrics)
    print(f"Benchmark result file not generated, fallback to template: {result_path}")
    return metrics


def append_to_aggregated_file(
    record: dict[str, Any],
    *,
    aggregated_result_file: Path,
    result_lock: threading.Lock,
) -> None:
    """Thread-safe append of *record* to a session-level aggregated JSON array file."""
    with result_lock:
        aggregated_result_file.parent.mkdir(parents=True, exist_ok=True)
        if aggregated_result_file.exists():
            with open(aggregated_result_file, encoding="utf-8") as f:
                records: list[dict[str, Any]] = json.load(f)
        else:
            records = []
        records.append(record)
        with open(aggregated_result_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Server CLI builders (JSON parsers stay elsewhere; these only assemble CLI)
# ---------------------------------------------------------------------------


def build_omni_server_cli_args_from_tuple(
    *,
    stage_config_path: str | None,
    stage_overrides: str | None,
    extra_cli_args: tuple[str, ...] | list[str],
    use_omni: bool,
    timeout_args: list[str] | None = DEFAULT_OMNI_SERVER_TIMEOUT_ARGS,
) -> list[str]:
    """Build ``OmniServer`` CLI args from omni perf parametrize tuple fields."""
    server_args: list[str] = []
    if use_omni and timeout_args:
        server_args += list(timeout_args)
    if stage_config_path:
        server_args = ["--deploy-config", stage_config_path] + server_args
    if stage_overrides:
        server_args = ["--stage-overrides", stage_overrides] + server_args
    if extra_cli_args:
        server_args = list(extra_cli_args) + server_args
    return normalize_server_cli_args(server_args)


def normalize_repo_relative_path(path: str) -> str:
    """Normalize legacy ``../vllm_omni/...`` paths for repo-root subprocess cwd."""
    if path.startswith("../vllm_omni/"):
        return path[3:]
    return path


def normalize_server_cli_args(args: list[str]) -> list[str]:
    """Fix deploy-config paths in a flat CLI arg list."""
    normalized = list(args)
    deploy_flags = {"--deploy-config", "--deploy_config"}
    for i, arg in enumerate(normalized):
        if arg in deploy_flags and i + 1 < len(normalized):
            normalized[i + 1] = normalize_repo_relative_path(normalized[i + 1])
    return normalized


def build_omni_server_cli_args_from_diffusion_cfg(
    server_cfg: dict[str, Any],
    *,
    timeout_args: list[str] | None = DEFAULT_OMNI_SERVER_TIMEOUT_ARGS,
) -> list[str]:
    """Build ``OmniServer`` CLI args from diffusion runner server config dict."""
    serve_args = normalize_server_cli_args(list(server_cfg["serve_args"]))
    if timeout_args:
        return list(timeout_args) + serve_args
    return serve_args


# ---------------------------------------------------------------------------
# Omni benchmark runner
# ---------------------------------------------------------------------------


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return [value] if not isinstance(value, (list, tuple)) else list(value)


@dataclass
class OmniBenchmarkSession:
    """Perf omni runner: one session JSON array file (same pattern as diffusion)."""

    result_dir: Path = field(default_factory=lambda: Path(os.environ.get("OMNI_BENCHMARK_DIR", str(PERF_RESULTS_DIR))))
    aggregated_result_file: Path | None = None
    result_lock: threading.Lock = field(default_factory=threading.Lock)


def build_omni_run_params(
    params: dict[str, Any],
    *,
    num_prompts: int,
    sweep_index: int | None = None,
    request_rate: Any | None = None,
    max_concurrency: Any | None = None,
) -> dict[str, Any]:
    """Materialize one omni sweep's benchmark_params (snake_case JSON dialect)."""
    run_params = {
        key: value for key, value in params.items() if key not in {"request_rate", "max_concurrency", "num_prompts"}
    }
    run_params["num_prompts"] = num_prompts
    if request_rate is not None:
        run_params["request_rate"] = request_rate
    if max_concurrency is not None:
        run_params["max_concurrency"] = max_concurrency
    if "baseline" in params:
        run_params["baseline"] = {
            metric: resolve_baseline_value(
                baseline_raw,
                sweep_index=sweep_index,
                max_concurrency=max_concurrency,
                request_rate=request_rate,
            )
            for metric, baseline_raw in params["baseline"].items()
        }
    return run_params


def iter_omni_sweep_runs(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand omni benchmark_params into request-rate and max-concurrency sweeps."""
    request_rate_list = to_list(params.get("request_rate"))
    num_prompt_list = to_list(params.get("num_prompts"))
    max_concurrency_list = to_list(params.get("max_concurrency"))

    max_len = max(len(request_rate_list), len(max_concurrency_list))
    if len(num_prompt_list) == 1 and max_len > 1:
        num_prompt_list = num_prompt_list * max_len
    elif max_len == 1 and len(num_prompt_list) > 1:
        if len(request_rate_list) == 1:
            request_rate_list = request_rate_list * len(num_prompt_list)
        if len(max_concurrency_list) == 1:
            max_concurrency_list = max_concurrency_list * len(num_prompt_list)
        max_len = max(len(request_rate_list), len(max_concurrency_list))
    elif len(num_prompt_list) != max_len and max_len > 0:
        raise ValueError("The number of prompts does not match the QPS or max_concurrency")

    sweep_runs: list[dict[str, Any]] = []

    for i, (request_rate, num_prompts) in enumerate(zip(request_rate_list, num_prompt_list)):
        sweep_runs.append(
            {
                "params": build_omni_run_params(
                    params,
                    request_rate=request_rate,
                    num_prompts=num_prompts,
                    sweep_index=i,
                ),
                "num_prompts": num_prompts,
                "sweep_index": i,
                "request_rate": request_rate,
                "max_concurrency": None,
                "flow": request_rate,
            }
        )

    for i, (max_concurrency, num_prompts) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        sweep_runs.append(
            {
                "params": build_omni_run_params(
                    params,
                    max_concurrency=max_concurrency,
                    num_prompts=num_prompts,
                    sweep_index=i,
                    request_rate="inf",
                ),
                "num_prompts": num_prompts,
                "sweep_index": i,
                "request_rate": None,
                "max_concurrency": max_concurrency,
                "flow": max_concurrency,
            }
        )

    return sweep_runs


def resolve_omni_benchmark_endpoint(params: dict[str, Any]) -> str:
    endpoint = params.get("endpoint")
    if endpoint:
        return normalize_endpoint(cast(str, endpoint))
    backend = params.get("backend")
    if backend:
        return normalize_endpoint(cast(str, backend))
    return "/v1/chat/completions"


def build_omni_report_record(
    *,
    test_name: str,
    endpoint: str,
    timestamp: str,
    server_params: dict[str, Any],
    benchmark_params: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a diffusion-style omni perf record without flat CI report fields."""
    return {
        "test_name": test_name,
        "endpoint": endpoint,
        "timestamp": timestamp,
        "server_params": server_params,
        "benchmark_params": benchmark_params,
        "result": metrics,
    }


def _extract_omni_raw_metrics(
    raw: dict[str, Any],
    *,
    random_input_len: Any | None = None,
    random_output_len: Any | None = None,
) -> dict[str, Any]:
    """Return vLLM bench metrics only (baseline stays in benchmark_params)."""
    metrics = {key: value for key, value in raw.items() if key != "baseline"}
    if random_input_len is not None:
        metrics["random_input_len"] = random_input_len
    if random_output_len is not None:
        metrics["random_output_len"] = random_output_len
    return metrics


def run_omni_benchmark(
    args: list[str],
    test_name: str,
    flow: Any,
    dataset_name: str,
    num_prompt: int,
    *,
    session: OmniBenchmarkSession | None = None,
    server_params: dict[str, Any] | None = None,
    benchmark_params: dict[str, Any] | None = None,
    endpoint: str | None = None,
    baseline_config: dict[str, Any] | None = None,
    sweep_index: int | None = None,
    request_rate: Any | None = None,
    max_concurrency: Any | None = None,
    random_input_len: Any | None = None,
    random_output_len: Any | None = None,
    result_template_path: Path = OMNI_RAW_RESULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Run one ``vllm bench serve --omni`` iteration and return flat metrics for assertions."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ri = safe_filename_token(random_input_len)
    ro = safe_filename_token(random_output_len)
    raw_result_filename = f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_in{ri}_out{ro}_{timestamp}.json"

    if session is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="omni_bench_tmp_", delete=False) as tmp:
            raw_result_path = Path(tmp.name)
        bench_result_dir = str(raw_result_path.parent)
        bench_result_filename = raw_result_path.name
    else:
        bench_result_dir = os.environ.get("BENCHMARK_DIR", "tests")
        bench_result_filename = raw_result_filename
        raw_result_path = Path(bench_result_dir) / bench_result_filename

    command = (
        ["vllm", "bench", "serve", "--omni"]
        + args
        + [
            "--num-warmups",
            "2",
            "--save-result",
            "--result-dir",
            bench_result_dir,
            "--result-filename",
            bench_result_filename,
        ]
    )
    run_subprocess_with_log(command, cwd=REPO_ROOT)

    try:
        raw_result = read_metrics_or_template(raw_result_path, result_template_path)
        metrics = _extract_omni_raw_metrics(
            raw_result,
            random_input_len=random_input_len,
            random_output_len=random_output_len,
        )

        if session is not None:
            assert server_params is not None
            assert benchmark_params is not None
            resolved_endpoint = endpoint or resolve_omni_benchmark_endpoint(benchmark_params)
            record = build_omni_report_record(
                test_name=test_name,
                endpoint=resolved_endpoint,
                timestamp=timestamp,
                server_params=server_params,
                benchmark_params=benchmark_params,
                metrics=metrics,
            )
            if session.aggregated_result_file is None:
                raise ValueError("OmniBenchmarkSession.aggregated_result_file must be set for perf runs")
            append_to_aggregated_file(
                record,
                aggregated_result_file=session.aggregated_result_file,
                result_lock=session.result_lock,
            )
            print(f"\n  Result appended to: {session.aggregated_result_file}")
            return metrics

        # Legacy flat JSON for stability and other non-perf callers.
        legacy_result = dict(metrics)
        if baseline_config:
            legacy_result["baseline"] = baseline_thresholds_for_step(
                baseline_config,
                sweep_index=sweep_index,
                request_rate=request_rate,
                max_concurrency=max_concurrency,
            )
        else:
            legacy_result["baseline"] = {}
        write_json_file(raw_result_path, legacy_result)
        return legacy_result
    finally:
        if session is not None:
            raw_result_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Diffusion benchmark reporting + runner
# ---------------------------------------------------------------------------


def get_branchpoint_commit_sha() -> str:
    """Return the branch-point commit SHA against main."""
    global _BRANCHPOINT_COMMIT_SHA
    if _BRANCHPOINT_COMMIT_SHA is not None:
        return _BRANCHPOINT_COMMIT_SHA

    try:
        sha = (
            subprocess.check_output(
                ["git", "merge-base", "HEAD", "origin/main"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.STDOUT,
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
        _BRANCHPOINT_COMMIT_SHA = sha
    except Exception as e:
        print(f"Warning: failed to get branch-point commit SHA: {e}")
        _BRANCHPOINT_COMMIT_SHA = ""
    return _BRANCHPOINT_COMMIT_SHA


def to_resolution_string(params: dict[str, Any]) -> str:
    width = params.get("width", "unknown width")
    height = params.get("height", "unknown height")
    return f"{width}x{height}"


def to_parallelism_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    parts: list[str] = []
    if framework == "vllm-omni":
        keys = [
            "num-gpus",
            "usp",
            "ulysses-degree",
            "ring",
            "ring-degree",
            "cfg-parallel-size",
            "vae-patch-parallel-size",
            "vae-use-tiling",
            "tensor-parallel-size",
        ]
        for key in keys:
            if key in serve_args_dict:
                parts.append(f"{key}={serve_args_dict[key]}")
    return ",".join(parts) if parts else "none"


def to_cache_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni" and "cache-backend" in serve_args_dict:
        return str(serve_args_dict["cache-backend"])
    return "disabled"


def to_offload_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    selected: list[str] = []
    if framework == "vllm-omni":
        for key in ("enable-cpu-offload", "enable-layerwise-offload"):
            if key in serve_args_dict:
                selected.append(key)
    return f"enabled({';'.join(selected)})" if selected else "disabled"


def to_compile_value(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni" and "enforce-eager" in serve_args_dict:
        return "disabled"
    if framework == "vllm-omni":
        return "enabled"
    return "disabled"


def to_quantization_value(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni":
        quant = serve_args_dict.get("quantization")
        return str(quant) if quant else "disabled"
    return "disabled"


def default_benchmark_endpoint_for_task(task: str) -> str:
    if task in {"t2v", "i2v", "ti2v", "v2v"}:
        return "/v1/videos"
    if task in {"t2i", "i2i", "ti2i"}:
        return "/v1/chat/completions"
    raise ValueError(f"Unsupported task for benchmark endpoint resolution: {task}")


def resolve_benchmark_endpoint(server_cfg: dict[str, Any], params: dict[str, Any]) -> str:
    configured = server_cfg.get("benchmark_endpoint")
    if configured:
        return normalize_endpoint(cast(str, configured))
    return default_benchmark_endpoint_for_task(cast(str, params.get("task", "t2i")))


@dataclass
class DiffusionBenchmarkSession:
    benchmark_script: str
    result_dir: Path
    aggregated_result_file: Path
    template_path: Path = DIFFUSION_RESULT_TEMPLATE_PATH
    result_lock: threading.Lock = field(default_factory=threading.Lock)


_STAGE_METRICS_ENDPOINTS = {"/v1/chat/completions"}
_DIFFUSION_PIPELINE_PROFILER_ARG = "enable-diffusion-pipeline-profiler"


def _extract_diffusion_template_metrics(template_payload: Any) -> dict[str, Any]:
    return cast(dict[str, Any], template_payload[0]["result"])


def build_diffusion_report_record(
    *,
    test_name: str,
    endpoint: str,
    timestamp: str,
    model: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    log_file: Path,
    server_cfg: dict[str, Any],
    source_file: str,
) -> dict[str, Any]:
    server_type = cast(str, server_cfg.get("server_type", "vllm-omni"))
    serve_args_dict = server_cfg.get("serve_args_dict", {})
    if not isinstance(serve_args_dict, dict):
        serve_args_dict = {}

    completed = metrics.get("completed_requests", metrics.get("completed", 0))
    failed = metrics.get("failed_requests", metrics.get("failed", 0))

    return {
        "test_name": test_name,
        "endpoint": endpoint,
        "timestamp": timestamp,
        "server_params": server_cfg.get("server_params"),
        "benchmark_params": params,
        "result": metrics,
        "log_file": str(log_file),
        "Model": model,
        "Framework": server_type,
        "API Endpoint": endpoint,
        "Hardware": "",
        "Deployment": "",
        "Task": params.get("task", "t2i"),
        "Dataset": params.get("dataset", "random"),
        "resolution": to_resolution_string(params),
        "Parallelism": to_parallelism_string(server_type, serve_args_dict),
        "max_concurrency": params.get("max-concurrency", ""),
        "Cache": to_cache_string(server_type, serve_args_dict),
        "Quantization": to_quantization_value(server_type, serve_args_dict),
        "offload": to_offload_string(server_type, serve_args_dict),
        "compile": to_compile_value(server_type, serve_args_dict),
        "Attn_backend": os.environ.get("DIFFUSION_ATTENTION_BACKEND", ""),
        "num_inference_steps": params.get("num-inference-steps", ""),
        "completed": completed,
        "failed": failed,
        "throughput_qps": metrics.get("throughput_qps"),
        "latency_mean": metrics.get("latency_mean"),
        "latency_median": metrics.get("latency_median"),
        "latency_p99": metrics.get("latency_p99"),
        "latency_p95": metrics.get("latency_p95"),
        "latency_p50": metrics.get("latency_p50"),
        "peak_memory_mb_max": metrics.get("peak_memory_mb_max"),
        "peak_memory_mb_mean": metrics.get("peak_memory_mb_mean"),
        "peak_memory_mb_median": metrics.get("peak_memory_mb_median"),
        "stage_durations_mean": metrics.get("stage_durations_mean"),
        "stage_durations_p50": metrics.get("stage_durations_p50"),
        "stage_durations_p99": metrics.get("stage_durations_p99"),
        "commit_sha": get_branchpoint_commit_sha(),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID", ""),
        "build_url": os.environ.get("BUILDKITE_BUILD_URL", ""),
        "source_file": source_file,
    }


def run_diffusion_benchmark(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    test_name: str,
    *,
    session: DiffusionBenchmarkSession | None = None,
    endpoint: str | None = None,
    server_cfg: dict[str, Any] | None = None,
    source_file: str = "",
    template_path: Path = DIFFUSION_RESULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Run one ``diffusion_benchmark_serving.py`` iteration.

    With *session*, append a perf report record to the aggregated JSON file.
    With ``session=None`` (stability loop), return flat metrics only.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    server_cfg = server_cfg or {}
    resolved_endpoint = normalize_endpoint(endpoint or resolve_benchmark_endpoint(server_cfg, params))

    if session is not None:
        log_dir = session.result_dir / "logs"
        endpoint_label = endpoint_filename_token(resolved_endpoint)
        log_file: Path | None = log_dir / f"{test_name}_{endpoint_label}_{timestamp}.log"
        benchmark_script = session.benchmark_script
        result_template_path = session.template_path
    else:
        log_file = None
        benchmark_script = DEFAULT_DIFFUSION_BENCHMARK_SCRIPT
        result_template_path = template_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="diffusion_bench_tmp_", delete=False) as tmp:
        tmp_result_file = Path(tmp.name)

    exclude_keys = {
        "baseline",
        "dataset",
        "task",
        "name",
        "skip-performance-assertion",
        "request_rate",
        "max_concurrency",
        "num_prompts",
        "duration_sec",
        "num_prompts_per_batch",
    }
    cmd = [
        sys.executable,
        benchmark_script,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--endpoint",
        resolved_endpoint,
        "--dataset",
        params.get("dataset", "random"),
        "--task",
        params.get("task", "t2i"),
        "--output-file",
        str(tmp_result_file),
    ]

    serve_args_dict = server_cfg.get("serve_args_dict")
    profiler_enabled = isinstance(serve_args_dict, dict) and bool(serve_args_dict.get(_DIFFUSION_PIPELINE_PROFILER_ARG))
    if resolved_endpoint in _STAGE_METRICS_ENDPOINTS and profiler_enabled:
        cmd.append("--return-stage-metrics")

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(value)])

    print(f"\nRunning benchmark (endpoint={resolved_endpoint}): {' '.join(cmd)}")
    if log_file is not None:
        print(f"  Log file: {log_file}")

    return_code = run_subprocess_with_log(
        cmd,
        cwd=REPO_ROOT,
        log_file=log_file,
        unbuffered=True,
    )
    if return_code != 0:
        tmp_result_file.unlink(missing_ok=True)
        print(f"ERROR:Benchmark script exited with code {return_code}")
        if session is None:
            return {
                "completed": 0,
                "failed": 1,
                "duration": 0.0,
                "errors": [f"diffusion_benchmark_serving.py exited {return_code}"],
            }

    try:
        if not tmp_result_file.is_file() and session is None:
            return {
                "completed": 0,
                "failed": 1,
                "duration": 0.0,
                "errors": [f"Missing benchmark output: {tmp_result_file}"],
            }
        metrics = read_metrics_or_template(
            tmp_result_file,
            result_template_path,
            extract_template_metrics=_extract_diffusion_template_metrics,
        )
    finally:
        tmp_result_file.unlink(missing_ok=True)

    if session is None:
        return metrics

    assert log_file is not None
    record = build_diffusion_report_record(
        test_name=test_name,
        endpoint=resolved_endpoint,
        timestamp=timestamp,
        model=model,
        params=params,
        metrics=metrics,
        log_file=log_file,
        server_cfg=server_cfg,
        source_file=source_file,
    )
    append_to_aggregated_file(
        record,
        aggregated_result_file=session.aggregated_result_file,
        result_lock=session.result_lock,
    )
    print(f"\n  Result appended to: {session.aggregated_result_file}")
    print(f"  Log saved to:       {log_file}")
    return metrics


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_omni_benchmark_result(
    result: dict[str, Any],
    params: dict[str, Any],
    num_prompt: int,
    *,
    assert_baseline: bool,
    sweep_index: int | None = None,
    max_concurrency: Any | None = None,
    request_rate: Any | None = None,
) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    if not assert_baseline:
        return

    baseline_data = params.get("baseline", {})
    for metric_name, baseline_raw in baseline_data.items():
        current_value = result[metric_name]
        baseline_value = resolve_baseline_value(
            baseline_raw,
            sweep_index=sweep_index,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
        )
        if "throughput" in metric_name:
            if current_value <= baseline_value:
                print(
                    f"ERROR: Throughput test results were below baseline: "
                    f"{metric_name}: {current_value} > {baseline_value}"
                )
        else:
            if current_value >= baseline_value:
                print(f"ERROR: Test results exceeded baseline: {metric_name}: {current_value} < {baseline_value}")


def assert_diffusion_benchmark_result(
    result: dict[str, Any],
    params: dict[str, Any],
    num_prompts: int,
    *,
    sweep_index: int | None = None,
    max_concurrency: Any = None,
    request_rate: Any = None,
    assert_baseline: bool = True,
) -> None:
    completed = result.get("completed_requests", result.get("completed", 0))
    assert completed == num_prompts, f"Expected {num_prompts} completed requests, got {completed}"

    if not assert_baseline:
        return
    if params.get("skip-performance-assertion", False):
        print("Skipping performance assertions.")
        return

    for metric, baseline_raw in params.get("baseline", {}).items():
        current = result.get(metric)
        assert current is not None, f"Metric '{metric}' not found in result: {list(result.keys())}"
        threshold = resolve_baseline_value(
            baseline_raw,
            sweep_index=sweep_index,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
        )
        if "throughput" in metric:
            assert current >= threshold, f"{metric}: {current:.4f} < baseline {threshold}"
        else:
            assert current <= threshold, f"{metric}: {current:.4f} > baseline {threshold}"


# ---------------------------------------------------------------------------
# Diffusion sweep helpers (kebab-case params)
# ---------------------------------------------------------------------------


def build_diffusion_run_params(
    params: dict[str, Any],
    *,
    num_prompts: int,
    sweep_index: int | None = None,
    request_rate: Any | None = None,
    max_concurrency: Any | None = None,
) -> dict[str, Any]:
    run_params = {
        key: value for key, value in params.items() if key not in {"request-rate", "max-concurrency", "num-prompts"}
    }
    run_params["num-prompts"] = num_prompts
    if request_rate is not None:
        run_params["request-rate"] = request_rate
    if max_concurrency is not None:
        run_params["max-concurrency"] = max_concurrency
    if "baseline" in params:
        run_params["baseline"] = {
            metric: resolve_baseline_value(
                baseline_raw,
                sweep_index=sweep_index,
                max_concurrency=max_concurrency,
                request_rate=request_rate,
            )
            for metric, baseline_raw in params["baseline"].items()
        }
    return run_params


def iter_diffusion_sweep_runs(params: dict[str, Any]) -> list[dict[str, Any]]:
    request_rate_list = to_list(params.get("request-rate"))
    num_prompt_list = to_list(params.get("num-prompts", 10))
    max_concurrency_list = to_list(params.get("max-concurrency"))

    max_len = max(len(request_rate_list), len(max_concurrency_list))
    if len(num_prompt_list) == 1 and max_len > 1:
        num_prompt_list = num_prompt_list * max_len
    elif max_len == 1 and len(num_prompt_list) > 1:
        if len(request_rate_list) == 1:
            request_rate_list = request_rate_list * len(num_prompt_list)
        if len(max_concurrency_list) == 1:
            max_concurrency_list = max_concurrency_list * len(num_prompt_list)
        max_len = max(len(request_rate_list), len(max_concurrency_list))
    elif len(num_prompt_list) != max_len and max_len > 0:
        raise ValueError("The number of prompts does not match the request-rate or max-concurrency")

    sweep_runs: list[dict[str, Any]] = []

    for i, (request_rate, num_prompts) in enumerate(zip(request_rate_list, num_prompt_list)):
        sweep_runs.append(
            {
                "params": build_diffusion_run_params(
                    params,
                    request_rate=request_rate,
                    num_prompts=num_prompts,
                    sweep_index=i,
                ),
                "num_prompts": num_prompts,
                "sweep_index": i,
                "request_rate": request_rate,
                "max_concurrency": None,
            }
        )

    for i, (max_concurrency, num_prompts) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        sweep_runs.append(
            {
                "params": build_diffusion_run_params(
                    params,
                    max_concurrency=max_concurrency,
                    num_prompts=num_prompts,
                    sweep_index=i,
                    request_rate="inf",
                ),
                "num_prompts": num_prompts,
                "sweep_index": i,
                "request_rate": None,
                "max_concurrency": max_concurrency,
            }
        )

    if not sweep_runs:
        default_num_prompts = num_prompt_list[0]
        sweep_runs.append(
            {
                "params": build_diffusion_run_params(params, num_prompts=default_num_prompts),
                "num_prompts": default_num_prompts,
                "sweep_index": None,
                "request_rate": None,
                "max_concurrency": None,
            }
        )

    return sweep_runs
