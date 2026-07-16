"""
Performance benchmark CI runner for diffusion models.

This runner separates two concepts:

1. ``server_type``: how the serving process is started.
   Currently only ``vllm-omni`` is supported here.
2. ``benchmark_endpoint``: which serving API the benchmark client calls.
   Examples: ``/v1/chat/completions`` and ``/v1/videos``.

A config JSON file is REQUIRED via --test-config-file:
  pytest run_diffusion_benchmark.py --test-config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json

Optional: ``--assert-baseline`` compares metrics to the ``baseline`` block in each benchmark entry (default: off).

All benchmark results for a session are consolidated into a single JSON file under
BENCHMARK_RESULT_DIR (override via the DIFFUSION_BENCHMARK_DIR environment variable).
Each entry in the file contains the test metadata (test_name, endpoint, benchmark_params,
timestamp) together with the raw metrics returned by the benchmark script.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from tests.dfx.perf.helpers import (
    DiffusionBenchmarkSession,
    assert_diffusion_benchmark_result,
    build_omni_server_cli_args_from_diffusion_cfg,
    iter_diffusion_sweep_runs,
    resolve_benchmark_endpoint,
    run_diffusion_benchmark,
)
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model, pytest.mark.local_model]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")


# ---------------------------------------------------------------------------
# Inline field processing (diffusion JSON dialect)
# ---------------------------------------------------------------------------
def _process_inline_fields(obj: Any, parent_key: str = "") -> None:
    """Recursively process '*-inline' fields into temp files."""

    if isinstance(obj, list):
        for item in obj:
            _process_inline_fields(item, parent_key)
        return

    if not isinstance(obj, dict):
        return

    import atexit

    import yaml

    for key in list(obj.keys()):
        value = obj[key]

        if not key.endswith("-inline"):
            _process_inline_fields(value, key)
            continue

        base_key = key[:-7]
        full_key = f"{parent_key}.{key}" if parent_key else key

        try:
            if not isinstance(value, dict):
                raise ValueError("must be a dict")

            file_type = value.get("type")
            content = value.get("content")

            if file_type not in {"yaml", "jsonl"}:
                raise ValueError(f"invalid type: {file_type}")

            fd, path = tempfile.mkstemp(
                suffix=f".{file_type}",
                prefix=f"{base_key}_",
            )

            atexit.register(Path(path).unlink, missing_ok=True)

            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if file_type == "jsonl":
                    items = content if isinstance(content, list) else [content]
                    f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
                else:
                    yaml.dump(
                        content,
                        f,
                        allow_unicode=True,
                        sort_keys=False,
                        indent=2,
                    )

            obj[base_key] = path
            del obj[key]

        except Exception as e:
            print(f"Warning: failed processing '{full_key}': {e}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_RESULT_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_RESULT_DIR = Path(os.environ.get("DIFFUSION_BENCHMARK_DIR", str(_DEFAULT_RESULT_DIR)))

BENCHMARK_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent.parent / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"
)

_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_server_lock = threading.Lock()


def _get_config_file_from_argv() -> str | None:
    """Read --test-config-file from sys.argv at import time so pytest parametrize can use it."""
    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    print("No config file provided, using default config file: tests/dfx/perf/tests/test_qwen_image_vllm_omni.json")
    CONFIG_FILE_PATH = "tests/dfx/perf/tests/test_qwen_image_vllm_omni.json"


# ---------------------------------------------------------------------------
# Config loading (diffusion JSON dialect)
# ---------------------------------------------------------------------------


def _resolve_refs(configs: list[dict[str, Any]], config_dir: Path) -> list[dict[str, Any]]:
    """Resolve {"$ref": "filename.json"} in benchmark_params fields."""
    for cfg in configs:
        bp = cfg.get("benchmark_params")
        if isinstance(bp, dict) and "$ref" in bp:
            ref_path = config_dir / bp["$ref"]
            try:
                with open(ref_path, encoding="utf-8") as f:
                    cfg["benchmark_params"] = json.load(f)
            except FileNotFoundError:
                raise ValueError(f"benchmark_params $ref not found: {ref_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing error in {ref_path}: {e}")
    return configs


def load_configs(config_path: str) -> list[dict[str, Any]]:
    """Load benchmark configs from JSON file and process inline fields."""
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)
        configs = _resolve_refs(configs, abs_path.parent)
        _process_inline_fields(configs)
        return configs
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


def _build_serve_args(serve_args_dict: dict[str, Any]) -> list[str]:
    """Convert a serve_args dict from diffusion test.json into a flat CLI argument list."""
    args: list[str] = []
    for key, value in serve_args_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            args.extend([flag, str(value)])
    return args


def _unique_server_params(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one server-config dict per unique test_name."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for cfg in configs:
        test_name = cfg["test_name"]
        if test_name in seen:
            continue
        seen.add(test_name)
        server_type = cfg.get("server_type", "vllm-omni")
        if server_type != "vllm-omni":
            raise ValueError(f"Unsupported server_type in config: {server_type}")
        serve_args_dict = cfg["server_params"].get("serve_args", {})
        result.append(
            {
                "test_name": test_name,
                "server_type": server_type,
                "model": cfg["server_params"]["model"],
                "serve_args_dict": serve_args_dict,
                "serve_args": _build_serve_args(serve_args_dict),
                "benchmark_endpoint": cfg.get("benchmark_endpoint", cfg.get("benchmark_backend")),
                "server_params": cfg["server_params"],
            }
        )
    return result


def _test_param_mapping(configs: list[dict[str, Any]]) -> dict[str, list[dict]]:
    mapping: dict[str, list[dict]] = {}
    for cfg in configs:
        name = cfg["test_name"]
        mapping.setdefault(name, [])
        mapping[name].extend(cfg["benchmark_params"])
    return mapping


BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)

_config_stem = Path(CONFIG_FILE_PATH).stem
AGGREGATED_RESULT_FILE = BENCHMARK_RESULT_DIR / f"diffusion_result_{_config_stem}_{_SESSION_TIMESTAMP}.json"
DIFFUSION_BENCHMARK_SESSION = DiffusionBenchmarkSession(
    benchmark_script=BENCHMARK_SCRIPT,
    result_dir=BENCHMARK_RESULT_DIR,
    aggregated_result_file=AGGREGATED_RESULT_FILE,
)

server_params = _unique_server_params(BENCHMARK_CONFIGS)
test_param_map = _test_param_mapping(BENCHMARK_CONFIGS)
benchmark_indices: list[int] = list(range(max(len(v) for v in test_param_map.values())))


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diffusion_server(request):
    """Start one vLLM-Omni server per unique test configuration."""
    with _server_lock:
        server_cfg: dict[str, Any] = request.param
        test_name = server_cfg["test_name"]
        server_type = server_cfg["server_type"]
        model = server_cfg["model"]

        print(f"\nStarting {server_type} server for test: {test_name}")

        server_args = build_omni_server_cli_args_from_diffusion_cfg(server_cfg)
        with OmniServer(model, server_args, use_omni=True) as server:
            server.test_name = test_name
            server.server_cfg = server_cfg
            print(f"{server_type} server started successfully")
            yield server
            print(f"{server_type} server stopping…")

    print(f"{server_type} server stopped")


@pytest.fixture
def benchmark_params(request, diffusion_server):
    """Yield the benchmark params dict for the current (server, index) pair."""
    param_index: int = request.param
    test_name = diffusion_server.test_name

    params_list = test_param_map.get(test_name, [])
    if not params_list:
        raise ValueError(f"No benchmark params for test: {test_name}")
    if param_index >= len(params_list):
        pytest.skip(f"Param index {param_index} out of range for {test_name} (has {len(params_list)} params)")

    current = param_index + 1
    total = len(params_list)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": params_list[param_index]}


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "diffusion_server",
    server_params,
    ids=[p["test_name"] for p in server_params],
    indirect=True,
)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_diffusion_performance_benchmark(diffusion_server, benchmark_params, request):
    """Run the diffusion performance benchmark and verify request completion."""
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    server_cfg = getattr(diffusion_server, "server_cfg", {})
    sweep_runs = iter_diffusion_sweep_runs(params)

    for sweep_run in sweep_runs:
        endpoint = resolve_benchmark_endpoint(server_cfg, sweep_run["params"])
        result = run_diffusion_benchmark(
            host=diffusion_server.host,
            port=diffusion_server.port,
            model=diffusion_server.model,
            params=sweep_run["params"],
            test_name=test_name,
            endpoint=endpoint,
            server_cfg=server_cfg,
            source_file=cast(str, CONFIG_FILE_PATH),
            session=DIFFUSION_BENCHMARK_SESSION,
        )

        print(f"\n{'=' * 60}")
        print(f"Results for {test_name} (server={server_cfg.get('server_type', 'vllm-omni')}, endpoint={endpoint}):")
        for key in (
            "throughput_qps",
            "latency_mean",
            "latency_median",
            "latency_p50",
            "latency_p99",
            "peak_memory_mb_max",
            "peak_memory_mb_mean",
            "peak_memory_mb_median",
        ):
            if key in result:
                print(f"  {key}: {result[key]:.4f}")

        print(f"\n  Aggregated results: {AGGREGATED_RESULT_FILE}")
        print("=" * 60)

        assert_baseline = request.config.getoption("--assert-baseline", default=False)

        assert_diffusion_benchmark_result(
            result,
            params,
            sweep_run["num_prompts"],
            sweep_index=sweep_run["sweep_index"],
            max_concurrency=sweep_run["max_concurrency"],
            request_rate=sweep_run["request_rate"],
            assert_baseline=assert_baseline,
        )
