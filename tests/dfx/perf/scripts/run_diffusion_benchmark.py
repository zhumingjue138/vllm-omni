"""
Performance benchmark CI runner for diffusion models.

This runner separates two concepts:

1. ``server_type``: how the serving process is started.
   Currently only ``vllm-omni`` is supported here.
2. ``benchmark_endpoint``: which serving API the benchmark client calls.
   Examples: ``/v1/chat/completions`` and ``/v1/videos``.

A config JSON file may be passed via --test-config-file. If omitted, every ``*.json`` under
``tests/dfx/perf/tests/`` is loaded and pytest ``-m`` filters by each case's ``mark``:
  pytest run_diffusion_benchmark.py -m "diffusion"
  pytest run_diffusion_benchmark.py --test-config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json

Optional: ``--assert-baseline`` compares metrics to the ``baseline`` block in each benchmark entry (default: off).

Optional JSON field ``mark`` is applied as pytest marks on that case via
``pytest.param`` (e.g. ``"mark": [{"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}}, "full_model", "diffusion"]``).

All benchmark results are written under BENCHMARK_RESULT_DIR (override via the
DIFFUSION_BENCHMARK_DIR environment variable). Each source JSON file gets one
aggregated ``diffusion_result_{config_stem}_{hardware}_{timestamp}.json`` (JSON array
of all runs from cases in that file). Bulk load without ``--test-config-file`` uses
the same per-file aggregation; ``-m`` only selects which cases run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_benchmark_pytest_params,
    get_runtime_resource_label,
    is_diffusion_perf_config,
    resolve_pytest_marks,
    resource_label_for_filename,
)
from tests.dfx.perf.helpers import (
    DiffusionBenchmarkSession,
    assert_diffusion_benchmark_result,
    build_omni_server_cli_args_from_diffusion_cfg,
    iter_diffusion_sweep_runs,
    normalize_repo_relative_path,
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

_DIFFUSION_SOURCE_CONFIG_KEY = "_source_config_file"
_PERF_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
_AGGREGATED_RESULT_FILES_BY_SOURCE: dict[str, Path] = {}


def _get_config_file_from_argv() -> str | None:
    """Read --test-config-file from sys.argv at import time so pytest parametrize can use it."""
    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


CONFIG_FILE_PATH = _get_config_file_from_argv()


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


def load_diffusion_benchmark_configs(
    config_path: str | None = None,
    *,
    config_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load one diffusion benchmark JSON, or merge all ``*.json`` under *config_dir*."""
    if config_path is not None:
        configs = load_configs(config_path)
        source = str(Path(config_path).resolve())
        for cfg in configs:
            cfg.setdefault(_DIFFUSION_SOURCE_CONFIG_KEY, source)
        return configs
    if config_dir is None:
        raise ValueError("load_diffusion_benchmark_configs requires config_path or config_dir")
    configs: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.json")):
        source = str(path.resolve())
        for cfg in load_configs(str(path)):
            cfg[_DIFFUSION_SOURCE_CONFIG_KEY] = source
            configs.append(cfg)
    if not configs:
        raise ValueError(f"No benchmark JSON files found under {config_dir}")
    return configs


if CONFIG_FILE_PATH is None:
    _all_configs = load_diffusion_benchmark_configs(config_dir=_PERF_TESTS_DIR)
    BENCHMARK_CONFIGS = [cfg for cfg in _all_configs if is_diffusion_perf_config(cfg)]
    print(
        f"No --test-config-file: loaded {len(BENCHMARK_CONFIGS)} diffusion case(s) from "
        f"{_PERF_TESTS_DIR}/*.json (skipped {len(_all_configs) - len(BENCHMARK_CONFIGS)} omni/tts; "
        f"use -m to filter, e.g. -m diffusion)"
    )
else:
    BENCHMARK_CONFIGS = load_diffusion_benchmark_configs(CONFIG_FILE_PATH)


def _normalized_source_path(source_file: str) -> str:
    return str(Path(source_file).resolve())


def _aggregated_result_file_for_source(source_file: str) -> Path:
    """One session aggregate per source JSON (same naming as single ``--test-config-file``)."""
    key = _normalized_source_path(source_file)
    if key not in _AGGREGATED_RESULT_FILES_BY_SOURCE:
        stem = Path(key).stem
        resource = resource_label_for_filename(get_runtime_resource_label())
        if resource:
            result_name = f"diffusion_result_{stem}_{resource}_{_SESSION_TIMESTAMP}.json"
        else:
            result_name = f"diffusion_result_{stem}_{_SESSION_TIMESTAMP}.json"
        _AGGREGATED_RESULT_FILES_BY_SOURCE[key] = BENCHMARK_RESULT_DIR / result_name
    return _AGGREGATED_RESULT_FILES_BY_SOURCE[key]


def _diffusion_session_for_source(source_file: str) -> DiffusionBenchmarkSession:
    return DiffusionBenchmarkSession(
        benchmark_script=BENCHMARK_SCRIPT,
        result_dir=BENCHMARK_RESULT_DIR,
        aggregated_result_file=_aggregated_result_file_for_source(source_file),
    )


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
            str_value = str(value)
            if key in {"deploy-config", "deploy_config"}:
                str_value = normalize_repo_relative_path(str_value)
            args.extend([flag, str_value])
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
        entry: dict[str, Any] = {
            "test_name": test_name,
            "server_type": server_type,
            "model": cfg["server_params"]["model"],
            "serve_args_dict": serve_args_dict,
            "serve_args": _build_serve_args(serve_args_dict),
            "benchmark_endpoint": cfg.get("benchmark_endpoint", cfg.get("benchmark_backend")),
            "server_params": cfg["server_params"],
            "mark": cfg.get("mark"),
        }
        if _DIFFUSION_SOURCE_CONFIG_KEY in cfg:
            entry[_DIFFUSION_SOURCE_CONFIG_KEY] = cfg[_DIFFUSION_SOURCE_CONFIG_KEY]
        result.append(entry)
    return result


def _test_param_mapping(configs: list[dict[str, Any]]) -> dict[str, list[dict]]:
    mapping: dict[str, list[dict]] = {}
    for cfg in configs:
        name = cfg["test_name"]
        mapping.setdefault(name, [])
        mapping[name].extend(cfg["benchmark_params"])
    return mapping


def _marks_by_test_name(configs: list[dict[str, Any]]) -> dict[str, list[pytest.MarkDecorator]]:
    return {str(cfg["test_name"]): resolve_pytest_marks(cfg.get("mark")) for cfg in configs}


def _paired_diffusion_benchmark_pytest_params(configs: list[dict[str, Any]]) -> list[Any]:
    """Paired params for ``run_diffusion_benchmark.py``; same shape as omni runner."""
    test_param_map = _test_param_mapping(configs)
    server_entries = [(cfg, cfg["test_name"]) for cfg in _unique_server_params(configs)]
    return create_paired_benchmark_pytest_params(server_entries, test_param_map, _marks_by_test_name(configs))


test_param_map = _test_param_mapping(BENCHMARK_CONFIGS)
paired_benchmark_params = _paired_diffusion_benchmark_pytest_params(BENCHMARK_CONFIGS)


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
def benchmark_params(request):
    """Benchmark params for the paired server/index parametrization."""
    test_name, param_index = request.param

    params_list = test_param_map.get(test_name, [])
    if not params_list:
        raise ValueError(f"No benchmark params for test: {test_name}")

    current = param_index + 1
    total = len(params_list)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": params_list[param_index]}


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "diffusion_server,benchmark_params",
    paired_benchmark_params,
    indirect=["diffusion_server", "benchmark_params"],
)
def test_diffusion_performance_benchmark(diffusion_server, benchmark_params, request):
    """Run the diffusion performance benchmark and verify request completion."""
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    server_cfg = getattr(diffusion_server, "server_cfg", {})
    sweep_runs = iter_diffusion_sweep_runs(params)
    source_file = str(
        server_cfg.get(
            _DIFFUSION_SOURCE_CONFIG_KEY,
            CONFIG_FILE_PATH or f"{_PERF_TESTS_DIR}/*.json",
        )
    )
    session = _diffusion_session_for_source(source_file)

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
            source_file=source_file,
            session=session,
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

        print(f"\n  Aggregated results: {_aggregated_result_file_for_source(source_file)}")
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
