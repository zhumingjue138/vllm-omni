import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    get_benchmark_params_for_server,
    is_diffusion_perf_config,
    load_benchmark_configs,
)
from tests.dfx.perf.helpers import (
    OmniBenchmarkSession,
    assert_omni_benchmark_result,
    build_omni_server_cli_args_from_tuple,
    iter_omni_sweep_runs,
    run_omni_benchmark,
)
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_DEFAULT_RESULT_DIR = Path(__file__).resolve().parent.parent / "results"
OMNI_BENCHMARK_RESULT_DIR = Path(os.environ.get("OMNI_BENCHMARK_DIR", str(_DEFAULT_RESULT_DIR)))
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def _get_config_file_from_argv() -> str | None:
    """Read ``--test-config-file`` from ``sys.argv`` at import time so parametrization can use it."""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


_PERF_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    _all_configs = load_benchmark_configs(config_dir=_PERF_TESTS_DIR)
    BENCHMARK_CONFIGS = [cfg for cfg in _all_configs if not is_diffusion_perf_config(cfg)]
    print(
        f"No --test-config-file: loaded {len(BENCHMARK_CONFIGS)} omni/tts case(s) from "
        f"{_PERF_TESTS_DIR}/*.json (skipped {len(_all_configs) - len(BENCHMARK_CONFIGS)} diffusion; "
        f"use -m to filter, e.g. -m tts)"
    )
else:
    BENCHMARK_CONFIGS = load_benchmark_configs(CONFIG_FILE_PATH)

_config_stem = Path(CONFIG_FILE_PATH).stem if CONFIG_FILE_PATH else "bulk"
AGGREGATED_RESULT_FILE = OMNI_BENCHMARK_RESULT_DIR / f"result_{_config_stem}_{_SESSION_TIMESTAMP}.json"
OMNI_BENCHMARK_SESSION = OmniBenchmarkSession(
    result_dir=OMNI_BENCHMARK_RESULT_DIR,
    aggregated_result_file=AGGREGATED_RESULT_FILE,
)

DEPLOY_CONFIGS_DIR = Path(__file__).parent.parent / "deploy"
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)
paired_benchmark_params = create_paired_omni_benchmark_pytest_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR)

_omni_server_lock = threading.Lock()


def _server_params_for_test_name(test_name: str) -> dict[str, Any]:
    for config in BENCHMARK_CONFIGS:
        if config.get("test_name") == test_name:
            return dict(config.get("server_params") or {})
    return {}


def _build_omni_bench_cli_args(
    params: dict[str, Any],
    *,
    host: str,
    port: int,
    test_name: str,
) -> list[str]:
    args = ["--host", host, "--port", str(port)]
    exclude_keys = {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "task",
        "enabled",
        "eval_phase",
        "trust_remote_code",
    }

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue

        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            json_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            args.extend([arg_name, json_str])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])

    server_params = _server_params_for_test_name(test_name)
    if server_params.get("trust_remote_code") or params.get("trust_remote_code"):
        args.append("--trust-remote-code")
    return args


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights."""
    with _omni_server_lock:
        test_name, model, stage_config_path, stage_overrides, extra_cli_args, use_omni = request.param

        print(f"Starting OmniServer with test: {test_name}, model: {model}")

        server_args = build_omni_server_cli_args_from_tuple(
            stage_config_path=stage_config_path,
            stage_overrides=stage_overrides,
            extra_cli_args=extra_cli_args,
            use_omni=use_omni,
        )
        with OmniServer(model, server_args, use_omni=use_omni) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


@pytest.fixture
def benchmark_params(request):
    """Benchmark parameters fixture; paired with ``omni_server`` via parametrization."""
    test_name, param_index = request.param

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)

    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")

    return {
        "test_name": test_name,
        "params": all_params[param_index],
    }


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "omni_server,benchmark_params",
    paired_benchmark_params,
    indirect=["omni_server", "benchmark_params"],
)
def test_performance_benchmark(omni_server, benchmark_params, request):
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    dataset_name = params.get("dataset_name", "")

    host = omni_server.host
    port = omni_server.port
    model = omni_server.model
    server_params = _server_params_for_test_name(test_name)

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    assert_baseline = request.config.getoption("--assert-baseline", default=False)
    base_args = _build_omni_bench_cli_args(params, host=host, port=port, test_name=test_name)

    for sweep_run in iter_omni_sweep_runs(params):
        run_params = sweep_run["params"]
        num_prompt = sweep_run["num_prompts"]
        sweep_args = list(base_args)
        if sweep_run["request_rate"] is not None:
            sweep_args += ["--request-rate", str(sweep_run["request_rate"]), "--num-prompts", str(num_prompt)]
        else:
            sweep_args += [
                "--max-concurrency",
                str(sweep_run["max_concurrency"]),
                "--num-prompts",
                str(num_prompt),
                "--request-rate",
                "inf",
            ]

        result = run_omni_benchmark(
            args=sweep_args,
            test_name=test_name,
            flow=sweep_run["flow"],
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            session=OMNI_BENCHMARK_SESSION,
            server_params=server_params,
            benchmark_params=run_params,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
        )
        assert_omni_benchmark_result(
            result,
            params,
            num_prompt,
            assert_baseline=assert_baseline,
            sweep_index=sweep_run["sweep_index"],
            max_concurrency=sweep_run["max_concurrency"],
            request_rate=sweep_run["request_rate"],
        )
