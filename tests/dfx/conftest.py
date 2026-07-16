import json
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.stage_config import modify_stage_config

# Backward-compatible alias for stability and other omni DFX suites.
from tests.dfx.perf.helpers import run_omni_benchmark as run_benchmark  # noqa: F401


def load_configs(config_path: str) -> list[dict[str, Any]]:
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)

        return configs

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


def _modify_stage(default_path: str, updates: dict[str, Any] | None, deletes: dict[str, Any] | None) -> str:
    kwargs: dict[str, Any] = {}
    if updates is not None:
        kwargs["updates"] = updates
    if deletes is not None:
        kwargs["deletes"] = deletes
    if kwargs:
        return modify_stage_config(default_path, **kwargs)
    return default_path


def _build_serve_args(serve_args: Any) -> list[str]:
    """Convert server_params.serve_args to a flat CLI args list."""
    if serve_args is None:
        return []
    if isinstance(serve_args, list):
        return [str(item) for item in serve_args]
    if not isinstance(serve_args, dict):
        raise TypeError(f"serve_args must be dict/list/None, got {type(serve_args).__name__}")

    args: list[str] = []
    for key, value in serve_args.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
            continue
        args.extend([flag, str(value)])
    return args


def create_unique_server_params(
    configs: list[dict[str, Any]],
    stage_configs_dir: Path,
) -> list[tuple[str, str, str | None, str | None, tuple[str, ...], bool]]:
    """Return one row per unique server configuration.

    ``(test_name, model, deploy_yaml_path, stage_overrides_json, extra_cli_args, use_omni)``.

    JSON ``server_params.serve_args`` (dict/list) is expanded via ``_build_serve_args``
    and **prepended** to ``extra_cli_args`` so perf / stability ``omni_server`` fixtures
    stay identical to main while still honoring ``serve_args`` in benchmark JSON.
    """
    unique_params: list[tuple[str, str, str | None, str | None, tuple[str, ...], bool]] = []
    seen: set[tuple[str, str, str | None, str | None, tuple[str, ...], bool]] = set()
    for config in configs:
        test_name = config["test_name"]
        server_params = config["server_params"]
        model = server_params["model"]
        stage_config_name = server_params.get("stage_config_name")
        if stage_config_name:
            stage_config_path = str(stage_configs_dir / stage_config_name)
            delete = server_params.get("delete", None)
            update = server_params.get("update", None)
            stage_config_path = _modify_stage(stage_config_path, update, delete)
        else:
            stage_config_path = None

        stage_overrides = server_params.get("stage_overrides")
        stage_overrides_json = json.dumps(stage_overrides) if stage_overrides else None

        serve_flat = _build_serve_args(server_params.get("serve_args"))
        raw_extra = tuple(server_params.get("extra_cli_args") or ())
        extra_cli_args = tuple(serve_flat) + raw_extra
        use_omni = bool(server_params.get("use_omni", True))

        server_param = (
            test_name,
            model,
            stage_config_path,
            stage_overrides_json,
            extra_cli_args,
            use_omni,
        )
        if server_param not in seen:
            seen.add(server_param)
            unique_params.append(server_param)

    return unique_params


def create_test_parameter_mapping(configs: list[dict[str, Any]]) -> dict[str, dict]:
    mapping = {}
    for config in configs:
        test_name = config["test_name"]
        if test_name not in mapping:
            mapping[test_name] = {
                "test_name": test_name,
                "benchmark_params": [],
            }
        for entry in config["benchmark_params"]:
            # Skip disabled entries
            if not entry.get("enabled", True):
                continue
            mapping[test_name]["benchmark_params"].append(entry)
    return mapping


def get_benchmark_params_for_server(test_name: str, server_to_benchmark_mapping: dict[str, dict]) -> list:
    if test_name not in server_to_benchmark_mapping:
        return []
    return server_to_benchmark_mapping[test_name]["benchmark_params"]


def create_benchmark_indices(
    benchmark_configs: list[dict[str, Any]],
    server_to_benchmark_mapping: dict[str, dict],
) -> list[tuple[str, int]]:
    indices = []
    seen = set()
    for config in benchmark_configs:
        test_name = config["test_name"]
        if test_name not in seen:
            seen.add(test_name)
            params_list = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)
            for idx in range(len(params_list)):
                indices.append((test_name, idx))

    return indices


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register shared CLI options for DFX benchmark suites."""
    parser.addoption(
        "--test-config-file",
        action="store",
        default=None,
        help=("Path to benchmark config JSON. Example: --test-config-file tests/dfx/perf/tests/test_tts.json"),
    )
    parser.addoption(
        "--assert-baseline",
        action="store_true",
        default=False,
        help=(
            "When set, omni/diffusion perf runners compare metrics against the baseline block in the JSON config "
            "(default: off)."
        ),
    )
