# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Conftest for e2e online_serving tests.

Loads L5 GPU monitor conftest so that GPU_MONITOR=1 pytest ... under
this directory uses L5 monitor hooks and gpu_monitor_data_root fixture.
"""
import importlib.util
from pathlib import Path

_l5_conftest_path = Path(__file__).resolve().parent / "L5" / "conftest.py"
if _l5_conftest_path.is_file():
    _spec = importlib.util.spec_from_file_location("l5_conftest", _l5_conftest_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
