# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the compatibility re-export of YuanrongTransferEngineConnector."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reexport_symbol():
    from vllm_omni.distributed.omni_connectors.connectors import yuanrong_transfer_engine_connector as pkg

    assert "YuanrongTransferEngineConnector" in pkg.__all__
    assert pkg.YuanrongTransferEngineConnector.__name__ == "YuanrongTransferEngineConnector"


def test_direct_import_path():
    from vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector,
    )
    from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector as NpuConnector,
    )

    assert YuanrongTransferEngineConnector is NpuConnector
