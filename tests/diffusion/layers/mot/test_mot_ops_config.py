# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU unit tests for MoT ops config loading and RMSNorm input validation."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.layers.mot.ops import mot_gemm, mot_rms_norm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def clear_mot_config_cache():
    mot_gemm._load_mot_config_file.cache_clear()
    mot_gemm.get_mot_configs.cache_clear()
    yield
    mot_gemm._load_mot_config_file.cache_clear()
    mot_gemm.get_mot_configs.cache_clear()


class TestMotGemmConfigHelpers:
    def test_build_config_filename(self):
        assert mot_gemm.build_config_filename("H100", "w16a16") == "device_name=H100,dtype=w16a16.json"

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.torch.cuda.get_device_name", return_value="NVIDIA H800")
    def test_get_device_name_strips_prefix_and_aliases(self, _mock_device):
        assert mot_gemm.get_device_name() == "H100"

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.torch.cuda.get_device_name", return_value="NVIDIA A800-SXM4")
    def test_get_device_name_a800_alias(self, _mock_device):
        assert mot_gemm.get_device_name() == "A100"

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.torch.cuda.get_device_name", return_value="NVIDIA H800-SXM5")
    def test_get_device_name_h800_alias_strips_variant_suffix(self, _mock_device):
        assert mot_gemm.get_device_name() == "H100"

    def test_get_mot_default_config_small_m(self):
        cfg = mot_gemm.get_mot_default_config(M=8, N=128, K=64)
        assert cfg["BLOCK_SIZE_M"] == 16
        assert cfg["BLOCK_SIZE_N"] == 64

    def test_get_mot_default_config_fp8_block_quant(self):
        cfg = mot_gemm.get_mot_default_config(
            M=128,
            N=256,
            K=128,
            dtype="fp8_w8a8",
            block_quant_shape=[128, 64],
        )
        assert cfg["BLOCK_SIZE_N"] == 128
        assert cfg["BLOCK_SIZE_K"] == 64

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.get_device_name", return_value="TestGPU")
    def test_load_mot_config_from_env_folder(self, _mock_device, tmp_path, monkeypatch):
        filename = mot_gemm.build_config_filename("TestGPU", "w16a16")
        config_data = {"64_128": {"128": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}}}
        config_path = tmp_path / filename
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        monkeypatch.setenv(mot_gemm._ENV_CONFIG_FOLDER, str(tmp_path))

        loaded = mot_gemm._load_mot_config_file("w16a16")
        assert loaded == config_data

        configs = mot_gemm.get_mot_configs(K=64, N=128, dtype_str="w16a16")
        assert configs is not None
        assert configs[128]["BLOCK_SIZE_M"] == 32

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.get_device_name", return_value="MissingGPU")
    def test_get_mot_configs_returns_none_without_file(self, _mock_device):
        assert mot_gemm.get_mot_configs(K=1, N=1, dtype_str="w16a16") is None

    @patch("vllm_omni.diffusion.layers.mot.ops.mot_gemm.get_mot_configs", return_value=None)
    def test_get_best_mot_config_falls_back_to_default(self, _mock_configs):
        m_key, cfg = mot_gemm.get_best_mot_config(M=256, N=128, K=64, dtype_str="w16a16")
        assert m_key == -1
        assert cfg["BLOCK_SIZE_M"] == 64

    @patch(
        "vllm_omni.diffusion.layers.mot.ops.mot_gemm.get_mot_configs",
        return_value={64: {"BLOCK_SIZE_M": 32}, 256: {"BLOCK_SIZE_M": 64}},
    )
    def test_get_best_mot_config_picks_closest_m(self, _mock_configs):
        m_key, cfg = mot_gemm.get_best_mot_config(M=200, N=128, K=64)
        assert m_key == 256
        assert cfg["BLOCK_SIZE_M"] == 64


class TestMotRmsNormValidation:
    def test_mismatched_text_weight_dim_raises(self):
        inp = torch.randn(4, 8)
        text_w = torch.randn(4)
        vae_w = torch.randn(8)
        text_idx = torch.tensor([0, 1])
        vae_idx = torch.tensor([2, 3])
        with pytest.raises(AssertionError, match="Text weight dimension"):
            mot_rms_norm.mot_rms_norm(inp, text_w, vae_w, text_idx, vae_idx)

    def test_token_count_mismatch_raises(self):
        inp = torch.randn(3, 8)
        weight = torch.randn(8)
        text_idx = torch.tensor([0, 1])
        vae_idx = torch.tensor([2, 3])
        with pytest.raises(AssertionError, match="batched_token_length"):
            mot_rms_norm.mot_rms_norm(inp, weight, weight, text_idx, vae_idx)

    def test_head_norm_requires_3d_input(self):
        inp = torch.randn(4, 8)
        weight = torch.randn(8)
        text_idx = torch.tensor([0, 1])
        vae_idx = torch.tensor([2, 3])
        with pytest.raises(AssertionError, match="head_norm=True"):
            mot_rms_norm.mot_rms_norm(inp, weight, weight, text_idx, vae_idx, head_norm=True)

    def test_head_norm_per_head_weight_mismatch_raises(self):
        inp = torch.randn(4, 2, 8)
        text_w = torch.randn(3, 8)
        vae_w = torch.randn(3, 8)
        text_idx = torch.tensor([0, 1])
        vae_idx = torch.tensor([2, 3])
        with pytest.raises(AssertionError, match="num of heads"):
            mot_rms_norm.mot_rms_norm(inp, text_w, vae_w, text_idx, vae_idx, head_norm=True)
