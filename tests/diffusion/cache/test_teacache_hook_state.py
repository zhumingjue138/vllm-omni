# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for TeaCache config, state, hook, and extractor registry."""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.hooks import HookRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestTeaCacheConfig:
    def test_default_qwen_coefficients(self):
        config = TeaCacheConfig(transformer_type="QwenImageTransformer2DModel")
        assert len(config.coefficients) == 5

    def test_invalid_rel_l1_thresh_raises(self):
        with pytest.raises(ValueError, match="rel_l1_thresh must be positive"):
            TeaCacheConfig(rel_l1_thresh=0.0)

    def test_unknown_transformer_type_raises(self):
        with pytest.raises(KeyError, match="Cannot find coefficients"):
            TeaCacheConfig(transformer_type="UnknownTransformer")

    def test_invalid_coefficient_length_raises(self):
        with pytest.raises(ValueError, match="exactly 5 elements"):
            TeaCacheConfig(coefficients=[1.0, 2.0, 3.0])


class TestTeaCacheState:
    def test_reset_clears_cached_values(self):
        state = TeaCacheState()
        state.cnt = 3
        state.accumulated_rel_l1_distance = 0.42
        state.previous_modulated_input = torch.tensor([1.0])
        state.previous_residual = torch.tensor([2.0])
        state.previous_residual_encoder = torch.tensor([3.0])

        state.reset()

        assert state.cnt == 0
        assert state.accumulated_rel_l1_distance == 0.0
        assert state.previous_modulated_input is None
        assert state.previous_residual is None
        assert state.previous_residual_encoder is None


class TestCacheContextValidation:
    def test_validate_rejects_non_tensor_modulated_input(self):
        ctx = CacheContext(
            modulated_input=Mock(),  # type: ignore[arg-type]
            hidden_states=torch.tensor([1.0]),
            encoder_hidden_states=None,
            temb=torch.tensor([0.0]),
            run_transformer_blocks=lambda: (torch.tensor([1.0]),),
            postprocess=lambda h: h,
        )
        with pytest.raises(TypeError, match="modulated_input must be torch.Tensor"):
            ctx.validate()

    def test_validate_rejects_batch_size_mismatch(self):
        ctx = CacheContext(
            modulated_input=torch.tensor([[1.0], [2.0]]),
            hidden_states=torch.tensor([[1.0]]),
            encoder_hidden_states=None,
            temb=torch.tensor([0.0]),
            run_transformer_blocks=lambda: (torch.tensor([1.0]),),
            postprocess=lambda h: h,
        )
        with pytest.raises(ValueError, match="Batch size mismatch"):
            ctx.validate()


class TestGetExtractor:
    def test_known_transformer_type(self):
        extractor = get_extractor("FluxTransformer2DModel")
        assert callable(extractor)

    def test_unknown_transformer_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_extractor("TotallyUnknownTransformer")


class TestTeaCacheHook:
    def _hook(self, rel_l1_thresh: float = 1.0) -> TeaCacheHook:
        config = TeaCacheConfig(
            rel_l1_thresh=rel_l1_thresh,
            coefficients=[0.0, 1.0, 0.0, 0.0, 0.0],
            transformer_type="QwenImageTransformer2DModel",
        )
        return TeaCacheHook(config)

    def test_first_step_always_computes(self):
        hook = self._hook()
        state = TeaCacheState()
        should_compute = hook._should_compute_full_transformer(state, torch.tensor([1.0]))
        assert should_compute is True
        assert state.accumulated_rel_l1_distance == 0.0

    def test_similar_inputs_accumulate_below_threshold(self):
        hook = self._hook(rel_l1_thresh=10.0)
        state = TeaCacheState()
        state.cnt = 1
        state.previous_modulated_input = torch.tensor([1.0, 1.0])

        should_compute = hook._should_compute_full_transformer(state, torch.tensor([1.01, 0.99]))
        assert should_compute is False

    def test_large_distance_resets_accumulator_and_computes(self):
        hook = self._hook(rel_l1_thresh=0.01)
        state = TeaCacheState()
        state.cnt = 1
        state.accumulated_rel_l1_distance = 0.005
        state.previous_modulated_input = torch.tensor([1.0, 1.0])

        should_compute = hook._should_compute_full_transformer(state, torch.tensor([10.0, 10.0]))
        assert should_compute is True
        assert state.accumulated_rel_l1_distance == 0.0

    @patch("vllm_omni.diffusion.cache.teacache.hook.get_extractor")
    def test_new_forward_uses_cached_residual(self, mock_get_extractor: Mock):
        hidden = torch.tensor([[1.0, 2.0]])
        cached_residual = torch.tensor([[0.5, 0.5]])

        def run_blocks() -> tuple[torch.Tensor, ...]:
            raise AssertionError("transformer blocks should not run on cache hit")

        ctx = CacheContext(
            modulated_input=torch.tensor([[1.0, 1.0]]),
            hidden_states=hidden.clone(),
            encoder_hidden_states=None,
            temb=torch.tensor([0.0]),
            run_transformer_blocks=run_blocks,
            postprocess=lambda h: h,
        )
        mock_get_extractor.return_value = lambda module, **kwargs: ctx

        hook = self._hook(rel_l1_thresh=10.0)
        module = Mock(do_true_cfg=False)
        hook.initialize_hook(module)

        hook.state_manager.set_context("teacache_positive")
        state = hook.state_manager.get_state()
        state.cnt = 1
        state.previous_modulated_input = torch.tensor([[1.0, 1.0]])
        state.previous_residual = cached_residual

        output = hook.new_forward(module)
        assert torch.allclose(output, hidden + cached_residual)

    def test_apply_teacache_hook_registers_hook(self):
        class DummyModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        module = DummyModule()
        config = TeaCacheConfig(transformer_type="QwenImageTransformer2DModel")

        apply_teacache_hook(module, config)

        registry = HookRegistry.get_or_create(module)
        assert registry.get_hook("teacache") is not None
