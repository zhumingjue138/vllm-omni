# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU unit tests for TeaCache extractor registry and early validation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.extractors import (
    EXTRACTOR_REGISTRY,
    CacheContext,
    extract_flux2_context,
    extract_flux_context,
    extract_qwen_context,
    extract_stable_audio_context,
    extract_zimage_context,
    get_extractor,
    register_extractor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestExtractorRegistry:
    def test_get_extractor_known_type(self):
        fn = get_extractor("FluxTransformer2DModel")
        assert fn is extract_flux_context

    def test_get_extractor_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_extractor("TotallyUnknownModel")

    def test_register_extractor_custom_type(self):
        sentinel = MagicMock(return_value="ctx")

        register_extractor("CustomTestModel", sentinel)
        try:
            assert get_extractor("CustomTestModel") is sentinel
            assert EXTRACTOR_REGISTRY["CustomTestModel"] is sentinel
        finally:
            EXTRACTOR_REGISTRY.pop("CustomTestModel", None)


class TestExtractorEarlyValidation:
    def test_qwen_requires_transformer_blocks(self):
        module = SimpleNamespace(transformer_blocks=[])
        with pytest.raises(ValueError, match="transformer_blocks"):
            extract_qwen_context(
                module,
                hidden_states=torch.randn(1, 4, 8),
                encoder_hidden_states=torch.randn(1, 2, 8),
                encoder_hidden_states_mask=None,
                timestep=torch.tensor([1.0]),
                img_shapes=[(1, 2, 2)],
                txt_seq_lens=[2],
            )

    def test_flux2_requires_transformer_blocks(self):
        module = SimpleNamespace()
        with pytest.raises(ValueError, match="transformer_blocks"):
            extract_flux2_context(
                module,
                hidden_states=torch.randn(1, 4, 8),
                encoder_hidden_states=torch.randn(1, 2, 8),
                timestep=torch.tensor([0.5]),
                img_ids=torch.zeros(1, 4, 3),
                txt_ids=torch.zeros(1, 2, 3),
            )

    def test_zimage_requires_layers(self):
        module = SimpleNamespace(layers=[])
        with pytest.raises(ValueError, match="main transformer layers"):
            extract_zimage_context(
                module,
                x=[torch.randn(1, 4, 4)],
                t=torch.tensor([1.0]),
                cap_feats=[torch.randn(2, 8)],
            )

    def test_stable_audio_requires_transformer_blocks(self):
        module = SimpleNamespace(transformer_blocks=[], dtype=torch.float32)
        with pytest.raises(ValueError, match="transformer_blocks"):
            extract_stable_audio_context(
                module,
                hidden_states=torch.randn(1, 8, 4),
                encoder_hidden_states=torch.randn(1, 4, 8),
                global_hidden_states=torch.randn(1, 8),
                timestep=torch.tensor([0.5]),
            )


class TestCacheContextValidate:
    def test_validate_rejects_missing_modulated_input(self):
        ctx = CacheContext(
            modulated_input=None,
            hidden_states=torch.randn(1, 4),
            encoder_hidden_states=None,
            temb=None,
            run_transformer_blocks=lambda: (torch.randn(1, 4),),
            postprocess=lambda h: h,
        )
        with pytest.raises(TypeError, match="modulated_input"):
            ctx.validate()

    def test_validate_accepts_complete_context(self):
        modulated = torch.randn(2, 4)
        hidden = torch.randn(2, 4)
        ctx = CacheContext(
            modulated_input=modulated,
            hidden_states=hidden,
            encoder_hidden_states=None,
            temb=torch.randn(2, 8),
            run_transformer_blocks=lambda: (hidden,),
            postprocess=lambda h: h,
        )
        ctx.validate()
