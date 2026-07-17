# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for TeaCache coefficient estimation utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    BagelAdapter,
    DataCollectionHook,
    DefaultAdapter,
    TeaCacheCoefficientEstimator,
    calculate_relative_l1,
    estimate_teacache_coefficients,
)
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestCalculateRelativeL1:
    def test_identical_arrays_zero_distance(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert calculate_relative_l1(arr, arr) == pytest.approx(0.0)

    def test_known_relative_distance(self):
        current = np.array([1.0, 1.0])
        nxt = np.array([2.0, 1.0])
        assert calculate_relative_l1(current, nxt) == pytest.approx(0.5)


class TestEstimateTeacacheCoefficients:
    def _trajectory(self, values: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
        traj: list[tuple[np.ndarray, np.ndarray]] = []
        for value in values:
            arr = np.array([[value]], dtype=np.float64)
            traj.append((arr, arr * 2.0))
        return traj

    def test_returns_polynomial_coefficients(self):
        collected = [self._trajectory([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        coeffs = estimate_teacache_coefficients(collected, poly_order=2)
        assert len(coeffs) == 3
        assert all(isinstance(c, float) for c in coeffs)

    def test_skips_single_step_trajectory(self):
        collected = [self._trajectory([1.0])]
        coeffs = estimate_teacache_coefficients(collected, poly_order=1)
        assert coeffs == pytest.approx([0.0, 0.0])


class TestDataCollectionHook:
    def _make_context(self, modulated: float, output: float) -> CacheContext:
        modulated_tensor = torch.tensor([modulated], dtype=torch.float32)
        hidden = torch.tensor([output], dtype=torch.float32)

        def run_blocks() -> tuple[torch.Tensor, ...]:
            return (hidden,)

        def postprocess(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        return CacheContext(
            modulated_input=modulated_tensor,
            hidden_states=hidden,
            encoder_hidden_states=None,
            temb=torch.tensor([0.0]),
            run_transformer_blocks=run_blocks,
            postprocess=postprocess,
        )

    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.get_extractor")
    def test_new_forward_records_trajectory(self, mock_get_extractor: Mock):
        mock_get_extractor.return_value = lambda module, **kwargs: self._make_context(1.0, 2.0)
        hook = DataCollectionHook("FluxTransformer2DModel")
        module = Mock()
        hook.initialize_hook(module)

        hook.start_collection()
        hook.new_forward(module)
        trajectory = hook.stop_collection()

        assert len(trajectory) == 1
        modulated, model_output = trajectory[0]
        assert modulated.shape == (1,)
        assert model_output.shape == (1,)
        assert model_output[0] == pytest.approx(2.0)

    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.get_extractor")
    def test_start_collection_clears_previous_trajectory(self, mock_get_extractor: Mock):
        mock_get_extractor.return_value = lambda module, **kwargs: self._make_context(1.0, 2.0)
        hook = DataCollectionHook("FluxTransformer2DModel")
        hook.initialize_hook(Mock())

        hook.start_collection()
        hook.new_forward(Mock())
        hook.start_collection()
        trajectory = hook.stop_collection()
        assert trajectory == []


class TestAdapters:
    def test_bagel_get_transformer(self):
        pipeline = Mock()
        pipeline.bagel = Mock()
        transformer, transformer_type = BagelAdapter.get_transformer(pipeline)
        assert transformer is pipeline.bagel
        assert transformer_type == "Bagel"

    def test_default_adapter_requires_model_class_name(self):
        with pytest.raises(ValueError, match="doesn't have a set class name"):
            DefaultAdapter.load_pipeline("dummy-model", "cpu", torch.float32)


class TestTeaCacheCoefficientEstimator:
    def test_unsupported_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            TeaCacheCoefficientEstimator(model_path="dummy", model_type="NotARealModel")

    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.install_hook")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.get_transformer")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.load_pipeline")
    def test_collect_from_prompt_appends_trajectory(
        self,
        mock_load_pipeline: Mock,
        mock_get_transformer: Mock,
        mock_install_hook: Mock,
    ):
        mock_pipeline = Mock()
        mock_load_pipeline.return_value = mock_pipeline
        mock_transformer = Mock()
        mock_transformer.__class__.__name__ = "Bagel"
        mock_get_transformer.return_value = (mock_transformer, "Bagel")

        estimator = TeaCacheCoefficientEstimator(model_path="dummy", model_type="Bagel")
        estimator.hook.current_trajectory = [(np.array([1.0]), np.array([2.0]))]

        with (
            patch.object(estimator.hook, "start_collection"),
            patch.object(estimator.hook, "stop_collection", return_value=[(np.array([3.0]), np.array([4.0]))]),
            patch("torch.accelerator.empty_cache"),
        ):
            estimator.collect_from_prompt("a cat")

        mock_pipeline.forward.assert_called_once()
        assert len(estimator.collected_data) == 1
        assert estimator.collected_data[0][0][0] == pytest.approx(3.0)

    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.install_hook")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.get_transformer")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.load_pipeline")
    def test_estimate_without_data_raises(
        self,
        mock_load_pipeline: Mock,
        mock_get_transformer: Mock,
        mock_install_hook: Mock,
    ):
        mock_load_pipeline.return_value = Mock()
        mock_get_transformer.return_value = (Mock(), "Bagel")

        estimator = TeaCacheCoefficientEstimator(model_path="dummy", model_type="Bagel")
        with pytest.raises(RuntimeError, match="No data collected"):
            estimator.estimate()

    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.install_hook")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.get_transformer")
    @patch("vllm_omni.diffusion.cache.teacache.coefficient_estimator.BagelAdapter.load_pipeline")
    def test_estimate_delegates_to_polyfit(
        self,
        mock_load_pipeline: Mock,
        mock_get_transformer: Mock,
        mock_install_hook: Mock,
    ):
        mock_load_pipeline.return_value = Mock()
        mock_get_transformer.return_value = (Mock(), "Bagel")

        estimator = TeaCacheCoefficientEstimator(model_path="dummy", model_type="Bagel")
        estimator.collected_data = [
            [
                (np.array([[1.0]]), np.array([[2.0]])),
                (np.array([[2.0]]), np.array([[4.0]])),
                (np.array([[3.0]]), np.array([[6.0]])),
                (np.array([[4.0]]), np.array([[8.0]])),
                (np.array([[5.0]]), np.array([[10.0]])),
                (np.array([[6.0]]), np.array([[12.0]])),
            ]
        ]

        coeffs = estimator.estimate(poly_order=2)
        assert len(coeffs) == 3
