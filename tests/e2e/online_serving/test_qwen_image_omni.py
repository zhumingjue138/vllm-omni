# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import base64
import concurrent.futures
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    OmniServerTest,
    convert_audio_to_text,
    cosine_similarity_text,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
    run_benchmark,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen-Image_Edit-2511"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }

@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_no_async_chunk_004(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 256
    stage_config_path = modify_stage_config(
        stage_config_path,
        updates={
            "stage_args":{
                0: {
                    "runtime.max_batch_size": num_concurrent_requests
                },
                1: {
                    "runtime.max_batch_size": num_concurrent_requests
                },
            },
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        request_rates = [0.1, 0.2, 0.3, 0.4]
        for request_rate in request_rates:
            args = [
                "--model",
                server.model,
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--dataset-name",
                "random-mm",
                "--request_rate",
                str(request_rate),
                "--random-input-len",
                "2500",
                "--random-output-len",
                "900",
                "--num-prompts",
                "100",
                "--endpoint",
                "/v1/chat/completions",
                "--backend",
                "openai-chat-omni",
                "--ignore-eos",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el,audio_ttfp,audio_rtf",
            ]
            result = run_benchmark(args)
            assert result.get("completed") == 100, "The request success rate did not reach 100%."


# Model and serve args for Qwen-Image-Edit-2511 (OmniServerTest, no stage config)
MODEL_IMAGE_EDIT_2511 = "Qwen/Qwen-Image_Edit-2511"
SERVE_ARGS_IMAGE_EDIT = [
    "--vae-use-tiling",
    "--vae-use-slicing",
    "--enforce-eager",
]
# GPU device id for this test (equivalent to export CUDA_VISIBLE_DEVICES="x")
# Override via env: CUDA_VISIBLE_DEVICES (e.g. pytest ... -E CUDA_VISIBLE_DEVICES=1)
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def get_gpu_memory(gpu_id: str = "0"):
    """Return (used_mb, total_mb) for the given GPU, or (None, None) if unavailable."""
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_id,
            ],
            stderr=subprocess.DEVNULL,
        )
        used, total = result.decode().strip().split(", ")
        return int(used), int(total)
    except Exception:
        return None, None


def test_image_edit_loop_omni_server_test() -> None:
    """Use OmniServerTest to start Qwen-Image_Edit-2511, then loop images.edit requests."""
    num_requests = 5
    env_dict = {"CUDA_VISIBLE_DEVICES": GPU_ID}
    with OmniServerTest(
        MODEL_IMAGE_EDIT_2511, SERVE_ARGS_IMAGE_EDIT, env_dict=env_dict
    ) as server:
        client_instance = openai.OpenAI(
            base_url=f"http://{server.host}:{server.port}/v1",
            api_key="EMPTY",
        )
        # Prepare a temp image file from synthetic image
        img_result = generate_synthetic_image(512, 512)
        image_bytes = base64.b64decode(img_result["base64"])
        image_path = None
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
            f.write(image_bytes)
            image_path = f.name
        try:
            success_count = 0
            for i in range(num_requests):
                width = random.randint(512, 768)
                height = random.randint(512, 768)
                size = f"{width}x{height}"
                try:
                    with open(image_path, "rb") as img_file:
                        result = client_instance.images.edit(
                            model=MODEL_IMAGE_EDIT_2511,
                            image=[img_file],
                            prompt=(
                                "Change the scene to a cozy reading room. "
                                "Keep the composition recognizable. "
                                "Use soft lighting and warm colors."
                            ),
                            size=size,
                            stream=False,
                            output_format="jpeg",
                            extra_body={
                                "num_inference_steps": 1,
                                "guidance_scale": 1.0,
                            },
                        )
                    image_base64 = result.data[0].b64_json
                    _ = base64.b64decode(image_base64)
                    success_count += 1
                except Exception as e:
                    pytest.fail(f"Request {i + 1} (size={size}) failed: {e}")
                used, total = get_gpu_memory(GPU_ID)
                if used is not None and total is not None:
                    print(f"  [Request {i + 1}/{num_requests}] size={size} GPU{GPU_ID}: {used}/{total} MB")
            assert success_count == num_requests, f"Expected {num_requests} successes, got {success_count}"
        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)