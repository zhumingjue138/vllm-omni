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



# Model and serve args for Qwen-Image-Edit-2511 (OmniServerTest, no stage config)
MODEL = ["/data/models/Qwen-Image-Edit-2511"]
SERVE_ARGS_IMAGE_EDIT = [
    "--vae-use-tiling",
    "--vae-use-slicing",
    "--enforce-eager",
]
# GPU device id for this test (equivalent to export CUDA_VISIBLE_DEVICES="x")
# Override via env: CUDA_VISIBLE_DEVICES (e.g. pytest ... -E CUDA_VISIBLE_DEVICES=1)
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "7")


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



@pytest.mark.parametrize("model_name", MODEL)
@pytest.mark.parametrize("gpu_id", GPU_ID)
def test_image_edit_loop_omni_server_test_multi_image(model_name: str, gpu_id: str) -> None:
    """Use OmniServerTest to start Qwen-Image_Edit-2511, then loop images.edit requests with multiple images."""
    num_requests = 3
    env_dict = {"CUDA_VISIBLE_DEVICES": gpu_id}
    
    # 本地多张图片路径列表
    local_image_paths = [
        "/home/z00939163/vllm-omni-main/vllm-omni/tests/e2e/online_serving/cat.png",
        "/home/z00939163/vllm-omni-main/vllm-omni/tests/e2e/online_serving/dog.jpg", 
        # 添加更多图片路径...
    ]
    
    # 检查图片文件是否存在
    for img_path in local_image_paths:
        if not os.path.exists(img_path):
            pytest.fail(f"Local image file not found: {img_path}")
    
    with OmniServerTest(
        model_name, SERVE_ARGS_IMAGE_EDIT, env_dict=env_dict
    ) as server:
        
        api_client = client(server)
        
        try:
            success_count = 0
            
            for i in range(num_requests):
                width = random.randint(100, 8000)
                height = random.randint(100, 8000)
                size = f"{width}x{height}"
                
                try:
                    # 打开所有图片文件
                    image_files = []
                    for img_path in local_image_paths:
                        with open(img_path, "rb") as img_file:
                            # 注意：这里需要确认API是否支持多张图片输入
                            # 有些API可能需要将多个图片合并为一个文件，或者使用不同的参数
                            image_files.append(img_file.read())
                    
                    # 根据API要求处理多张图片
                    # 假设API支持传入图片列表
                    result = api_client.images.edit(
                        model=MODEL,
                        image=image_files,  # 传入图片列表
                        prompt=(
                            "将两张图片的角色放在一起打架"
                            #"将第二张图中人脸/猫狗脸转换为3D卡通形象，质量要求：毛发纹理自然电影级色彩校准，注意保留原图的外貌、肤色、发型特征，仅替换第一张财神爷/猫狗财神的用绿框框住的头部区域，耳朵样式和原图的3D卡通形象一致， （帽子紧贴头顶，未贴合部分可调整帽子摆放角度达到完全贴合）， 头部与身体过渡必须自然，无拼接痕迹，无额外元素出现在头部四周，纯白背景，不要出现绿色圆圈"
                        ),
                        size=size,
                        stream=False,
                        output_format="jpeg",
                        extra_body={
                            "num_inference_steps": 20,
                            "guidance_scale": 1.0,
                        },
                    )

                    

                    # 处理返回结果
                    for j, data_item in enumerate(result.data):
                        image_base64 = data_item.b64_json
                        image_bytes = base64.b64decode(image_base64)
                        output_path = f"test_{i}_{size}.jpeg"
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)

                        print(f"✓ Image saved: {output_path}")

                        print(f"  Processed output {j+1} for request {i+1}")
                    
                    success_count += 1
                    
                except Exception as e:
                    pytest.fail(f"Request {i + 1} (size={size}) failed: {e}")
                
                used, total = get_gpu_memory(gpu_id)
                if used is not None and total is not None:
                    print(f"  [Request {i + 1}/{num_requests}] size={size} GPU{gpu_id}: {used}/{total} MB")
            
            assert success_count == num_requests, f"Expected {num_requests} successes, got {success_count}"
            
        except Exception as e:
            pytest.fail(f"Test failed: {e}")