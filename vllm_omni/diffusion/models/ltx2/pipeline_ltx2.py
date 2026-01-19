# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import torch
from diffusers import LTXVideoPipeline
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest


def get_ltx2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-process function for LTX-2: convert video output to numpy array."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_ltx2_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for LTX-2: validate and prepare image inputs for I2V mode."""
    import PIL.Image

    def pre_process_func(requests: list[OmniDiffusionRequest]) -> list[OmniDiffusionRequest]:
        for req in requests:
            # Load image if path is provided
            if req.image_path is not None and req.pil_image is None:
                req.pil_image = PIL.Image.open(req.image_path).convert("RGB")

            # For image-to-video, store the image for later use
            if req.pil_image is not None:
                # LTX-2 will handle image resizing internally
                pass

        return requests

    return pre_process_func


class LTX2Pipeline(nn.Module):
    """
    Wrapper for Lightricks LTX-2 text-to-video pipeline.
    
    This model uses the diffusers LTXVideoPipeline for efficient video generation.
    Supports text-to-video and image-to-video generation.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        
        model = od_config.model
        local_files_only = os.path.exists(model)
        
        # Load the LTX-2 pipeline from diffusers
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        
        self.pipe = LTXVideoPipeline.from_pretrained(
            model,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        
        # Store configuration
        self.output_type = self.od_config.output_type
        
        # No weights_sources needed since we use the diffusers pipeline directly
        self.weights_sources = []

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 4.0,
        frame_num: Optional[int] = None,
        output_type: Optional[str] = "np",
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> DiffusionOutput:
        """
        Generate video from text prompt.
        
        Args:
            req: OmniDiffusionRequest containing generation parameters
            prompt: Text prompt for video generation
            negative_prompt: Negative text prompt
            height: Video height (must be divisible by 32)
            width: Video width (must be divisible by 32)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for classifier-free guidance
            frame_num: Number of frames to generate
            output_type: Output format ("np", "pt", or "pil")
            generator: Random generator for reproducibility
            **kwargs: Additional arguments passed to the pipeline
            
        Returns:
            DiffusionOutput containing the generated video frames
        """
        # Extract parameters from request
        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt
        
        if prompt is None:
            raise ValueError("Prompt is required for LTX-2 video generation.")
        
        # Set default values for video generation
        height = req.height or height or 512
        width = req.width or width or 768
        num_frames = req.num_frames if req.num_frames else frame_num or 121
        num_steps = req.num_inference_steps or num_inference_steps or 40
        
        # Ensure dimensions are divisible by 32 (LTX-2 requirement)
        height = (height // 32) * 32
        width = (width // 32) * 32
        
        # Ensure num_frames follows LTX-2 convention (divisible by 8 + 1)
        # e.g., 25, 41, 81, 121, etc.
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
        num_frames = max(num_frames, 25)  # Minimum 25 frames
        
        # Prepare generator
        if generator is None:
            seed = req.seed if hasattr(req, 'seed') and req.seed is not None else 42
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare image input for I2V mode if available
        image = None
        if hasattr(req, 'pil_image') and req.pil_image is not None:
            image = req.pil_image
        
        # Call the diffusers pipeline
        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": output_type,
        }
        
        # Add image if provided (for I2V mode)
        if image is not None:
            pipeline_kwargs["image"] = image
        
        # Add any extra kwargs
        pipeline_kwargs.update(kwargs)
        
        # Generate video
        output = self.pipe(**pipeline_kwargs)
        
        # Extract frames from the pipeline output
        if hasattr(output, "frames"):
            frames = output.frames
        elif hasattr(output, "videos"):
            frames = output.videos
        else:
            # Fallback: assume the output itself is the frames
            frames = output
        
        # Return as DiffusionOutput
        return DiffusionOutput(output=frames)
