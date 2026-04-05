"""
RunPod Serverless Handler for LTX-Video 0.9.8 Image-to-Video Generation.
Uses the latest LTX model with spatial upscaler for high quality output.
"""

import os
import io
import base64
import tempfile
import runpod
import torch
import requests
from PIL import Image

# ── Load models ONCE at startup ──────────────────────────────

print("Loading LTX-Video 0.9.8 pipeline...")

from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video

MODEL_ID = "Lightricks/LTX-Video-0.9.8-distilled"
UPSCALER_ID = "Lightricks/ltxv-spatial-upscaler-0.9.8"

pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.vae.enable_tiling()

pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    UPSCALER_ID,
    vae=pipe.vae,
    torch_dtype=torch.bfloat16,
)
pipe_upsample.to("cuda")

print("LTX-Video 0.9.8 + upscaler loaded and ready!")


# ── Helper Functions ──────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def decode_base64_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ── Handler ───────────────────────────────────────────────────

def handler(job):
    """
    Input:
        {
            "image": "https://..." or "data:image/jpeg;base64,...",
            "prompt": "A character slowly turns their head...",
            "negative_prompt": "worst quality, blurry",
            "num_frames": 97,
            "width": 768,
            "height": 1344,
            "num_inference_steps": 30,
            "guidance_scale": 3.5,
            "upscale": true,
            "seed": 42
        }
    """
    job_input = job["input"]

    # Get image
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "No image provided"}

    if image_input.startswith("http"):
        image = download_image(image_input)
    else:
        image = decode_base64_image(image_input)

    # Parameters
    prompt = job_input.get("prompt", "gentle camera movement, subtle motion")
    negative_prompt = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery")
    num_frames = job_input.get("num_frames", 97)
    width = job_input.get("width", 768)
    height = job_input.get("height", 1344)
    num_inference_steps = job_input.get("num_inference_steps", 20)
    seed = job_input.get("seed", 42)
    do_upscale = job_input.get("upscale", False)

    # Prepare image as video condition (first frame)
    image_resized = image.resize((width, height), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_img_path = f.name
        image_resized.save(temp_img_path)

    loaded_image = load_image(temp_img_path)
    video_cond = load_video(export_to_video([loaded_image], fps=24))
    condition = LTXVideoCondition(video=video_cond, frame_index=0)

    os.unlink(temp_img_path)

    print(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate video at base resolution
    if do_upscale:
        # Generate at lower res, then upscale
        base_w = width // 2
        base_h = height // 2
        # Ensure divisible by 32
        base_w = (base_w // 32) * 32
        base_h = (base_h // 32) * 32

        latents = pipe(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=base_w,
            height=base_h,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="latent",
        ).frames

        print("Base generation done, upscaling...")

        upscaled_latents = pipe_upsample(
            latents=latents,
            output_type="latent",
        ).frames

        video = pipe(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=base_w * 2,
            height=base_h * 2,
            num_frames=num_frames,
            denoise_strength=0.4,
            num_inference_steps=10,
            latents=upscaled_latents,
            generator=generator,
            output_type="pil",
        ).frames[0]
    else:
        # Direct generation at target resolution
        result = pipe(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
        )
        video = result.frames[0]

    # Export to MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    export_to_video(video, output_path, fps=24)

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(output_path)

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration_seconds = num_frames / 24

    print(f"Video generated: {len(video_bytes)} bytes, {duration_seconds:.1f}s")

    return {
        "video_base64": video_base64,
        "duration_seconds": duration_seconds,
    }


runpod.serverless.start({"handler": handler})
