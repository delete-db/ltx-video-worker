"""
RunPod Serverless Handler for LTX-2.3 Image-to-Video Generation.
Uses the official ltx-pipelines package for best quality.
"""

import os
import io
import sys
import base64
import tempfile
import requests
from PIL import Image

# Add LTX-2 to path
sys.path.insert(0, "/app/ltx2/packages/ltx-pipelines/src")
sys.path.insert(0, "/app/ltx2/packages/ltx-core/src")

import runpod

# ── Load model ONCE at startup ──────────────────────────────

print("Loading LTX-2.3 pipeline...")

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ti2vid_two_stages import ImageConditioningInput
from ltx_core.components.guiders import MultiModalGuiderParams

# Model paths - will be downloaded from HuggingFace on first run
HF_MODEL = "Lightricks/LTX-2.3"
CACHE_DIR = "/tmp/ltx-models"

# Download model files if not cached
from huggingface_hub import hf_hub_download

os.makedirs(CACHE_DIR, exist_ok=True)

print("Downloading LTX-2.3 checkpoint...")
checkpoint_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-22b-distilled.safetensors",
    cache_dir=CACHE_DIR,
)

print("Downloading distilled LoRA...")
distilled_lora_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    cache_dir=CACHE_DIR,
)

print("Downloading spatial upscaler...")
spatial_upscaler_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    cache_dir=CACHE_DIR,
)

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

distilled_lora = [
    LoraPathStrengthAndSDOps(
        distilled_lora_path,
        0.6,
        LTXV_LORA_COMFY_RENAMING_MAP,
    ),
]

print("Initializing pipeline...")
pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=checkpoint_path,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=spatial_upscaler_path,
    loras=[],
)

print("LTX-2.3 pipeline ready!")


# ── Helper Functions ──────────────────────────────────────────

def download_image(url: str, save_path: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img.save(save_path)
    return save_path


def decode_base64_image(data: str, save_path: str) -> str:
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(save_path)
    return save_path


# ── Handler ───────────────────────────────────────────────────

def handler(job):
    job_input = job["input"]

    # Get image
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "No image provided"}

    # Save input image to temp file
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_img_path = temp_img.name
    temp_img.close()

    if image_input.startswith("http"):
        download_image(image_input, temp_img_path)
    else:
        decode_base64_image(image_input, temp_img_path)

    # Parameters
    prompt = job_input.get("prompt", "gentle camera movement, subtle motion")
    num_frames = job_input.get("num_frames", 97)
    width = job_input.get("width", 768)
    height = job_input.get("height", 1344)
    num_inference_steps = job_input.get("num_inference_steps", 8)
    seed = job_input.get("seed", 42)

    # Ensure dimensions divisible by 32, frames by 8+1
    width = (width // 32) * 32
    height = (height // 32) * 32
    num_frames = ((num_frames - 1) // 8) * 8 + 1

    print(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")

    # Output path
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Guider params for distilled model
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0,
        stg_scale=0.0,
        rescale_scale=0.0,
        modality_scale=1.0,
        stg_blocks=[],
    )

    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0,
        stg_scale=0.0,
        rescale_scale=0.0,
        modality_scale=1.0,
        stg_blocks=[],
    )

    # Generate
    pipeline(
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=25.0,
        num_inference_steps=num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[ImageConditioningInput(temp_img_path, 0, 1.0, 33)],
        generate_audio=False,
    )

    # Read output
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup
    os.unlink(temp_img_path)
    os.unlink(output_path)

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration_seconds = num_frames / 25

    print(f"Video generated: {len(video_bytes)} bytes, {duration_seconds:.1f}s")

    return {
        "video_base64": video_base64,
        "duration_seconds": duration_seconds,
    }


runpod.serverless.start({"handler": handler})
