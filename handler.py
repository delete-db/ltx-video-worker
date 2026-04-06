"""
RunPod Serverless Handler for LTX-2.3 Image-to-Video Generation.
Uses TI2VidTwoStagesPipeline with the full dev model for cinematic quality.

Pipeline: stage 1 generates at half resolution with CFG+STG guidance (30 steps),
stage 2 upsamples 2x and refines with distilled LoRA (4 steps).

Memory strategy for RTX 6000 Ada 48GB:
- FP8 quantization (~22GB model weights instead of ~44GB)
- Layer streaming (transformer layers stay on CPU, stream to GPU on demand)
- StateDictRegistry (shared weights between stage 1 and stage 2)
"""

import os
import io
import sys
import base64
import tempfile
import torch
import requests
from PIL import Image

# Add LTX-2 packages to path
sys.path.insert(0, "/app/ltx2/packages/ltx-pipelines/src")
sys.path.insert(0, "/app/ltx2/packages/ltx-core/src")

import runpod

# ── Configuration ────────────────────────────────────────────

CACHE_DIR = "/tmp/ltx-models"
HF_MODEL = "Lightricks/LTX-2.3"
GEMMA_MODEL = "google/gemma-3-12b-it-qat-q4_0-unquantized"

os.makedirs(CACHE_DIR, exist_ok=True)

# Set HF token if available (needed for Gemma license)
hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN"))
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print(f"HF_TOKEN set ({hf_token[:8]}...)")
else:
    print("WARNING: No HF_TOKEN found. Gemma download may fail if license not accepted.")

# ── Download Models ──────────────────────────────────────────

from huggingface_hub import hf_hub_download, snapshot_download

print("Downloading LTX-2.3 dev checkpoint...")
checkpoint_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-22b-dev.safetensors",
    cache_dir=CACHE_DIR,
    token=hf_token,
)

print("Downloading distilled LoRA (for stage 2 refinement)...")
distilled_lora_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    cache_dir=CACHE_DIR,
    token=hf_token,
)

print("Downloading spatial upscaler...")
spatial_upscaler_path = hf_hub_download(
    repo_id=HF_MODEL,
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    cache_dir=CACHE_DIR,
    token=hf_token,
)

print("Downloading Gemma 3 12B text encoder (this may take a while)...")
gemma_root = snapshot_download(
    repo_id=GEMMA_MODEL,
    cache_dir=CACHE_DIR,
    token=hf_token,
)
print(f"Gemma downloaded to: {gemma_root}")

# ── Initialize Pipeline ─────────────────────────────────────

print("Initializing LTX-2.3 TI2VidTwoStagesPipeline (dev model)...")

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.loader.registry import StateDictRegistry

quantization = QuantizationPolicy.fp8_cast()
print("Using FP8 quantization")

registry = StateDictRegistry()

distilled_lora = [
    LoraPathStrengthAndSDOps(
        distilled_lora_path,
        0.6,
        LTXV_LORA_COMFY_RENAMING_MAP,
    ),
]

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=checkpoint_path,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=spatial_upscaler_path,
    gemma_root=gemma_root,
    loras=[],
    quantization=quantization,
    registry=registry,
)

print("LTX-2.3 pipeline ready!")

# ── Official LTX-2.3 recommended guider params ──────────────

# From ltx_pipelines.utils.constants.LTX_2_3_PARAMS
VIDEO_GUIDER_PARAMS = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[28],
)

AUDIO_GUIDER_PARAMS = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=[28],
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


# ── Helper Functions ─────────────────────────────────────────

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


# ── Handler ──────────────────────────────────────────────────

def handler(job):
    job_input = job["input"]

    # Get image input
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "No image provided. Send 'image' as URL or base64."}

    # Save input image to temp file
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_img_path = temp_img.name
    temp_img.close()

    try:
        if image_input.startswith("http"):
            download_image(image_input, temp_img_path)
        else:
            decode_base64_image(image_input, temp_img_path)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    # Parameters with sensible defaults
    prompt = job_input.get("prompt", "gentle camera movement, subtle motion")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    num_frames = job_input.get("num_frames", 97)      # ~3.8 sec at 25fps
    width = job_input.get("width", 768)
    height = job_input.get("height", 1344)
    num_inference_steps = job_input.get("num_inference_steps", 30)  # LTX-2.3 default
    seed = job_input.get("seed", 42)
    frame_rate = job_input.get("frame_rate", 25.0)

    # Two-stage pipeline requires dimensions divisible by 64, frames by 8+1
    width = (width // 64) * 64
    height = (height // 64) * 64
    num_frames = ((num_frames - 1) // 8) * 8 + 1

    print(f"Generating: {width}x{height}, {num_frames} frames, {num_inference_steps} steps, seed={seed}")

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        video_frames_iter, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=VIDEO_GUIDER_PARAMS,
            audio_guider_params=AUDIO_GUIDER_PARAMS,
            images=[ImageConditioningInput(temp_img_path, 0, 1.0, 33)],
            streaming_prefetch_count=2,
        )

        # Encode to MP4
        video_chunks_number = get_video_chunks_number(num_frames)
        encode_video(
            video=video_frames_iter,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

    except Exception as e:
        if os.path.exists(temp_img_path):
            os.unlink(temp_img_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return {"error": f"Generation failed: {str(e)}"}

    # Read output video
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup temp files
    os.unlink(temp_img_path)
    os.unlink(output_path)

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration_seconds = num_frames / frame_rate

    print(f"Done: {len(video_bytes)} bytes, {duration_seconds:.1f}s")

    return {
        "video_base64": video_base64,
        "duration_seconds": duration_seconds,
    }


runpod.serverless.start({"handler": handler})
