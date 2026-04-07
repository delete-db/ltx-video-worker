"""
RunPod Serverless handler for LTX-2.3 video generation.
Uses ltx-pipelines directly (no ComfyUI) for maximum speed.
Models are loaded once at startup and kept in GPU memory.
"""

import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import runpod
import torch
from PIL import Image

# ── Configuration ───────────────────────────────────────────

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/runpod-volume/ComfyUI/models"))
USE_FP8 = os.environ.get("USE_FP8", "false").lower() == "true"
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(MODELS_ROOT / "checkpoints" / "ltx-2.3-22b-dev.safetensors"),
)
DISTILLED_LORA_PATH = os.environ.get(
    "DISTILLED_LORA_PATH",
    str(MODELS_ROOT / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"),
)
UPSAMPLER_PATH = os.environ.get(
    "UPSAMPLER_PATH",
    str(MODELS_ROOT / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
)
GEMMA_ROOT = os.environ.get(
    "GEMMA_ROOT",
    str(MODELS_ROOT / "text_encoders" / "gemma-3-12b-it"),
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/ltx_output"))
WORKER_VERSION = "direct-pipeline-v1-fp8" if USE_FP8 else "direct-pipeline-v1"

# ── Load Pipeline Once ──────────────────────────────────────

print(f"Worker version: {WORKER_VERSION}")
print(f"Loading pipeline...")
print(f"  Checkpoint:     {CHECKPOINT_PATH}")
print(f"  Distilled LoRA: {DISTILLED_LORA_PATH}")
print(f"  Upsampler:      {UPSAMPLER_PATH}")
print(f"  Gemma root:     {GEMMA_ROOT}")

load_start = time.time()

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video

try:
    from ltx_pipelines.utils.args import ImageConditioningInput
except ImportError:
    from collections import namedtuple
    ImageConditioningInput = namedtuple(
        "ImageConditioningInput", ["path", "frame_idx", "strength", "crf"]
    )

distilled_lora = [
    LoraPathStrengthAndSDOps(
        DISTILLED_LORA_PATH,
        0.5,
        LTXV_LORA_COMFY_RENAMING_MAP,
    ),
]

PIPELINE = TI2VidTwoStagesPipeline(
    checkpoint_path=CHECKPOINT_PATH,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=UPSAMPLER_PATH,
    gemma_root=GEMMA_ROOT,
    loras=[],
    quantization=QuantizationPolicy.fp8_cast() if USE_FP8 else None,
)

load_elapsed = time.time() - load_start
print(f"Pipeline loaded in {load_elapsed:.1f}s")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────

def clamp_dimensions(width: int, height: int) -> tuple[int, int]:
    return max(64, (width // 32) * 32), max(64, (height // 32) * 32)


def clamp_num_frames(num_frames: int) -> int:
    return max(9, ((num_frames - 1) // 8) * 8 + 1)


def save_input_image(image_input: str) -> str:
    """Decode base64 or download URL image, save to temp file, return path."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        import requests
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        payload = image_input.split(",", 1)[1] if "," in image_input else image_input
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

    tmp_path = os.path.join(tempfile.gettempdir(), f"input_{int(time.time())}_{os.getpid()}.png")
    image.save(tmp_path)
    return tmp_path


# ── Handler ─────────────────────────────────────────────────

@torch.inference_mode()
def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})
    mode = job_input.get("mode", "t2v").lower()

    if mode not in {"t2v", "i2v"}:
        return {"error": f"Invalid mode '{mode}'. Supported: t2v, i2v"}

    # Parse inputs
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required input: prompt"}

    negative_prompt = job_input.get("negative_prompt", "").strip()
    width = int(job_input.get("width", 768))
    height = int(job_input.get("height", 1344))
    width, height = clamp_dimensions(width, height)

    fps = float(job_input.get("fps", 25))
    duration_seconds = float(job_input.get("duration_seconds", 4))
    num_frames = int(job_input.get("num_frames", round(duration_seconds * fps)))
    num_frames = clamp_num_frames(num_frames)

    seed = int(job_input.get("seed", 42))
    cfg = float(job_input.get("cfg", 3.0))
    skip_audio = bool(job_input.get("skip_audio", False))

    # Handle i2v image
    images = []
    input_image_path = None
    if mode == "i2v":
        image_input = job_input.get("image")
        if not image_input:
            return {"error": "Missing required input: image for i2v mode"}
        input_image_path = save_input_image(image_input)
        images = [ImageConditioningInput(
            path=input_image_path,
            frame_idx=0,
            strength=1.0,
            crf=18,
        )]

    print(f"Generating {mode}: {width}x{height}, {num_frames} frames, seed={seed}, cfg={cfg}")
    gen_start = time.time()

    try:
        video_guider = MultiModalGuiderParams(
            cfg_scale=cfg,
            stg_scale=0.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[],
        )
        audio_guider = MultiModalGuiderParams(
            cfg_scale=cfg,
            stg_scale=0.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[],
        )
        tiling = TilingConfig.default()

        video, audio = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=30,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=images,
            tiling_config=tiling,
        )

        # Encode to MP4
        output_path = str(OUTPUT_DIR / f"ltx23_{mode}_{int(time.time())}.mp4")
        video_chunks = get_video_chunks_number(num_frames, tiling)
        encode_video(
            video=video,
            fps=fps,
            audio=None if skip_audio else audio,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )

        gen_elapsed = time.time() - gen_start
        print(f"Generation complete in {gen_elapsed:.1f}s → {output_path}")

        # Read and encode to base64
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Cleanup
        os.remove(output_path)

        response = {
            "mode": mode,
            "video_base64": video_base64,
            "duration_seconds": num_frames / fps,
            "generation_time_seconds": round(gen_elapsed, 1),
        }
        return response

    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if input_image_path and os.path.exists(input_image_path):
            os.remove(input_image_path)


runpod.serverless.start({"handler": handler})
