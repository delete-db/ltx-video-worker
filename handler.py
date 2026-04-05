"""
RunPod Serverless Handler for LTX-Video Image-to-Video Generation.
Receives an image URL + motion prompt, returns a video URL.
"""

import os
import io
import base64
import tempfile
import runpod
import torch
import requests
from PIL import Image

# ── Load model ONCE at startup (not per request) ──────────────

print("Loading LTX-Video pipeline...")

from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Lightricks/LTX-Video"

pipe = LTXImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.enable_attention_slicing()

print("LTX-Video pipeline loaded and ready!")


# ── Helper Functions ──────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
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
            "num_frames": 97,    # optional, default 97 (~4 sec at 25fps)
            "width": 768,        # optional
            "height": 1344,      # optional (9:16 vertical)
            "num_inference_steps": 30,  # optional
            "guidance_scale": 3.5      # optional
        }
    Output:
        {
            "video_base64": "...",   # base64 encoded MP4
            "duration_seconds": 3.88
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

    # Get parameters
    prompt = job_input.get("prompt", "gentle camera movement, subtle motion")
    num_frames = job_input.get("num_frames", 97)
    width = job_input.get("width", 768)
    height = job_input.get("height", 1344)
    num_inference_steps = job_input.get("num_inference_steps", 30)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", 42)

    # Resize image to match target dimensions
    image = image.resize((width, height), Image.LANCZOS)

    print(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")

    # Generate video
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    video_frames = result.frames[0]

    # Export to MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    export_to_video(video_frames, output_path, fps=25)

    # Read and encode as base64
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(output_path)

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    fps = 25
    duration_seconds = num_frames / fps

    print(f"Video generated: {len(video_bytes)} bytes, {duration_seconds:.1f}s")

    return {
        "video_base64": video_base64,
        "duration_seconds": duration_seconds,
    }


# ── Start RunPod Serverless ───────────────────────────────────

runpod.serverless.start({"handler": handler})
