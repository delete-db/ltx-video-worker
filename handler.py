import base64
import copy
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import runpod
from PIL import Image


COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/runpod-volume/ComfyUI/output"))
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/runpod-volume/ComfyUI/input"))

VOLUME_WORKFLOW_DIR = Path(
    os.environ.get("VOLUME_WORKFLOW_DIR", "/runpod-volume/ComfyUI/workflows")
)
WORKFLOW_DIR = Path(os.environ.get("WORKFLOW_DIR", "/app/workflows"))
T2V_WORKFLOW_NAME = os.environ.get("LTX_T2V_WORKFLOW_NAME", "ltx23_t2v_2stage_api.json")
I2V_WORKFLOW_NAME = os.environ.get("LTX_I2V_WORKFLOW_NAME", "ltx23_i2v_2stage_api.json")

REQUEST_TIMEOUT_SEC = int(os.environ.get("REQUEST_TIMEOUT_SEC", "1800"))
POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "2.5"))
RETURN_BASE64 = os.environ.get("RETURN_BASE64", "true").lower() == "true"
WORKER_VERSION = os.environ.get("WORKER_VERSION", "volume-workflow-v2-vhs")


def wait_for_comfyui(timeout_sec: int = 300) -> None:
    deadline = time.time() + timeout_sec
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if response.ok:
                print("ComfyUI is ready")
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"ComfyUI did not become ready in {timeout_sec}s: {last_error}")


def load_workflow(mode: str) -> dict[str, Any]:
    filename = T2V_WORKFLOW_NAME if mode == "t2v" else I2V_WORKFLOW_NAME
    candidates = [
        VOLUME_WORKFLOW_DIR / filename,
        WORKFLOW_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

    raise FileNotFoundError(
        f"Workflow file not found in any expected location: {', '.join(str(path) for path in candidates)}"
    )


def write_temp_image_from_input(image_input: str) -> str:
    """Save input image to ComfyUI's input directory and return just the filename."""
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"input_{int(time.time())}_{os.getpid()}.png"
    save_path = COMFY_INPUT_DIR / filename

    try:
        if image_input.startswith("http://") or image_input.startswith("https://"):
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            payload = image_input.split(",", 1)[1] if "," in image_input else image_input
            image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

        image.save(str(save_path))
        return filename  # ComfyUI LoadImage expects just the filename
    except Exception:
        if save_path.exists():
            save_path.unlink()
        raise


def clamp_dimensions(width: int, height: int) -> tuple[int, int]:
    return max(64, (width // 64) * 64), max(64, (height // 64) * 64)


def clamp_num_frames(num_frames: int) -> int:
    return max(9, ((num_frames - 1) // 8) * 8 + 1)


def build_template_values(job_input: dict[str, Any]) -> dict[str, str]:
    mode = job_input.get("mode", "t2v").lower()
    width = int(job_input.get("width", 768))
    height = int(job_input.get("height", 1344))
    width, height = clamp_dimensions(width, height)

    fps = float(job_input.get("fps", 25))
    duration_seconds = float(job_input.get("duration_seconds", 4))
    num_frames = int(job_input.get("num_frames", round(duration_seconds * fps)))
    num_frames = clamp_num_frames(num_frames)

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
      raise ValueError("Missing required input: prompt")

    negative_prompt = job_input.get("negative_prompt", "").strip()
    seed = int(job_input.get("seed", 42))
    cfg = float(job_input.get("cfg", 3.0))
    filename_prefix = job_input.get("filename_prefix", f"ltx23_{mode}")

    values = {
        "PROMPT": prompt,
        "NEGATIVE_PROMPT": negative_prompt,
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "NUM_FRAMES": str(num_frames),
        "FPS": str(fps),
        "SEED": str(seed),
        "CFG": str(cfg),
        "FILENAME_PREFIX": filename_prefix,
    }

    image_input = job_input.get("image")
    if mode == "i2v":
        if not image_input:
            raise ValueError("Missing required input: image for i2v mode")
        image_path = write_temp_image_from_input(image_input)
        values["IMAGE_PATH"] = image_path

    return values


def render_template(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: render_template(inner, replacements) for key, inner in value.items()}
    if isinstance(value, list):
        return [render_template(item, replacements) for item in value]
    if isinstance(value, str):
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", replacement)

        lowered = rendered.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        try:
            if rendered.isdigit() or (rendered.startswith("-") and rendered[1:].isdigit()):
                return int(rendered)
            return float(rendered) if "." in rendered else rendered
        except ValueError:
            return rendered
    return value


def queue_prompt(prompt: dict[str, Any]) -> str:
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": prompt},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
    return prompt_id


def fetch_object_info(node_class: str | None = None) -> dict[str, Any]:
    response = requests.get(f"{COMFY_URL}/object_info", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if node_class:
        return {node_class: payload.get(node_class)}
    return payload


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.time() + REQUEST_TIMEOUT_SEC
    last_payload = None

    while time.time() < deadline:
        response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload

        if prompt_id in payload:
            result = payload[prompt_id]
            status = result.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {json.dumps(result, ensure_ascii=True)[:2000]}")
            outputs = result.get("outputs")
            if outputs:
                return result

        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}. Last payload: {str(last_payload)[:2000]}")


def extract_video_file(history: dict[str, Any]) -> tuple[str, str]:
    outputs = history.get("outputs", {})
    for _, node_output in outputs.items():
        gifs = node_output.get("gifs") or []
        if gifs:
            item = gifs[0]
            filename = item["filename"]
            subfolder = item.get("subfolder", "")
            return filename, subfolder

        videos = node_output.get("videos") or []
        if videos:
            item = videos[0]
            filename = item["filename"]
            subfolder = item.get("subfolder", "")
            return filename, subfolder

        images = node_output.get("images") or []
        if images and images[0].get("filename", "").lower().endswith(".mp4"):
            item = images[0]
            filename = item["filename"]
            subfolder = item.get("subfolder", "")
            return filename, subfolder

    raise RuntimeError(f"No output video found in ComfyUI history: {json.dumps(history, ensure_ascii=True)[:2000]}")


def resolve_output_path(filename: str, subfolder: str) -> Path:
    if subfolder:
        return COMFY_OUTPUT_DIR / subfolder / filename
    return COMFY_OUTPUT_DIR / filename


def cleanup_temp_inputs(replacements: dict[str, str]) -> None:
    image_filename = replacements.get("IMAGE_PATH")
    if image_filename:
        full_path = COMFY_INPUT_DIR / image_filename
        if full_path.exists():
            full_path.unlink()


wait_for_comfyui()
print(f"Worker version: {WORKER_VERSION}")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})
    mode = job_input.get("mode", "t2v").lower()
    if mode == "object_info":
        try:
            node_class = job_input.get("node_class")
            return fetch_object_info(node_class)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    if mode not in {"t2v", "i2v"}:
        return {"error": "Invalid mode. Supported values: t2v, i2v, object_info"}

    replacements = {}
    try:
        workflow = load_workflow(mode)
        replacements = build_template_values(job_input)
        prompt = render_template(copy.deepcopy(workflow), replacements)

        prompt_id = queue_prompt(prompt)
        history = wait_for_history(prompt_id)
        filename, subfolder = extract_video_file(history)
        output_path = resolve_output_path(filename, subfolder)

        if not output_path.exists():
            raise FileNotFoundError(f"Expected ComfyUI output missing: {output_path}")

        response = {
            "prompt_id": prompt_id,
            "mode": mode,
            "duration_seconds": int(replacements["NUM_FRAMES"]) / float(replacements["FPS"]),
            "file_name": filename,
            "subfolder": subfolder,
            "output_path": str(output_path),
        }

        if RETURN_BASE64:
            video_bytes = output_path.read_bytes()
            response["video_base64"] = base64.b64encode(video_bytes).decode("utf-8")

        return response
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    finally:
        cleanup_temp_inputs(replacements)


runpod.serverless.start({"handler": handler})
