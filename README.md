# RunPod ComfyUI Worker

This worker is designed for `LTX-2.3` on `RunPod Serverless` using `ComfyUI`, with support for:

- two-stage `text-to-video`
- two-stage `image-to-video`
- `dev` checkpoint workflows
- `VHS_VideoCombine` mp4 output
- API debug modes for node schema and rendered workflow inspection

## Important

This worker does not hardcode the full ComfyUI graph. Instead it loads exported ComfyUI API workflow JSON files from:

- `workflows/ltx23_t2v_2stage_api.json`
- `workflows/ltx23_i2v_2stage_api.json`

That is deliberate. The exact node graph and node ids vary across ComfyUI / LTXVideo versions, and the stable integration point for serverless is the exported API workflow.

## What To Export From ComfyUI

Create two workflows in ComfyUI:

1. `LTX-2.3 two-stage text-to-video`
2. `LTX-2.3 two-stage image-to-video`

Export each workflow in API format and replace the placeholder JSON files in `workflows/`.

## Placeholder Tokens Currently Used

Keep these literal placeholders in the exported workflow values where appropriate:

- `{{PROMPT}}`
- `{{NEGATIVE_PROMPT}}`
- `{{WIDTH}}`
- `{{HEIGHT}}`
- `{{NUM_FRAMES}}`
- `{{FPS}}`
- `{{SEED}}`
- `{{CFG}}`
- `{{FILENAME_PREFIX}}`
- `{{IMAGE_PATH}}` for image-to-video only

## Current Workflow Shape

The checked-in workflows currently use:

- `CheckpointLoaderSimple`
- `ltx-2.3-22b-dev.safetensors`
- `LTXVLatentUpsampler`
- `VHS_VideoCombine` with `video/h264-mp4`

These workflows do not currently use:

- `LowVRAMCheckpointLoader`
- `LTXVApplySTG`
- request-driven sampler/scheduler placeholders
- request-driven stage step placeholders

## Request Contract

### Text-to-video

```json
{
  "input": {
    "mode": "t2v",
    "prompt": "cinematic prompt here",
    "negative_prompt": "optional",
    "width": 768,
    "height": 1344,
    "duration_seconds": 4,
    "fps": 25,
    "seed": 42,
    "cfg": 3.0,
    "filename_prefix": "ltx23_t2v"
  }
}
```

### Image-to-video

```json
{
  "input": {
    "mode": "i2v",
    "image": "https://... or base64",
    "prompt": "motion prompt here",
    "negative_prompt": "optional",
    "width": 768,
    "height": 1344,
    "duration_seconds": 4,
    "fps": 25,
    "seed": 42,
    "cfg": 3.0,
    "filename_prefix": "ltx23_i2v"
  }
}
```

## Response

The handler returns:

- `prompt_id`
- `mode`
- `duration_seconds`
- `file_name`
- `subfolder`
- `output_path`
- `video_base64` when `RETURN_BASE64=true`

## Debug Modes

The worker also supports these non-generation modes:

- `object_info`
  - returns ComfyUI node schema for a specific `node_class`, or the full object info payload
- `rendered_workflow`
  - returns the rendered prompt JSON after placeholder replacement
  - supports optional `node_ids` filtering

## Model Storage

On RunPod, attach a network volume and put ComfyUI models under:

- `/runpod-volume/ComfyUI/models/checkpoints`
- `/runpod-volume/ComfyUI/models/loras`
- `/runpod-volume/ComfyUI/models/latent_upscale_models`
- `/runpod-volume/ComfyUI/models/text_encoders`

Recommended files:

- `ltx-2.3-22b-dev.safetensors`
- `ltx-2.3-22b-dev-fp8.safetensors` as fallback
- `ltx-2.3-22b-distilled-lora-384.safetensors` if your chosen two-stage workflow uses the official LoRA refinement
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `gemma_3_12B_it_fp4_mixed.safetensors`
