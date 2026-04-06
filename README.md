# RunPod ComfyUI Worker

This worker is designed for `LTX-2.3` on `RunPod Serverless` using `ComfyUI`, with support for:

- two-stage `text-to-video`
- two-stage `image-to-video`
- `dev` checkpoint workflows
- `STG` placeholders
- low-VRAM loading via your chosen ComfyUI workflow

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

## Placeholder Tokens

Keep these literal placeholders in the exported workflow values where appropriate:

- `{{PROMPT}}`
- `{{NEGATIVE_PROMPT}}`
- `{{WIDTH}}`
- `{{HEIGHT}}`
- `{{NUM_FRAMES}}`
- `{{FPS}}`
- `{{SEED}}`
- `{{STAGE1_STEPS}}`
- `{{STAGE2_STEPS}}`
- `{{CFG}}`
- `{{STG_SCALE}}`
- `{{STG_BLOCKS}}`
- `{{SAMPLER}}`
- `{{SCHEDULER}}`
- `{{FILENAME_PREFIX}}`
- `{{IMAGE_PATH}}` for image-to-video only

## Recommended Workflow Shape

For your requested setup, use:

- `LowVRAMCheckpointLoader`
- `ltx-2.3-22b-dev.safetensors`
- `LTXVApplySTG`
- `LTXVLatentUpsampler`
- save node that outputs `mp4`

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
    "stage1_steps": 30,
    "stage2_steps": 6,
    "cfg": 3.0,
    "stg_scale": 1.0,
    "stg_blocks": "14,19",
    "sampler": "euler",
    "scheduler": "simple",
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
    "stage1_steps": 30,
    "stage2_steps": 6,
    "cfg": 3.0,
    "stg_scale": 1.0,
    "stg_blocks": "14,19",
    "sampler": "euler",
    "scheduler": "simple",
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
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- `gemma_3_12B_it_fp4_mixed.safetensors`
