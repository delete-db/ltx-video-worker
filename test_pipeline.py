import torch
import time
from pathlib import Path
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video

M = Path("/workspace/ComfyUI/models")

print("Loading pipeline...")
t0 = time.time()
P = TI2VidTwoStagesPipeline(
    checkpoint_path=str(M / "checkpoints/ltx-2.3-22b-dev.safetensors"),
    distilled_lora=[
        LoraPathStrengthAndSDOps(
            str(M / "loras/ltx-2.3-22b-distilled-lora-384.safetensors"),
            0.5,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ],
    spatial_upsampler_path=str(M / "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
    gemma_root=str(M / "text_encoders/gemma-3-12b-it"),
    loras=[],
)
print(f"Pipeline loaded in {time.time() - t0:.1f}s")

g = MultiModalGuiderParams(
    cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.7,
    modality_scale=3.0, stg_blocks=[],
)

print("Generating video...")
t1 = time.time()
with torch.inference_mode():
    v, a = P(
        prompt="A golden eagle soaring through misty mountains at sunrise",
        negative_prompt="",
        seed=42,
        height=512,
        width=768,
        num_frames=97,
        frame_rate=25.0,
        num_inference_steps=30,
        video_guider_params=g,
        audio_guider_params=g,
        images=[],
        tiling_config=TilingConfig.default(),
    )
    encode_video(
        video=v, fps=25.0, audio=a,
        output_path="/workspace/test.mp4",
        video_chunks_number=get_video_chunks_number(97, TilingConfig.default()),
    )
print(f"Generation done in {time.time() - t1:.1f}s")
print("Output: /workspace/test.mp4")
