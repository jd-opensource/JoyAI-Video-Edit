"""Compute the condition embeddings for one editing session and save them to .npy.

The DiT is conditioned on hidden states of the MiMo-VL-7B encoder evaluated on
(first frame, instruction).  That encoder runs exactly once per session, so on an edge
device it does not need to be resident: run this script wherever the 7B model fits
(the device itself with enough RAM, a workstation, a server) and hand the ~6 MB result
to the CPU runtime.

    python -m joyedge.encode_prompt --text-encoder deploy/deps/checkpoints/MiMo-VL-7B-RL-2508 \
        --image first_frame.png --width 432 --height 240 \
        --prompt "Turn the video into a watercolor wash style." --out prompt.npy

It reuses the reference ``Pipeline.encode_prompt`` code path unchanged.
NOTE: this helper was not exercised end-to-end in the environment where the edge runtime
was developed (no MiMo-VL weights there); it is a thin wrapper around stock code.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JOYOMNI_FA4_ATTN", "0")

import numpy as np
import torch
from PIL import Image

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))


def encode(text_encoder_ckpt: str, image: Image.Image, prompt: str, *, width: int, height: int,
           device: str = "cpu", dtype: torch.dtype = torch.float32, max_sequence_length: int = 1024) -> np.ndarray:
    from xvideo.config import ExpConfig
    from xvideo.models.models import load_text_encoder
    from xvideo.models.pipeline import Pipeline
    from xvideo.models.scheduler import FlowMatchDiscreteScheduler
    from xvideo.models.vae import XVAEChunkCausal

    cfg = ExpConfig()
    cfg.text_encoder_arch_config["params"]["text_encoder_ckpt"] = text_encoder_ckpt
    tokenizer, text_encoder = load_text_encoder(text_encoder_ckpt, device=torch.device(device), torch_dtype=dtype)
    # Pipeline.__init__ only reads the VAE's compression factors; a placeholder is enough here.
    placeholder_vae = XVAEChunkCausal(
        in_channels=3, out_channels=3, patch_size=2, latent_channels=4, layers_per_block=1,
        block_in_channels=(8, 8), temporal_downsample=(False, False), chunk_size=1,
    )
    pipe = Pipeline(vae=placeholder_vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=None,
                    scheduler=FlowMatchDiscreteScheduler(), args=cfg)
    frame = image.convert("RGB").resize((width, height), getattr(Image, "Resampling", Image).BICUBIC)
    with torch.no_grad():
        embeds, _ = pipe.encode_prompt(
            prompt=[f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n"],
            images=[frame],
            device=torch.device(device),
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            template_type="video",
        )
    return embeds[0].to(torch.float32).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--image", required=True, help="first frame of the stream / video")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    embeds = encode(args.text_encoder, Image.open(args.image), args.prompt, width=args.width, height=args.height,
                    device=args.device, dtype=dtype)
    np.save(args.out, embeds)
    print(f"wrote {args.out}: {embeds.shape}")


if __name__ == "__main__":
    main()
