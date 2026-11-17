"""Time the streaming VAE (encode / decode / pseudo re-encode) of one chunk on this CPU.

    python -m joyedge.bench_vae --vae deploy/deps/checkpoints/JoyAI-Video-Edit/vae --width 432 --height 240

The DiT is only one of the three stages of a chunk; on a CPU the convolutional VAE can
dominate once the DiT is a small student.  Run this with the released ``vae/`` folder to
see where your device stands (``--random`` uses a toy configuration, for smoke tests only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JOYOMNI_VAE_COMPILE", "0")

import torch

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae", help="folder with config.json + diffusion_pytorch_model.safetensors")
    ap.add_argument("--random", action="store_true", help="toy random VAE instead of real weights")
    ap.add_argument("--width", type=int, default=432)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from xvideo.models.vae import XVAEChunkCausal

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.random:
        vae = XVAEChunkCausal(
            in_channels=3, out_channels=3, patch_size=2, latent_channels=64, layers_per_block=1,
            block_in_channels=(16, 32, 64, 64), temporal_downsample=(True, True, True, False), chunk_size=8,
        )
    elif args.vae:
        vae = XVAEChunkCausal.from_pretrained(args.vae, torch_dtype=dtype)
    else:
        raise SystemExit("pass --vae <folder> or --random")
    vae = vae.to(dtype).eval().requires_grad_(False)

    stem = vae.stem
    ft, fs = int(vae.ffactor_temporal), int(vae.ffactor_spatial)
    lat_h = args.height * stem.group // stem.stride // fs
    lat_w = args.width * stem.group // stem.stride // fs
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE {n_params / 1e6:.1f}M params | {args.width}x{args.height} -> latent {lat_w}x{lat_h} "
          f"x{vae.config.latent_channels}ch | {ft} frames per chunk | threads {torch.get_num_threads()}")

    enc = dec = pse = 0.0
    with torch.no_grad():
        for k in range(args.chunks):
            window = torch.randn(1, 3, 1 + ft, args.height, args.width, dtype=dtype).clamp(-1, 1)
            t0 = time.perf_counter()
            vae.encode(window).latent_dist.mode()
            t1 = time.perf_counter()
            z = torch.randn(1, vae.config.latent_channels, 2, lat_h, lat_w, dtype=dtype)
            pixels = vae.decode(z, return_dict=False)[0]
            t2 = time.perf_counter()
            vae.encode(pixels[:, :, -1:]).latent_dist.mode()
            t3 = time.perf_counter()
            print(f"chunk {k}: encode {t1 - t0:7.3f}s  decode {t2 - t1:7.3f}s  pseudo re-encode {t3 - t2:7.3f}s")
            if k > 0 or args.chunks == 1:
                enc += t1 - t0; dec += t2 - t1; pse += t3 - t2
    n = max(1, args.chunks - 1)
    total = (enc + dec + pse) / n
    print(f"\nVAE per chunk: {total:.3f}s (encode {enc / n:.3f} + decode {dec / n:.3f} + pseudo {pse / n:.3f})"
          f"  -> VAE-only ceiling {ft / total:.2f} fps")


if __name__ == "__main__":
    main()
