"""End-to-end on CPU, frames in -> frames out, with random weights (no checkpoint):

  A. the stock ``JoyOmniV2VStreamingSession`` (deploy/) running on a CUDA-less machine
     thanks to the portable op backend -- PyTorch DiT + PyTorch VAE;
  B. ``EdgeV2VSession`` -- the ggml engine for the DiT + the same PyTorch VAE.

Both must produce the same video.

    edge/engine/build.sh && python edge/tests/test_cpu_streaming.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["JOYOMNI_PORTABLE_OPS"] = "1"
os.environ["JOYOMNI_FA4_ATTN"] = "0"
os.environ["JOYOMNI_PRECISION"] = "fp32"
os.environ["JOYOMNI_VAE_COMPILE"] = "0"
os.environ["JOYOMNI_FP8_IMG"] = "0"
os.environ["JOYOMNI_FP8_TXT"] = "0"

import numpy as np
import torch
from PIL import Image

EDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE))
sys.path.insert(0, str(EDGE.parent / "deploy"))

from joyedge.engine import EdgeDiT  # noqa: E402
from joyedge.export_gguf import write_gguf  # noqa: E402
from joyedge.pipeline import EdgeV2VSession, LitePipeline  # noqa: E402
from joyedge.reference import build_random_dit  # noqa: E402

from xvideo.config import ExpConfig  # noqa: E402
from xvideo.models.scheduler import FlowMatchDiscreteScheduler  # noqa: E402
from xvideo.models.vae import vae as vae_mod  # noqa: E402
from xvideo.serving.joyomni_streaming import (  # noqa: E402
    JoyOmniRuntime, StreamingSettings, _clone_vae_shared_weights,
)

W, H, LATENT_C, TEXT_DIM, N_TXT, N_CHUNKS = 240, 144, 16, 64, 9, 4


def build_vae() -> torch.nn.Module:
    torch.manual_seed(1)
    vae = vae_mod.XVAEChunkCausal(
        in_channels=3, out_channels=3, patch_size=2, latent_channels=LATENT_C, layers_per_block=1,
        block_in_channels=(16, 32, 64, 64), temporal_downsample=(True, True, True, False), chunk_size=8,
        latents_mean=tuple(np.linspace(-0.2, 0.2, LATENT_C).tolist()),
        latents_std=tuple(np.linspace(0.8, 1.2, LATENT_C).tolist()),
    )
    assert vae.ffactor_spatial == 16 and vae.ffactor_temporal == 8
    return vae.eval().requires_grad_(False)


def main() -> int:
    # deterministic posterior so that both runs see identical latents
    vae_mod.DiagonalGaussianDistribution.sample = lambda self, generator=None: self.mean

    torch.manual_seed(0)
    vae = build_vae()
    dit = build_random_dit(hidden_size=256, heads_num=2, depth=3, channels=LATENT_C, text_dim=TEXT_DIM)
    prompt_embeds = torch.randn(N_TXT, TEXT_DIM)

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(H // 8, W // 8, 3), dtype=np.uint8)
    frames = []
    for i in range(1 + 8 * (N_CHUNKS - 1)):
        img = Image.fromarray(np.roll(base, shift=i, axis=1)).resize((W, H), Image.BILINEAR)
        frames.append(img)
    ref_image = Image.fromarray(rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8))

    ok = True
    for use_ref in (False, True):
        print(f"=== reference image: {use_ref}")
        # ---- A: stock streaming session on CPU ---------------------------------- #
        pipe = LitePipeline(vae=vae, transformer=dit, scheduler=FlowMatchDiscreteScheduler(shift=5.159),
                            prompt_encoder=lambda text, images: prompt_embeds)
        runtime = JoyOmniRuntime(ExpConfig(), pipe, torch.device("cpu"),
                                 decode_vae=_clone_vae_shared_weights(vae),
                                 pseudo_encode_vae=_clone_vae_shared_weights(vae),
                                 postprocess_device=torch.device("cpu"))
        settings = StreamingSettings(height=H, width=W, num_inference_steps=2, seed=7,
                                     freeze_kv_on_static=False, output_codec="h264")
        sess = runtime.create_v2v_session("make it snow", settings=settings,
                                          ref_image=ref_image if use_ref else None)
        t0 = time.perf_counter()
        results = []
        for f in frames:
            results += sess.push_frame(f)
        deadline = time.time() + 300
        while len(results) < N_CHUNKS and time.time() < deadline:
            r = sess.wait_async_result(timeout=0.1)
            if r is not None:
                results.append(r)
        sess.close()
        ref_video = np.concatenate([np.stack(r.jpegs) for r in results], axis=0)
        t_ref = time.perf_counter() - t0

        # ---- B: ggml engine for the DiT ----------------------------------------- #
        with tempfile.TemporaryDirectory() as tmp:
            gguf_path = Path(tmp) / "tiny.gguf"
            write_gguf(dit.state_dict(), gguf_path, dtype="f32", verbose=False)
            engine = EdgeDiT(str(gguf_path), lat_h=H // 24, lat_w=W // 24, n_steps=2, kv_f32=True)
            edge = EdgeV2VSession(vae, engine, width=W, height=H, seed=7, sample_posterior=False)
            edge.set_prompt(prompt_embeds)
            if use_ref:
                edge.set_reference_image(ref_image)
            t0 = time.perf_counter()
            out, i = [], 0
            while i < len(frames):
                n = edge.frames_per_next_chunk
                out.append(edge.push_frames(frames[i:i + n]))
                i += n
            edge_video = np.concatenate(out, axis=0)
            t_edge = time.perf_counter() - t0
            engine.close()

        assert ref_video.shape == edge_video.shape == (len(frames), H, W, 3), (ref_video.shape, edge_video.shape)
        diff = np.abs(ref_video.astype(np.int16) - edge_video.astype(np.int16))
        mse = float((diff.astype(np.float64) ** 2).mean())
        psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        spread = float(ref_video.std())
        print(f"  frames {ref_video.shape}  pixel std {spread:.1f}  max |diff| {int(diff.max())}  "
              f"mean |diff| {diff.mean():.4f}  PSNR {psnr:.1f} dB")
        print(f"  stock session {t_ref:.2f}s   edge session {t_edge:.2f}s")
        good = spread > 5.0 and psnr > 45.0
        print("  OK" if good else "  FAIL")
        ok &= good

    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
