"""CPU video-to-video streaming built on the joyedge DiT engine.

Two pieces:

* ``LitePipeline`` - a drop-in for ``xvideo.models.pipeline.Pipeline`` that holds only
  the VAE, the DiT and the scheduler.  Condition embeddings come from a callable, so the
  7B MLLM encoder does not have to live in the same process (or on the same machine).
  It lets the stock ``JoyOmniV2VStreamingSession`` run on a CPU-only box.

* ``EdgeV2VSession`` - the same streaming algorithm, single-threaded and CUDA-free, with
  the DiT replaced by the quantized ggml engine (``EdgeDiT``).  The VAE stays in PyTorch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

os.environ.setdefault("JOYOMNI_FA4_ATTN", "0")

import numpy as np
import torch
from PIL import Image

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from .engine import EdgeDiT

PromptEncoder = Callable[[str, Optional[Sequence[Image.Image]]], torch.Tensor]


def _pipeline_cls():
    from xvideo.models.pipeline import Pipeline
    return Pipeline


class LitePipeline:
    """VAE + DiT + scheduler; the methods the streaming session needs are borrowed
    verbatim from the reference ``Pipeline`` class."""

    _BORROWED = (
        "_kv_cache_memory_id", "_KV_CACHE_ID_REF_IMAGE", "_get_chunk_window", "_chunk_frame_bounds",
        "_gather_window_temporal_ids", "_prefill_static_reference_kv_cache",
        "_store_clean_chunk_kv_cache", "_resolve_streaming_chunk_size", "_resolve_global_sink_chunk",
        "_encode_vae_single", "normalize_latents", "denormalize_latents",
    )

    def __init__(self, *, vae, transformer, scheduler, prompt_encoder: PromptEncoder):
        cls = type(self)
        if not getattr(cls, "_borrowed_done", False):
            ref = _pipeline_cls()
            for name in self._BORROWED:
                setattr(cls, name, ref.__dict__[name])
            cls._borrowed_done = True
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.prompt_encoder = prompt_encoder
        self.vae_scale_factor = vae.ffactor_spatial
        self.vae_scale_factor_temporal = vae.ffactor_temporal

    def encode_prompt(self, prompt, images=None, device=None, **_):
        text = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
        embeds = self.prompt_encoder(text, images)
        if embeds.ndim == 2:
            embeds = embeds[None]
        param = next(self.transformer.parameters())
        embeds = embeds.to(device=param.device, dtype=param.dtype)
        return embeds, torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)

    def set_progress_bar_config(self, **_):
        pass


def _to_tensor(frames: Sequence[Image.Image], width: int, height: int) -> torch.Tensor:
    resample = getattr(Image, "Resampling", Image).BICUBIC
    arrs = []
    for f in frames:
        f = f.convert("RGB")
        if f.size != (width, height):
            f = f.resize((width, height), resample)
        arrs.append(np.asarray(f, dtype=np.uint8))
    u8 = torch.from_numpy(np.stack(arrs, axis=0))           # [T, H, W, 3]
    return u8.permute(3, 0, 1, 2)[None].to(torch.float32) / 127.5 - 1.0   # [1, 3, T, H, W]


class EdgeV2VSession:
    """Causal streaming video editing on CPU.

    ``push_frames`` takes source frames as they arrive (1 for the first call, then
    ``frames_per_chunk`` at a time) and returns the edited frames as uint8 [T, H, W, 3].
    """

    def __init__(self, vae, dit: EdgeDiT, *, width: int, height: int, seed: int = 42,
                 enable_denormalization: bool = True, sample_posterior: bool = True,
                 vae_dtype: torch.dtype = torch.float32):
        self.vae = vae.eval()
        self.dit = dit
        self.width, self.height = width, height
        self.ffactor_t = int(vae.ffactor_temporal)
        stem = vae.stem
        if height % (stem.stride * 8) or width % (stem.stride * 8):
            raise ValueError(f"size must be a multiple of {stem.stride * 8}, got {width}x{height}")
        self.lat_h = height * stem.group // stem.stride // int(vae.ffactor_spatial)
        self.lat_w = width * stem.group // stem.stride // int(vae.ffactor_spatial)
        if (self.lat_h, self.lat_w) != (dit.lat_h, dit.lat_w):
            raise ValueError(f"DiT session grid {(dit.lat_h, dit.lat_w)} != VAE grid {(self.lat_h, self.lat_w)}")
        self.latent_channels = int(vae.config.latent_channels)
        self.enable_denormalization = enable_denormalization
        self.sample_posterior = sample_posterior
        self.vae_dtype = vae_dtype
        self.generator = torch.Generator().manual_seed(seed)
        self._mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        self._std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        self.chunk_idx = 0
        self._prev_source: Optional[torch.Tensor] = None
        self._pseudo_latent: Optional[torch.Tensor] = None
        self.timings: dict = {}

    @property
    def frames_per_next_chunk(self) -> int:
        return 1 if self.chunk_idx == 0 else self.ffactor_t

    # -- VAE helpers ---------------------------------------------------------- #
    @torch.no_grad()
    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        dist = self.vae.encode(pixels.to(self.vae_dtype)).latent_dist
        return (dist.sample() if self.sample_posterior else dist.mode()).to(torch.float32)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._mean) / self._std if self.enable_denormalization else z

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._std + self._mean if self.enable_denormalization else z

    # -- conditioning ----------------------------------------------------------- #
    def set_prompt(self, embeds) -> None:
        e = embeds.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(embeds) else np.asarray(embeds)
        self.dit.set_prompt(e[0] if e.ndim == 3 else e)

    @torch.no_grad()
    def set_reference_image(self, image: Image.Image, basesize: int = 768) -> None:
        from xvideo.config import generate_video_image_bucket
        from xvideo.utils import _dynamic_resize_from_bucket

        buckets = generate_video_image_bucket(img_basesize=basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0)
        resized, _ = _dynamic_resize_from_bucket(image.convert("RGB"), bucket_configs=buckets, num_frames=1,
                                                 num_items=1, return_bucket=True)
        pixel = torch.from_numpy(np.array(resized)).permute(2, 0, 1).to(torch.float32) / 127.5 - 1.0
        z = self._normalize(self._encode(pixel[None, :, None]))[:, :, :1]
        self.dit.set_reference_image(z[0, :, 0].numpy())

    # -- streaming ------------------------------------------------------------- #
    @torch.no_grad()
    def push_frames(self, frames: Sequence[Image.Image], *, static: bool = False) -> np.ndarray:
        import time

        if len(frames) != self.frames_per_next_chunk:
            raise ValueError(f"expected {self.frames_per_next_chunk} frames, got {len(frames)}")
        t0 = time.perf_counter()
        new = _to_tensor(frames, self.width, self.height)
        window = new if self.chunk_idx == 0 else torch.cat([self._prev_source, new], dim=2)
        self._prev_source = window[:, :, -1:].clone()
        src = self._normalize(self._encode(window))[:, :, -1:]
        t1 = time.perf_counter()

        noise = torch.randn((1, self.latent_channels, 1, self.lat_h, self.lat_w), generator=self.generator)
        out = self.dit.process_chunk(noise[0, :, 0].numpy(), src[0, :, 0].numpy(), update_history=not static)
        t2 = time.perf_counter()

        z = self._denormalize(torch.from_numpy(out)[None, :, None])
        if self.chunk_idx > 0:
            z = torch.cat([self._pseudo_latent, z], dim=2)
        pixels = self.vae.decode(z.to(self.vae_dtype), return_dict=False)[0].to(torch.float32)
        if self.chunk_idx > 0:
            pixels = pixels[:, :, -self.ffactor_t:]
        # the decoder is causal: the next chunk is decoded behind a re-encoding of this chunk's last frame
        self._pseudo_latent = self._encode(pixels[:, :, -1:])
        t3 = time.perf_counter()

        self.chunk_idx += 1
        self.timings = {"vae_encode_s": t1 - t0, "dit_s": t2 - t1, "vae_decode_s": t3 - t2, "total_s": t3 - t0}
        u8 = (pixels / 2 + 0.5).clamp(0, 1).mul(255.0).round().to(torch.uint8)
        return u8[0].permute(1, 2, 3, 0).contiguous().numpy()
