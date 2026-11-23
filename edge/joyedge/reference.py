"""Pure-PyTorch streaming rollout of the deploy DiT (CPU friendly).

This is the eager path of ``deploy/xvideo/serving/joyomni_streaming.py``
(``JoyOmniV2VStreamingSession._denoise_chunk``) reduced to the DiT: no VAE, no text
encoder, no CUDA graph.  It is the numerical reference for the ggml engine and a
runnable (slow) fallback on any device.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("JOYOMNI_FA4_ATTN", "0")

import torch

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from xvideo.models.dit import SELF_ATTN_MODE_REF_IMAGE_CACHE, Transformer3DModel  # noqa: E402
from xvideo.models.scheduler import FlowMatchDiscreteScheduler  # noqa: E402

REF_IMAGE_MEM_ID = -1


def chunk_window(chunk_idx: int, window_size: int, global_sink_chunk: bool) -> list[int]:
    """Same selection as Pipeline._get_chunk_window."""
    if global_sink_chunk and chunk_idx > 0:
        tail_start = max(1, chunk_idx - max(window_size - 1, 1) + 1)
        return [0] + list(range(tail_start, chunk_idx + 1))
    return list(range(max(0, chunk_idx - window_size + 1), chunk_idx + 1))


def window_temporal_ids(selected: list[int], max_temporal_ids: Optional[int]) -> torch.Tensor:
    """Same positions as Pipeline._gather_window_temporal_ids for chunk_size == 1."""
    if max_temporal_ids is None:
        return torch.arange(len(selected), dtype=torch.long)
    abs_ids = torch.tensor(selected, dtype=torch.long)
    shift = (abs_ids.max() - int(max_temporal_ids)).clamp_min(0)
    return (abs_ids - shift).clamp_min(0)


class TorchStreamingDiT:
    """Chunk-by-chunk editing with the reference PyTorch transformer."""

    def __init__(self, model: Transformer3DModel, *, n_steps: int = 2, shift: float = 5.159,
                 max_temporal_ids: Optional[int] = 2):
        self.model = model.eval()
        self.n_steps = n_steps
        self.mti = max_temporal_ids
        self.scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=shift)
        self.window_size = int(model.config.local_window_size)
        self.global_sink = bool(model.config.global_sink_chunk)
        self.prompt: Optional[torch.Tensor] = None
        self.has_ref = False
        self.chunk_idx = 0
        self.model.reset_inference_kv_cache()

    def set_prompt(self, embeds: torch.Tensor) -> None:
        self.prompt = embeds if embeds.ndim == 3 else embeds[None]

    @torch.no_grad()
    def set_reference_image(self, latent: torch.Tensor) -> None:
        """latent: [C, h, w]"""
        ref = latent[None, :, None]
        with self.model.cache_context("cond"):
            self.model(
                hidden_states=ref,
                timestep=torch.zeros((1,), dtype=ref.dtype),
                encoder_hidden_states=self.prompt,
                current_temporal_ids=torch.zeros((1, 1), dtype=torch.long),
                kv_cache_mode="store",
                kv_cache_scope="cond",
                kv_cache_chunk_id=REF_IMAGE_MEM_ID,
                kv_cache_selected_chunk_ids=[],
                self_attn_input_mode=SELF_ATTN_MODE_REF_IMAGE_CACHE,
                skip_text_stream=True,
            )
        self.has_ref = True

    @torch.no_grad()
    def process_chunk(self, noise: torch.Tensor, src_latent: torch.Tensor) -> torch.Tensor:
        """noise / src_latent: [C, h, w] -> edited latent [C, h, w]"""
        k = self.chunk_idx
        x = noise[None, :, None]
        src = src_latent[None, :, None]

        selected = chunk_window(k, self.window_size, self.global_sink)
        history = selected[:-1]
        ids = window_temporal_ids(selected, self.mti)
        cur_ids, cached_ids = ids[-1:], (ids[:-1] if len(ids) > 1 else None)
        memory_ids = list(history) + ([REF_IMAGE_MEM_ID] if self.has_ref else [])

        self.scheduler.set_timesteps(self.n_steps)
        for t in self.scheduler.timesteps:
            with self.model.cache_context("cond"):
                pred = self.model(
                    hidden_states=x,
                    timestep=t.repeat(1),
                    encoder_hidden_states=self.prompt,
                    ref_video_latent=src,
                    current_temporal_ids=cur_ids[None],
                    cached_temporal_ids=cached_ids[None] if cached_ids is not None else None,
                    kv_cache_mode="reuse",
                    kv_cache_scope="cond",
                    kv_cache_chunk_id=k,
                    kv_cache_selected_chunk_ids=memory_ids,
                    kv_cache_pre_rope=True,
                )[0]
            x = self.scheduler.step(pred, t, x.clone(), return_dict=False)[0].to(noise.dtype)

        keep = set(history) | ({REF_IMAGE_MEM_ID} if self.has_ref else set())
        self.model.evict_kv_cache_chunks(keep)
        with self.model.cache_context("cond"):
            self.model(
                hidden_states=x,
                timestep=torch.zeros((1,), dtype=x.dtype),
                encoder_hidden_states=self.prompt,
                current_temporal_ids=cur_ids[None],
                cached_temporal_ids=None,
                kv_cache_mode="store",
                kv_cache_scope="cond",
                kv_cache_chunk_id=k,
                kv_cache_selected_chunk_ids=[],
                kv_cache_pre_rope=True,
                skip_text_stream=True,
            )
        nxt = set(chunk_window(k + 1, self.window_size, self.global_sink))
        self.model.evict_kv_cache_chunks(nxt | ({REF_IMAGE_MEM_ID} if self.has_ref else set()))
        self.chunk_idx += 1
        return x[0, :, 0]


def build_random_dit(*, hidden_size: int = 256, heads_num: int = 2, depth: int = 3, channels: int = 16,
                     text_dim: int = 64, rope_dim_list=(16, 56, 56), seed: int = 0,
                     dtype: torch.dtype = torch.float32) -> Transformer3DModel:
    """A randomly initialised DiT with the released topology (for tests / benchmarks)."""
    g = torch.Generator().manual_seed(seed)
    model = Transformer3DModel(
        hidden_size=hidden_size, in_channels=channels, out_channels=channels, heads_num=heads_num,
        mm_double_blocks_depth=depth, patch_size=[1, 1, 1], rope_dim_list=list(rope_dim_list),
        text_states_dim=text_dim, theta=256, chunk_size=1, causal=True, local_window_size=3,
        global_sink_chunk=True, use_inference_kv_cache=True,
        source_id_rope_dim=min(128, hidden_size // heads_num),
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("_norm.weight"):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            elif name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            elif name.endswith("modulate_table"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            elif p.ndim >= 2:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=g) * (0.8 / fan_in ** 0.5))
    return model.to(dtype).eval()
