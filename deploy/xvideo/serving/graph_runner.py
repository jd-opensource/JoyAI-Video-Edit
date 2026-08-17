from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import torch

from xvideo.models.dit.rope import apply_rotary_emb

GRAPH_ENV = "JOYOMNI_CUDA_GRAPH"

# The captured graph's KV pool is laid out as exactly [sink, prev1(, ref)]:
# an eager window [0, k-1, k] of this length is the only shape it can serve.
GRAPH_WINDOW_CHUNKS = 3


def graph_env_enabled() -> bool:
    return os.environ.get(GRAPH_ENV, "1").lower() in {"1", "true", "yes", "on"}


class StreamingGraphRunner:
    def __init__(
        self,
        transformer,
        *,
        chunk_tokens: int,
        ref_tokens: int,
        latent_shape: tuple[int, ...],
        txt_len: int,
        max_temporal_ids: int,
        pos_freqs: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        autocast_ctx: Callable[[], Any],
    ):
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.autocast_ctx = autocast_ctx
        self.num_layers = len(transformer.double_blocks)
        heads = int(transformer.config.heads_num)
        head_dim = int(transformer.config.hidden_size) // heads

        self.chunk_tokens = chunk_tokens
        self.ref_tokens = int(ref_tokens)
        self.txt_len = int(txt_len)
        self.max_temporal_ids = int(max_temporal_ids)
        self.sink_off = 0
        self.prev1_off = chunk_tokens
        self.ref_off = 2 * chunk_tokens
        self.pool_len = 2 * chunk_tokens + self.ref_tokens

        L = self.num_layers
        self.pool_k = torch.zeros(L, 1, self.pool_len, heads, head_dim, device=device, dtype=dtype)
        self.pool_v = torch.zeros_like(self.pool_k)
        self.stage_k = torch.zeros(L, 1, chunk_tokens, heads, head_dim, device=device, dtype=dtype)
        self.stage_v = torch.zeros_like(self.stage_k)
        self.pub_k = torch.zeros(2, L, 1, chunk_tokens, heads, head_dim, device=device, dtype=dtype)
        self.pub_v = torch.zeros_like(self.pub_k)
        self._pub_parity = 0

        cos, sin = pos_freqs
        ct = chunk_tokens
        # Rope table for cached-KV positions 0..mti-1: row p = freqs of one chunk at temporal position p.
        self.pos_cos = cos.squeeze(0).view(self.max_temporal_ids, ct, -1).contiguous()
        self.pos_sin = sin.squeeze(0).view(self.max_temporal_ids, ct, -1).contiguous()
        # Writable buffer read by run_commit inside the graph; row is swapped per chunk via set_positions.
        self.commit_cos = self.pos_cos[self.max_temporal_ids - 1].clone().unsqueeze(0)
        self.commit_sin = self.pos_sin[self.max_temporal_ids - 1].clone().unsqueeze(0)

        self.in_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_timestep = torch.zeros((latent_shape[0],), device=device, dtype=torch.float32)
        self.in_ref_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_store_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_store_timestep = torch.zeros((latent_shape[0],), device=device, dtype=dtype)
        self.in_prompt = torch.zeros((1, self.txt_len, int(transformer.config.text_states_dim)),
                                     device=device, dtype=dtype)
        self.in_current_ids = torch.full((1, latent_shape[2]), self.max_temporal_ids,
                                         device=device, dtype=torch.long)

        self.full_graph: Optional[torch.cuda.CUDAGraph] = None
        self.in_noise = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.out_latents: Optional[torch.Tensor] = None
        self.pool_dirty = True
        self.ready = False
        self.bound_token: Optional[int] = None
        self.last_graph_chunk = -(10 ** 9)

    def _pool_assembler(self, layer_idx, *, device, dtype, cached_freqs_cis=None):
        return self.pool_k[layer_idx], self.pool_v[layer_idx]

    def _stage_writer(self, layer_idx, key, value):
        self.stage_k[layer_idx].copy_(key)
        self.stage_v[layer_idx].copy_(value)

    def capture(self, *, timesteps: torch.Tensor, sigmas: torch.Tensor) -> None:
        tf = self.transformer
        t0 = time.time()
        mem_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize(self.device)

        def run_denoise():
            with self.autocast_ctx():
                with tf.cache_context("cond"):
                    return tf(
                        hidden_states=self.in_latent,
                        timestep=self.in_timestep,
                        encoder_hidden_states=self.in_prompt,
                        ref_video_latent=self.in_ref_latent,
                        current_temporal_ids=self.in_current_ids,
                        cached_temporal_ids=None,
                        kv_cache_mode="reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=None,
                        kv_cache_selected_chunk_ids=[],
                        kv_cache_pre_rope=True,
                    )[0]

        def run_store():
            with self.autocast_ctx():
                with tf.cache_context("cond"):
                    tf(
                        hidden_states=self.in_store_latent,
                        timestep=self.in_store_timestep,
                        encoder_hidden_states=self.in_prompt,
                        current_temporal_ids=self.in_current_ids,
                        cached_temporal_ids=None,
                        kv_cache_mode="store",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=None,
                        kv_cache_selected_chunk_ids=[],
                        kv_cache_pre_rope=True,
                        skip_text_stream=True,
                    )

        def run_commit():
            seg = slice(self.prev1_off, self.prev1_off + self.chunk_tokens)
            for li in range(self.num_layers):
                roped = apply_rotary_emb(self.stage_k[li], (self.commit_cos, self.commit_sin))
                self.pool_k[li, :, seg].copy_(roped)
                self.pool_v[li, :, seg].copy_(self.stage_v[li])

        ts_vals = [float(t) for t in timesteps]
        dt_vals = [float(sigmas[i + 1] - sigmas[i]) for i in range(len(ts_vals))]

        def run_full():
            lat32 = self.in_noise.to(torch.float32)
            tf._graph_kv_assembler = self._pool_assembler
            tf._graph_kv_writer = None
            for t_val, dt in zip(ts_vals, dt_vals):
                self.in_latent.copy_(lat32.to(self.dtype))
                self.in_timestep.fill_(t_val)
                pred = run_denoise()
                lat32 = lat32 + pred.to(torch.float32) * dt
            self.in_store_latent.copy_(lat32.to(self.dtype))
            tf._graph_kv_assembler = None
            tf._graph_kv_writer = self._stage_writer
            try:
                run_store()
            finally:
                tf._graph_kv_writer = None
            run_commit()
            return lat32

        try:
            run_full()
            torch.cuda.synchronize(self.device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=mem_pool, capture_error_mode="thread_local"):
                self.out_latents = run_full()
            self.full_graph = g
        finally:
            tf._graph_kv_assembler = None
            tf._graph_kv_writer = None

        torch.cuda.synchronize(self.device)
        self.ready = True
        static_gib = sum(t.numel() * t.element_size() for t in (
            self.pool_k, self.pool_v, self.stage_k, self.stage_v, self.pub_k, self.pub_v)) / 2 ** 30
        print(f"#####[GRAPH] captured whole-chunk graph in {time.time() - t0:.1f}s "
              f"(static bufs {static_gib:.2f} GiB, txt_len={self.txt_len}, "
              f"ref_tokens={self.ref_tokens}, mti={self.max_temporal_ids})", flush=True)

    def bind_session(self, token: int, prompt_embeds: torch.Tensor) -> None:
        self.in_prompt.copy_(prompt_embeds.to(dtype=self.dtype))
        self.bound_token = token
        self.pool_dirty = True
        self.last_graph_chunk = -(10 ** 9)

    def _pos_freqs(self, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos_cos[pos].unsqueeze(0), self.pos_sin[pos].unsqueeze(0)

    def set_positions(self, chunk_idx: int) -> None:
        """Point the graph's position-dependent buffers at chunk_idx's temporal layout.

        Mirrors _gather_window_temporal_ids for the window [0, k-1, k]:
        current sits at min(k, mti), the committed KV (prev1 of chunk k+1) at min(k, mti-1).
        """
        mti = self.max_temporal_ids
        self.in_current_ids.fill_(min(chunk_idx, mti))
        commit_pos = min(chunk_idx, mti - 1)
        self.commit_cos.copy_(self.pos_cos[commit_pos].unsqueeze(0))
        self.commit_sin.copy_(self.pos_sin[commit_pos].unsqueeze(0))

    def _seed_segment(self, chunk_store: dict, off: int, length: int,
                      freqs: Optional[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for li in range(self.num_layers):
            entry = chunk_store[li]
            k = entry["key"].to(device=self.device, dtype=self.dtype)
            v = entry["value"].to(device=self.device, dtype=self.dtype)
            if entry.get("pre_rope", False):
                if freqs is None:
                    raise RuntimeError("graph seed: pre-rope entry in a no-rope segment")
                k = apply_rotary_emb(k, freqs)
            self.pool_k[li, :, off:off + length].copy_(k)
            self.pool_v[li, :, off:off + length].copy_(v)

    def seed_from_cache(self, scope_store: dict, *, sink_mem_id: int,
                        prev1_mem_id: int, prev1_pos: int, ref_mem_id: Optional[int]) -> None:
        for mem_id, off, freqs in (
            (sink_mem_id, self.sink_off, self._pos_freqs(0)),
            (prev1_mem_id, self.prev1_off, self._pos_freqs(prev1_pos)),
        ):
            store = scope_store.get(mem_id)
            if store is None:
                raise RuntimeError(f"graph seed: python cache missing mem_id={mem_id}")
            self._seed_segment(store, off, self.chunk_tokens, freqs)
        if self.ref_tokens > 0:
            store = scope_store.get(ref_mem_id)
            if store is None:
                raise RuntimeError(f"graph seed: python cache missing ref mem_id={ref_mem_id}")
            self._seed_segment(store, self.ref_off, self.ref_tokens, None)
        self.pool_dirty = False

    def publish_to_cache(self, scope_store: dict, mem_id: int) -> None:
        p = self._pub_parity
        self._pub_parity ^= 1
        self.pub_k[p].copy_(self.stage_k)
        self.pub_v[p].copy_(self.stage_v)
        chunk_store = {}
        for li in range(self.num_layers):
            chunk_store[li] = {
                "key": self.pub_k[p, li],
                "value": self.pub_v[p, li],
                "pre_rope": True,
            }
        scope_store[mem_id] = chunk_store
