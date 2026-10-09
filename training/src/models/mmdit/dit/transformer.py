"""Chunk-causal MMDiT for video-editing training and cached inference."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional, Union
from einops import rearrange
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.utils import BaseOutput

from src.distributed.parallel_states import sp_enabled

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

from .posemb_layers import apply_rotary_emb, get_1d_rotary_pos_embed, get_nd_rotary_pos_embed
from .modulate_layers import load_modulation, modulate, apply_gate

BlockMask = Any

_FLEX_COMPILE = os.environ.get("WAN_FLEX_COMPILE", "1").lower() not in {"0", "false", "off", "no"}
_FLEX_LOW_RESOURCE = os.environ.get("WAN_FLEX_LOW_RESOURCE", "1").lower() not in {"0", "false", "off", "no"}

if flex_attention is not None and _FLEX_COMPILE:
    try:
        flex_attention = torch.compile(flex_attention, dynamic=True, mode="default")
    except Exception as exc:
        warnings.warn(
            f"Failed to compile flex_attention, falling back to eager flex attention: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _pad_sequence_length(tensor: torch.Tensor, multiple: int = 128) -> tuple[torch.Tensor, int]:
    seq_len = tensor.shape[2]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return tensor, 0
    return F.pad(tensor, (0, 0, 0, pad_len)), pad_len


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    hidden_states = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return hidden_states.transpose(1, 2)


def _flash_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if flash_attn_func is None:
        raise ImportError("flash_attn is not available.")

    original_dtype = query.dtype
    if query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

    return flash_attn_func(query, key, value, dropout_p=0.0, causal=False).to(original_dtype)


def _run_flash_or_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if flash_attn_func is None or attention_mask is not None:
        return _sdpa_attention(query, key, value, attention_mask=attention_mask)
    try:
        return _flash_attention(query, key, value)
    except (RuntimeError, NotImplementedError, ImportError):
        return _sdpa_attention(query, key, value, attention_mask=attention_mask)


def _get_flex_kernel_options() -> Optional[Dict[str, Any]]:
    if not _FLEX_LOW_RESOURCE:
        return None
    return {
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 32,
        "bwd_BLOCK_M2": 32,
        "bwd_BLOCK_N2": 32,
        "bwd_num_warps": 4,
        "bwd_num_stages": 2,
    }


def _run_flex_or_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Optional[BlockMask],
    attention_mask: Optional[torch.Tensor],
    lazy_dense_mask_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run chunk-causal attention."""
    if block_mask is None:
        if attention_mask is None and lazy_dense_mask_fn is not None:
            attention_mask = lazy_dense_mask_fn()
        return _sdpa_attention(query, key, value, attention_mask=attention_mask)
    if flex_attention is None:
        raise RuntimeError("A block mask requires flex_attention; use a dense mask for SDPA.")

    query_for_flex, pad_len = _pad_sequence_length(query.transpose(1, 2))
    key_for_flex, _ = _pad_sequence_length(key.transpose(1, 2))
    value_for_flex, _ = _pad_sequence_length(value.transpose(1, 2))
    try:
        hidden_states = flex_attention(
            query_for_flex,
            key_for_flex,
            value_for_flex,
            block_mask=block_mask,
            kernel_options=_get_flex_kernel_options(),
        )
    except (RuntimeError, NotImplementedError, ImportError) as exc:
        raise RuntimeError("flex_attention failed; dense fallback is disabled to avoid allocating a full attention matrix.") from exc
    if pad_len > 0:
        hidden_states = hidden_states[:, :, :-pad_len, :]
    return hidden_states.transpose(1, 2)


def _concat_kv_entries(
    entries: Iterable[Dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    keys = []
    values = []
    pre_rope_offset = 0
    for entry in entries:
        key = entry.get("key")
        value = entry.get("value")
        if key is None or value is None:
            continue
        key = key.to(device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            segment_end = pre_rope_offset + key.shape[1]
            segment_freqs = (
                cos_all[..., pre_rope_offset:segment_end, :],
                sin_all[..., pre_rope_offset:segment_end, :],
            )
            key, _ = apply_rotary_emb(key, key, segment_freqs, head_first=False)
            pre_rope_offset = segment_end
        keys.append(key)
        values.append(value)
    if not keys:
        return None, None
    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


def _chunk_causal_visibility(
    query_types: torch.Tensor,
    key_types: torch.Tensor,
    query_chunk_ids: torch.Tensor,
    key_chunk_ids: torch.Tensor,
    active_chunk_ids: torch.Tensor,
    has_clean_tokens: torch.Tensor,
    history_size: int,
    global_sink_chunk: bool,
    relax_history_window: bool,
) -> torch.Tensor:
    """Build causal masks with chunk-local clean self-attention."""
    same_chunk = query_chunk_ids == key_chunk_ids
    chunk_distance = query_chunk_ids - key_chunk_ids
    past_window = (chunk_distance >= 1) & (chunk_distance <= history_size)
    if global_sink_chunk:
        past_window = past_window | ((key_chunk_ids == 0) & (query_chunk_ids >= 1))

    query_is_noisy = query_types == 0
    key_is_noisy = key_types == 0
    query_is_clean = query_types == 3
    key_is_clean = key_types == 3
    key_is_reference = key_types == 1
    query_is_conditioned = query_is_noisy | (query_types == 2) | (query_types == 4)
    key_is_conditioned = key_is_noisy | (key_types == 2) | (key_types == 4)
    reference_self_attention = (query_types == 1) & key_is_reference

    forcing_visible = (
        query_is_conditioned
        & ((key_is_conditioned & same_chunk) | (key_is_clean & past_window) | key_is_reference)
    ) | (query_is_clean & key_is_clean & same_chunk) | reference_self_attention

    query_is_active = query_chunk_ids == active_chunk_ids
    noisy_history_visible = query_is_noisy & key_is_noisy & (
        same_chunk | (past_window & (query_is_active | relax_history_window))
    )
    active_context_visible = query_is_conditioned & query_is_active & (
        (key_is_noisy & (same_chunk | past_window))
        | (((key_types == 2) | (key_types == 4)) & same_chunk)
        | key_is_reference
    )
    rollout_visible = noisy_history_visible | active_context_visible | reference_self_attention
    return torch.where(has_clean_tokens, forcing_visible, rollout_visible) & (query_types >= 0) & (key_types >= 0)


def _modulate_stream(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
    if shift is not None and shift.ndim == x.ndim:
        if scale is None:
            return x + shift
        return x * (1 + scale) + shift
    if scale is not None and scale.ndim == x.ndim:
        return x * (1 + scale) if shift is None else x * (1 + scale) + shift
    return modulate(x, shift=shift, scale=scale)


def _apply_stream_gate(x: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is not None and gate.ndim == x.ndim:
        return x * gate
    return apply_gate(x, gate=gate)


def _run_stream_modulation(module: nn.Module, vec: torch.Tensor, factor: int = 6) -> list[torch.Tensor]:
    if vec.ndim == 4:
        if hasattr(module, "modulate_table"):
            table = module.modulate_table.view(1, 1, factor, -1).to(device=vec.device, dtype=vec.dtype)
            return [part.squeeze(2) for part in (table + vec).chunk(factor, dim=2)]
        return [part.squeeze(2) for part in vec.chunk(factor, dim=2)]
    return module(vec)


@dataclass
class Transformer3DModelOutput(BaseOutput):
    img: torch.Tensor
    txt: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """Initialize root-mean-square normalization with optional affine weight."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """Normalize the last dimension in float32."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMS normalization while preserving the input dtype."""
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):
    """Two-stream MM-DiT block with chunk-causal or cached attention."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: Optional[str] = "wanx",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dit_modulation_type = dit_modulation_type
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

        self.txt_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        txt_freqs_cis: tuple = None,
        attention_mask: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        kv_cache_reader: Optional[Callable[[Optional[int]], Iterable[Dict[str, Any]]]] = None,
        kv_cache_writer: Optional[Callable[[Optional[int], torch.Tensor, torch.Tensor], None]] = None,
        layer_idx: Optional[int] = None,
        skip_text: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: Optional[tuple] = None,
        text_vec_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_modulation = _run_stream_modulation(self.img_mod, vec)
        if vec.ndim == 4:
            if vec.shape[1] < img.shape[1]:
                raise ValueError(
                    f"Per-token modulation length {vec.shape[1]} is shorter than image sequence length {img.shape[1]}."
                )
            img_modulation = [part[:, :img.shape[1]] for part in img_modulation]
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_modulation

        if not skip_text:
            txt_modulation = _run_stream_modulation(self.txt_mod, vec)
            if vec.ndim == 4:
                if text_vec_indices is None:
                    txt_modulation = [part[:, :1] for part in txt_modulation]
                else:
                    indices = text_vec_indices.to(device=vec.device, dtype=torch.long)
                    if indices.ndim == 1:
                        indices = indices.unsqueeze(0).expand(vec.shape[0], -1)
                    gather_indices = indices.unsqueeze(-1).expand(-1, -1, txt_modulation[0].shape[-1])
                    txt_modulation = [torch.gather(part, 1, gather_indices) for part in txt_modulation]
            txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_modulation

        img_modulated = _modulate_stream(self.img_norm1(img), shift=img_shift1, scale=img_scale1)
        img_query, img_key, img_value = rearrange(
            self.img_attn_qkv(img_modulated), "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        img_query = self.img_attn_q_norm(img_query).to(img_value)
        img_key = self.img_attn_k_norm(img_key).to(img_value)
        img_key_for_cache = img_key if kv_cache_pre_rope else None
        if vis_freqs_cis is not None:
            img_query, img_key = apply_rotary_emb(img_query, img_key, vis_freqs_cis, head_first=False)
        if img_key_for_cache is None:
            img_key_for_cache = img_key

        query, key, value = img_query, img_key, img_value
        if not skip_text:
            txt_modulated = _modulate_stream(self.txt_norm1(txt), shift=txt_shift1, scale=txt_scale1)
            txt_query, txt_key, txt_value = rearrange(
                self.txt_attn_qkv(txt_modulated), "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )
            txt_query = self.txt_attn_q_norm(txt_query).to(txt_value)
            txt_key = self.txt_attn_k_norm(txt_key).to(txt_value)
            if txt_freqs_cis is not None:
                txt_query, txt_key = apply_rotary_emb(txt_query, txt_key, txt_freqs_cis, head_first=False)
            query = torch.cat((img_query, txt_query), dim=1)
            key = torch.cat((img_key, txt_key), dim=1)
            value = torch.cat((img_value, txt_value), dim=1)

        use_kv_cache = kv_cache_reader is not None or kv_cache_writer is not None
        if use_kv_cache:
            if kv_cache_writer is not None:
                kv_cache_writer(layer_idx, img_key_for_cache, img_value)
            if kv_cache_reader is not None:
                cached_key, cached_value = _concat_kv_entries(
                    kv_cache_reader(layer_idx),
                    device=query.device,
                    dtype=query.dtype,
                    cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
                )
                if cached_key is not None:
                    key = torch.cat((cached_key, key), dim=1)
                    value = torch.cat((cached_value, value), dim=1)
            attn = _run_flash_or_sdpa(query, key, value)
        else:
            attn = _run_flex_or_sdpa(query, key, value, block_mask, attention_mask)

        attn = attn.flatten(2, 3)
        img_attn = attn[:, :img.shape[1]]
        img = img + _apply_stream_gate(self.img_attn_proj(img_attn), gate=img_gate1)
        img = img + _apply_stream_gate(
            self.img_mlp(_modulate_stream(self.img_norm2(img), shift=img_shift2, scale=img_scale2)),
            gate=img_gate2,
        )
        if not skip_text:
            txt_attn = attn[:, img.shape[1]:]
            txt = txt + _apply_stream_gate(self.txt_attn_proj(txt_attn), gate=txt_gate1)
            txt = txt + _apply_stream_gate(
                self.txt_mlp(_modulate_stream(self.txt_norm2(txt), shift=txt_shift2, scale=txt_scale2)),
                gate=txt_gate2,
            )
        return img, txt


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep_shape = timestep.shape
        timestep_is_sequence = timestep.ndim > 1
        if timestep_is_sequence:
            timestep = timestep.flatten()
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        if timestep_is_sequence:
            temb = temb.reshape(*timestep_shape, temb.shape[-1])
            timestep_proj = timestep_proj.reshape(*timestep_shape, timestep_proj.shape[-1])

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """Chunk-causal MM-DiT for forcing, DMD rollouts, and KV-cache inference."""

    _fsdp_shard_conditions: list = [
        lambda name, module: isinstance(module, (MMDoubleStreamBlock))]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        rope_type: str = 'rope',
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: str = "wanx",
        attn_backend: str = 'flash_attn',
        unpatchify_new: bool = False,
        theta: int = 256,
        chunk_size: Optional[int] = 1,
        local_window_size: int = 4,
        global_sink_chunk: bool = True,
        causal: bool = True,
        use_inference_kv_cache: bool = False,
    ):
        if not causal:
            raise ValueError("Only chunk-causal SFT, DMD rollouts, and streaming inference are supported; set `causal=True`.")
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        self.args = args
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.dit_modulation_type = dit_modulation_type
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.attn_backend = attn_backend
        self.rope_type = rope_type
        self.unpatchify_new = unpatchify_new
        self.theta = theta
        self._initial_use_inference_kv_cache = use_inference_kv_cache

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6 if dit_modulation_type != 'adaLN' else hidden_size,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    dit_modulation_type=self.dit_modulation_type,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            hidden_size, self.out_channels * math.prod(patch_size),
            **factory_kwargs)

        self.enable_source_id_rope = bool(
            getattr(args, "enable_source_id_rope", False)
        )
        self.source_id_rope_dim = int(
            getattr(args, "source_id_rope_dim", hidden_size // heads_num)
        )
        self.source_id_rope_theta = float(
            getattr(args, "source_id_rope_theta", 10000.0)
        )

        self.gradient_checkpointing = args.enable_activation_checkpointing
        self._ensure_kv_cache_state()
        self._use_inference_kv_cache = use_inference_kv_cache

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Ignore unsupported reference-projector and type-embedding weights.
        legacy_keys = (
            "img_in_ref.weight",
            "img_in_ref.bias",
            "cond_type_embed",
            "target_type_embed",
            "clean_type_embed",
            "noise_type_embed",
            "ref_image_type_embed",
            "ref_video_type_embed",
        )
        if any(k in state_dict for k in legacy_keys):
            if not isinstance(state_dict, dict):
                state_dict = dict(state_dict)
            for k in legacy_keys:
                state_dict.pop(k, None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


    def _ensure_kv_cache_state(self) -> None:
        if not hasattr(self, "_inference_kv_cache"):
            self._inference_kv_cache = {"cond": {}, "uncond": {}}
        if not hasattr(self, "_use_inference_kv_cache"):
            self._use_inference_kv_cache = bool(getattr(self, "_initial_use_inference_kv_cache", False))
        if not hasattr(self, "_kv_cache_mode"):
            self._kv_cache_mode = None
        if not hasattr(self, "_kv_cache_scope"):
            self._kv_cache_scope = None
        if not hasattr(self, "_kv_cache_chunk_id"):
            self._kv_cache_chunk_id = None
        if not hasattr(self, "_kv_cache_selected_chunk_ids"):
            self._kv_cache_selected_chunk_ids = None
        if not hasattr(self, "_kv_cache_pre_rope"):
            self._kv_cache_pre_rope = False

    def reset_inference_kv_cache(self) -> None:
        self._ensure_kv_cache_state()
        self._inference_kv_cache = {"cond": {}, "uncond": {}}
        self._use_inference_kv_cache = bool(getattr(self.config, "use_inference_kv_cache", False))
        self._kv_cache_mode = None
        self._kv_cache_scope = None
        self._kv_cache_chunk_id = None
        self._kv_cache_selected_chunk_ids = None
        self._kv_cache_pre_rope = False

    def configure_inference_kv_cache(
        self,
        *,
        scope: Optional[str],
        mode: Optional[str],
        chunk_id: Optional[int] = None,
        selected_chunk_ids: Optional[list[int]] = None,
        pre_rope: bool = False,
    ) -> None:
        self._ensure_kv_cache_state()
        self._kv_cache_scope = scope
        self._kv_cache_mode = mode
        self._kv_cache_chunk_id = chunk_id
        self._kv_cache_selected_chunk_ids = list(selected_chunk_ids) if selected_chunk_ids is not None else None
        self._use_inference_kv_cache = mode in {"reuse", "store", "reuse_store"}
        self._kv_cache_pre_rope = bool(pre_rope)

    @contextmanager
    def cache_context(self, scope: Optional[str]):
        self._ensure_kv_cache_state()
        previous_scope = self._kv_cache_scope
        self._kv_cache_scope = scope
        try:
            yield self
        finally:
            self._kv_cache_scope = previous_scope

    def _get_cache_scope_store(self) -> Optional[Dict[int, Dict[int, Dict[str, Any]]]]:
        self._ensure_kv_cache_state()
        if self._kv_cache_scope is None:
            return None
        return self._inference_kv_cache.setdefault(self._kv_cache_scope, {})

    def _read_layer_kv_cache(self, layer_idx: Optional[int]) -> list[Dict[str, Any]]:
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None:
            return []
        selected_chunk_ids = self._kv_cache_selected_chunk_ids or []
        layer_entries = []
        for selected_chunk_id in selected_chunk_ids:
            chunk_store = scope_store.get(selected_chunk_id)
            if chunk_store is None:
                continue
            entry = chunk_store.get(layer_idx)
            if entry is not None:
                layer_entries.append(entry)
        return layer_entries

    def _write_layer_kv_cache(
        self,
        layer_idx: Optional[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None or self._kv_cache_chunk_id is None:
            return
        chunk_store = scope_store.setdefault(self._kv_cache_chunk_id, {})
        chunk_store[layer_idx] = {
            "key": key.detach().clone(),
            "value": value.detach().clone(),
            "pre_rope": bool(self._kv_cache_pre_rope),
        }

    def _get_inference_cache_chunk_ids(self, scope: Optional[str]) -> list[int]:
        self._ensure_kv_cache_state()
        if scope is None:
            return []
        scope_store = self._inference_kv_cache.get(scope, {})
        return sorted(scope_store.keys())

    def evict_kv_cache_chunks(self, chunk_ids_to_keep: set[int]) -> None:
        self._ensure_kv_cache_state()
        for scope in ("cond", "uncond"):
            scope_store = self._inference_kv_cache.get(scope)
            if scope_store is None:
                continue
            evict_ids = [cid for cid in scope_store if cid not in chunk_ids_to_keep]
            for cid in evict_ids:
                del scope_store[cid]


    def get_rotary_pos_embed(self, vis_rope_size, txt_rope_size=None):
        target_ndim = 3

        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)
                                   ) + vis_rope_size
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim //
                             target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        vis_freqs, txt_freqs = get_nd_rotary_pos_embed(
            rope_dim_list,
            vis_rope_size,
            txt_rope_size=txt_rope_size,
            theta=self.theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return vis_freqs, txt_freqs

    def get_rotary_pos_embed_from_ids(
        self,
        *,
        frame_ids: torch.Tensor,
        spatial_shape: Tuple[int, int],
        txt_rope_size: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        post_patch_height, post_patch_width = spatial_shape
        # Keep RoPE indices on the input device.
        device = frame_ids.device
        frame_ids_dev = frame_ids.to(dtype=torch.float32)
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if frame_ids_dev.numel() % spatial_tokens_per_frame != 0:
            raise ValueError(
                f"`frame_ids` length {frame_ids_dev.numel()} is not divisible by spatial token count {spatial_tokens_per_frame}."
            )

        temporal_positions = frame_ids_dev
        h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
        w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
        h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
        h_positions = h_grid.reshape(-1).repeat(frame_ids_dev.numel() // spatial_tokens_per_frame)
        w_positions = w_grid.reshape(-1).repeat(frame_ids_dev.numel() // spatial_tokens_per_frame)

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // 3 for _ in range(3)]
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal to head_dim of attention layer")

        cos_list = []
        sin_list = []
        for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions)):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                positions,
                theta=self.theta,
                use_real=True,
            )
            cos_list.append(cos)
            sin_list.append(sin)
        vis_freqs = (torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1))

        txt_freqs = None
        if txt_rope_size is not None:
            max_vis_id = torch.stack(
                [temporal_positions.max(), h_positions.max(), w_positions.max()]
            ).max()
            txt_positions = torch.arange(txt_rope_size, dtype=torch.float32, device=device) + max_vis_id + 1
            txt_cos_list = []
            txt_sin_list = []
            for dim in rope_dim_list:
                cos, sin = get_1d_rotary_pos_embed(
                    dim,
                    txt_positions,
                    theta=self.theta,
                    use_real=True,
                )
                txt_cos_list.append(cos)
                txt_sin_list.append(sin)
            txt_freqs = (torch.cat(txt_cos_list, dim=1), torch.cat(txt_sin_list, dim=1))

        return vis_freqs, txt_freqs

    def generate_source_id_rope_from_types(
        self,
        token_types: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build source RoPE: video=1, reference image=2, other tokens=0."""
        role_dim = max(0, min(int(self.source_id_rope_dim), int(head_dim)))
        if role_dim % 2 == 1:
            role_dim -= 1

        half_head = head_dim // 2
        cos_half = torch.ones(*token_types.shape, half_head, device=device, dtype=torch.float32)
        sin_half = torch.zeros(*token_types.shape, half_head, device=device, dtype=torch.float32)
        if role_dim == 0:
            return (
                cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
                sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
            )

        inv_freq = 1.0 / (
            self.source_id_rope_theta
            ** (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
        )

        source_id = torch.zeros_like(token_types, dtype=torch.float32)
        source_id = torch.where(token_types == 1, torch.full_like(source_id, 2.0), source_id)
        source_id = torch.where(token_types == 2, torch.ones_like(source_id), source_id)

        angles = source_id.unsqueeze(-1) * inv_freq
        cos_half[..., : role_dim // 2] = torch.cos(angles)
        sin_half[..., : role_dim // 2] = torch.sin(angles)
        return (
            cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
            sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def _get_patch_shape(latent: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        _, _, num_frames, height, width = latent.shape
        return (
            num_frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )

    @staticmethod
    def _get_token_frame_ids(
        post_patch_shape: Tuple[int, int, int],
        device: torch.device,
        temporal_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_frames, post_patch_height, post_patch_width = post_patch_shape
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_ids is None:
            frame_ids = torch.arange(num_frames, device=device, dtype=torch.long)
        else:
            frame_ids = torch.as_tensor(temporal_ids, device=device, dtype=torch.long)
            if frame_ids.ndim != 1 or frame_ids.numel() != num_frames:
                raise ValueError(
                    f"`temporal_ids` must be 1D with length {num_frames}, got {tuple(frame_ids.shape)}."
                )
        return frame_ids.repeat_interleave(spatial_tokens_per_frame)

    @staticmethod
    def _build_segment_chunk_causal_attention_mask(
        frame_ids: torch.Tensor,
        token_types: torch.Tensor,
        chunk_size: int,
        local_window_size: int,
        global_sink_chunk: bool,
        device: torch.device,
        relax_history_window: bool = False,
    ) -> torch.Tensor:
        if chunk_size <= 0 or local_window_size <= 0:
            raise ValueError("`chunk_size` and `local_window_size` must be positive.")
        frame_ids = torch.as_tensor(frame_ids, device=device, dtype=torch.long)
        token_types = torch.as_tensor(token_types, device=device, dtype=torch.long)
        if frame_ids.shape != token_types.shape:
            raise ValueError("`frame_ids` and `token_types` must have matching shapes.")
        squeeze_batch = frame_ids.ndim == 1
        if squeeze_batch:
            frame_ids = frame_ids.unsqueeze(0)
            token_types = token_types.unsqueeze(0)
        elif frame_ids.ndim != 2:
            raise ValueError("`frame_ids` and `token_types` must be 1D or 2D.")

        chunk_ids = torch.div(frame_ids.clamp_min(0), chunk_size, rounding_mode="floor")
        mask = _chunk_causal_visibility(
            query_types=token_types[:, :, None],
            key_types=token_types[:, None, :],
            query_chunk_ids=chunk_ids[:, :, None],
            key_chunk_ids=chunk_ids[:, None, :],
            active_chunk_ids=chunk_ids.amax(dim=1)[:, None, None],
            has_clean_tokens=(token_types == 3).any(dim=1)[:, None, None],
            history_size=max(local_window_size - (2 if global_sink_chunk else 1), 0),
            global_sink_chunk=global_sink_chunk,
            relax_history_window=relax_history_window,
        ).unsqueeze(1)
        return mask.squeeze(0) if squeeze_batch else mask

    @staticmethod
    def _build_segment_chunk_causal_block_mask(
        frame_ids: torch.Tensor,
        token_types: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor],
        chunk_size: int,
        local_window_size: int,
        global_sink_chunk: bool,
        device: torch.device,
        relax_history_window: bool = False,
    ):
        if create_block_mask is None:
            return None
        if chunk_size <= 0 or local_window_size <= 0:
            raise ValueError("`chunk_size` and `local_window_size` must be positive.")
        frame_ids = torch.as_tensor(frame_ids, device=device, dtype=torch.long)
        token_types = torch.as_tensor(token_types, device=device, dtype=torch.long)
        if frame_ids.ndim != 2 or frame_ids.shape != token_types.shape:
            raise ValueError("`frame_ids` and `token_types` must have matching 2D shapes.")
        if hidden_states_mask is None:
            hidden_states_mask = torch.ones_like(token_types, dtype=torch.bool)
        else:
            hidden_states_mask = torch.as_tensor(hidden_states_mask, device=device, dtype=torch.bool)
            if hidden_states_mask.shape != token_types.shape:
                raise ValueError("`hidden_states_mask` must match `token_types`.")

        batch_size, seq_len = token_types.shape
        padded_seq_len = ((seq_len + 127) // 128) * 128
        chunk_ids = torch.div(frame_ids.clamp_min(0), chunk_size, rounding_mode="floor")
        active_chunk_ids = chunk_ids.amax(dim=1)
        has_clean_tokens = (token_types == 3).any(dim=1)
        history_size = max(local_window_size - (2 if global_sink_chunk else 1), 0)

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            valid_query = query_idx < seq_len
            valid_key = key_idx < seq_len
            safe_query_idx = torch.where(valid_query, query_idx, 0)
            safe_key_idx = torch.where(valid_key, key_idx, 0)
            valid = valid_query & valid_key & hidden_states_mask[batch_idx, safe_query_idx] & hidden_states_mask[batch_idx, safe_key_idx]
            visible = _chunk_causal_visibility(
                query_types=token_types[batch_idx, safe_query_idx],
                key_types=token_types[batch_idx, safe_key_idx],
                query_chunk_ids=chunk_ids[batch_idx, safe_query_idx],
                key_chunk_ids=chunk_ids[batch_idx, safe_key_idx],
                active_chunk_ids=active_chunk_ids[batch_idx],
                has_clean_tokens=has_clean_tokens[batch_idx],
                history_size=history_size,
                global_sink_chunk=global_sink_chunk,
                relax_history_window=relax_history_window,
            )
            return valid & visible

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=padded_seq_len,
            KV_LEN=padded_seq_len,
            device=device,
            BLOCK_SIZE=128,
            _compile=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,  # Diffusion time in [0, 1000].
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        clean_video_latent: Optional[torch.Tensor] = None,
        ref_video_latent: Optional[torch.Tensor] = None,
        ref_image_latent: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        noisy_temporal_ids: Optional[torch.Tensor] = None,
        cached_temporal_ids: Optional[torch.Tensor] = None,
        kv_cache_mode: Optional[str] = None,
        kv_cache_scope: Optional[str] = None,
        kv_cache_chunk_id: Optional[int] = None,
        kv_cache_selected_chunk_ids: Optional[list[int]] = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: Optional[str] = None,
        skip_text_stream: bool = False,
        relax_history_window: bool = False,
        active_chunk_only_conditioning: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | Transformer3DModelOutput:
        """Run forcing or DMD rollouts with kv_cache_mode=None; otherwise store or reuse clean-history KV."""
        if sp_enabled():
            raise NotImplementedError("Chunk-causal attention requires sequence parallel size 1.")
        if self_attn_input_mode not in {None, "ref_image_cache"}:
            raise ValueError(f"Unsupported cache prefill mode: {self_attn_input_mode!r}.")
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states` must be provided for MMDiT.")

        self._ensure_kv_cache_state()
        if kv_cache_mode is not None:
            self.configure_inference_kv_cache(
                scope=kv_cache_scope,
                mode=kv_cache_mode,
                chunk_id=kv_cache_chunk_id,
                selected_chunk_ids=kv_cache_selected_chunk_ids,
                pre_rope=kv_cache_pre_rope,
            )

        if hidden_states.ndim == 6:
            if hidden_states.shape[1] == 1:
                hidden_states = hidden_states[:, 0]
            else:
                raise ValueError(
                    f"`hidden_states` must be 5D (B,C,T,H,W) or 6D with N=1, got shape {tuple(hidden_states.shape)}. "
                    f"Pass reference latents via `ref_video_latent` / `ref_image_latent` instead."
                )
        elif hidden_states.ndim != 5:
            raise ValueError(
                f"`hidden_states` must have shape [B,C,T,H,W], got {tuple(hidden_states.shape)}."
            )

        batch_size = hidden_states.shape[0]
        patch_size = tuple(self.patch_size)
        noisy_patch_shape = self._get_patch_shape(hidden_states, patch_size)
        noisy_seq_len = math.prod(noisy_patch_shape)
        device = hidden_states.device

        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(
                (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )
        else:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(
                device=encoder_hidden_states.device,
                dtype=torch.bool,
            )

        noisy_temporal_ids_tensor = None
        if noisy_temporal_ids is not None:
            noisy_temporal_ids_tensor = torch.as_tensor(noisy_temporal_ids, device=device, dtype=torch.long)
            if noisy_temporal_ids_tensor.ndim == 1:
                noisy_temporal_ids_tensor = noisy_temporal_ids_tensor.unsqueeze(0).expand(batch_size, -1)
            if noisy_temporal_ids_tensor.shape != (batch_size, noisy_patch_shape[0]):
                raise ValueError(
                    f"`noisy_temporal_ids` must have shape {(batch_size, noisy_patch_shape[0])}, "
                    f"got {tuple(noisy_temporal_ids_tensor.shape)}."
                )

        def _project_latent(latent: torch.Tensor) -> torch.Tensor:
            return self.img_in(latent).flatten(2).transpose(1, 2)

        def _rotary_from_shape(
            post_patch_shape: Tuple[int, int, int],
            token_frame_ids: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.get_rotary_pos_embed_from_ids(
                frame_ids=token_frame_ids,
                spatial_shape=(post_patch_shape[1], post_patch_shape[2]),
                txt_rope_size=None,
            )[0]

        def _expand_ref_image_latents(
            latent: Optional[Union[torch.Tensor, list[torch.Tensor]]],
        ) -> list[Optional[torch.Tensor]]:
            if latent is None:
                return [None] * batch_size
            if torch.is_tensor(latent):
                if latent.shape[0] != batch_size:
                    raise ValueError(
                        f"Ref image latent batch size {latent.shape[0]} does not match hidden states batch size {batch_size}."
                    )
                return [latent[idx: idx + 1] for idx in range(batch_size)]
            if len(latent) != batch_size:
                raise ValueError(
                    f"Ref image latent list length {len(latent)} does not match hidden states batch size {batch_size}."
                )
            return latent

        hidden_tokens = _project_latent(hidden_states)
        if self_attn_input_mode == "ref_image_cache":
            noisy_token_types = torch.ones(noisy_seq_len, device=device, dtype=torch.long)
            cache_temporal_ids = torch.zeros(noisy_patch_shape[0], device=device, dtype=torch.long)
            noisy_frame_ids = self._get_token_frame_ids(noisy_patch_shape, device, temporal_ids=cache_temporal_ids)
        else:
            noisy_token_types = torch.zeros(noisy_seq_len, device=device, dtype=torch.long)

        ref_image_latents = _expand_ref_image_latents(ref_image_latent if self_attn_input_mode is None else None)

        clean_tokens = None
        if clean_video_latent is not None:
            if clean_video_latent.shape != hidden_states.shape:
                raise ValueError(
                    f"Clean video latent shape {tuple(clean_video_latent.shape)} does not match "
                    f"hidden states shape {tuple(hidden_states.shape)}."
                )
            clean_tokens = _project_latent(clean_video_latent)

        # Limit source and text conditioning to the active chunk.
        apply_active_only = (
            bool(active_chunk_only_conditioning)
            and clean_video_latent is None
            and self_attn_input_mode is None
        )

        ref_video_tokens = None
        ref_video_patch_shape = None
        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"Ref video latent batch size {ref_video_latent.shape[0]} does not match hidden states batch size {batch_size}."
                )
            ref_video_patch_shape = self._get_patch_shape(ref_video_latent, patch_size)
            if ref_video_patch_shape[1:] != noisy_patch_shape[1:]:
                raise ValueError(
                    "Ref video latent spatial patch shape must match noisy latent spatial patch shape: "
                    f"{ref_video_patch_shape[1:]} != {noisy_patch_shape[1:]}."
                )
            ref_video_tokens = _project_latent(ref_video_latent)

        sample_hidden_states = []
        sample_rotary_cos = []
        sample_rotary_sin = []
        sample_frame_ids = []
        sample_token_types = []
        sample_noisy_indices = []
        sample_ref_video_indices = []
        sample_ref_video_src_noisy_indices = []
        sequence_lengths = []
        max_visual_seq_len = 0
        max_ref_video_len = 0

        for idx in range(batch_size):
            latents_for_sample = []
            rotary_for_sample = []
            frame_ids_for_sample = []
            token_types_for_sample = []

            temporal_ids = noisy_temporal_ids_tensor[idx] if noisy_temporal_ids_tensor is not None else None
            current_noisy_frame_ids = self._get_token_frame_ids(noisy_patch_shape, device, temporal_ids=temporal_ids)
            if self_attn_input_mode == "ref_image_cache":
                current_noisy_frame_ids = noisy_frame_ids
            clean_or_noisy_rotary = _rotary_from_shape(noisy_patch_shape, current_noisy_frame_ids)

            if clean_tokens is not None:
                latents_for_sample.append(clean_tokens[idx: idx + 1])
                rotary_for_sample.append(clean_or_noisy_rotary)
                frame_ids_for_sample.append(current_noisy_frame_ids)
                token_types_for_sample.append(torch.full_like(current_noisy_frame_ids, 3))

            noisy_start = sum(sample.shape[1] for sample in latents_for_sample)
            latents_for_sample.append(hidden_tokens[idx: idx + 1])
            rotary_for_sample.append(clean_or_noisy_rotary)
            frame_ids_for_sample.append(current_noisy_frame_ids)
            token_types_for_sample.append(noisy_token_types)

            image_latent = ref_image_latents[idx]
            if image_latent is not None:
                if image_latent.shape[0] != 1:
                    raise ValueError(f"Each ref image latent must have batch size 1, got {image_latent.shape[0]}.")
                image_latent = image_latent.to(device=device, dtype=hidden_states.dtype)
                image_patch_shape = self._get_patch_shape(image_latent, patch_size)
                image_tokens = _project_latent(image_latent)
                image_temporal_ids = torch.zeros(image_patch_shape[0], device=device, dtype=torch.long)
                image_frame_ids = self._get_token_frame_ids(image_patch_shape, device, temporal_ids=image_temporal_ids)
                latents_for_sample.append(image_tokens)
                rotary_for_sample.append(_rotary_from_shape(image_patch_shape, image_frame_ids))
                frame_ids_for_sample.append(image_frame_ids)
                token_types_for_sample.append(torch.ones(image_tokens.shape[1], device=device, dtype=torch.long))

            ref_video_start = sum(sample.shape[1] for sample in latents_for_sample)
            ref_video_len = 0
            if ref_video_tokens is not None:
                video_patch_shape = ref_video_patch_shape
                shares_noisy_layout = video_patch_shape[0] == noisy_patch_shape[0]
                if shares_noisy_layout:
                    video_temporal_ids = temporal_ids
                else:
                    video_temporal_ids = None

                ref_tok = ref_video_tokens[idx: idx + 1]
                if apply_active_only and shares_noisy_layout:
                    rv_spatial = video_patch_shape[1] * video_patch_shape[2]
                    if video_temporal_ids is not None:
                        rv_frame_vals = torch.as_tensor(video_temporal_ids, device=device, dtype=torch.long)
                    else:
                        rv_frame_vals = torch.arange(video_patch_shape[0], device=device, dtype=torch.long)
                    rv_frame_chunk = rv_frame_vals.clamp_min(0) // self.config.chunk_size
                    active_frame_mask = rv_frame_chunk == int(rv_frame_chunk.max())
                    active_tok_mask = active_frame_mask.repeat_interleave(rv_spatial)
                    video_patch_shape = (
                        int(active_frame_mask.sum()),
                        video_patch_shape[1],
                        video_patch_shape[2],
                    )
                    video_temporal_ids = rv_frame_vals[active_frame_mask]
                    ref_tok = ref_tok[:, active_tok_mask, :]

                video_frame_ids = self._get_token_frame_ids(video_patch_shape, device, temporal_ids=video_temporal_ids)
                latents_for_sample.append(ref_tok)
                rotary_for_sample.append(_rotary_from_shape(video_patch_shape, video_frame_ids))
                frame_ids_for_sample.append(video_frame_ids)
                token_types_for_sample.append(torch.full((ref_tok.shape[1],), 2, device=device, dtype=torch.long))
                ref_video_len = ref_tok.shape[1]

                # Align source-token timesteps with the corresponding target frames.
                if shares_noisy_layout:
                    if apply_active_only:
                        ref_video_src = torch.arange(noisy_seq_len, device=device, dtype=torch.long)[active_tok_mask]
                    else:
                        ref_video_src = torch.arange(ref_video_len, device=device, dtype=torch.long)
                else:
                    ref_video_src = None
            else:
                ref_video_src = None

            sample_hidden_state = torch.cat(latents_for_sample, dim=1)
            sample_hidden_states.append(sample_hidden_state)
            sample_rotary_cos.append(torch.cat([rotary[0] for rotary in rotary_for_sample], dim=0).unsqueeze(0))
            sample_rotary_sin.append(torch.cat([rotary[1] for rotary in rotary_for_sample], dim=0).unsqueeze(0))
            sample_frame_ids.append(torch.cat(frame_ids_for_sample, dim=0))
            sample_token_types.append(torch.cat(token_types_for_sample, dim=0))
            sample_noisy_indices.append(torch.arange(noisy_seq_len, device=device) + noisy_start)
            if ref_video_src is not None:
                sample_ref_video_indices.append(
                    torch.arange(ref_video_len, device=device, dtype=torch.long) + ref_video_start
                )
                sample_ref_video_src_noisy_indices.append(ref_video_src)
            else:
                sample_ref_video_indices.append(torch.empty(0, device=device, dtype=torch.long))
                sample_ref_video_src_noisy_indices.append(torch.empty(0, device=device, dtype=torch.long))
            max_ref_video_len = max(max_ref_video_len, ref_video_len)
            sequence_lengths.append(sample_hidden_state.shape[1])
            max_visual_seq_len = max(max_visual_seq_len, sample_hidden_state.shape[1])

        for idx in range(batch_size):
            pad_len = max_visual_seq_len - sequence_lengths[idx]
            if pad_len <= 0:
                continue
            sample_hidden_states[idx] = F.pad(sample_hidden_states[idx], (0, 0, 0, pad_len))
            sample_frame_ids[idx] = F.pad(sample_frame_ids[idx], (0, pad_len))
            sample_token_types[idx] = F.pad(sample_token_types[idx], (0, pad_len), value=-1)
            sample_rotary_cos[idx] = torch.cat(
                [
                    sample_rotary_cos[idx],
                    sample_rotary_cos[idx].new_zeros(1, pad_len, sample_rotary_cos[idx].shape[-1]),
                ],
                dim=1,
            )
            sample_rotary_sin[idx] = torch.cat(
                [
                    sample_rotary_sin[idx],
                    sample_rotary_sin[idx].new_zeros(1, pad_len, sample_rotary_sin[idx].shape[-1]),
                ],
                dim=1,
            )

        img = torch.cat(sample_hidden_states, dim=0)
        visual_lengths = torch.tensor(sequence_lengths, device=device)
        visual_hidden_mask = torch.arange(max_visual_seq_len, device=device).unsqueeze(0) < visual_lengths.unsqueeze(1)
        visual_frame_ids = torch.stack(sample_frame_ids, dim=0)
        visual_token_types = torch.stack(sample_token_types, dim=0)
        noisy_indices = torch.stack(sample_noisy_indices, dim=0)
        vis_freqs_cis = (torch.cat(sample_rotary_cos, dim=0), torch.cat(sample_rotary_sin, dim=0))

        # Source RoPE IDs: target/text=0, source video=1, reference image=2.
        if self.enable_source_id_rope:
            head_dim = self.hidden_size // self.heads_num
            cos_3d, sin_3d = vis_freqs_cis
            cos_role, sin_role = self.generate_source_id_rope_from_types(
                token_types=visual_token_types,
                head_dim=head_dim,
                device=cos_3d.device,
                dtype=cos_3d.dtype,
            )
            new_cos = cos_3d * cos_role - sin_3d * sin_role
            new_sin = sin_3d * cos_role + cos_3d * sin_role
            vis_freqs_cis = (new_cos, new_sin)

        if timestep.ndim == 2:
            if timestep.shape[0] != batch_size:
                raise ValueError(f"`timestep` batch size {timestep.shape[0]} does not match {batch_size}.")
            if timestep.shape[1] != noisy_seq_len:
                raise ValueError(
                    f"Sequence timestep length {timestep.shape[1]} does not match noisy sequence length {noisy_seq_len}."
                )
            timestep_dev = timestep.to(device=device)
            timestep_for_embed = timestep.new_zeros((batch_size, max_visual_seq_len))
            timestep_for_embed.scatter_(1, noisy_indices, timestep_dev)
            if max_ref_video_len > 0:
                for b in range(batch_size):
                    rv_idx = sample_ref_video_indices[b]
                    rv_src = sample_ref_video_src_noisy_indices[b]
                    if rv_idx.numel() == 0:
                        continue
                    timestep_for_embed[b].index_copy_(0, rv_idx, timestep_dev[b].index_select(0, rv_src))
        else:
            timestep_for_embed = timestep

        _, vec, txt = self.condition_embedder(timestep_for_embed, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(-1, (6, -1))

        # Bind each text replica to one target chunk.
        per_replica_text_len = txt.shape[1]
        chunk_size_cfg = self.config.chunk_size
        noisy_only_frame_ids = noisy_temporal_ids_tensor if noisy_temporal_ids_tensor is not None else None
        if noisy_only_frame_ids is None:
            base_ids = torch.arange(noisy_patch_shape[0], device=device, dtype=torch.long)
            noisy_only_frame_ids = base_ids.unsqueeze(0).expand(batch_size, -1)
        noisy_chunk_ids_per_sample = torch.div(
            noisy_only_frame_ids.clamp_min(0), chunk_size_cfg, rounding_mode="floor"
        )
        if apply_active_only:
            active_id_text = int(noisy_chunk_ids_per_sample[0].max())
            text_chunk_ids = torch.tensor([active_id_text], device=device, dtype=torch.long)
            n_text_replicas = 1
        else:
            text_chunk_ids = torch.unique(noisy_chunk_ids_per_sample[0])
            n_text_replicas = int(text_chunk_ids.numel())

        if n_text_replicas > 1:
            txt = txt.repeat(1, n_text_replicas, 1)
            encoder_hidden_states_mask_replicated = encoder_hidden_states_mask.repeat(1, n_text_replicas)
        else:
            encoder_hidden_states_mask_replicated = encoder_hidden_states_mask

        # Use each chunk's timestep for its text replica.
        text_vec_indices = None
        if timestep.ndim == 2:
            chunk_size_cfg = self.config.chunk_size
            spatial_tokens_per_frame = noisy_patch_shape[1] * noisy_patch_shape[2]
            text_vec_idx_list = []
            for b in range(batch_size):
                per_chunk_first_pos = []
                for replica_c in text_chunk_ids.tolist():
                    chunk_frame_local_idx = (noisy_chunk_ids_per_sample[b] == replica_c).nonzero(as_tuple=False)
                    first_frame_local = int(chunk_frame_local_idx[0].item())
                    first_noisy_pos_in_visual = int(noisy_indices[b, first_frame_local * spatial_tokens_per_frame].item())
                    per_chunk_first_pos.append(first_noisy_pos_in_visual)
                replica_indices = torch.tensor(per_chunk_first_pos, device=device, dtype=torch.long)
                replica_indices = replica_indices.repeat_interleave(per_replica_text_len)
                text_vec_idx_list.append(replica_indices)
            text_vec_indices = torch.stack(text_vec_idx_list, dim=0)

        txt_seq_len = txt.shape[1]

        txt_freqs_cis = None
        if self.rope_type == "mrope":
            _, txt_freqs_cis = self.get_rotary_pos_embed(
                vis_rope_size=noisy_patch_shape,
                txt_rope_size=per_replica_text_len,
            )
            if n_text_replicas > 1 and txt_freqs_cis is not None:
                txt_cos, txt_sin = txt_freqs_cis
                txt_freqs_cis = (
                    txt_cos.repeat(n_text_replicas, 1),
                    txt_sin.repeat(n_text_replicas, 1),
                )

        replica_frame_id_values = text_chunk_ids * self.config.chunk_size
        text_frame_ids = replica_frame_id_values.repeat_interleave(per_replica_text_len)
        text_frame_ids = text_frame_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        text_token_types = torch.full((batch_size, txt_seq_len), 4, device=device, dtype=torch.long)
        full_frame_ids = torch.cat([visual_frame_ids, text_frame_ids], dim=1)
        full_token_types = torch.cat([visual_token_types, text_token_types], dim=1)
        full_hidden_mask = torch.cat([visual_hidden_mask, encoder_hidden_states_mask_replicated.to(device)], dim=1)

        attention_mask = None
        block_mask = None
        use_kv_cache = bool(self._use_inference_kv_cache and kv_cache_mode in {"reuse", "store", "reuse_store"})
        if not use_kv_cache:
            mask_kwargs = dict(
                frame_ids=full_frame_ids,
                token_types=full_token_types,
                chunk_size=self.config.chunk_size,
                local_window_size=self.config.local_window_size,
                global_sink_chunk=self.config.global_sink_chunk,
                device=device,
                relax_history_window=relax_history_window,
            )
            block_mask = self._build_segment_chunk_causal_block_mask(
                hidden_states_mask=full_hidden_mask, **mask_kwargs
            )
            if block_mask is None:
                attention_mask = self._build_segment_chunk_causal_attention_mask(**mask_kwargs)
                attention_mask = attention_mask & full_hidden_mask[:, None, :, None] & full_hidden_mask[:, None, None, :]

        cached_freqs_cis = None
        if kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_ids_tensor = torch.as_tensor(cached_temporal_ids, device=device, dtype=torch.long)
            if cached_ids_tensor.ndim == 1:
                cached_ids_tensor = cached_ids_tensor.unsqueeze(0).expand(batch_size, -1)
            if cached_ids_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"`cached_temporal_ids` batch {cached_ids_tensor.shape[0]} does not match {batch_size}."
                )
            cached_cos_list = []
            cached_sin_list = []
            for idx in range(batch_size):
                cached_frame_ids = self._get_token_frame_ids(
                    (cached_ids_tensor.shape[1], noisy_patch_shape[1], noisy_patch_shape[2]),
                    device,
                    temporal_ids=cached_ids_tensor[idx],
                )
                cached_vis_freqs, _ = self.get_rotary_pos_embed_from_ids(
                    frame_ids=cached_frame_ids,
                    spatial_shape=(noisy_patch_shape[1], noisy_patch_shape[2]),
                    txt_rope_size=None,
                )
                cached_cos_list.append(cached_vis_freqs[0].unsqueeze(0))
                cached_sin_list.append(cached_vis_freqs[1].unsqueeze(0))
            cached_freqs_cis = (
                torch.cat(cached_cos_list, dim=0),
                torch.cat(cached_sin_list, dim=0),
            )

        for layer_idx, block in enumerate(self.double_blocks):
            block_kwargs = dict(
                attention_mask=attention_mask,
                block_mask=block_mask,
                layer_idx=layer_idx,
                text_vec_indices=text_vec_indices,
            )
            if use_kv_cache:
                block_kwargs.update(
                    kv_cache_reader=self._read_layer_kv_cache if kv_cache_mode in {"reuse", "reuse_store"} else None,
                    kv_cache_writer=self._write_layer_kv_cache if kv_cache_mode in {"store", "reuse_store"} else None,
                    skip_text=skip_text_stream,
                    kv_cache_pre_rope=kv_cache_pre_rope,
                    cached_freqs_cis=cached_freqs_cis,
                )
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                txt_freqs_cis,
                **block_kwargs,
            )

        img = self.proj_out(self.norm_out(img))

        gather_index = noisy_indices.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img = torch.gather(img, dim=1, index=gather_index)
        img = self.unpatchify(img, noisy_patch_shape[0], noisy_patch_shape[1], noisy_patch_shape[2])

        if not return_dict:
            return (img, txt)

        return Transformer3DModelOutput(img=img, txt=txt)

    def unpatchify(self, x, t, h, w):
        """Restore latent frames from spatial patches."""
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        if self.unpatchify_new:
            x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
            x = torch.einsum("nthwopqc->nctohpwq", x)
        else:
            x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
            x = torch.einsum("nthwcopq->nctohpwq", x)

        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs
