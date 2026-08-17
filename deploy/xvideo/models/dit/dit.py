from contextlib import contextmanager
import logging
import os
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps


try:
    from flash_attn.cute import flash_attn_func as _fa4_func
    _FA4_IMPORTED = True
except Exception:  # noqa: BLE001
    # flash-attn-4 is optional. On GPUs/toolchains where it isn't available
    # (e.g. no FA4 build for this arch), fall back to SDPA (cuDNN/flash) at
    # runtime instead of failing at import time.
    _fa4_func = None
    _FA4_IMPORTED = False


SOURCE_ID_TARGET = 0.0
SOURCE_ID_EDIT_CONDITION = 1.0
SOURCE_ID_EXTRA_REF_IMAGE = 2.0

TIME_FREQ_DIM = 256

NORM_EPS = 1e-6
NUM_MODULATION_CHUNKS = 6

SELF_ATTN_MODE_REF_IMAGE_CACHE = "ref_image_cache"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


_FP8_IMG_ENABLED = _env_on("JOYOMNI_FP8_IMG")
_FP8_TXT_ENABLED = _env_on("JOYOMNI_FP8_TXT")
_TXT_PARALLEL_ENABLED = _env_on("JOYOMNI_TXT_PARALLEL")


def _fp8_stream_wanted(stream: str) -> bool:
    if stream == "img":
        return _FP8_IMG_ENABLED
    if stream == "txt":
        return _FP8_TXT_ENABLED
    return False


def _maybe_install_fp8_stream(block, stream: str) -> None:
    """Quantize the attn qkv/proj + mlp up/down Linears of one stream ("img" or "txt")
    to FP8. Idempotent per (block, stream); the original bf16 Linears are left in place."""
    if not _fp8_stream_wanted(stream):
        return
    installed_flag = f"_fp8_{stream}_installed"
    if getattr(block, installed_flag, False):
        return
    from xvideo.models.dit.fp8_linear import Fp8Linear
    setattr(block, f"_fp8_{stream}_attn_qkv",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_qkv")))
    setattr(block, f"_fp8_{stream}_attn_proj",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_proj")))
    ff = getattr(block, f"{stream}_mlp")
    up_lin = down_lin = None
    gelu_mod = None
    for m in ff.net:
        if isinstance(m, nn.Linear) and down_lin is None and up_lin is not None:
            down_lin = m
            continue
        if isinstance(m, nn.Linear) and up_lin is None:
            up_lin = m
            continue
        if hasattr(m, "proj") and isinstance(m.proj, nn.Linear) and up_lin is None:
            up_lin = m.proj
            gelu_mod = m
    if up_lin is None or down_lin is None:
        raise RuntimeError(f"could not locate mlp up/down Linears in {ff}")
    setattr(block, f"_fp8_{stream}_mlp_up", Fp8Linear.from_linear(up_lin))
    setattr(block, f"_fp8_{stream}_mlp_down", Fp8Linear.from_linear(down_lin))
    _approx = getattr(gelu_mod, "approximate", "tanh") if gelu_mod is not None else "tanh"
    setattr(block, f"_{stream}_mlp_act",
            lambda x, _a=_approx: torch.nn.functional.gelu(x, approximate=_a))
    setattr(block, installed_flag, True)


def _fp8_stream_enabled(block, stream: str) -> bool:
    return bool(getattr(block, f"_fp8_{stream}_installed", False))



from xvideo.models.dit import sgl_fused_ops as _sgl_fused

from xvideo.models.dit.rope import apply_rotary_emb, get_1d_rotary_pos_embed


class ModulateWan(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size,
                        dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


logger = logging.getLogger(__name__)


def _clone_kv_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone()


_FA4_DISABLED = not _FA4_IMPORTED

_SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

_SAGE_ATTN_ENABLED = _env_on("JOYOMNI_SAGE_ATTN")
_SAGE_MIN_KV_LEN = 4096
_sage_attn_func = None
if _SAGE_ATTN_ENABLED:
    try:
        from sageattention import sageattn as _sage_attn_func
    except Exception as _sage_exc:  # noqa: BLE001
        _sage_attn_func = None
        print(f"[dit] JOYOMNI_SAGE_ATTN=1 but sageattention unavailable: {_sage_exc!r}")


def _sdpa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if (
        _sage_attn_func is not None
        and key.shape[1] >= _SAGE_MIN_KV_LEN
        and query.shape[-1] <= 128
    ):
        return _sage_attn_func(query, key, value, tensor_layout="NHD", is_causal=False)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    with sdpa_kernel(_SDPA_BACKENDS, set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.transpose(1, 2)


def _flash_attention4(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    global _FA4_DISABLED
    original_dtype = query.dtype
    if query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)
    if not _FA4_DISABLED:
        try:
            out = _fa4_func(query, key, value, causal=False)
            if isinstance(out, tuple):
                out = out[0]
            return out.to(original_dtype)
        except Exception as exc:  # noqa: BLE001
            _FA4_DISABLED = True
            logger.warning(f"[{__name__}] FA4 unavailable, falling back to SDPA (flash/cuDNN): {exc!r}")
    return _sdpa_attention(query, key, value).to(original_dtype)


def warmup_attention_backend(
    heads_num: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    q_len: int = 4096,
    kv_len: int = 4096,
) -> None:
    if device is None:
        device = torch.device("cuda")
    try:
        q = torch.zeros(1, q_len, heads_num, head_dim, device=device, dtype=dtype)
        k = torch.zeros(1, kv_len, heads_num, head_dim, device=device, dtype=dtype)
        _flash_attention4(q, k, k)
        torch.cuda.synchronize(device)
        if not _FA4_DISABLED:
            backend = "FA4"
        elif _sage_attn_func is not None and kv_len >= _SAGE_MIN_KV_LEN:
            backend = "sage"
        else:
            backend = "cuDNN"
        logger.warning(f"[{__name__}] attention warmup done via {backend} (q={q_len}, kv={kv_len}, heads={heads_num}, dim={head_dim})")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{__name__}] attention warmup failed (will lazily compile in-loop): {exc}")


def _concat_kv_entries(
    entries: Iterable[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    keys = []
    values = []
    pre_rope_offset = 0

    for entry in entries:
        if entry is None:
            continue
        key = entry.get("key")
        value = entry.get("value")
        if key is None or value is None:
            continue

        key = key.to(device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)

        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            seg_len = key.shape[1]
            cos_seg = cos_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            sin_seg = sin_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            key = apply_rotary_emb(key, (cos_seg, sin_seg))
            pre_rope_offset += seg_len

        keys.append(key)
        values.append(value)

    if not keys:
        return None, None

    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = NORM_EPS,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu-approximate",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

        self.txt_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        kv_cache_reader: Optional[Callable[[Optional[int]], Iterable[Dict[str, torch.Tensor]]]] = None,
        kv_cache_writer: Optional[Callable[[Optional[int], torch.Tensor, torch.Tensor], None]] = None,
        kv_cache_assembler: Optional[Callable[..., Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        layer_idx: Optional[int] = None,
        skip_text_stream: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: Optional[tuple] = None,
        txt_side_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _maybe_install_fp8_stream(self, "img")
        _fp8_on = _fp8_stream_enabled(self, "img")
        if not skip_text_stream:
            _maybe_install_fp8_stream(self, "txt")
        _fp8_txt_on = (not skip_text_stream) and _fp8_stream_enabled(self, "txt")
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        if not skip_text_stream:
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec)

        _txt_par = txt_side_stream is not None and not skip_text_stream
        if _txt_par:
            _fork_ev = torch.cuda.Event()
            _fork_ev.record()
            with torch.cuda.stream(txt_side_stream):
                txt_side_stream.wait_event(_fork_ev)
                txt_modulated = _sgl_fused.fused_layernorm_modulate(
                    txt, shift=txt_mod1_shift, scale=txt_mod1_scale,
                    weight=self.txt_norm1.weight if self.txt_norm1.elementwise_affine else None,
                    bias=self.txt_norm1.bias if self.txt_norm1.elementwise_affine else None,
                    eps=self.txt_norm1.eps,
                )
                txt_qkv = self._fp8_txt_attn_qkv(txt_modulated) if _fp8_txt_on else self.txt_attn_qkv(txt_modulated)
                _tv = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.heads_num, -1)
                txt_q, txt_k, txt_v = _tv[:, :, 0], _tv[:, :, 1], _tv[:, :, 2]
                txt_q = self.txt_attn_q_norm(txt_q).to(txt_v.dtype)
                txt_k = self.txt_attn_k_norm(txt_k).to(txt_v.dtype)
                _txt_qkv_ev = torch.cuda.Event()
                _txt_qkv_ev.record(txt_side_stream)

        img_modulated = _sgl_fused.fused_layernorm_modulate(
            img, shift=img_mod1_shift, scale=img_mod1_scale,
            weight=self.img_norm1.weight if self.img_norm1.elementwise_affine else None,
            bias=self.img_norm1.bias if self.img_norm1.elementwise_affine else None,
            eps=self.img_norm1.eps,
        )
        img_qkv = self._fp8_img_attn_qkv(img_modulated) if _fp8_on else self.img_attn_qkv(img_modulated)
        _iv = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.heads_num, -1)
        img_q, img_k, img_v = _iv[:, :, 0], _iv[:, :, 1], _iv[:, :, 2]
        if kv_cache_pre_rope:
            img_k_for_cache = _sgl_fused.rmsnorm_qk_bf16(
                img_k, self.img_attn_k_norm.weight, eps=self.img_attn_k_norm.eps
            )
        img_q, img_k = _sgl_fused.fused_qk_norm_rope_3d(
            img_q, img_k,
            q_norm_weight=self.img_attn_q_norm.weight,
            k_norm_weight=self.img_attn_k_norm.weight,
            freqs_cis=vis_freqs_cis,
            eps=self.img_attn_q_norm.eps,
        )
        if not kv_cache_pre_rope:
            img_k_for_cache = img_k

        if _txt_par:
            torch.cuda.current_stream().wait_event(_txt_qkv_ev)
            for _t in (txt_q, txt_k, txt_v):
                _t.record_stream(torch.cuda.current_stream())
        elif not skip_text_stream:
            txt_modulated = _sgl_fused.fused_layernorm_modulate(
                txt, shift=txt_mod1_shift, scale=txt_mod1_scale,
                weight=self.txt_norm1.weight if self.txt_norm1.elementwise_affine else None,
                bias=self.txt_norm1.bias if self.txt_norm1.elementwise_affine else None,
                eps=self.txt_norm1.eps,
            )
            txt_qkv = self._fp8_txt_attn_qkv(txt_modulated) if _fp8_txt_on else self.txt_attn_qkv(txt_modulated)
            _tv = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.heads_num, -1)
            txt_q, txt_k, txt_v = _tv[:, :, 0], _tv[:, :, 1], _tv[:, :, 2]
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v.dtype)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v.dtype)

        if skip_text_stream:
            q = img_q
            k = img_k
            v = img_v
        else:
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)

        if kv_cache_writer is not None:
            kv_cache_writer(layer_idx, img_k_for_cache, img_v)

        if kv_cache_assembler is not None:
            cached_key, cached_value = kv_cache_assembler(
                layer_idx,
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
        elif kv_cache_reader is not None:
            cached_key, cached_value = _concat_kv_entries(
                kv_cache_reader(layer_idx),
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
        else:
            cached_key = cached_value = None

        if cached_key is not None:
            k = torch.cat([cached_key, k], dim=1)
            v = torch.cat([cached_value, v], dim=1)

        attn = _flash_attention4(q, k, v)
        attn = attn.flatten(2, 3)
        if skip_text_stream:
            img_attn = attn
        else:
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        def _img_proj_call(x):
            return self._fp8_img_attn_proj(x) if _fp8_on else self.img_attn_proj(x)

        def _img_mlp_call(x):
            if _fp8_on:
                h = self._fp8_img_mlp_up(x)
                h = self._img_mlp_act(h)
                return self._fp8_img_mlp_down(h)
            return self.img_mlp(x)

        def _txt_proj_call(x):
            return self._fp8_txt_attn_proj(x) if _fp8_txt_on else self.txt_attn_proj(x)

        def _txt_mlp_call(x):
            if _fp8_txt_on:
                h = self._fp8_txt_mlp_up(x)
                h = self._txt_mlp_act(h)
                return self._fp8_txt_mlp_down(h)
            return self.txt_mlp(x)

        def _txt_tail(txt_in):
            t = _sgl_fused.fused_add_gate(
                txt_in, _txt_proj_call(txt_attn),
                txt_mod1_gate,
            )
            t_mod2 = _sgl_fused.fused_layernorm_modulate(
                t, shift=txt_mod2_shift, scale=txt_mod2_scale,
                weight=self.txt_norm2.weight if self.txt_norm2.elementwise_affine else None,
                bias=self.txt_norm2.bias if self.txt_norm2.elementwise_affine else None,
                eps=self.txt_norm2.eps,
            )
            return _sgl_fused.fused_add_gate(
                t, _txt_mlp_call(t_mod2),
                txt_mod2_gate,
            )

        if _txt_par:
            _attn_ev = torch.cuda.Event()
            _attn_ev.record()
            with torch.cuda.stream(txt_side_stream):
                txt_side_stream.wait_event(_attn_ev)
                txt_out = _txt_tail(txt)
                _txt_done_ev = torch.cuda.Event()
                _txt_done_ev.record(txt_side_stream)

        img = _sgl_fused.fused_add_gate(
            img, _img_proj_call(img_attn),
            img_mod1_gate,
        )
        img_mod2_modulated = _sgl_fused.fused_layernorm_modulate(
            img, shift=img_mod2_shift, scale=img_mod2_scale,
            weight=self.img_norm2.weight if self.img_norm2.elementwise_affine else None,
            bias=self.img_norm2.bias if self.img_norm2.elementwise_affine else None,
            eps=self.img_norm2.eps,
        )
        img = _sgl_fused.fused_add_gate(
            img, _img_mlp_call(img_mod2_modulated),
            img_mod2_gate,
        )

        if _txt_par:
            torch.cuda.current_stream().wait_event(_txt_done_ev)
            txt_out.record_stream(torch.cuda.current_stream())
            txt = txt_out
        elif not skip_text_stream:
            txt = _txt_tail(txt)

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
            timestep_proj = timestep_proj.reshape(*timestep_shape, timestep_proj.shape[-1])

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return timestep_proj, encoder_hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        theta: int = 256,
        chunk_size: Optional[int] = None,
        local_window_size: int = 4,
        global_sink_chunk: bool = True,
        causal: bool = False,
        use_inference_kv_cache: bool = False,
        source_id_rope_dim: int = 128,
        source_id_rope_theta: float = 256.0,
    ):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive when provided, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        if causal and chunk_size is None:
            raise ValueError("`chunk_size` must be provided when `causal=True`.")
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.theta = theta
        self.chunk_size = chunk_size
        self.local_window_size = local_window_size
        self.global_sink_chunk = global_sink_chunk
        self.causal = causal
        self._initial_use_inference_kv_cache = use_inference_kv_cache

        self.source_id_rope_dim = int(source_id_rope_dim)
        self.source_id_rope_theta = float(source_id_rope_theta)

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=TIME_FREQ_DIM,
            time_proj_dim=hidden_size * NUM_MODULATION_CHUNKS,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS
        )
        self.proj_out = nn.Linear(
            hidden_size, self.out_channels * math.prod(patch_size),
            **factory_kwargs)

        self._ensure_kv_cache_state()
        self._use_inference_kv_cache = use_inference_kv_cache

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
        if not hasattr(self, "_kv_cache_generation"):
            self._kv_cache_generation = 0
        if not hasattr(self, "_kv_assembly_version"):
            self._kv_assembly_version = None
        if not hasattr(self, "_kv_assembly_cache"):
            self._kv_assembly_cache = {}
        if not hasattr(self, "_kv_assembly_freqs"):
            self._kv_assembly_freqs = None

    def reset_inference_kv_cache(self) -> None:
        self._ensure_kv_cache_state()
        self._inference_kv_cache = {"cond": {}, "uncond": {}}
        self._use_inference_kv_cache = bool(getattr(self.config, "use_inference_kv_cache", False))
        self._kv_cache_mode = None
        self._kv_cache_scope = None
        self._kv_cache_chunk_id = None
        self._kv_cache_selected_chunk_ids = None
        self._kv_cache_pre_rope = False
        self._kv_cache_generation = 0
        self._kv_assembly_version = None
        self._kv_assembly_cache = {}
        self._kv_assembly_freqs = None

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

    def _get_cache_scope_store(self) -> Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]]:
        self._ensure_kv_cache_state()
        if self._kv_cache_scope is None:
            return None
        return self._inference_kv_cache.setdefault(self._kv_cache_scope, {})

    def _read_layer_kv_cache(self, layer_idx: Optional[int]) -> list[Dict[str, torch.Tensor]]:
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

    def _assemble_layer_kv_cache_cached(
        self,
        layer_idx: Optional[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if layer_idx in self._kv_assembly_cache:
            return self._kv_assembly_cache[layer_idx]
        entries = self._read_layer_kv_cache(layer_idx)
        cached_key, cached_value = _concat_kv_entries(
            entries, device=device, dtype=dtype, cached_freqs_cis=cached_freqs_cis
        )
        self._kv_assembly_cache[layer_idx] = (cached_key, cached_value)
        return cached_key, cached_value

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
            "key": _clone_kv_tensor(key),
            "value": _clone_kv_tensor(value),
            "pre_rope": bool(self._kv_cache_pre_rope),
        }
        self._kv_cache_generation += 1

    def evict_kv_cache_chunks(self, chunk_ids_to_keep: set[int]) -> None:
        self._ensure_kv_cache_state()
        for scope in ("cond", "uncond"):
            scope_store = self._inference_kv_cache.get(scope)
            if scope_store is None:
                continue
            evict_ids = [cid for cid in scope_store if cid not in chunk_ids_to_keep]
            for cid in evict_ids:
                del scope_store[cid]
            if evict_ids:
                self._kv_cache_generation += 1

    def get_rotary_pos_embed_from_ids(
        self,
        *,
        frame_ids: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        post_patch_height, post_patch_width = spatial_shape
        device = frame_ids.device
        temporal_positions = frame_ids.to(dtype=torch.float32)
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_positions.numel() % spatial_tokens_per_frame != 0:
            raise ValueError(
                f"`frame_ids` length {temporal_positions.numel()} is not divisible by spatial token count {spatial_tokens_per_frame}."
            )

        h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
        w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
        h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
        h_positions = h_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)
        w_positions = w_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal to head_dim of attention layer")

        cos_list = []
        sin_list = []
        for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions)):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                positions,
                theta=self.theta,
            )
            cos_list.append(cos)
            sin_list.append(sin)
        vis_freqs = (torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1))

        return vis_freqs

    def generate_source_id_rope(
        self,
        source_id: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        role_dim = max(0, min(int(self.source_id_rope_dim), int(head_dim)))

        half_head = head_dim // 2
        cos_half = torch.ones(*source_id.shape, half_head, device=device, dtype=torch.float32)
        sin_half = torch.zeros(*source_id.shape, half_head, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (
            self.source_id_rope_theta **
            (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
        )

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_video_latent: Optional[torch.Tensor] = None,
        current_temporal_ids: Optional[torch.Tensor] = None,
        cached_temporal_ids: Optional[torch.Tensor] = None,
        kv_cache_mode: Optional[str] = None,
        kv_cache_scope: Optional[str] = None,
        kv_cache_chunk_id: Optional[int] = None,
        kv_cache_selected_chunk_ids: Optional[list[int]] = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: Optional[str] = None,
        skip_text_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._ensure_kv_cache_state()
        self.configure_inference_kv_cache(
            scope=kv_cache_scope,
            mode=kv_cache_mode,
            chunk_id=kv_cache_chunk_id,
            selected_chunk_ids=kv_cache_selected_chunk_ids,
            pre_rope=kv_cache_pre_rope,
        )

        batch_size = hidden_states.shape[0]
        patch_size = tuple(self.patch_size)
        current_patch_shape = self._get_patch_shape(hidden_states, patch_size)
        current_seq_len = math.prod(current_patch_shape)
        device = hidden_states.device

        hidden_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temporal_ids = None
        if current_temporal_ids is not None:
            current_temporal_ids = torch.as_tensor(current_temporal_ids, device=device, dtype=torch.long)
            assert current_temporal_ids.shape == (batch_size, current_patch_shape[0]), (
                f"`current_temporal_ids` must have shape {(batch_size, current_patch_shape[0])}, "
                f"got {tuple(current_temporal_ids.shape)}."
            )
            temporal_ids = current_temporal_ids[0]
        if self_attn_input_mode == SELF_ATTN_MODE_REF_IMAGE_CACHE:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_EXTRA_REF_IMAGE, device=device, dtype=torch.float32)
        else:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_TARGET, device=device, dtype=torch.float32)
        current_frame_ids = self._get_token_frame_ids(current_patch_shape, device, temporal_ids=temporal_ids)
        current_rotary = self.get_rotary_pos_embed_from_ids(
            frame_ids=current_frame_ids,
            spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
        )

        latent_segments = [hidden_tokens]
        rotary_segments = [current_rotary]
        source_id_segments = [current_source_id]

        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"Ref video latent batch size {ref_video_latent.shape[0]} does not match hidden states batch size {batch_size}."
                )
            ref_video_patch_shape = self._get_patch_shape(ref_video_latent, patch_size)
            if ref_video_patch_shape[1:] != current_patch_shape[1:]:
                raise ValueError(
                    "Ref video latent spatial patch shape must match noisy latent spatial patch shape: "
                    f"{ref_video_patch_shape[1:]} != {current_patch_shape[1:]}."
                )
            ref_video_tokens = self.img_in(ref_video_latent).flatten(2).transpose(1, 2)
            video_frame_ids = self._get_token_frame_ids(ref_video_patch_shape, device, temporal_ids=temporal_ids)
            latent_segments.append(ref_video_tokens)
            rotary_segments.append(self.get_rotary_pos_embed_from_ids(
                frame_ids=video_frame_ids,
                spatial_shape=(ref_video_patch_shape[1], ref_video_patch_shape[2]),
            ))
            source_id_segments.append(
                torch.full((ref_video_tokens.shape[1],), SOURCE_ID_EDIT_CONDITION, device=device, dtype=torch.float32)
            )

        img = torch.cat(latent_segments, dim=1)
        visual_source_id = torch.cat(source_id_segments, dim=0).unsqueeze(0)
        current_indices = torch.arange(current_seq_len, device=device).unsqueeze(0)
        vis_freqs_cis = (
            torch.cat([rotary[0] for rotary in rotary_segments], dim=0).unsqueeze(0),
            torch.cat([rotary[1] for rotary in rotary_segments], dim=0).unsqueeze(0),
        )

        head_dim = self.hidden_size // self.heads_num
        cos_3d, sin_3d = vis_freqs_cis
        cos_role, sin_role = self.generate_source_id_rope(
            source_id=visual_source_id,
            head_dim=head_dim,
            device=cos_3d.device,
            dtype=cos_3d.dtype,
        )
        new_cos = cos_3d * cos_role - sin_3d * sin_role
        new_sin = sin_3d * cos_role + cos_3d * sin_role
        vis_freqs_cis = (new_cos, new_sin)

        vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(-1, (NUM_MODULATION_CHUNKS, -1))

        use_memo = bool(kv_cache_pre_rope and
                        cached_temporal_ids is not None and
                        kv_cache_mode in {"reuse", "reuse_store"})
        _assembly_stamp = None
        if use_memo:
            _cached_ids_src = torch.as_tensor(cached_temporal_ids, dtype=torch.long)
            _temporal_sig = (tuple(_cached_ids_src.shape), tuple(_cached_ids_src.reshape(-1).tolist()))
            _assembly_stamp = (
                self._kv_cache_scope,
                tuple(self._kv_cache_selected_chunk_ids or ()),
                bool(self._kv_cache_pre_rope),
                _temporal_sig,
                self._kv_cache_generation,
            )

        cached_freqs_cis = None
        if use_memo and _assembly_stamp == self._kv_assembly_version:
            cached_freqs_cis = self._kv_assembly_freqs
        elif kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_ids_tensor = torch.as_tensor(cached_temporal_ids, device=device, dtype=torch.long)
            cached_frame_ids = self._get_token_frame_ids(
                (cached_ids_tensor.shape[1], current_patch_shape[1], current_patch_shape[2]),
                device,
                temporal_ids=cached_ids_tensor[0],
            )
            cached_vis_freqs = self.get_rotary_pos_embed_from_ids(
                frame_ids=cached_frame_ids,
                spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
            )
            cached_freqs_cis = (
                cached_vis_freqs[0].unsqueeze(0),
                cached_vis_freqs[1].unsqueeze(0),
            )
            if use_memo:
                self._kv_assembly_version = _assembly_stamp
                self._kv_assembly_freqs = cached_freqs_cis
                self._kv_assembly_cache = {}

        _graph_assembler = getattr(self, "_graph_kv_assembler", None)
        _graph_writer = getattr(self, "_graph_kv_writer", None)
        _txt_stream = None
        if (
            not skip_text_stream
            and _TXT_PARALLEL_ENABLED
            and device.type == "cuda"
            and _graph_assembler is not None
        ):
            _txt_stream = getattr(self, "_txt_side_stream", None)
            if _txt_stream is None:
                _txt_stream = torch.cuda.Stream(device=device)
                self._txt_side_stream = _txt_stream
        for layer_idx, block in enumerate(self.double_blocks):
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                layer_idx=layer_idx,
                kv_cache_reader=None if _graph_assembler is not None else (
                    self._read_layer_kv_cache if kv_cache_mode in {"reuse", "reuse_store"} else None),
                kv_cache_writer=_graph_writer if _graph_writer is not None else (
                    self._write_layer_kv_cache if kv_cache_mode in {"store", "reuse_store"} else None),
                kv_cache_assembler=_graph_assembler if _graph_assembler is not None else (
                    self._assemble_layer_kv_cache_cached if use_memo else None),
                skip_text_stream=skip_text_stream,
                kv_cache_pre_rope=kv_cache_pre_rope,
                cached_freqs_cis=cached_freqs_cis,
                txt_side_stream=_txt_stream,
            )

        img = self.proj_out(self.norm_out(img))

        gather_index = current_indices.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img = torch.gather(img, dim=1, index=gather_index)
        img = self.unpatchify(img, current_patch_shape[0], current_patch_shape[1], current_patch_shape[2])

        return (img, txt)

    def unpatchify(self, x, t, h, w):
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs
