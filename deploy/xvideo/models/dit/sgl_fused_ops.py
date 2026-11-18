"""Fused DiT helper ops with two interchangeable backends.

* CUDA  -- the hand-written ``joyomni_ops`` kernels (bf16, in-place), used when the
  extension is importable and the tensors live on a CUDA device.
* torch -- pure-PyTorch reference implementations with the same numerics
  (fp32 accumulate).  They run on any device/dtype (CPU, MPS, older GPUs), so the
  DiT no longer hard-requires a CUDA toolchain / ``joyomni_ops`` build.

Set ``JOYOMNI_PORTABLE_OPS=1`` to force the torch backend even on CUDA.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


_AVAILABLE: Optional[bool] = None
_FUSED_NORM_SCALE_SHIFT = None
_FUSED_QK_NORM_ROPE_3D = None
_RMSNORM = None

_FORCE_PORTABLE = os.environ.get("JOYOMNI_PORTABLE_OPS", "").strip().lower() in {
    "1", "true", "yes", "on"
}


def available() -> bool:
    return _try_import()


def _try_import() -> bool:
    global _AVAILABLE, _FUSED_NORM_SCALE_SHIFT, _FUSED_QK_NORM_ROPE_3D, _RMSNORM
    if _AVAILABLE is not None:
        return _AVAILABLE
    if _FORCE_PORTABLE:
        _AVAILABLE = False
        return _AVAILABLE
    try:
        from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
    except Exception:  # noqa: BLE001 - not built / no CUDA toolchain: use the torch backend
        _AVAILABLE = False
        return _AVAILABLE
    _RMSNORM = rmsnorm
    _FUSED_NORM_SCALE_SHIFT = fused_norm_scale_shift
    _FUSED_QK_NORM_ROPE_3D = fused_qk_norm_rope_3d_paired
    _AVAILABLE = True
    return _AVAILABLE


_WARNED_SLOW_CUDA = False


def _use_cuda_kernels(x: torch.Tensor) -> bool:
    global _WARNED_SLOW_CUDA
    if not x.is_cuda:
        return False
    if _try_import():
        return True
    if not _FORCE_PORTABLE and not _WARNED_SLOW_CUDA:
        _WARNED_SLOW_CUDA = True
        print("[dit] joyomni_ops is not available: running the fused DiT ops in pure PyTorch on CUDA "
              "(correct but slower). Build deploy/joyomni_ops for real-time serving.", flush=True)
    return False


def backend(device: torch.device | str = "cpu") -> str:
    """Name of the backend that will serve tensors on `device`."""
    return "joyomni_ops" if (torch.device(device).type == "cuda" and _try_import()) else "torch"


# --------------------------------------------------------------------------- #
# LayerNorm + adaLN modulate:  out = LN(x) * (1 + scale) + shift
# --------------------------------------------------------------------------- #
def fused_layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, L, D = x.shape

    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

    if not _use_cuda_kernels(x):
        out = F.layer_norm(
            x.float(), (D,),
            weight.float() if weight is not None else None,
            bias.float() if bias is not None else None,
            eps,
        )
        out = torch.addcmul(shift.float(), out, 1.0 + scale.float())
        return out.to(x.dtype)

    x_2d = x.reshape(-1, D)
    if scale.shape[1] == 1 and L != 1:
        scale = scale.expand(B, L, D)
        shift = shift.expand(B, L, D)
    scale_2d = scale.reshape(-1, D).contiguous()
    shift_2d = shift.reshape(-1, D).contiguous()

    out = _FUSED_NORM_SCALE_SHIFT(x_2d, weight, bias, scale_2d, shift_2d, "layer", eps)
    return out.reshape(B, L, D)


# --------------------------------------------------------------------------- #
# per-head RMSNorm(q, k) + interleaved 3D RoPE
# --------------------------------------------------------------------------- #
def _half_freqs(freqs_cis: Tuple[torch.Tensor, torch.Tensor], head_dim: int, device):
    """Normalise cos/sin to [L, head_dim/2] (one entry per rotated pair)."""
    cos, sin = freqs_cis
    cos = cos.to(device)
    sin = sin.to(device)
    while cos.dim() > 2:
        if cos.shape[0] != 1:
            raise RuntimeError(f"freqs_cis cos has non-singleton leading dim: {cos.shape}")
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.shape[-1] == head_dim:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    return cos.contiguous(), sin.contiguous()


def _rms_normalize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * weight.float()


def _rotate_pairs(xf: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # xf: [B, L, H, D] fp32, cos/sin: [L, D/2]
    c = cos.float()[None, :, None, :]
    s = sin.float()[None, :, None, :]
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


def fused_qk_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, H, D = q.shape

    if not _use_cuda_kernels(q):
        cos, sin = _half_freqs(freqs_cis, D, q.device)
        q_out = _rotate_pairs(_rms_normalize(q, q_norm_weight, eps), cos, sin).to(q.dtype)
        k_out = _rotate_pairs(_rms_normalize(k, k_norm_weight, eps), cos, sin).to(k.dtype)
        return q_out, k_out

    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"fused_qk_norm_rope_3d (CUDA kernel) requires bf16 q/k, got {q.dtype}"
        )

    q = q.contiguous()
    k = k.contiguous()
    q_r = q.view(B, L * H, D)
    k_r = k.view(B, L * H, D)

    cos, sin = _half_freqs(freqs_cis, D, q.device)
    cos_bf16 = cos.to(torch.bfloat16)
    sin_bf16 = sin.to(torch.bfloat16)
    qw = q_norm_weight.to(torch.bfloat16)
    kw = k_norm_weight.to(torch.bfloat16)

    _FUSED_QK_NORM_ROPE_3D(q_r, k_r, L, H, eps, qw, kw, cos_bf16, sin_bf16)
    return q, k


def rmsnorm_qk_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not _use_cuda_kernels(x) or _RMSNORM is None:
        return _rms_normalize(x, weight, eps).to(x.dtype)
    assert x.dtype == torch.bfloat16, f"rmsnorm_qk_bf16 needs bf16, got {x.dtype}"
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.reshape(-1, D).contiguous()
    w = weight.to(dtype=torch.bfloat16)
    out = _RMSNORM(x_flat, w, eps)
    return out.reshape(orig_shape)


def fused_add_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.addcmul(residual, x, gate.unsqueeze(1))
