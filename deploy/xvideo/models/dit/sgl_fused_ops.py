from __future__ import annotations

from typing import Optional, Tuple

import torch


_AVAILABLE: Optional[bool] = None
_FUSED_NORM_SCALE_SHIFT = None
_FUSED_QK_NORM_ROPE_3D = None
_RMSNORM = None


def available() -> bool:
    return _try_import()


def _try_import() -> bool:
    global _AVAILABLE, _FUSED_NORM_SCALE_SHIFT, _FUSED_QK_NORM_ROPE_3D, _RMSNORM
    if _AVAILABLE is not None:
        return _AVAILABLE
    from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
    _RMSNORM = rmsnorm
    _FUSED_NORM_SCALE_SHIFT = fused_norm_scale_shift
    _FUSED_QK_NORM_ROPE_3D = fused_qk_norm_rope_3d_paired
    _AVAILABLE = True
    return _AVAILABLE


def fused_layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not _try_import():
        raise RuntimeError("joyomni_ops not available")
    B, L, D = x.shape
    x_2d = x.reshape(-1, D)

    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    if scale.shape[1] == 1 and L != 1:
        scale = scale.expand(B, L, D)
        shift = shift.expand(B, L, D)
    scale_2d = scale.reshape(-1, D).contiguous()
    shift_2d = shift.reshape(-1, D).contiguous()

    out = _FUSED_NORM_SCALE_SHIFT(x_2d, weight, bias, scale_2d, shift_2d, "layer", eps)
    return out.reshape(B, L, D)


def fused_qk_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _try_import():
        raise RuntimeError("joyomni_ops not available")
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"fused_qk_norm_rope_3d requires bf16 q/k, got {q.dtype}"
        )

    B, L, H, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    q_r = q.view(B, L * H, D)
    k_r = k.view(B, L * H, D)

    cos, sin = freqs_cis
    cos = cos.to(q.device)
    sin = sin.to(q.device)
    while cos.dim() > 2:
        if cos.shape[0] != 1:
            raise RuntimeError(f"freqs_cis cos has non-singleton leading dim: {cos.shape}")
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.shape[-1] == D:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()
    cos_bf16 = cos.to(torch.bfloat16)
    sin_bf16 = sin.to(torch.bfloat16)
    qw = q_norm_weight.to(torch.bfloat16)
    kw = k_norm_weight.to(torch.bfloat16)

    _FUSED_QK_NORM_ROPE_3D(q_r, k_r, L, H, eps, qw, kw, cos_bf16, sin_bf16)
    return q, k


def rmsnorm_qk_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not _try_import() or _RMSNORM is None:
        raise RuntimeError("joyomni_ops rmsnorm not available")
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
