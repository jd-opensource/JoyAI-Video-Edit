from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from joyomni_ops import fp8_scaled_mm, sgl_per_token_quant_fp8
    _SGL_KERNEL_OK = True
except Exception:
    fp8_scaled_mm = None
    sgl_per_token_quant_fp8 = None
    _SGL_KERNEL_OK = False


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def available() -> bool:
    return _SGL_KERNEL_OK


def _quantize_weight_per_channel(w_bf16: torch.Tensor):
    w_kn = w_bf16.t().contiguous()
    K, N = w_kn.shape
    absmax = w_kn.abs().amax(dim=0).to(torch.float32).clamp_min(1e-8)
    scale = absmax / FP8_MAX
    w_q_rm = (w_kn.float() / scale.unsqueeze(0)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    w_q_cm = w_q_rm.t().contiguous().t()
    assert w_q_cm.shape == (K, N) and w_q_cm.stride() == (1, K)
    return w_q_cm, scale


class Fp8Linear(nn.Module):

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor,
                 bias: Optional[torch.Tensor], out_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.contiguous(), persistent=False)
        else:
            self.bias = None
        self.out_dtype = out_dtype
        K, N = weight_fp8.shape
        self.in_features = K
        self.out_features = N

    @classmethod
    def from_linear(cls, lin: nn.Linear, out_dtype: torch.dtype = torch.bfloat16) -> "Fp8Linear":
        assert available(), "joyomni_ops not importable; cannot build Fp8Linear"
        w = lin.weight.data
        w_q, w_s = _quantize_weight_per_channel(w.to(torch.bfloat16))
        b = None
        if lin.bias is not None:
            b = lin.bias.data.to(dtype=out_dtype)
        return cls(w_q, w_s, b, out_dtype=out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, K = x_2d.shape
        assert K == self.in_features, f"in features mismatch {K} vs {self.in_features}"
        x_q = torch.empty((M, K), device=x_2d.device, dtype=FP8_DTYPE)
        x_scale = torch.empty((M, 1), device=x_2d.device, dtype=torch.float32)
        sgl_per_token_quant_fp8(x_2d, x_q, x_scale)
        y = fp8_scaled_mm(x_q, self.weight_fp8, x_scale, self.weight_scale,
                          out_dtype=self.out_dtype, bias=self.bias)
        return y.reshape(*orig_shape[:-1], self.out_features)
