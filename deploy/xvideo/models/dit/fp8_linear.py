from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from joyomni_ops import fp8_scaled_mm, sgl_per_token_quant_fp8

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

# GeForce runs fp16-accumulate tensor MMAs at full rate.
_FAST_ACCUM = os.environ.get("JOYOMNI_FP8_FAST_ACCUM", "").strip().lower() in {
    "1", "true", "yes", "on"
}
_FA_QMAX_A = 7.0
_FA_QMAX_W = 14.0
_FA_PROMOTE = 4
assert _FA_PROMOTE * 128 * _FA_QMAX_A * _FA_QMAX_W <= 65504.0

if _FAST_ACCUM:
    import triton
    import triton.language as tl

    @triton.jit
    def _fa_mm_kernel(A, B, C, XS, WS, BIAS, M, N, K,
                      sam, sak, sbk, sbn, scm, scn,
                      HAS_BIAS: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP: tl.constexpr, PROMOTE: tl.constexpr):
        # Promote every 512 elements of K: 512 * 7 * 14 = 50,176 < fp16 max.
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GROUP * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP, GROUP)
        pid_m = group_id * GROUP + (pid % group_size)
        pid_n = (pid % width) // group_size
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        A_ptr = A + (rm[:, None] * sam + rk[None, :] * sak)
        B_ptr = B + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        part = tl.zeros((BM, BN), dtype=tl.float16)
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(A_ptr, mask=(rm[:, None] < M) & ((rk[None, :] + k * BK) < K), other=0.0)
            b = tl.load(B_ptr, mask=((rk[:, None] + k * BK) < K) & (rn[None, :] < N), other=0.0)
            part = tl.dot(a, b, part, out_dtype=tl.float16)
            if (k + 1) % PROMOTE == 0:
                acc += part.to(tl.float32)
                part = tl.zeros((BM, BN), dtype=tl.float16)
            A_ptr += BK * sak
            B_ptr += BK * sbk
        acc += part.to(tl.float32)
        xs = tl.load(XS + rm, mask=rm < M, other=0.0)
        ws = tl.load(WS + rn, mask=rn < N, other=0.0)
        out = acc * xs[:, None] * ws[None, :]
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            out = out + bias[None, :]
        C_ptr = C + rm[:, None] * scm + rn[None, :] * scn
        tl.store(C_ptr, out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _fa_quant_kernel(X, XQ, XS, M, K, sxm, sxk,
                         BLOCK: tl.constexpr, QMAX: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = tl.zeros((BLOCK,), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(K, BLOCK)):
            idx = k0 * BLOCK + offs
            x = tl.load(X + row * sxm + idx * sxk, mask=idx < K, other=0.0).to(tl.float32)
            m = tl.maximum(m, tl.abs(x))
        amax = tl.max(m, axis=0)
        scale = tl.maximum(amax, 1e-8) / QMAX
        tl.store(XS + row, scale)
        inv = 1.0 / scale
        for k0 in range(0, tl.cdiv(K, BLOCK)):
            idx = k0 * BLOCK + offs
            x = tl.load(X + row * sxm + idx * sxk, mask=idx < K, other=0.0).to(tl.float32)
            q = (x * inv).to(tl.float8e4nv)
            tl.store(XQ + row * K + idx, q, mask=idx < K)

    def _fa_pick_config(M, N, K):
        if K >= 8192:  # mlp down projections
            return (64, 256, 8, 3) if M >= 1000 else (64, 128, 8, 3)
        if N >= 12288:  # attn qkv / mlp up
            return (128, 256, 8, 3) if M >= 500 else (64, 256, 8, 3)
        return (64, 256, 8, 3) if M >= 1800 else (64, 128, 8, 3)


def _quantize_weight_per_channel(w_bf16: torch.Tensor):
    w_kn = w_bf16.t().contiguous()
    K, N = w_kn.shape
    absmax = w_kn.abs().amax(dim=0).to(torch.float32).clamp_min(1e-8)
    scale = absmax / (_FA_QMAX_W if _FAST_ACCUM else FP8_MAX)
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
        if _FAST_ACCUM:
            _fa_quant_kernel[(M,)](x_2d, x_q, x_scale, M, K,
                                   x_2d.stride(0), x_2d.stride(1),
                                   BLOCK=1024, QMAX=_FA_QMAX_A, num_warps=4)
            N = self.out_features
            y = torch.empty((M, N), device=x_2d.device, dtype=self.out_dtype)
            xs_eff = x_scale.view(-1)
            bm, bn, warps, stages = _fa_pick_config(M, N, K)
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            _fa_mm_kernel[grid](
                x_q, self.weight_fp8, y, xs_eff, self.weight_scale,
                self.bias if self.bias is not None else y,
                M, N, K,
                x_q.stride(0), x_q.stride(1),
                self.weight_fp8.stride(0), self.weight_fp8.stride(1),
                y.stride(0), y.stride(1),
                HAS_BIAS=self.bias is not None,
                BM=bm, BN=bn, BK=128, GROUP=8, PROMOTE=_FA_PROMOTE,
                num_warps=warps, num_stages=stages,
            )
            return y.reshape(*orig_shape[:-1], self.out_features)
        sgl_per_token_quant_fp8(x_2d, x_q, x_scale)
        y = fp8_scaled_mm(x_q, self.weight_fp8, x_scale, self.weight_scale,
                          out_dtype=self.out_dtype, bias=self.bias)
        return y.reshape(*orig_shape[:-1], self.out_features)
