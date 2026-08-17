"""Benchmark + accuracy test for joyomni_ops vs pure-PyTorch reference.

For each op:
  - correctness: max abs / mean rel error vs a pure-torch (fp32-accumulate) impl,
    and vs sgl_kernel when available (should be ~0).
  - speed: CUDA-event timed, joyomni_ops kernel vs the equivalent eager torch
    ops, reporting the speedup.

Usage:
  python tests/test_bench.py
  python tests/test_bench.py --dtype fp16 --iters 200
"""
import argparse

import torch

import joyomni_ops as jo

try:
    import sgl_kernel as sk
    HAVE_SGL = True
except Exception:
    HAVE_SGL = False

DEV = "cuda"


def bench(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def errs(a, b):
    a = a.float()
    b = b.float()
    max_abs = (a - b).abs().max().item()
    rel = ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-9)).item()
    return max_abs, rel


def line(name, max_abs, rel, t_ker, t_ref):
    sp = t_ref / t_ker if t_ker > 0 else float("inf")
    print(f"{name:26s} | max_abs={max_abs:9.3e} rel={rel:8.2e} | "
          f"kernel={t_ker*1000:7.1f}us torch={t_ref*1000:7.1f}us speedup={sp:5.2f}x")


# ---------------------------------------------------------------------------
# rmsnorm
# ---------------------------------------------------------------------------
def run_rmsnorm(dt, iters):
    M, N = 4096, 4096
    x = torch.randn(M, N, device=DEV, dtype=dt)
    w = torch.randn(N, device=DEV, dtype=dt)
    eps = 1e-6

    def torch_ref():
        xf = x.float()
        return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(dt)

    out = jo.rmsnorm(x, w, eps)
    max_abs, rel = errs(out, torch_ref())
    t_ker = bench(lambda: jo.rmsnorm(x, w, eps), iters=iters)
    t_ref = bench(torch_ref, iters=iters)
    line("rmsnorm", max_abs, rel, t_ker, t_ref)
    if HAVE_SGL:
        s = errs(out, sk.rmsnorm(x, w, eps))
        print(f"{'  └ vs sgl_kernel':26s} | max_abs={s[0]:9.3e}")


# ---------------------------------------------------------------------------
# fused_norm_scale_shift (LayerNorm + adaLN)
# ---------------------------------------------------------------------------
def run_norm_scale_shift(dt, iters):
    M, N = 4096, 4096
    x = torch.randn(M, N, device=DEV, dtype=dt)
    gamma = torch.randn(N, device=DEV, dtype=dt)
    beta = torch.randn(N, device=DEV, dtype=dt)
    scale = torch.randn(M, N, device=DEV, dtype=dt)
    shift = torch.randn(M, N, device=DEV, dtype=dt)
    eps = 1e-5

    def torch_ref():
        xf = x.float()
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        ln = (xf - mean) * torch.rsqrt(var + eps) * gamma.float() + beta.float()
        return (ln * (1 + scale.float()) + shift.float()).to(dt)

    out = jo.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", eps)
    max_abs, rel = errs(out, torch_ref())
    t_ker = bench(lambda: jo.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", eps), iters=iters)
    t_ref = bench(torch_ref, iters=iters)
    line("fused_norm_scale_shift", max_abs, rel, t_ker, t_ref)
    if HAVE_SGL:
        s = errs(out, sk.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", eps))
        print(f"{'  └ vs sgl_kernel':26s} | max_abs={s[0]:9.3e}")


# ---------------------------------------------------------------------------
# fused_qk_norm_rope_3d_paired (bf16 only — the kernel is bf16)
# ---------------------------------------------------------------------------
def run_qk_norm_rope(iters):
    dt = torch.bfloat16
    B, seq_len, num_heads, head_dim = 1, 1560, 32, 128  # steady-state DiT shape
    M = seq_len * num_heads
    q0 = torch.randn(B, M, head_dim, device=DEV, dtype=dt)
    k0 = torch.randn(B, M, head_dim, device=DEV, dtype=dt)
    qw = torch.randn(head_dim, device=DEV, dtype=dt)
    kw = torch.randn(head_dim, device=DEV, dtype=dt)
    cos = torch.randn(seq_len, head_dim // 2, device=DEV, dtype=dt)
    sin = torch.randn(seq_len, head_dim // 2, device=DEV, dtype=dt)
    eps = 1e-6

    def torch_ref_one(x, w):
        xf = x.float().view(B, seq_len, num_heads, head_dim)
        xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        c = cos.float().view(seq_len, 1, head_dim // 2)
        s = sin.float().view(seq_len, 1, head_dim // 2)
        x1, x2 = xn[..., 0::2], xn[..., 1::2]
        o = torch.empty_like(xn)
        o[..., 0::2] = x1 * c - x2 * s
        o[..., 1::2] = x1 * s + x2 * c
        return o.view(B, M, head_dim).to(dt)

    # kernel is in-place; work on clones
    qj, kj = q0.clone(), k0.clone()
    jo.fused_qk_norm_rope_3d_paired(qj, kj, seq_len, num_heads, eps, qw, kw, cos, sin)
    max_abs, rel = errs(qj, torch_ref_one(q0, qw))

    def ker():
        q, k = q0.clone(), k0.clone()
        jo.fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, qw, kw, cos, sin)

    def ref():
        torch_ref_one(q0, qw)
        torch_ref_one(k0, kw)

    t_ker = bench(ker, iters=iters)
    t_ref = bench(ref, iters=iters)
    line("qk_norm_rope_3d_paired", max_abs, rel, t_ker, t_ref)
    if HAVE_SGL:
        qs, ks = q0.clone(), k0.clone()
        sk.fused_qk_norm_rope_3d_paired(qs, ks, seq_len, num_heads, eps, qw, kw, cos, sin)
        s = errs(qj, qs)
        print(f"{'  └ vs sgl_kernel':26s} | max_abs={s[0]:9.3e}")


# ---------------------------------------------------------------------------
# fp8 linear: quant + fp8_scaled_mm  vs  bf16/fp16 torch matmul
# ---------------------------------------------------------------------------
def run_fp8_linear(dt, iters):
    if not jo.has_fp8():
        print("fp8_scaled_mm            | (not built — JOYOMNI_OPS_NO_FP8)")
        return
    M, K, N = 4096, 4096, 4096
    x = torch.randn(M, K, device=DEV, dtype=dt)
    w = torch.randn(N, K, device=DEV, dtype=dt)  # weight [N, K] row-major

    # Pre-quant the weight once (as an Fp8Linear would at load time).
    wq = torch.empty(N, K, device=DEV, dtype=torch.float8_e4m3fn)
    wsc = torch.empty(N, device=DEV, dtype=torch.float32)
    jo.sgl_per_token_quant_fp8(w, wq, wsc)
    wq_t = wq.t()  # [K, N] col-major

    xq = torch.empty(M, K, device=DEV, dtype=torch.float8_e4m3fn)
    xsc = torch.empty(M, device=DEV, dtype=torch.float32)

    def fp8_path():
        jo.sgl_per_token_quant_fp8(x, xq, xsc)  # dynamic per-token quant of activation
        return jo.fp8_scaled_mm(xq, wq_t, xsc, wsc, dt, None)

    def torch_path():
        return torch.matmul(x, w.t())  # eager bf16/fp16 GEMM

    out = fp8_path()
    ref = torch_path()
    max_abs, rel = errs(out, ref)
    t_ker = bench(fp8_path, iters=iters)
    t_ref = bench(torch_path, iters=iters)
    line(f"fp8 linear (quant+mm)", max_abs, rel, t_ker, t_ref)
    # mm only (weight already quantized, activation pre-quantized) — the steady cost
    jo.sgl_per_token_quant_fp8(x, xq, xsc)

    def mm_only():
        return jo.fp8_scaled_mm(xq, wq_t, xsc, wsc, dt, None)

    t_mm = bench(mm_only, iters=iters)
    line(f"fp8 mm only", *errs(mm_only(), ref), t_mm, t_ref)
    if HAVE_SGL:
        s = errs(mm_only(), sk.fp8_scaled_mm(xq, wq_t, xsc, wsc, dt, None))
        print(f"{'  └ vs sgl_kernel':26s} | max_abs={s[0]:9.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"device={torch.cuda.get_device_name(0)}  dtype={args.dtype}  "
          f"iters={args.iters}  sgl_kernel={'yes' if HAVE_SGL else 'no'}  has_fp8={jo.has_fp8()}")
    print("-" * 100)
    run_rmsnorm(dt, args.iters)
    run_norm_scale_shift(dt, args.iters)
    run_qk_norm_rope(args.iters)  # bf16 only
    run_fp8_linear(dt, args.iters)
    print("-" * 100)
    print("Notes: fp8 rel-err vs torch is the inherent fp8 quantization error (~few %),")
    print("       not a bug. Correctness vs sgl_kernel should be ~0.")


if __name__ == "__main__":
    main()
