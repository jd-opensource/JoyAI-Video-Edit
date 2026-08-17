"""Numerical parity: joyomni_ops vs sgl_kernel for the 3 light ops.
Run on a CUDA GPU with both packages importable:
  python tests/test_parity_light.py
"""
import torch
import joyomni_ops as jo

try:
    import sgl_kernel as sk
    HAVE_SGL = True
except Exception as e:  # noqa: BLE001
    HAVE_SGL = False
    print(f"[warn] sgl_kernel not importable ({e!r}); running self-consistency only")

dev = "cuda"
torch.manual_seed(0)


def report(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print(f"{name:34s} max_diff={d:.3e}  {'OK' if d < 1e-2 else 'FAIL'}")
    return d


def test_rmsnorm():
    M, N = 512, 4096
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, device=dev, dtype=torch.bfloat16)
    out = jo.rmsnorm(x, w, 1e-6)
    # reference in fp32
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    report("rmsnorm vs fp32-ref", out, ref)
    if HAVE_SGL:
        report("rmsnorm vs sgl", out, sk.rmsnorm(x, w, 1e-6))


def test_fused_norm_scale_shift():
    M, N = 512, 4096
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(N, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(N, device=dev, dtype=torch.bfloat16)
    scale = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    shift = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    out = jo.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-5)
    # fp32 ref: LayerNorm then modulate
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    ln = (xf - mean) * torch.rsqrt(var + 1e-5) * gamma.float() + beta.float()
    ref = ln * (1 + scale.float()) + shift.float()
    report("norm_scale_shift vs fp32-ref", out, ref)
    if HAVE_SGL:
        report("norm_scale_shift vs sgl", out, sk.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-5))


def test_qk_norm_rope():
    B, seq_len, num_heads, head_dim = 1, 128, 8, 128
    M = seq_len * num_heads
    q = torch.randn(B, M, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, M, head_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(head_dim, device=dev, dtype=torch.bfloat16)
    kw = torch.randn(head_dim, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(seq_len, head_dim // 2, device=dev, dtype=torch.bfloat16)
    sin = torch.randn(seq_len, head_dim // 2, device=dev, dtype=torch.bfloat16)
    if HAVE_SGL:
        q_j, k_j = q.clone(), k.clone()
        q_s, k_s = q.clone(), k.clone()
        jo.fused_qk_norm_rope_3d_paired(q_j, k_j, seq_len, num_heads, 1e-6, qw, kw, cos, sin)
        sk.fused_qk_norm_rope_3d_paired(q_s, k_s, seq_len, num_heads, 1e-6, qw, kw, cos, sin)
        report("qk_norm_rope q vs sgl", q_j, q_s)
        report("qk_norm_rope k vs sgl", k_j, k_s)
    else:
        q_j, k_j = q.clone(), k.clone()
        jo.fused_qk_norm_rope_3d_paired(q_j, k_j, seq_len, num_heads, 1e-6, qw, kw, cos, sin)
        # fp32 ref
        def ref_one(x, w):
            xf = x.float().view(B, seq_len, num_heads, head_dim)
            xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            c = cos.float().view(seq_len, 1, head_dim // 2)
            s = sin.float().view(seq_len, 1, head_dim // 2)
            x1 = xn[..., 0::2]
            x2 = xn[..., 1::2]
            o = torch.empty_like(xn)
            o[..., 0::2] = x1 * c - x2 * s
            o[..., 1::2] = x1 * s + x2 * c
            return o.view(B, M, head_dim)
        report("qk_norm_rope q vs fp32-ref", q_j, ref_one(q, qw))
        report("qk_norm_rope k vs fp32-ref", k_j, ref_one(k, kw))


if __name__ == "__main__":
    print("has_fp8:", jo.has_fp8())
    test_rmsnorm()
    test_fused_norm_scale_shift()
    test_qk_norm_rope()
