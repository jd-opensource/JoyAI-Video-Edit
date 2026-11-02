"""Symmetric GPTQ expert kernels; compile on CPU, launch only inside a GPU lease."""

import torch
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@tr.jit
def _gemv(
    X,
    Q,
    S,
    Id,
    W,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    DOWN: tl.constexpr,
    BN: tl.constexpr,
):
    epos = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    expert = tl.load(Id + epos)
    x = tl.load(X + (epos * K if DOWN else 0) + k)
    q = tl.load(Q + expert * N * (K // 8) + n[:, None] * (K // 8) + k[None, :] // 8)
    q = (q >> 4 * (k[None, :] % 8) & 15).to(tl.float32) - 8.0
    scale = tl.load(
        S + expert * N * (K // 128) + n[:, None] * (K // 128) + k[None, :] // 128
    ).to(tl.float32)
    weight = (q * scale).to(tl.bfloat16).to(tl.float32)
    y = tl.sum(weight * x[None, :].to(tl.float32), 1)
    if DOWN:
        y *= tl.load(W + epos).to(tl.float32)
    tl.store(Y + epos * N + n, y)


@tr.jit
def _gemm(
    X,
    Q,
    S,
    Order,
    Counts,
    Starts,
    Blocks,
    Id,
    W,
    Y,
    T,
    K: tl.constexpr,
    N: tl.constexpr,
    DOWN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    bid = tl.program_id(0)
    es = tl.arange(0, 256)
    ends = tl.load(Blocks + es)
    eid = tl.sum((bid >= ends).to(tl.int32))
    if eid < 256:
        prev = tl.load(Blocks + eid - 1, mask=eid > 0, other=0)
        start = tl.load(Starts + eid)
        count = tl.load(Counts + eid)
        rows = (bid - prev) * BM + tl.arange(0, BM)
        assign = tl.load(Order + start + rows, mask=rows < count, other=0)
        nr = tl.program_id(1) * BN + tl.arange(0, BN)
        kr = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for kb in range(K // BK):
            k = kb * BK + kr
            ar = assign if DOWN else assign // 8
            a = tl.load(
                X + ar[:, None] * K + k[None, :], mask=(rows < count)[:, None], other=0
            )
            packed = tl.load(
                Q + eid * N * (K // 8) + nr[None, :] * (K // 8) + k[:, None] // 8
            )
            q = (packed >> 4 * (k[:, None] % 8) & 15).to(tl.float32) - 8.0
            scale = tl.load(
                S + eid * N * (K // 128) + nr[None, :] * (K // 128) + k[:, None] // 128
            ).to(tl.float32)
            b = (q * scale).to(tl.bfloat16)
            acc += tl.dot(a, b)
        if DOWN:
            acc *= tl.load(W + assign, mask=rows < count, other=0)[:, None].to(
                tl.float32
            )
        tl.store(
            Y + assign[:, None] * N + nr[None, :], acc, mask=(rows < count)[:, None]
        )


_compiled = None


def prepare_kernels():
    global _compiled
    if _compiled is not None:
        return _compiled
    target = GPUTarget("cuda", 120, 32)
    kernels = {}
    for down, k, n in [(False, 2048, 1024), (True, 512, 2048)]:
        sig = dict(X="*bf16", Q="*i32", S="*bf16", Id="*i64", W="*bf16", Y="*bf16")
        kernels["v", down] = tr.compile(
            ASTSource(_gemv, sig, constexprs=dict(K=k, N=n, DOWN=down, BN=8)),
            target=target,
            options={"num_warps": 4},
        )
        sig = dict(
            X="*bf16",
            Q="*i32",
            S="*bf16",
            Order="*i64",
            Counts="*i64",
            Starts="*i64",
            Blocks="*i64",
            Id="*i64",
            W="*bf16",
            Y="*bf16",
            T="i32",
        )
        kernels["m", down] = tr.compile(
            ASTSource(
                _gemm, sig, constexprs=dict(K=k, N=n, DOWN=down, BM=16, BN=64, BK=64)
            ),
            target=target,
            options={"num_warps": 4},
        )
    # Build host launchers during CPU startup, before the short GPU lease.
    from triton.backends.nvidia.driver import CudaLauncher, CudaUtils

    CudaUtils()
    for kernel in kernels.values():
        CudaLauncher(kernel.src, kernel.metadata)
    _compiled = kernels
    return kernels


class Int4Experts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
        ) == (256, 2048, 512, 8)
        for name, n, k in [("gate_up", 1024, 2048), ("down", 2048, 512)]:
            self.register_buffer(
                name + "_q", torch.empty(256, n, k // 8, dtype=torch.int32)
            )
            self.register_buffer(
                name + "_s", torch.empty(256, n, k // 128, dtype=torch.bfloat16)
            )

    def forward(self, x, ids, weights):
        kernels = prepare_kernels()
        m = x.shape[0]
        t = m * 8
        up = torch.empty(t, 1024, device=x.device, dtype=x.dtype)
        down = torch.empty(t, 2048, device=x.device, dtype=x.dtype)
        if m == 1:
            kernels["v", False][8, 128, 1](
                x, self.gate_up_q, self.gate_up_s, ids, weights, up
            )
            mid = torch.nn.functional.silu(up[:, :512]) * up[:, 512:]
            kernels["v", True][8, 256, 1](
                mid, self.down_q, self.down_s, ids, weights, down
            )
        else:
            flat = ids.flatten()
            (_, order) = flat.sort()
            counts = torch.zeros(256, device=x.device, dtype=torch.int64).scatter_add_(
                0, flat, torch.ones_like(flat)
            )
            ends = counts.cumsum(0)
            starts = ends - counts
            blocks = ((counts + 15) // 16).cumsum(0)
            grid = (tr.cdiv(t, 16) + 256, 16, 1)
            kernels["m", False][grid](
                x,
                self.gate_up_q,
                self.gate_up_s,
                order,
                counts,
                starts,
                blocks,
                ids,
                weights,
                up,
                t,
            )
            mid = torch.nn.functional.silu(up[:, :512]) * up[:, 512:]
            kernels["m", True][grid[0], 32, 1](
                mid,
                self.down_q,
                self.down_s,
                order,
                counts,
                starts,
                blocks,
                ids,
                weights,
                down,
                t,
            )
        return down.view(m, 8, 2048).sum(1)
