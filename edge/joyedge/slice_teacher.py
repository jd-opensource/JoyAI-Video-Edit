"""Carve a small student DiT out of a large teacher checkpoint ("weight selection" init).

    python -m joyedge.slice_teacher --teacher joyai_video_edit_dit_1101_face++.pth \
        --hidden 1024 --heads 8 --depth 16 --out student_s_init.pth

The result is only an *initialisation* for training (training/configs/edge_student_*):
a sliced network is not a working editor on its own.  Selecting evenly spaced blocks,
whole attention heads and evenly spaced residual / MLP channels keeps the statistics of
the pretrained weights, which trains faster than a random start
(Xu et al., "Initializing Models with Larger Ones", 2023).

The teacher is opened with mmap and processed tensor by tensor.
"""
from __future__ import annotations

import argparse
import re

import torch

_BLOCK = re.compile(r"^double_blocks\.(\d+)\.(.*)$")


def _even(n_src: int, n_dst: int) -> torch.Tensor:
    if n_dst > n_src:
        raise ValueError(f"cannot select {n_dst} of {n_src}")
    return torch.linspace(0, n_src - 1, n_dst).round().long()


def slice_state_dict(sd: dict, *, hidden: int, heads: int, depth: int, mlp_ratio: float = 4.0) -> dict:
    t_hidden = sd["img_in.weight"].shape[0]
    head_dim = sd["double_blocks.0.img_attn_q_norm.weight"].shape[0]
    t_heads = t_hidden // head_dim
    t_depth = 1 + max(int(m.group(1)) for k in sd if (m := _BLOCK.match(k)))
    t_mlp = sd["double_blocks.0.img_mlp.net.0.proj.weight"].shape[0]
    if hidden != heads * head_dim:
        raise ValueError(f"hidden ({hidden}) must equal heads ({heads}) x teacher head_dim ({head_dim})")

    res = _even(t_hidden, hidden)                         # residual-stream channels
    head_sel = _even(t_heads, heads)                      # whole heads, so q/k norms and rope pairs stay intact
    attn = (head_sel[:, None] * head_dim + torch.arange(head_dim)[None]).reshape(-1)
    mlp = _even(t_mlp, int(hidden * mlp_ratio))
    blocks = _even(t_depth, depth).tolist()

    def rows(t, idx):
        return t.index_select(0, idx)

    def cols(t, idx):
        return t.index_select(1, idx)

    def chunked(t, idx, n_chunks, width):
        """index every one of `n_chunks` equal slices of dim 0 with `idx`"""
        full = torch.cat([idx + c * width for c in range(n_chunks)])
        return t.index_select(0, full)

    out: dict = {}

    def put(name, t):
        out[name] = t.contiguous().clone()

    # embedders and head
    put("img_in.weight", rows(sd["img_in.weight"], res))
    put("img_in.bias", rows(sd["img_in.bias"], res))
    ce = "condition_embedder."
    put(ce + "time_embedder.linear_1.weight", rows(sd[ce + "time_embedder.linear_1.weight"], res))
    put(ce + "time_embedder.linear_1.bias", rows(sd[ce + "time_embedder.linear_1.bias"], res))
    put(ce + "time_embedder.linear_2.weight", cols(rows(sd[ce + "time_embedder.linear_2.weight"], res), res))
    put(ce + "time_embedder.linear_2.bias", rows(sd[ce + "time_embedder.linear_2.bias"], res))
    put(ce + "time_proj.weight", cols(chunked(sd[ce + "time_proj.weight"], res, 6, t_hidden), res))
    put(ce + "time_proj.bias", chunked(sd[ce + "time_proj.bias"], res, 6, t_hidden))
    put(ce + "text_embedder.linear_1.weight", rows(sd[ce + "text_embedder.linear_1.weight"], res))
    put(ce + "text_embedder.linear_1.bias", rows(sd[ce + "text_embedder.linear_1.bias"], res))
    put(ce + "text_embedder.linear_2.weight", cols(rows(sd[ce + "text_embedder.linear_2.weight"], res), res))
    put(ce + "text_embedder.linear_2.bias", rows(sd[ce + "text_embedder.linear_2.bias"], res))
    put("proj_out.weight", cols(sd["proj_out.weight"], res))
    put("proj_out.bias", sd["proj_out.bias"])

    for new_idx, old_idx in enumerate(blocks):
        src, dst = f"double_blocks.{old_idx}.", f"double_blocks.{new_idx}."
        for st in ("img", "txt"):
            put(dst + f"{st}_mod.modulate_table", sd[src + f"{st}_mod.modulate_table"].index_select(-1, res))
            put(dst + f"{st}_attn_qkv.weight", cols(chunked(sd[src + f"{st}_attn_qkv.weight"], attn, 3, t_hidden), res))
            put(dst + f"{st}_attn_qkv.bias", chunked(sd[src + f"{st}_attn_qkv.bias"], attn, 3, t_hidden))
            put(dst + f"{st}_attn_q_norm.weight", sd[src + f"{st}_attn_q_norm.weight"])
            put(dst + f"{st}_attn_k_norm.weight", sd[src + f"{st}_attn_k_norm.weight"])
            put(dst + f"{st}_attn_proj.weight", cols(rows(sd[src + f"{st}_attn_proj.weight"], res), attn))
            put(dst + f"{st}_attn_proj.bias", rows(sd[src + f"{st}_attn_proj.bias"], res))
            put(dst + f"{st}_mlp.net.0.proj.weight", cols(rows(sd[src + f"{st}_mlp.net.0.proj.weight"], mlp), res))
            put(dst + f"{st}_mlp.net.0.proj.bias", rows(sd[src + f"{st}_mlp.net.0.proj.bias"], mlp))
            put(dst + f"{st}_mlp.net.2.weight", cols(rows(sd[src + f"{st}_mlp.net.2.weight"], res), mlp))
            put(dst + f"{st}_mlp.net.2.bias", rows(sd[src + f"{st}_mlp.net.2.bias"], res))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = torch.load(args.teacher, map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get("model", sd)
    for prefix in ("model.", "module.", "transformer."):
        if any(k.startswith(prefix) for k in sd):
            sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    student = slice_state_dict(sd, hidden=args.hidden, heads=args.heads, depth=args.depth)
    torch.save(student, args.out)
    n = sum(v.numel() for v in student.values())
    print(f"wrote {args.out}: {n / 1e9:.3f}B parameters, {len(student)} tensors")


if __name__ == "__main__":
    main()
