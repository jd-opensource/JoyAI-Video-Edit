"""Convert a JoyAI-Video-Edit DiT checkpoint (.pth) to GGUF for the joyedge engine.

    python -m joyedge.export_gguf --dit-ckpt joyai_video_edit_dit_1101_face++.pth \
        --out joyai-dit-q8_0.gguf --type q8_0

The checkpoint is opened with ``mmap`` and tensors are converted and written one at a
time, so peak memory is roughly one weight matrix (a few hundred MiB for the released
16B model), not the 32 GiB checkpoint.

``--type`` f32 / f16 / q8_0 are produced directly.  For 4-bit files write f16 or q8_0
here and run ``joyedge-quantize in.gguf out.gguf q4_0`` (C++, multi-threaded).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import gguf

ARCH = "joyai-video-edit-dit"
_TYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}
_BLOCK_RE = re.compile(r"^double_blocks\.(\d+)\.")


def infer_config(sd: dict, *, rope_dims: Iterable[int] | None = None, depth: int | None = None) -> dict:
    """Read the architecture back from tensor shapes."""
    hidden = int(sd["img_in.weight"].shape[0])
    in_channels = int(np.prod(sd["img_in.weight"].shape[1:]))
    if tuple(sd["img_in.weight"].shape[2:]) not in {(), (1, 1, 1)}:
        raise ValueError("joyedge supports patch_size [1, 1, 1] only (the released configuration).")
    head_dim = int(sd["double_blocks.0.img_attn_q_norm.weight"].shape[0])
    blocks = {int(m.group(1)) for k in sd if (m := _BLOCK_RE.match(k))}
    rope = [int(x) for x in (rope_dims or [16, 56, 56])]
    if sum(rope) != head_dim:
        raise ValueError(f"rope dims {rope} must sum to head_dim {head_dim}; pass --rope-dims.")
    return {
        "hidden_size": hidden,
        "heads_num": hidden // head_dim,
        "depth": int(depth or (max(blocks) + 1)),
        "in_channels": in_channels,
        "out_channels": int(sd["proj_out.weight"].shape[0]),
        "text_dim": int(sd["condition_embedder.text_embedder.linear_1.weight"].shape[1]),
        "mlp_hidden": int(sd["double_blocks.0.img_mlp.net.0.proj.weight"].shape[0]),
        "time_freq_dim": int(sd["condition_embedder.time_embedder.linear_1.weight"].shape[1]),
        "rope_dims": rope,
    }


def _gguf_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if name == "img_in.weight":
        return (shape[0], int(np.prod(shape[1:])))          # Conv3d 1x1x1 == Linear
    if name.endswith("modulate_table"):
        return (shape[-2], shape[-1])                       # [1, 6, D] -> [6, D]
    return tuple(shape)


def _target_type(name: str, shape: tuple[int, ...], want: gguf.GGMLQuantizationType) -> gguf.GGMLQuantizationType:
    F32 = gguf.GGMLQuantizationType.F32
    if len(shape) != 2 or not name.endswith(".weight") or math.prod(shape) < 65536:
        return F32
    if name.startswith(("img_in.", "proj_out.")):
        return F32
    if want == gguf.GGMLQuantizationType.Q8_0 and shape[1] % 32 != 0:
        return F32
    return want


def _convert(t: torch.Tensor, shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    arr = t.detach().to(torch.float32).reshape(shape).contiguous().numpy()
    if qtype == gguf.GGMLQuantizationType.F32:
        return arr
    if qtype == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16)
    return gguf.quants.quantize(arr, qtype)


def write_gguf(
    sd: dict,
    out: str | Path,
    *,
    dtype: str = "f16",
    rope_dims: Iterable[int] | None = None,
    depth: int | None = None,
    rope_theta: float = 256.0,
    source_id_rope_dim: int = 128,
    source_id_rope_theta: float = 256.0,
    scheduler_shift: float = 5.159,
    num_train_timesteps: int = 1000,
    verbose: bool = True,
) -> dict:
    """Write `sd` (a DiT state dict; tensors may be mmap-backed) to `out`.

    `depth` larger than the number of stored blocks marks a benchmark file whose blocks
    are reused cyclically by the engine.
    """
    want = _TYPES[dtype]
    for prefix in ("model.", "module.", "transformer."):
        if any(k.startswith(prefix) for k in sd):
            sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    cfg = infer_config(sd, rope_dims=rope_dims, depth=depth)
    head_dim = cfg["hidden_size"] // cfg["heads_num"]

    w = gguf.GGUFWriter(str(out), ARCH)
    w.add_string("joyedge.source_dtype", dtype)
    for key in ("hidden_size", "heads_num", "depth", "in_channels", "out_channels", "text_dim",
                "mlp_hidden", "time_freq_dim"):
        w.add_uint32(f"joyedge.{key}", cfg[key])
    w.add_array("joyedge.rope_dims", cfg["rope_dims"])
    w.add_float32("joyedge.rope_theta", float(rope_theta))
    w.add_uint32("joyedge.source_id_rope_dim", min(int(source_id_rope_dim), head_dim))
    w.add_float32("joyedge.source_id_rope_theta", float(source_id_rope_theta))
    w.add_float32("joyedge.scheduler_shift", float(scheduler_shift))
    w.add_uint32("joyedge.num_train_timesteps", int(num_train_timesteps))

    # Block tensors are written block by block so that each block is one contiguous
    # byte range in the file -- that is what the engine pages in and out.
    def order(name: str) -> tuple[int, int, str]:
        m = _BLOCK_RE.match(name)
        return (1, int(m.group(1)), name) if m else (0, 0, name)

    names = sorted(sd.keys(), key=order)
    plan = []
    for name in names:
        shape = _gguf_shape(name, tuple(sd[name].shape))
        qtype = _target_type(name, shape, want)
        byte_shape = gguf.quant_shape_to_byte_shape(shape, qtype)
        if qtype == gguf.GGMLQuantizationType.F32:
            np_dtype, nbytes = np.dtype(np.float32), 4 * math.prod(shape)
        elif qtype == gguf.GGMLQuantizationType.F16:
            np_dtype, nbytes = np.dtype(np.float16), 2 * math.prod(shape)
        else:
            np_dtype, nbytes = np.dtype(np.uint8), math.prod(byte_shape)
        # gguf-py wants the *byte* shape for block-quantized tensors and derives the logical one
        info_shape = byte_shape if np_dtype == np.dtype(np.uint8) else shape
        w.add_tensor_info(name, info_shape, np_dtype, nbytes, raw_dtype=qtype)
        plan.append((name, shape, qtype))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = 0
    for i, (name, shape, qtype) in enumerate(plan):
        arr = _convert(sd[name], shape, qtype)
        w.write_tensor_data(arr)
        total += arr.nbytes
        if verbose and (len(shape) == 2 and math.prod(shape) >= 1 << 22) and (i % 16 == 0):
            print(f"  [{i + 1:5d}/{len(plan)}] {name:<52s} {qtype.name:<5s} {total / 2**30:7.2f} GiB written",
                  file=sys.stderr, flush=True)
    w.close()
    cfg["bytes"] = total
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit-ckpt", required=True, help="DiT .pth checkpoint (released weights or a student)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="f16", choices=sorted(_TYPES))
    ap.add_argument("--rope-dims", type=int, nargs=3, default=None)
    ap.add_argument("--scheduler-shift", type=float, default=5.159)
    args = ap.parse_args()

    sd = torch.load(args.dit_ckpt, map_location="cpu", weights_only=True, mmap=True)
    if "model" in sd:
        sd = sd["model"]
    if "generator_lora" in sd:
        raise SystemExit("This is a LoRA-only Stage2 checkpoint; merge it into the Stage1 backbone first "
                         "(training/src/models/__init__.py::_merge_generator_lora) and export the merged weights.")
    cfg = write_gguf(sd, args.out, dtype=args.type, rope_dims=args.rope_dims, scheduler_shift=args.scheduler_shift)
    print(f"wrote {args.out}: {cfg}")


if __name__ == "__main__":
    main()
