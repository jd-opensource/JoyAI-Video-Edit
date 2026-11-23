"""Compute model for the streaming DiT: how much arithmetic one chunk costs, and what
that means on a device with a given sustained throughput.

    python -m joyedge.budget                      # table for the built-in presets
    python -m joyedge.budget --tflops 0.6         # ... on a device sustaining 0.6 TFLOP/s

All numbers are floating-point operations (1 multiply-accumulate = 2 FLOPs) of the DiT
only.  Measure the real throughput of a device with ``joyedge-bench``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str
    hidden: int
    heads: int
    depth: int
    mlp_ratio: float = 4.0
    note: str = ""

    @property
    def stream_params_per_block(self) -> float:
        # qkv (3 D^2) + proj (D^2) + mlp up/down (2 r D^2)
        return (4 + 2 * self.mlp_ratio) * self.hidden ** 2

    @property
    def params(self) -> float:
        return 2 * self.stream_params_per_block * self.depth

    @property
    def runtime_params(self) -> float:
        """Parameters touched when the text stream is not evaluated (image stream only)."""
        return self.stream_params_per_block * self.depth


PRESETS = {
    "full": Arch("full", 4096, 32, 40, note="released JoyAI-Video-Edit DiT (16B)"),
    "student-l": Arch("student-l", 2048, 16, 24, note="~2.4B, edge GPU / NPU / Apple-silicon class"),
    "student-m": Arch("student-m", 1536, 12, 20, note="~1.1B, desktop CPU with VNNI/AMX, M-series"),
    "student-s": Arch("student-s", 1024, 8, 16, note="~400M, laptop CPU"),
    "student-xs": Arch("student-xs", 768, 6, 12, note="~170M, phone / SBC class CPU"),
}


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """Pixels -> latent tokens: x2/3 stem, then 16x spatial VAE (24x overall)."""
    if width % 24 or height % 24:
        raise ValueError("width and height must be multiples of 24")
    return height // 24, width // 24


def chunk_flops(arch: Arch, lat_h: int, lat_w: int, *, n_txt: int = 400, steps: int = 2,
                history_chunks: int = 2, ref_tokens: int = 0, text_stream: bool = True) -> dict:
    """FLOPs of one streaming chunk (steps x denoise forward + 1 clean-chunk store forward)."""
    n = lat_h * lat_w
    d = arch.hidden
    lin = arch.stream_params_per_block          # MACs per token per block (one stream)
    txt = n_txt if text_stream else 0

    q_den = 2 * n + txt                         # noisy chunk + source chunk (+ text)
    k_den = q_den + history_chunks * n + ref_tokens
    den_lin = (2 * n + txt) * lin * arch.depth
    den_att = 2 * q_den * k_den * d * arch.depth   # QK^T and PV
    store_lin = n * lin * arch.depth
    store_att = 2 * n * n * d * arch.depth

    macs = steps * (den_lin + den_att) + store_lin + store_att
    return {
        "tokens_per_chunk": n,
        "linear_flops": 2 * (steps * den_lin + store_lin),
        "attention_flops": 2 * (steps * den_att + store_att),
        "flops": 2 * macs,
    }


def realtime_tflops(arch: Arch, width: int, height: int, fps: float, *, frames_per_chunk: int = 8,
                    **kw) -> float:
    lat_h, lat_w = latent_grid(width, height)
    return chunk_flops(arch, lat_h, lat_w, **kw)["flops"] * (fps / frames_per_chunk) / 1e12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflops", type=float, default=None,
                    help="sustained DiT throughput of the target device, in TFLOP/s (from joyedge-bench)")
    ap.add_argument("--txt", type=int, default=400)
    args = ap.parse_args()

    cases = [
        ("840x480 @24, 2 steps", 840, 480, 24.0, dict(steps=2)),
        ("624x360 @24, 2 steps", 624, 360, 24.0, dict(steps=2)),
        ("432x240 @12, 2 steps", 432, 240, 12.0, dict(steps=2)),
        ("432x240 @12, 1 step, no text stream", 432, 240, 12.0, dict(steps=1, text_stream=False)),
    ]
    print(f"{'model':<11s}{'params':>8s}  " + "  ".join(f"{c[0]:>36s}" for c in cases))
    for arch in PRESETS.values():
        cells = []
        for _, w, h, fps, kw in cases:
            need = realtime_tflops(arch, w, h, fps, n_txt=args.txt, **kw)
            lat_h, lat_w = latent_grid(w, h)
            per_chunk = chunk_flops(arch, lat_h, lat_w, n_txt=args.txt, **kw)["flops"] / 1e12
            if args.tflops:
                got_fps = 8.0 / (per_chunk / args.tflops)
                cells.append(f"{per_chunk:7.2f} TF/chunk -> {got_fps:8.2f} fps")
            else:
                cells.append(f"{per_chunk:7.2f} TF/chunk, RT needs {need:7.2f} TF/s")
        print(f"{arch.name:<11s}{arch.params / 1e9:7.2f}B  " + "  ".join(f"{c:>36s}" for c in cells))


if __name__ == "__main__":
    main()
