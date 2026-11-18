"""Write a randomly initialised DiT GGUF for latency benchmarking (no checkpoint needed).

    # the released 16B topology; only 1 of the 40 blocks is stored and reused cyclically
    python -m joyedge.make_bench_model --preset full --stored-blocks 1 --out full-f16.gguf
    # a distillation-student topology, stored in full
    python -m joyedge.make_bench_model --preset student-s --out student-s-f16.gguf

Then quantize with ``joyedge-quantize`` and time with ``joyedge-bench``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JOYOMNI_PORTABLE_OPS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joyedge.budget import PRESETS  # noqa: E402
from joyedge.export_gguf import write_gguf  # noqa: E402
from joyedge.reference import build_random_dit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="full", choices=sorted(PRESETS))
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--stored-blocks", type=int, default=0, help="store only this many blocks (0 = all)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--text-dim", type=int, default=4096)
    ap.add_argument("--type", default="f16", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    hidden = args.hidden or preset.hidden
    heads = args.heads or preset.heads
    depth = args.depth or preset.depth
    stored = args.stored_blocks or depth
    model = build_random_dit(hidden_size=hidden, heads_num=heads, depth=stored, channels=args.channels,
                             text_dim=args.text_dim)
    cfg = write_gguf(model.state_dict(), args.out, dtype=args.type, depth=depth)
    print(f"wrote {args.out}: {cfg}")


if __name__ == "__main__":
    main()
