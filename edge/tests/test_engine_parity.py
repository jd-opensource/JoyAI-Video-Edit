"""Parity: ggml CPU engine (libjoyedge) vs the PyTorch reference rollout.

Random weights with the released topology, several streaming chunks, with and without a
reference image, plus the quantized variants.  No checkpoint needed.

    edge/engine/build.sh && python edge/tests/test_engine_parity.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["JOYOMNI_PORTABLE_OPS"] = "1"
os.environ["JOYOMNI_FA4_ATTN"] = "0"

import numpy as np
import torch

EDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE))

from joyedge.engine import EdgeDiT  # noqa: E402
from joyedge.export_gguf import write_gguf  # noqa: E402
from joyedge.reference import TorchStreamingDiT, build_random_dit  # noqa: E402

QUANTIZE = EDGE / "engine" / "build" / "joyedge-quantize"


def rollout_torch(model, prompt, noises, srcs, ref, mti):
    ref_run = TorchStreamingDiT(model, n_steps=2, max_temporal_ids=mti)
    ref_run.set_prompt(prompt)
    if ref is not None:
        ref_run.set_reference_image(ref)
    return [ref_run.process_chunk(n, s).numpy() for n, s in zip(noises, srcs)]


def rollout_engine(path, prompt, noises, srcs, ref, mti, lat, **kw):
    eng = EdgeDiT(str(path), lat_h=lat[0], lat_w=lat[1], n_steps=2, max_temporal_ids=mti, **kw)
    eng.set_prompt(prompt.numpy())
    if ref is not None:
        eng.set_reference_image(ref.numpy())
    outs = [eng.process_chunk(n.numpy(), s.numpy()) for n, s in zip(noises, srcs)]
    eng.close()
    return outs


def report(tag, got, want, tol):
    worst = 0.0
    for k, (g, w) in enumerate(zip(got, want)):
        rel = float(np.abs(g - w).max() / (np.abs(w).max() + 1e-12))
        cos = float((g * w).sum() / (np.linalg.norm(g) * np.linalg.norm(w) + 1e-12))
        worst = max(worst, rel)
        print(f"    chunk {k}: max rel err {rel:.3e}  cosine {cos:.6f}")
    status = "OK " if worst < tol else "FAIL"
    print(f"  {status} {tag}: worst rel err {worst:.3e} (tol {tol:g})")
    return worst < tol


def main() -> int:
    torch.manual_seed(0)
    C, lat, n_txt, n_chunks = 16, (6, 10), 9, 5
    model = build_random_dit(hidden_size=256, heads_num=2, depth=3, channels=C, text_dim=64)
    prompt = torch.randn(n_txt, 64)
    noises = [torch.randn(C, *lat) for _ in range(n_chunks)]
    srcs = [torch.randn(C, *lat) for _ in range(n_chunks)]
    ref_img = torch.randn(C, 4, 5)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        f32 = Path(tmp) / "tiny-f32.gguf"
        cfg = write_gguf(model.state_dict(), f32, dtype="f32", verbose=False)
        print(f"exported {cfg}")

        for mti in (2, 8):
            for ref in (None, ref_img):
                tag = f"mti={mti} ref_image={'yes' if ref is not None else 'no'}"
                want = rollout_torch(model, prompt, noises, srcs, ref, mti)
                print(f"[fp32 weights, fp32 KV] {tag}")
                got = rollout_engine(f32, prompt, noises, srcs, ref, mti, lat, kv_f32=True)
                ok &= report(tag, got, want, 1e-3)
                print(f"[fp32 weights, fp16 KV, weight streaming window=1] {tag}")
                got = rollout_engine(f32, prompt, noises, srcs, ref, mti, lat, stream_window=1)
                ok &= report(tag, got, want, 1e-3)

        want = rollout_torch(model, prompt, noises, srcs, ref_img, 2)
        # python-side q8_0 export must agree with the C++ quantizer
        q8_py = Path(tmp) / "tiny-q8-py.gguf"
        write_gguf(model.state_dict(), q8_py, dtype="q8_0", verbose=False)
        got_py = rollout_engine(q8_py, prompt, noises, srcs, ref_img, 2, lat)
        for qt, tol in (("f16", 1e-3), ("q8_0", 2e-2), ("q6_k", 5e-2), ("q4_k", 0.2), ("q4_0", 0.2)):
            out = Path(tmp) / f"tiny-{qt}.gguf"
            subprocess.run([str(QUANTIZE), str(f32), str(out), qt], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[{qt} weights] size {out.stat().st_size / 1024:.0f} KiB (fp32 {f32.stat().st_size / 1024:.0f} KiB)")
            got = rollout_engine(out, prompt, noises, srcs, ref_img, 2, lat)
            ok &= report(qt, got, want, tol)
            if qt == "q8_0":
                same = max(float(np.abs(a - b).max()) for a, b in zip(got, got_py))
                print(f"    python q8_0 export vs joyedge-quantize: max abs diff {same:.3e}")
                ok &= same < 1e-4

        # ---- optional text-stream K/V cache ------------------------------------- #
        want = rollout_torch(model, prompt, noises, srcs, ref_img, 2)
        print("[text-stream cache, interval 3] chunk 0 is evaluated jointly and must stay exact")
        got = rollout_engine(f32, prompt, noises, srcs, ref_img, 2, lat, kv_f32=True, txt_cache_interval=3)
        ok &= report("chunk 0 exact", got[:1], want[:1], 1e-3)
        for k in range(1, n_chunks):
            cos = float((got[k] * want[k]).sum() / (np.linalg.norm(got[k]) * np.linalg.norm(want[k])))
            print(f"    chunk {k}: cosine to the exact rollout {cos:.4f} (approximation; random weights)")
            ok &= bool(np.isfinite(got[k]).all())
        same = rollout_engine(f32, prompt, noises, srcs, ref_img, 2, lat, kv_f32=True, txt_cache_interval=1)
        ok &= report("interval 1 == exact", same, want, 1e-3)

    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
