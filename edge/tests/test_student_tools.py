"""Student tooling: a checkpoint produced by the *training* code base with a student
architecture must export to GGUF and run in the CPU engine, and ``slice_teacher`` must
produce a state dict that the student architecture loads strictly.

    edge/engine/build.sh && python edge/tests/test_student_tools.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("WAN_FLEX_COMPILE", "0")
os.environ["JOYOMNI_PORTABLE_OPS"] = "1"
os.environ["JOYOMNI_FA4_ATTN"] = "0"

import numpy as np
import torch

EDGE = Path(__file__).resolve().parents[1]
ROOT = EDGE.parent
sys.path.insert(0, str(EDGE))

from joyedge.engine import EdgeDiT  # noqa: E402
from joyedge.export_gguf import write_gguf  # noqa: E402
from joyedge.reference import TorchStreamingDiT, build_random_dit  # noqa: E402
from joyedge.slice_teacher import slice_state_dict  # noqa: E402


def load_training_dit():
    pkg_dir = ROOT / "training" / "src" / "models" / "mmdit" / "dit"
    pkg = types.ModuleType("joy_train_dit")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["joy_train_dit"] = pkg
    spec = importlib.util.spec_from_file_location("joy_train_dit.transformer", pkg_dir / "transformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    torch.manual_seed(0)
    ok = True

    # 1) training-code student -> GGUF -> engine == deploy-code reference
    train = load_training_dit()
    args = types.SimpleNamespace(enable_source_id_rope=True, source_id_rope_dim=128, source_id_rope_theta=256.0,
                                 enable_activation_checkpointing=False)
    hidden, heads, depth, C, text_dim = 384, 3, 2, 16, 48      # head_dim 128 like every student preset
    student = train.Transformer3DModel(
        args, patch_size=[1, 1, 1], in_channels=C, out_channels=C, hidden_size=hidden, heads_num=heads,
        text_states_dim=text_dim, mm_double_blocks_depth=depth, rope_dim_list=[16, 56, 56], theta=256,
        chunk_size=1, local_window_size=3, global_sink_chunk=True, causal=True,
    ).eval()
    with torch.no_grad():
        for name, p in student.named_parameters():
            if name.endswith("_norm.weight"):
                p.uniform_(0.5, 1.5)
            elif p.ndim >= 2:
                p.normal_(0, 0.8 / p[0].numel() ** 0.5)
            else:
                p.normal_(0, 0.02)
    sd = student.state_dict()

    deploy_model = build_random_dit(hidden_size=hidden, heads_num=heads, depth=depth, channels=C, text_dim=text_dim)
    missing, unexpected = deploy_model.load_state_dict(sd, strict=True)
    print(f"[1] training-code state dict loads strictly into the deploy model ({len(sd)} tensors)")

    lat, n_txt = (5, 9), 7
    prompt = torch.randn(n_txt, text_dim)
    noises = [torch.randn(C, *lat) for _ in range(3)]
    srcs = [torch.randn(C, *lat) for _ in range(3)]
    ref = TorchStreamingDiT(deploy_model, n_steps=2, max_temporal_ids=2)
    ref.set_prompt(prompt)
    want = [ref.process_chunk(n, s).numpy() for n, s in zip(noises, srcs)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "student.gguf"
        cfg = write_gguf(sd, path, dtype="f32", verbose=False)
        eng = EdgeDiT(str(path), lat_h=lat[0], lat_w=lat[1], n_steps=2, kv_f32=True)
        eng.set_prompt(prompt.numpy())
        got = [eng.process_chunk(n.numpy(), s.numpy()) for n, s in zip(noises, srcs)]
        eng.close()
    worst = max(float(np.abs(g - w).max() / np.abs(w).max()) for g, w in zip(got, want))
    print(f"    exported {cfg['hidden_size']}x{cfg['depth']} student, engine vs reference worst rel err {worst:.3e}")
    ok &= worst < 1e-3

    # 2) slice_teacher: carve a student out of a bigger random "teacher"
    teacher = build_random_dit(hidden_size=512, heads_num=4, depth=6, channels=C, text_dim=text_dim, seed=3)
    sliced = slice_state_dict(teacher.state_dict(), hidden=256, heads=2, depth=3)
    small = build_random_dit(hidden_size=256, heads_num=2, depth=3, channels=C, text_dim=text_dim, seed=4)
    small.load_state_dict(sliced, strict=True)
    with torch.no_grad():
        out = small(
            hidden_states=torch.randn(1, C, 1, *lat), timestep=torch.tensor([500.0]),
            encoder_hidden_states=prompt[None], current_temporal_ids=torch.zeros((1, 1), dtype=torch.long),
        )[0]
    n_t = sum(v.numel() for v in teacher.state_dict().values())
    n_s = sum(v.numel() for v in sliced.values())
    print(f"[2] slice_teacher: {n_t / 1e6:.1f}M -> {n_s / 1e6:.1f}M parameters, loads strictly, forward finite: "
          f"{bool(torch.isfinite(out).all())}")
    ok &= bool(torch.isfinite(out).all())

    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
