"""Parity: deploy DiT block on the portable (pure-torch) op backend vs the training
reference block.  Runs on CPU with random weights; no checkpoint, CUDA or joyomni_ops
needed.

    python edge/tests/test_portable_block.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("WAN_FLEX_COMPILE", "0")
os.environ["JOYOMNI_PORTABLE_OPS"] = "1"
os.environ["JOYOMNI_FA4_ATTN"] = "0"

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))


def _load_training_dit():
    """Import training/src/models/mmdit/dit as an isolated package (its parent
    packages pull in the whole training stack, which this test does not need)."""
    pkg_dir = ROOT / "training" / "src" / "models" / "mmdit" / "dit"
    pkg = types.ModuleType("joy_train_dit")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["joy_train_dit"] = pkg
    spec = importlib.util.spec_from_file_location(
        "joy_train_dit.transformer", pkg_dir / "transformer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    torch.manual_seed(0)
    train = _load_training_dit()
    from xvideo.models.dit import dit as deploy
    from xvideo.models.dit import sgl_fused_ops

    assert sgl_fused_ops.backend("cpu") == "torch"

    hidden, heads = 256, 2            # head_dim 128, same as the released model
    head_dim = hidden // heads
    n_img, n_txt, n_cache = 24, 7, 12

    ref_block = train.MMDoubleStreamBlock(hidden, heads, mlp_width_ratio=4.0).eval()
    new_block = deploy.MMDoubleStreamBlock(hidden, heads, mlp_width_ratio=4.0).eval()
    with torch.no_grad():
        for p in ref_block.parameters():
            p.normal_(0, 0.05)
        for name in ("img_attn_q_norm", "img_attn_k_norm", "txt_attn_q_norm", "txt_attn_k_norm"):
            getattr(ref_block, name).weight.uniform_(0.5, 1.5)
    missing, unexpected = new_block.load_state_dict(ref_block.state_dict(), strict=True)
    assert not missing and not unexpected

    img = torch.randn(1, n_img, hidden)
    txt = torch.randn(1, n_txt, hidden)
    vec = torch.randn(1, 6, hidden) * 0.2
    ang = torch.rand(n_img, head_dim // 2) * 6.28
    cos = ang.cos().repeat_interleave(2, dim=1)
    sin = ang.sin().repeat_interleave(2, dim=1)
    ang_c = torch.rand(n_cache, head_dim // 2) * 6.28
    cos_c = ang_c.cos().repeat_interleave(2, dim=1)
    sin_c = ang_c.sin().repeat_interleave(2, dim=1)

    cache_entry = {
        "key": torch.randn(1, n_cache, heads, head_dim),
        "value": torch.randn(1, n_cache, heads, head_dim),
        "pre_rope": True,
    }
    worst = 0.0

    def check(tag, a, b, tol=2e-4):
        nonlocal worst
        err = (a - b).abs().max().item()
        rel = err / (b.abs().max().item() + 1e-12)
        worst = max(worst, rel)
        print(f"  {tag:<34s} max_abs={err:.3e} rel={rel:.3e}")
        assert rel < tol, f"{tag}: rel err {rel}"

    with torch.no_grad():
        print("[1] joint img+txt attention, no cache")
        r_img, r_txt = ref_block(img, txt, vec, vis_freqs_cis=(cos, sin))
        n_img_o, n_txt_o = new_block(img, txt, vec, (cos[None], sin[None]))
        check("img", n_img_o, r_img)
        check("txt", n_txt_o, r_txt)

        print("[2] reuse pre-rope KV cache + store (writer)")
        written_ref, written_new = {}, {}
        r_img, r_txt = ref_block(
            img, txt, vec, vis_freqs_cis=(cos, sin),
            kv_cache_reader=lambda li: [cache_entry],
            kv_cache_writer=lambda li, k, v: written_ref.update(k=k, v=v),
            layer_idx=0, kv_cache_pre_rope=True, cached_freqs_cis=(cos_c, sin_c),
        )
        n_img_o, n_txt_o = new_block(
            img, txt, vec, (cos[None], sin[None]),
            kv_cache_reader=lambda li: [cache_entry],
            kv_cache_writer=lambda li, k, v: written_new.update(k=k, v=v),
            layer_idx=0, kv_cache_pre_rope=True, cached_freqs_cis=(cos_c[None], sin_c[None]),
        )
        check("img", n_img_o, r_img)
        check("txt", n_txt_o, r_txt)
        check("stored key (pre-rope)", written_new["k"], written_ref["k"])
        check("stored value", written_new["v"], written_ref["v"])

        print("[3] clean-chunk store pass (text stream skipped)")
        r_img, _ = ref_block(img, txt, vec, vis_freqs_cis=(cos, sin), skip_text=True,
                             kv_cache_writer=lambda li, k, v: written_ref.update(k=k, v=v),
                             layer_idx=0, kv_cache_pre_rope=True)
        n_img_o, _ = new_block(img, txt, vec, (cos[None], sin[None]), skip_text_stream=True,
                               kv_cache_writer=lambda li, k, v: written_new.update(k=k, v=v),
                               layer_idx=0, kv_cache_pre_rope=True)
        check("img", n_img_o, r_img)
        check("stored key (pre-rope)", written_new["k"], written_ref["k"])

    print(f"OK  portable block == training reference (worst rel err {worst:.2e})")


if __name__ == "__main__":
    main()
