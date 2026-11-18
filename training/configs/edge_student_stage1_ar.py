"""Stage1 autoregressive training of a small "edge" student DiT.

The released model is a 16B dense transformer; on a CPU it cannot run in real time at any
quantization (see edge/README.md).  This config trains a student with the same topology
(same VAE latent space, same condition encoder, same chunk-causal KV protocol), so the
checkpoint drops straight into the joyedge CPU engine:

    python -m joyedge.export_gguf --dit-ckpt <student>.pth --out student.gguf --type f16
    joyedge-quantize student.gguf student-q6_k.gguf q6_k

Pick the size with JOYAI_STUDENT (xs | s | m | l) and the training resolution with
JOYAI_STUDENT_RES="H,W" (multiples of 24):

    JOYAI_STUDENT=s JOYAI_STUDENT_RES=240,432 \
        bash scripts/train/stage1_ar.sh configs/edge_student_stage1_ar.py 8

Initialisation: JOYAI_STUDENT_INIT may point to a checkpoint with the student's shapes
(for example the output of edge/joyedge/slice_teacher.py, which carves a student out of
the 16B teacher).  Unset means training from scratch.
"""

import os
from dataclasses import dataclass, field

from configs.stage1_ar import Stage1ARConfig as BaseStage1ARConfig
from src.config import build_editing_bucket_configs

# name: (hidden_size, heads_num, depth)  -- head_dim stays 128 so rope_dim_list is unchanged
STUDENTS = {
    "xs": (768, 6, 12),     # ~0.17B
    "s": (1024, 8, 16),     # ~0.40B
    "m": (1536, 12, 20),    # ~1.1B
    "l": (2048, 16, 24),    # ~2.4B
}


def _student() -> tuple[int, int, int]:
    name = os.getenv("JOYAI_STUDENT", "s").lower()
    if name not in STUDENTS:
        raise ValueError(f"JOYAI_STUDENT must be one of {sorted(STUDENTS)}, got {name!r}")
    return STUDENTS[name]


def _resolution() -> tuple[int, int]:
    h, w = (int(v) for v in os.getenv("JOYAI_STUDENT_RES", "240,432").split(","))
    if h % 24 or w % 24:
        raise ValueError("JOYAI_STUDENT_RES must be multiples of 24")
    return h, w


def _arch() -> dict:
    hidden, heads, depth = _student()
    return {
        "target": "src.models.mmdit.dit.Transformer3DModel",
        "params": {
            "hidden_size": hidden,
            "in_channels": 64,
            "heads_num": heads,
            "mm_double_blocks_depth": depth,
            "out_channels": 64,
            "patch_size": [1, 1, 1],
            "rope_dim_list": [16, 56, 56],
            "text_states_dim": 4096,
            "rope_type": "rope",
            "dit_modulation_type": "wanx",
            "unpatchify_new": False,
            "theta": 256,
            "chunk_size": 1,
            "causal": True,
            "local_window_size": 3,
            "global_sink_chunk": True,
        },
    }


@dataclass
class EdgeStudentStage1ARConfig(BaseStage1ARConfig):
    exp_name: str = field(default_factory=lambda: f"edge_student_{os.getenv('JOYAI_STUDENT', 's').lower()}")
    dit_ckpt: str | None = field(default_factory=lambda: os.getenv("JOYAI_STUDENT_INIT") or None)
    dit_arch_config: dict = field(default_factory=_arch)
    bucket_configs: list[tuple[int, int, int, int, int]] = field(
        default_factory=lambda: build_editing_bucket_configs(
            resolution=_resolution(),
            frame_buckets=range(49, 122, 8),
            bs_img=8,
            bs_mimg=4,
            bs_vid=2,
            bs_mvid=2,
        )
    )
    # a from-scratch / sliced student needs a real learning rate, not the SFT one
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 2000
