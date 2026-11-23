"""Stage2 two-step distillation of an edge student (see configs/edge_student_stage1_ar.py).

As released, the DMD trainer builds generator, fake-score and real-score models from one
``dit_arch_config``, so this config distils the student against *its own* Stage1 weights
(self-distillation to 2 steps), exactly like the released recipe does for the 16B model:

    export JOYAI_STAGE1_CKPT=<student stage1>.pth
    JOYAI_STUDENT=s JOYAI_STUDENT_RES=240,432 \
        bash scripts/train/stage2_distillation.sh configs/edge_student_stage2_distillation.py 8

Using the 16B model as the real-score teacher for a small generator needs a second arch
config for the real-score role in src/train_t2v_dmd.py (load_stage1_dit); that change is
described in edge/README.md and is not part of this config.
"""

import os
from dataclasses import dataclass, field

from configs.edge_student_stage1_ar import _arch, _resolution
from configs.stage2_distillation import Stage2DistillationConfig as BaseStage2DistillationConfig
from src.config import build_editing_bucket_configs


@dataclass
class EdgeStudentStage2DistillationConfig(BaseStage2DistillationConfig):
    exp_name: str = field(default_factory=lambda: f"edge_student_{os.getenv('JOYAI_STUDENT', 's').lower()}_dmd")
    dit_arch_config: dict = field(default_factory=_arch)
    bucket_configs: list[tuple[int, int, int, int, int]] = field(
        default_factory=lambda: build_editing_bucket_configs(
            resolution=_resolution(),
            frame_buckets=range(49, 90, 8),
            bs_img=4,
            bs_mimg=4,
            bs_vid=2,
            bs_mvid=2,
        )
    )
