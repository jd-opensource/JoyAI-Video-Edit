"""Long-horizon autoregressive distillation from a Stage2 checkpoint."""

import os
from dataclasses import dataclass, field

from configs.stage2_distillation import Stage2DistillationConfig as BaseStage2DistillationConfig


@dataclass
class Stage2DistillationConfig(BaseStage2DistillationConfig):
    exp_name: str = "default_longtuning"
    resume_from_checkpoint: str | None = field(
        default_factory=lambda: os.getenv("JOYAI_STAGE2_CKPT", "checkpoints/stage2.pth")
    )
    dmd_generator_learning_rate: float = 2e-5
    dmd_rollout_num_chunks: list[int] | int = 91
    dmd_rollout_use_relative_temporal_ids: bool = False
    max_temporal_ids: int | None = 8
