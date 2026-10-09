"""Autoregressive SFT with teacher forcing or resampling forcing."""

import os
from dataclasses import dataclass, field

from src.config import ExpConfig, build_editing_bucket_configs, generate_video_image_bucket

TRAIN_RESOLUTION = (720, 1248)

@dataclass
class Stage1ARConfig(ExpConfig):
    seed: int = 42
    exp_name: str = "default"
    output_dir: str = field(default_factory=lambda: os.getenv("JOYAI_OUTPUT_DIR", "./outputs/stage1_ar"))
    resume_from_checkpoint: str | None = None
    resume_optimizer: bool = False
    resume_dataloader: bool = False
    auto_resume: bool = False
    pipeline_arch_config: dict = field(
        default_factory=lambda: {
            "target": "src.models.common.diffusion.pipelines.Pipeline",
            "params": {},
        }
    )
    dit_ckpt: str | None = field(default_factory=lambda: os.getenv("JOYAI_BASE_MODEL", "checkpoints/base_model.pth"))
    dit_ckpt_type: str = "pt"
    dit_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.dit.Transformer3DModel",
        "params": {
            "hidden_size": 4096,
            "in_channels": 64,
            "heads_num": 32,
            "mm_double_blocks_depth": 40,
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
        }
    }
    )
    dit_precision: str = "bf16"
    enable_source_id_rope: bool = True
    source_id_rope_dim: int = 128
    source_id_rope_theta: float = 256.0
    clean_use_noisy_source_id: bool = True
    vae_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.vae.XVAEChunkCausal",
        "pretrained": os.getenv("JOYAI_VAE", "checkpoints/vae"),
    }
    )
    vae_precision: str = "bf16"
    enable_denormalization: bool = True
    text_encoder_arch_config: dict = field(
        default_factory=lambda: {
            "target": "src.models.mmdit.text_encoder.load_text_encoder",
            "params": {
                "text_encoder_ckpt": os.getenv("JOYAI_TEXT_ENCODER", "checkpoints/text_encoder"),
            },
        }
    )
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 1024
    scheduler_arch_config: dict = field(
        default_factory=lambda: {
            "target": "src.models.common.diffusion.schedulers.FlowMatchDiscreteScheduler",
            "params": {
                "num_train_timesteps": 1000,
                "shift": 5.159,
            },
        }
    )
    train_image_data_files: str | list[str] | None = field(
        default_factory=lambda: os.getenv("JOYAI_T2I_DATA", "data/t2i.json")
    )
    train_multiple_images_data_files: str | list[str] | None = field(
        default_factory=lambda: os.getenv("JOYAI_I2I_DATA", "data/i2i.json")
    )
    train_video_data_files: str | list[str] | None = field(
        default_factory=lambda: os.getenv("JOYAI_T2V_DATA", "data/t2v.json")
    )
    train_multiple_videos_data_files: str | list[str] | None = field(
        default_factory=lambda: os.getenv("JOYAI_V2V_DATA", "data/v2v.json")
    )
    image_sampling_prob: float = 1.0
    multiple_images_sampling_prob: float = 2.0
    video_sampling_prob: float = 1.0
    multiple_videos_sampling_prob: float = 4.0
    train_image_caption_keys: list[str] = field(
        default_factory=lambda: ["caption"]
    )
    train_image_caption_sampling_prob: list[float] = field(
        default_factory=lambda: [1.0]
    )
    train_multiple_images_caption_keys: list[str] = field(
        default_factory=lambda: ["caption"]
    )
    train_multiple_images_caption_sampling_prob: list[float] = field(
        default_factory=lambda: [1.0]
    )
    train_video_caption_keys: list[str] = field(
        default_factory=lambda: ["caption"]
    )
    train_video_caption_sampling_prob: list[float] = field(
        default_factory=lambda: [1.0]
    )
    train_multiple_videos_caption_keys: list[str] = field(
        default_factory=lambda: ["caption"]
    )
    train_multiple_videos_caption_sampling_prob: list[float] = field(
        default_factory=lambda: [1.0]
    )
    tar_files_shuffle_seed: int = 110
    micro_batch_size: int = 1
    bucket_configs: list[tuple[int, int, int, int, int]] = field(
        default_factory=lambda: build_editing_bucket_configs(
            img_basesize=TRAIN_RESOLUTION[0],
            min_temporal=49,
            max_temporal=121,
            min_items=2,
            max_items=2,
            bs_img=1,
            bs_mimg=1,
            bs_vid=1,
            bs_mvid=1,
            vid_basesizes=[TRAIN_RESOLUTION],
            spatial_multiple=24,
        )
    )
    ref_image_basesize: int = TRAIN_RESOLUTION[0]
    ref_image_bucket_configs: list[tuple[int, int, int, int, int]] = field(
        default_factory=lambda: generate_video_image_bucket(
            img_basesizes=[TRAIN_RESOLUTION],
            spatial_multiple=24,
            bs_img=1,
            bs_vid=0,
            bs_mimg=0,
            bs_mvid=0,
        )
    )
    prioritize_frame_matching: bool = True
    ensure_divisible_shards: bool = False
    shuffle: bool = True
    num_workers: int = 0
    fps: int = 24
    weighting_scheme: str = "lognorm"
    train_flow_shift: float = 5.159
    cfg_rate: float = 0.01
    ref_cfg_rate: float = 0.01
    max_train_steps: int = 10000000000
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 500
    grad_check_interval: int = 1000
    gc_interval: int = 5000
    log_interval: int = 1
    optimizer_name: str = 'adamw'
    learning_rate: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-10
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    use_vit: bool = True
    only_i2v: bool = False
    enable_activation_checkpointing: bool = True
    activation_checkpointing_skip_interval: int = 1
    causal_forcing_strategy: str = "resampling_forcing"
    teacher_forcing_noise_max_sigma: float = 0.03
    teacher_forcing_noise_min_sigma: float = 0.0
    resampling_forcing_shift_s: float = 0.6
    resampling_forcing_warmup_steps: int = 7000
    num_inference_steps: int = 30
    guidance_scale: float = 3.5
    source_guidance_scale: float = 1.5
    sp_size: int = 1
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = True
