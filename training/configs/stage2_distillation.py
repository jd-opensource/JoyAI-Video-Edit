"""Resilient-DMD with long-horizon autoregressive distillation (LHAD)."""

import os
from dataclasses import dataclass, field

from src.config import DEFAULT_VIDEO_RESOLUTION, ExpConfig, build_editing_bucket_configs, generate_video_image_bucket

TRAIN_RESOLUTION = DEFAULT_VIDEO_RESOLUTION

@dataclass
class Stage2DistillationConfig(ExpConfig):
    seed: int = 42
    exp_name: str = "default"
    output_dir: str = field(default_factory=lambda: os.getenv("JOYAI_OUTPUT_DIR", "./outputs/stage2_distillation"))
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
    dit_ckpt: str | None = None
    dit_ckpt_type: str = "pt"
    dit_arch_config: dict = field(
        default_factory=lambda: {
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
            },
        }
    )
    dit_precision: str = "bf16"
    enable_source_id_rope: bool = True
    source_id_rope_dim: int = 128
    source_id_rope_theta: float = 256.0
    vae_arch_config: dict = field(
        default_factory=lambda: {
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
    tar_files_shuffle_seed: int = 99
    micro_batch_size: int = 1
    bucket_configs: list[tuple[int, int, int, int, int]] = field(
        default_factory=lambda: build_editing_bucket_configs(
            img_basesize=TRAIN_RESOLUTION[0],
            min_temporal=49,

            max_temporal=89,

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
    ensure_divisible_shards: bool = True
    shuffle: bool = True
    num_workers: int = 0
    fps: int = 24
    weighting_scheme: str = "lognorm"
    train_flow_shift: float = 5.0
    cfg_rate: float = 0.0
    max_train_steps: int = 10000000000
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 100
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
    lr_warmup_steps: int = 50
    use_vit: bool = True
    only_i2v: bool = False
    enable_activation_checkpointing: bool = True
    activation_checkpointing_skip_interval: int = 1
    causal_forcing_strategy: str = "teacher_forcing"
    teacher_forcing_noise_max_sigma: float = 0.003
    num_inference_steps: int = 2
    guidance_scale: float = 1.0
    sp_size: int = 1
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = True
    dmd_stage1_ckpt: str | None = field(default_factory=lambda: os.getenv("JOYAI_STAGE1_CKPT", "checkpoints/stage1.pth"))
    dmd_generator_ckpt: str | None = None
    dmd_fake_score_ckpt: str | None = None
    dmd_real_score_ckpt: str | None = None
    dmd_generator_ckpt_type: str = "pt"
    dmd_fake_score_ckpt_type: str = "pt"
    dmd_real_score_ckpt_type: str = "pt"
    dmd_use_lora: bool = True
    dmd_base_ckpt: str | None = None
    dmd_base_ckpt_type: str = "pt"
    dmd_generator_lora_adapter_name: str = "generator"
    dmd_fake_score_lora_adapter_name: str = "fake_score"
    dmd_lora_rank: int = 64
    dmd_lora_alpha: int = 64
    dmd_lora_dropout: float = 0.0
    dmd_lora_bias: str = "none"
    dmd_lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "img_attn_qkv",
            "img_attn_proj",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_attn_qkv",
            "txt_attn_proj",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]
    )
    dmd_generator_learning_rate: float = 5e-5
    dmd_fake_score_learning_rate: float = 2e-5
    dmd_fake_gen_update_ratio: int = 5
    dmd_generator_num_inference_steps: int = 2
    dmd_generator_window_size: int | None = None
    dmd_generator_chunk_size: int | None = None
    dmd_generator_global_sink_chunk: bool | None = None
    dmd_rollout_num_chunks: list[int] | int = -1
    dmd_grad_window_chunks: int | None = 13
    dmd_critic_rollout_num_chunks: list[int] | int = -1
    dmd_rollout_use_relative_temporal_ids: bool = True
    max_temporal_ids: int | None = 2
    dmd_real_guidance_scale: float = 2.5
    dmd_real_source_guidance_scale: float = 0.5
    dmd_source_scale_ema_beta: float = 0.99
    dmd_attn_rho_kappa: float = 2.0
    dmd_attn_rho_ratio_margin: float = 0.05
    dmd_negative_prompt: str = (
        "An abstract, computer-generated scene with distorted and blurry visuals. "
        "A deformed, disfigured figure without specific features, depicted as an illustration. "
        "The background is a collage of grainy textures and striped patterns, lacking clear visual content. "
        "The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. "
        "The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, grainy effects, and includes logos and text elements. "
        "The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition."
    )
    dmd_min_sigma: float = 0.02
    dmd_max_sigma: float = 0.98
    dmd_normalize_gradient: bool = True
    dmd_normalizer_eps: float = 1e-6
