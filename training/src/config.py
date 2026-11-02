from dataclasses import dataclass
import json
import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any, Type

from src.distributed.parallel_states import print_rank0

DEFAULT_VIDEO_RESOLUTION = (720, 1248)
DEFAULT_REFERENCE_IMAGE_BASE_SIZE = 768
VAE_SPATIAL_FACTOR = 24
VAE_TEMPORAL_FACTOR = 8
DEFAULT_SPATIAL_BUCKETS = (
    (960, 960),
    (1104, 816),
    (1152, 768),
    (1248, 720),
    (1440, 624),
    (720, 1248),
    (768, 1152),
    (816, 1104),
    (624, 1440),
)


@dataclass
class ExpConfig:
    seed: int = 42
    exp_name: str = "test"
    output_dir: str = "./output"

    # Resume
    resume_from_checkpoint: str | None = None
    resume_optimizer: bool = True
    resume_dataloader: bool = True
    auto_resume: bool = False

    # DIT
    dit_ckpt: str | None = None
    dit_ckpt_type: str = "pt"  # "safetensor" or "pt"
    dit_arch_config: dict[str, Any] | None = None
    dit_precision: str = "bf16"
    dit_train_modules: str | list[str] | None = None

    # VAE
    vae_ckpt: str | None = None
    vae_precision: str = "fp16"
    enable_denormalization: bool = True

    # Text Encoder
    text_encoder_arch_config: dict[str, Any] | None = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 512

    # Pipeline
    pipeline_arch_config: dict[str, Any] | None = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

    # Scheduler
    scheduler_arch_config: dict[str, Any] | None = None

    # Data
    train_image_data_files: str | list[str] | None = None
    train_multiple_images_data_files: str | list[str] | None = None
    train_multiple_videos_data_files: str | list[str] | None = None
    train_video_data_files: str | list[str] | None = None
    train_image_caption_keys: list[str] | None = None
    train_image_caption_sampling_prob: list[float] | None = None
    train_multiple_images_caption_keys: list[str] | None = None
    train_multiple_images_caption_sampling_prob: list[float] | None = None
    train_multiple_videos_caption_keys: list[str] | None = None
    train_multiple_videos_caption_sampling_prob: list[float] | None = None
    train_video_caption_keys: list[str] | None = None
    train_video_caption_sampling_prob: list[float] | None = None
    image_sampling_prob: float = 0.0
    multiple_images_sampling_prob: float = 0.0
    multiple_videos_sampling_prob: float = 0.0
    video_sampling_prob: float = 0.0
    bucket_configs: list[tuple[int, int, int, int, int]] | None = None
    ensure_divisible_shards: bool = True
    shuffle: bool = True
    tar_files_shuffle_seed: int | None = None
    num_workers: int = 2
    fps: int = -1
    rec_aug_rate: float = 0.0

    # Training
    weighting_scheme: str = "lognorm"
    train_flow_shift: int = 1
    train_flow_base: int = 1
    cfg_rate: float = 0.1
    ref_cfg_rate: float = 0.0
    use_vit: bool = False

    causal_forcing_strategy: str = "teacher_forcing"
    teacher_forcing_noise_max_sigma: float = 0.0
    teacher_forcing_noise_min_sigma: float = 0.0

    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    checkpoint_interval: int = 1000
    gc_interval: int = 1000
    log_interval: int = 10

    optimizer_name: str = 'adamw'
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0
    adam_epsilon: float = 1e-10
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100

    enable_activation_checkpointing: bool = False
    activation_checkpointing_skip_interval: int = 1

    # FSDP2
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False  # zero2=False, zero3=True
    cpu_offload: bool = False
    pin_cpu_memory: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.resume_from_checkpoint and self.dit_ckpt:
            raise ValueError(
                "Cannot specify both 'resume_from_checkpoint' and 'dit_ckpt'. Choose one.")
        buckets = list(self.bucket_configs or [])
        if buckets:
            validate_bucket_configs(buckets)
            patch_size = (self.dit_arch_config or {}).get("params", {}).get("patch_size", [1, 1, 1])
            if len(patch_size) != 3 or any(type(size) is not int or size <= 0 for size in patch_size):
                raise ValueError("DiT patch_size must contain three positive integers.")
            for bucket in buckets:
                _, _, frames, height, width = bucket
                latent_frames = (frames - 1) // VAE_TEMPORAL_FACTOR + 1
                if latent_frames % patch_size[0] or height % (VAE_SPATIAL_FACTOR * patch_size[1]) or width % (VAE_SPATIAL_FACTOR * patch_size[2]):
                    raise ValueError(f"Bucket {bucket} is incompatible with DiT patch_size {patch_size}.")

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def load_config_class_from_pyfile(file_path: str) -> Type[ExpConfig]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ExpConfig) and obj is not ExpConfig and obj.__module__ == module.__name__:
            print_rank0(
                f"Dynamically loaded config class: '{obj.__name__}' from '{file_path}'")
            return obj

    raise ValueError(
        f"No class inheriting from 'ExpConfig' was found in '{file_path}'.")


def scale_spatial_buckets(resolution: tuple[int, int]) -> list[tuple[int, int]]:
    """Scale the shared spatial buckets to an aligned pixel-area budget."""
    if len(resolution) != 2 or any(type(size) is not int or size <= 0 or size % VAE_SPATIAL_FACTOR for size in resolution):
        raise ValueError(f"Resolution must contain two positive multiples of {VAE_SPATIAL_FACTOR}.")
    scale = math.sqrt(math.prod(resolution) / math.prod(DEFAULT_VIDEO_RESOLUTION))
    return list(dict.fromkeys(
        tuple(max(VAE_SPATIAL_FACTOR, round(size * scale / VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR) for size in bucket)
        for bucket in DEFAULT_SPATIAL_BUCKETS
    ))


def validate_bucket_configs(buckets: list[tuple[int, int, int, int, int]]) -> None:
    if not buckets:
        raise ValueError("At least one size bucket is required.")
    for bucket in buckets:
        if len(bucket) != 5 or any(type(value) is not int or value <= 0 for value in bucket):
            raise ValueError(f"Bucket must contain five positive integers, got {bucket}.")
        _, _, frames, height, width = bucket
        if (frames - 1) % VAE_TEMPORAL_FACTOR or height % VAE_SPATIAL_FACTOR or width % VAE_SPATIAL_FACTOR:
            raise ValueError(f"Bucket requires {VAE_TEMPORAL_FACTOR}n+1 frames and spatial multiples of {VAE_SPATIAL_FACTOR}, got {bucket}.")


def build_editing_bucket_configs(
    *,
    resolution: tuple[int, int],
    frame_buckets: list[int] | range,
    bs_img: int = 1,
    bs_vid: int = 1,
    bs_mimg: int = 1,
    bs_mvid: int = 1,
) -> list[tuple[int, int, int, int, int]]:
    """Use shared spatial buckets and fixed batch sizes for each task."""
    if any(type(size) is not int or size < 0 for size in (bs_img, bs_vid, bs_mimg, bs_mvid)):
        raise ValueError("Batch sizes must be non-negative integers.")
    frames = list(frame_buckets)
    if any(type(count) is not int or count <= 1 or (count - 1) % VAE_TEMPORAL_FACTOR for count in frames):
        raise ValueError(f"Video frame buckets must contain {VAE_TEMPORAL_FACTOR}n+1 integers greater than one.")
    if (bs_vid or bs_mvid) and not frames:
        raise ValueError("Video tasks require at least one frame-count bucket.")
    spatial_buckets = scale_spatial_buckets(resolution)
    buckets = [
        (batch_size, num_items, count, height, width)
        for batch_size, num_items, counts in (
            (bs_img, 1, [1]),
            (bs_mimg, 2, [1]),
            (bs_vid, 1, sorted(set(frames))),
            (bs_mvid, 2, sorted(set(frames))),
        )
        if batch_size
        for count in counts
        for height, width in spatial_buckets
    ]
    validate_bucket_configs(buckets)
    return buckets
