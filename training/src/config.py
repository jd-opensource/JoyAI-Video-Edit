from dataclasses import dataclass
import json
import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any, Type

from src.distributed.parallel_states import print_rank0

DEFAULT_VIDEO_RESOLUTION = (720, 1248)
VAE_SPATIAL_FACTOR = 24
VAE_TEMPORAL_FACTOR = 8


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
    bucket_configs_options: list[tuple[int, int, int, int, int]] | None = None
    bucket_configs_options_prob: float = 0.5
    ref_image_basesize: int = DEFAULT_VIDEO_RESOLUTION[0]
    ref_image_bucket_configs: list[tuple[int, int, int, int, int]] | None = None
    prioritize_frame_matching: bool = True
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
    only_i2v: bool = False

    clean_use_noisy_source_id: bool = True
    causal_forcing_strategy: str = "teacher_forcing"
    teacher_forcing_noise_max_sigma: float = 0.0
    teacher_forcing_noise_min_sigma: float = 0.0

    micro_batch_size: int = 1
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    checkpoint_interval: int = 1000
    grad_check_interval: int = 1000
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

    # Parallelism
    sp_size: int = 1

    # FSDP2
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False  # zero2=False, zero3=True
    cpu_offload: bool = False
    pin_cpu_memory: bool = False
    enable_torch_compile: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.resume_from_checkpoint and self.dit_ckpt:
            raise ValueError(
                "Cannot specify both 'resume_from_checkpoint' and 'dit_ckpt'. Choose one.")

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
        if issubclass(obj, ExpConfig) and obj is not ExpConfig:
            print_rank0(
                f"Dynamically loaded config class: '{obj.__name__}' from '{file_path}'")
            return obj

    raise ValueError(
        f"No class inheriting from 'ExpConfig' was found in '{file_path}'.")


BASE_BUCKET_SIZE = 256
SUPPORTED_BASE_SIZES = {BASE_BUCKET_SIZE * scale for scale in (1, 2, 3, 4)}
VIDEO_BUCKET_ASPECT_RATIOS = (
    (1, 1),
    (4, 3),
    (3, 2),
    (16, 9),
    (21, 9),
    (9, 16),
    (2, 3),
    (3, 4),
)


def _generate_hw_buckets(
    base_height: int = BASE_BUCKET_SIZE,
    base_width: int = BASE_BUCKET_SIZE,
    step_width: int = 16,
    step_height: int = 16,
    max_ratio: float = 4.0,
) -> list[tuple[int, int, int, int, int]]:
    """Generate (bs, num_items, num_frames, h, w) buckets for a fixed pixel budget."""
    if base_height <= 0 or base_width <= 0:
        raise ValueError("base_height and base_width must be positive.")
    if step_width <= 0 or step_height <= 0:
        raise ValueError("step_width and step_height must be positive.")
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be >= 1.0.")

    buckets: list[tuple[int, int, int, int, int]] = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            buckets.append((1, 1, 1, height, width))
        next_width = width + step_width
        if height * next_width <= target_pixels:
            width = next_width
        else:
            height -= step_height

    return buckets


def _generate_video_hw_buckets_from_ratios(
    base_height: int,
    base_width: int,
    aspect_ratios: tuple[tuple[int, int], ...] = VIDEO_BUCKET_ASPECT_RATIOS,
    align: int = VAE_SPATIAL_FACTOR,
) -> list[tuple[int, int]]:
    """Generate aligned buckets for preset ratios and both orientations of the base size."""
    if base_height <= 0 or base_width <= 0:
        raise ValueError("base_height and base_width must be positive.")
    if align <= 0:
        raise ValueError("align must be positive.")

    target_pixels = base_height * base_width
    hw_list: list[tuple[int, int]] = []
    seen_hw: set[tuple[int, int]] = set()

    for ratio_h, ratio_w in (*aspect_ratios, (base_height, base_width), (base_width, base_height)):
        if ratio_h <= 0 or ratio_w <= 0:
            raise ValueError("Aspect ratios must be positive.")

        height = math.sqrt(target_pixels * ratio_h / ratio_w)
        width = math.sqrt(target_pixels * ratio_w / ratio_h)
        aligned_height = max(align, int(round(height / align)) * align)
        aligned_width = max(align, int(round(width / align)) * align)
        hw = (aligned_height, aligned_width)
        if hw not in seen_hw:
            seen_hw.add(hw)
            hw_list.append(hw)

    return hw_list


def generate_video_image_bucket(
    img_basesize: int = DEFAULT_VIDEO_RESOLUTION[0],
    min_temporal: int = 65,
    max_temporal: int = 129,
    bs_img: int = 8,
    bs_vid: int = 1,
    bs_mimg: int = 4,
    bs_mvid: int = 0,
    min_items: int = 1,
    max_items: int = 1,
    vid_basesizes: list[tuple[int, int]] | None = None,
    img_basesizes: list[tuple[int, int]] | None = None,
    vae_temporal_compression: int = VAE_TEMPORAL_FACTOR,
    spatial_multiple: int = VAE_SPATIAL_FACTOR,
) -> list[tuple[int, int, int, int, int]]:
    """Generate (batch_size, num_items, num_frames, height, width) bucket configs."""
    if spatial_multiple <= 0 or vae_temporal_compression <= 0:
        raise ValueError("VAE compression factors must be positive.")
    if img_basesizes is None and img_basesize == DEFAULT_VIDEO_RESOLUTION[0]:
        img_basesizes = [DEFAULT_VIDEO_RESOLUTION]
    use_ratio_based_img = img_basesizes is not None
    use_default_base_buckets = not use_ratio_based_img and (
        (bs_img > 0 or bs_mimg > 0)
        or (vid_basesizes is None and (bs_vid > 0 or bs_mvid > 0))
    )
    if use_default_base_buckets and img_basesize not in SUPPORTED_BASE_SIZES:
        raise ValueError(
            f"[generate_video_image_bucket] wrong img_basesize {img_basesize}")
    if bs_img < 0 or bs_vid < 0 or bs_mimg < 0 or bs_mvid < 0:
        raise ValueError("Batch sizes must be non-negative.")
    if bs_img == 0 and bs_vid == 0 and bs_mimg == 0 and bs_mvid == 0:
        raise ValueError("At least one bucket type must be enabled.")
    if (bs_vid > 0 or bs_mvid > 0) and (
        min_temporal <= 0 or max_temporal <= 0 or min_temporal > max_temporal
    ):
        raise ValueError("Invalid temporal range.")
    if (bs_mimg > 0 or bs_mvid > 0) and (
        min_items <= 0 or max_items <= 0 or min_items > max_items
    ):
        raise ValueError("Invalid multiple-item range.")
    if vid_basesizes is not None:
        if len(vid_basesizes) == 0:
            raise ValueError("vid_basesizes must not be empty.")
        for base_height, base_width in vid_basesizes:
            if base_height <= 0 or base_width <= 0:
                raise ValueError("Video bucket base sizes must be positive.")
    if img_basesizes is not None:
        if len(img_basesizes) == 0:
            raise ValueError("img_basesizes must not be empty.")
        for base_height, base_width in img_basesizes:
            if base_height <= 0 or base_width <= 0:
                raise ValueError("Image bucket base sizes must be positive.")

    bucket_list: list[tuple[int, int, int, int, int]] = []
    scale_ratio = img_basesize // BASE_BUCKET_SIZE

    def scaled_hw(height: int, width: int) -> tuple[int, int]:
        return (
            max(spatial_multiple, round(height * scale_ratio / spatial_multiple) * spatial_multiple),
            max(spatial_multiple, round(width * scale_ratio / spatial_multiple) * spatial_multiple),
        )

    if use_ratio_based_img:
        seen_img_hw: set[tuple[int, int]] = set()
        image_hw_list: list[tuple[int, int]] = []
        for base_height, base_width in img_basesizes:
            for h, w in _generate_video_hw_buckets_from_ratios(
                base_height=base_height,
                base_width=base_width,
                align=spatial_multiple,
            ):
                hw = (h, w)
                if hw not in seen_img_hw:
                    seen_img_hw.add(hw)
                    image_hw_list.append(hw)
    else:
        image_hw_bucket_list = _generate_hw_buckets()

    if vid_basesizes is None:
        video_hw_bucket_list = (
            image_hw_list if use_ratio_based_img
            else [scaled_hw(h, w) for _, _, _, h, w in image_hw_bucket_list]
        )
    else:
        seen_video_hw: set[tuple[int, int]] = set()
        video_hw_bucket_list: list[tuple[int, int]] = []
        for base_height, base_width in vid_basesizes:
            for h, w in _generate_video_hw_buckets_from_ratios(
                base_height=base_height,
                base_width=base_width,
                align=spatial_multiple,
            ):
                hw = (h, w)
                if hw not in seen_video_hw:
                    seen_video_hw.add(hw)
                    video_hw_bucket_list.append(hw)

    # image buckets
    if bs_img > 0:
        if use_ratio_based_img:
            for h, w in image_hw_list:
                bucket_list.append((bs_img, 1, 1, h, w))
        else:
            for _, _, _, h, w in image_hw_bucket_list:
                sh, sw = scaled_hw(h, w)
                bucket_list.append((bs_img, 1, 1, sh, sw))

    temporal_step = vae_temporal_compression
    aligned_min = ((min_temporal - 1 + temporal_step - 1) // temporal_step) * temporal_step + 1
    aligned_max = ((max_temporal - 1) // temporal_step) * temporal_step + 1
    if (bs_vid > 0 or bs_mvid > 0) and aligned_min > aligned_max:
        raise ValueError("Temporal range contains no VAE-aligned frame counts.")

    # video buckets
    if bs_vid > 0:
        for temporal in range(aligned_min, aligned_max + 1, temporal_step):
            video_bs = (aligned_max + 1) // temporal * bs_vid
            for h, w in video_hw_bucket_list:
                bucket_list.append((video_bs, 1, temporal, h, w))

    # multiple-image buckets
    if bs_mimg > 0:
        if use_ratio_based_img:
            for num_items in range(min_items, max_items + 1):
                for h, w in image_hw_list:
                    bucket_list.append((bs_mimg, num_items, 1, h, w))
        else:
            for num_items in range(min_items, max_items + 1):
                for _, _, _, h, w in image_hw_bucket_list:
                    sh, sw = scaled_hw(h, w)
                    bucket_list.append((bs_mimg, num_items, 1, sh, sw))

    # multiple-video buckets
    if bs_mvid > 0:
        for num_items in range(min_items, max_items + 1):
            for temporal in range(aligned_min, aligned_max + 1, temporal_step):
                video_bs = (aligned_max + 1) // temporal * bs_mvid
                for h, w in video_hw_bucket_list:
                    bucket_list.append((video_bs, num_items, temporal, h, w))

    return list(dict.fromkeys(bucket_list))


def build_editing_bucket_configs(
    img_basesize: int,
    min_temporal: int,
    max_temporal: int,
    min_items: int,
    max_items: int,
    bs_img: int = 0,
    bs_vid: int = 0,
    bs_mimg: int = 0,
    bs_mvid: int = 1,
    vid_basesizes: list[tuple[int, int]] | None = None,
    spatial_multiple: int = VAE_SPATIAL_FACTOR,
) -> list[tuple[int, int, int, int, int]]:
    img_basesizes = None
    if img_basesize not in SUPPORTED_BASE_SIZES and vid_basesizes is not None:
        img_basesizes = vid_basesizes

    bucket_configs = generate_video_image_bucket(
        img_basesize=img_basesize,
        min_temporal=min_temporal,
        max_temporal=max_temporal,
        bs_img=bs_img,
        bs_vid=bs_vid,
        bs_mimg=bs_mimg,
        bs_mvid=bs_mvid,
        min_items=min_items,
        max_items=max_items,
        vid_basesizes=vid_basesizes,
        img_basesizes=img_basesizes,
        spatial_multiple=spatial_multiple,
    )
    return [
        bucket
        for bucket in bucket_configs
        if (bucket[2] - 1) % VAE_TEMPORAL_FACTOR == 0
        and bucket[3] % spatial_multiple == 0
        and bucket[4] % spatial_multiple == 0
    ]
