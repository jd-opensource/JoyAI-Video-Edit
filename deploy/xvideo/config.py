from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


# Default checkpoint locations (users clone weights into deps/checkpoints).
_CKPT_ROOT = Path(__file__).resolve().parents[1] / "deps" / "checkpoints"
_DEFAULT_VAE_PATH = _CKPT_ROOT / "JoyAI-Video-Edit" / "vae"
_DEFAULT_TEXT_ENCODER_PATH = _CKPT_ROOT / "MiMo-VL-7B-RL-2508"


@dataclass
class ExpConfig:

    seed: int = 42

    dit_ckpt: str | None = None
    dit_arch_config: dict[str, Any] = field(
        default_factory=lambda: {
            "params": {
                "hidden_size": 4096,
                "in_channels": 64,
                "heads_num": 32,
                "mm_double_blocks_depth": 40,
                "out_channels": 64,
                "patch_size": [1, 1, 1],
                "rope_dim_list": [16, 56, 56],
                "text_states_dim": 4096,
                "theta": 256,
                "chunk_size": 1,
                "causal": True,
                "local_window_size": 3,
                "global_sink_chunk": True,
                "source_id_rope_dim": 128,
                "source_id_rope_theta": 256.0,
            },
        }
    )
    dit_precision: str = "bf16"

    vae_arch_config: dict[str, Any] = field(
        default_factory=lambda: {"pretrained": str(_DEFAULT_VAE_PATH)}
    )
    vae_precision: str = "bf16"
    enable_denormalization: bool = True

    text_encoder_arch_config: dict[str, Any] = field(
        default_factory=lambda: {
            "params": {"text_encoder_ckpt": str(_DEFAULT_TEXT_ENCODER_PATH)},
        }
    )
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 1024

    pipeline_arch_config: dict[str, Any] = field(default_factory=lambda: {"params": {}})
    scheduler_arch_config: dict[str, Any] = field(
        default_factory=lambda: {"params": {"num_train_timesteps": 1000, "shift": 5.159}}
    )

    ref_image_basesize: int = 768


BASE_BUCKET_SIZE = 256
SUPPORTED_BASE_SIZES = {BASE_BUCKET_SIZE * scale for scale in (1, 2, 3, 4)}
VIDEO_TEMPORAL_BUCKET_STEP = 8
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
    align: int = 32,
) -> list[tuple[int, int]]:
    if base_height <= 0 or base_width <= 0:
        raise ValueError("base_height and base_width must be positive.")
    if align <= 0:
        raise ValueError("align must be positive.")

    target_pixels = base_height * base_width
    hw_list: list[tuple[int, int]] = []
    seen_hw: set[tuple[int, int]] = set()

    for ratio_h, ratio_w in aspect_ratios:
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
    img_basesize: int = BASE_BUCKET_SIZE,
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
) -> list[tuple[int, int, int, int, int]]:
    use_ratio_based_img = img_basesizes is not None
    use_default_base_buckets = (
        (not use_ratio_based_img and (bs_img > 0 or bs_mimg > 0)) or
        (vid_basesizes is None and (bs_vid > 0 or bs_mvid > 0))
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

    def scaled_hw(h: int, w: int) -> tuple[int, int]:
        if scale_ratio == 1:
            return h, w
        return h * scale_ratio, w * scale_ratio

    if use_ratio_based_img:
        seen_img_hw: set[tuple[int, int]] = set()
        image_hw_pairs: list[tuple[int, int]] = []
        for base_height, base_width in img_basesizes:
            for h, w in _generate_video_hw_buckets_from_ratios(
                base_height=base_height,
                base_width=base_width,
            ):
                hw = (h, w)
                if hw not in seen_img_hw:
                    seen_img_hw.add(hw)
                    image_hw_pairs.append(hw)
    else:
        image_hw_pairs = [
            scaled_hw(h, w) for _, _, _, h, w in _generate_hw_buckets()
        ]

    if vid_basesizes is None:
        video_hw_bucket_list = image_hw_pairs
    else:
        seen_video_hw: set[tuple[int, int]] = set()
        video_hw_bucket_list = []
        for base_height, base_width in vid_basesizes:
            for h, w in _generate_video_hw_buckets_from_ratios(
                base_height=base_height,
                base_width=base_width,
            ):
                hw = (h, w)
                if hw not in seen_video_hw:
                    seen_video_hw.add(hw)
                    video_hw_bucket_list.append(hw)

    if bs_img > 0:
        for h, w in image_hw_pairs:
            bucket_list.append((bs_img, 1, 1, h, w))

    temporal_step = VIDEO_TEMPORAL_BUCKET_STEP

    if bs_vid > 0:
        for temporal in range(min_temporal, max_temporal + 1, temporal_step):
            video_bs = (max_temporal + 1) // temporal * bs_vid
            for h, w in video_hw_bucket_list:
                bucket_list.append((video_bs, 1, temporal, h, w))

    if bs_mimg > 0:
        for num_items in range(min_items, max_items + 1):
            for h, w in image_hw_pairs:
                bucket_list.append((bs_mimg, num_items, 1, h, w))

    if bs_mvid > 0:
        for num_items in range(min_items, max_items + 1):
            for temporal in range(min_temporal, max_temporal + 1, temporal_step):
                video_bs = (max_temporal + 1) // temporal * bs_mvid
                for h, w in video_hw_bucket_list:
                    bucket_list.append((video_bs, num_items, temporal, h, w))

    return bucket_list
