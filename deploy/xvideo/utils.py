import math
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from xvideo.config import BASE_BUCKET_SIZE, generate_video_image_bucket


def seed_everything(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


_ASPECT_RATIO_TIE_EPS = 1e-4


class _BucketGroup:

    def __init__(self, bucket_configs, prioritize_frame_matching: bool = True):
        self.bucket_configs = [tuple(b) for b in bucket_configs]
        self.prioritize_frame_matching = prioritize_frame_matching

    def find_best_bucket(self, media_shape):
        num_items, num_frames, height, width = media_shape
        target_aspect_ratio = height / width

        if num_frames == 1:
            valid_buckets = [
                b for b in self.bucket_configs if b[1] == num_items and b[2] == 1
            ]
            if not valid_buckets:
                raise ValueError(f"No image buckets found for shape {media_shape}")
            return min(
                valid_buckets,
                key=lambda b: abs((b[3] / b[4]) - target_aspect_ratio),
            )

        valid_buckets = [
            b for b in self.bucket_configs
            if b[1] == num_items and 1 < b[2] <= num_frames
        ]
        if not valid_buckets:
            raise ValueError(f"No video buckets found for shape {media_shape}")

        if self.prioritize_frame_matching:
            max_frame_count = max(b[2] for b in valid_buckets)
            max_frame_buckets = [b for b in valid_buckets if b[2] == max_frame_count]
            return min(
                max_frame_buckets,
                key=lambda b: abs((b[3] / b[4]) - target_aspect_ratio),
            )

        min_ratio_diff = min(
            abs((b[3] / b[4]) - target_aspect_ratio) for b in valid_buckets
        )
        best_ratio_buckets = [
            b for b in valid_buckets
            if abs((b[3] / b[4]) - target_aspect_ratio) <= min_ratio_diff + _ASPECT_RATIO_TIE_EPS
        ]
        return max(best_ratio_buckets, key=lambda b: b[2])


def _dynamic_resize_from_bucket(
    image: Image.Image | torch.Tensor,
    bucket_configs: list[tuple[int, int, int, int, int]] | None = None,
    img_basesize: int | None = BASE_BUCKET_SIZE * 2,
    num_frames: int = 1,
    num_items: int = 1,
    prioritize_frame_matching: bool = True,
    return_bucket: bool = False,
    multiple_videos: bool = False,
    vid_basesizes: list[tuple[int, int]] | None = None,
    img_basesizes: list[tuple[int, int]] | None = None,
):
    def resize_center_crop(
        img: Image.Image | torch.Tensor,
        target_size: tuple[int, int],
        src_hw: tuple[int, int],
    ) -> Image.Image | torch.Tensor:
        h, w = src_hw
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
        img = TF.resize(img, (resize_h, resize_w),
                        interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, target_size)
        return img

    if isinstance(image, Image.Image):
        img_w, img_h = image.size
        num_frames = 1
    elif torch.is_tensor(image):
        if image.dim() != 4:
            raise ValueError(
                "Video tensor passed to `_dynamic_resize_from_bucket` must have shape (t, c, h, w)."
            )
        img_h, img_w = image.shape[-2:]
        if num_frames <= 1:
            num_frames = int(image.shape[0])
        if multiple_videos:
            num_items = 2
    else:
        raise TypeError(f"Unsupported media type for bucket resize: {type(image)}")

    if bucket_configs is None:
        if img_basesize is None and vid_basesizes is None:
            raise ValueError("Either `bucket_configs` or `img_basesize` or `vid_basesizes` must be provided.")

        is_video = num_frames > 1
        is_multiple_items = num_items > 1
        bucket_configs = generate_video_image_bucket(
            img_basesize=img_basesize,
            min_temporal=num_frames,
            max_temporal=num_frames,
            bs_img=1 if not is_video and not is_multiple_items else 0,
            bs_vid=1 if is_video and not is_multiple_items else 0,
            bs_mimg=1 if not is_video and is_multiple_items else 0,
            bs_mvid=1 if is_video and is_multiple_items else 0,
            min_items=num_items,
            max_items=num_items,
            vid_basesizes=vid_basesizes,
            img_basesizes=img_basesizes,
        )

    bucket_group = _BucketGroup(
        bucket_configs,
        prioritize_frame_matching=prioritize_frame_matching,
    )
    bucket = bucket_group.find_best_bucket((num_items, num_frames, img_h, img_w))
    target_height, target_width = bucket[-2], bucket[-1]
    img_proc = resize_center_crop(image, (target_height, target_width), (img_h, img_w))
    if return_bucket:
        return img_proc, bucket
    return img_proc
