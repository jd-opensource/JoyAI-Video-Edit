import math

from src.config import DEFAULT_REFERENCE_IMAGE_BASE_SIZE, VAE_SPATIAL_FACTOR, validate_bucket_configs


class BucketGroup:
    """Select the longest usable clip, then the closest spatial aspect ratio."""

    def __init__(self, bucket_configs: list[tuple[int, int, int, int, int]]):
        validate_bucket_configs(bucket_configs)
        self.bucket_configs = [tuple(bucket) for bucket in bucket_configs]

    def find_best_bucket(self, media_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
        if len(media_shape) != 4 or any(type(value) is not int or value <= 0 for value in media_shape):
            raise ValueError(f"Media shape must contain four positive integers, got {media_shape}.")
        num_items, num_frames, height, width = media_shape
        valid = [
            bucket for bucket in self.bucket_configs
            if bucket[1] == num_items
            and ((num_frames == 1 and bucket[2] == 1) or (num_frames > 1 and 1 < bucket[2] <= num_frames))
        ]
        if not valid:
            raise ValueError(f"No compatible size bucket for shape {media_shape}.")
        selected_frames = max(bucket[2] for bucket in valid)
        return min(
            (bucket for bucket in valid if bucket[2] == selected_frames),
            key=lambda bucket: abs(bucket[3] / bucket[4] - height / width),
        )

    def __repr__(self) -> str:
        return f"BucketGroup(configs={self.bucket_configs})"


def reference_size(height: int, width: int) -> tuple[int, int]:
    """Choose a reference bucket with a 768²-pixel budget on a 24-pixel grid."""
    if height <= 0 or width <= 0:
        raise ValueError("Reference dimensions must be positive.")
    max_area = DEFAULT_REFERENCE_IMAGE_BASE_SIZE ** 2
    multiple = VAE_SPATIAL_FACTOR
    choices = []
    h, w = max_area // multiple, multiple
    while h >= multiple:
        if max(h, w) / min(h, w) <= 4:
            choices.append((h, w))
        if h * (w + multiple) <= max_area:
            w += multiple
        else:
            h -= multiple
    return min(choices, key=lambda hw: abs(hw[0] / hw[1] - height / width))


def resize_pixels(pixels, height: int, width: int):
    """Cover the target rectangle, then apply the same center crop to every frame."""
    from torchvision.transforms import functional as TF

    h, w = pixels.shape[-2:]
    scale = max(height / h, width / w)
    pixels = TF.resize(pixels, (math.ceil(h * scale), math.ceil(w * scale)),
                       interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return TF.center_crop(pixels, (height, width))
