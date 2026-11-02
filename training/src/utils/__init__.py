import random
import glob
import re
from pathlib import Path
from typing import Any


import numpy as np
from einops import rearrange
from PIL import Image
import torch
import torchvision.io
from src.config import VAE_TEMPORAL_FACTOR


def seed_everything(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_video_tensor(result: Any) -> torch.Tensor:
    if hasattr(result, "frames"):
        result = result.frames
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        if len(result) == 0:
            raise ValueError("Pipeline returned an empty video result.")
        result = result[0]
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError("Pipeline returned an empty nested video result.")
            result = result[0]
    if torch.is_tensor(result):
        if result.dim() == 6:
            result = result[0, 0]
        if result.dim() == 5:
            result = result[0]
        if result.shape[-1] == 3:
            result = result.permute(0, 3, 1, 2)
        if result.dtype != torch.uint8:
            if result.max() <= 1.0:
                result = (result * 255).clamp(0, 255)
            result = result.to(torch.uint8)
        return result.cpu()
    raise TypeError(f"Unsupported video output type: {type(result)}")


def save_video(tensors: torch.Tensor | list[torch.Tensor], save_path: str, fps: int = 30) -> None:
    if not isinstance(tensors, list):
        tensors = [tensors]

    processed_tensors = []
    for tensor in tensors:
        if tensor.dtype != torch.uint8:
            raise ValueError("Input Tensor dtype must be uint8")
        if tensor.dim() != 4:
            raise ValueError(
                f"Input Tensor must be 4-dimensional (t, c, h, w), but got {tensor.dim()}")

        processed_tensors.append(tensor)

    final_tensor = torch.cat(processed_tensors, dim=3)

    if final_tensor.shape[0] == 1:
        image_tensor = rearrange(final_tensor, "1 c h w -> h w c")
        img = Image.fromarray(image_tensor.cpu().numpy())
        img.save(save_path)
    else:
        video_tensor = rearrange(final_tensor, "t c h w -> t h w c")
        options={
            "crf": "8",
        }
        torchvision.io.write_video(save_path, video_tensor, fps=fps, options=options)


def get_evenly_divisible_files(
    path_input: str | list[str], world_size: int, logger=None, is_skip_glob=False
) -> list[str]:
    """Trim shard lists for equal distribution across ranks."""
    if world_size <= 0:
        raise ValueError("world_size must be greater than 0")
    if is_skip_glob:
        unique_files = path_input
    else:
        patterns = [path_input] if isinstance(path_input, str) else path_input

        all_files = []
        for pattern in patterns:
            recursive = "**" in pattern
            all_files.extend(glob.glob(pattern, recursive=recursive))
        unique_files = sorted(set(all_files))

    num_files = len(unique_files)
    num_to_keep = num_files - (num_files % world_size)

    if logger is not None:
        logger.info(
            f"Input files: {num_files}, keeping: {num_to_keep} files, skipping {num_files - num_to_keep} files")

    return unique_files[:num_to_keep]


def get_from_dict(data: dict, path: str):
    """Look up a dot-separated key path."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary, got {type(data).__name__}")

    keys = path.split(".")
    current_level = data

    for key in keys:
        current_level = current_level[key]

    return current_level


def find_latest_checkpoint(ckpt_dir: Path,  pattern: str = r"global_step(\d+)") -> Path | None:
    if not ckpt_dir.exists():
        return None

    step_pattern = re.compile(pattern)
    latest_step = -1
    latest_ckpt = None

    for subdir in ckpt_dir.iterdir():
        if subdir.is_dir():
            match = step_pattern.match(subdir.name)
            if match:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    latest_ckpt = subdir

    return latest_ckpt.resolve() if latest_ckpt else None


def _resolve_full_length_target_frames(
    total_frames: int,
    source_fps: float | None,
    target_fps: float | None,
) -> int:
    """Resample the source duration and floor to a valid 8n+1 frame count."""
    if total_frames <= 0:
        raise ValueError("A source video must contain at least one frame.")
    if source_fps and target_fps and source_fps > 0 and target_fps > 0:
        resampled = int(round((total_frames - 1) * target_fps / source_fps)) + 1
    else:
        resampled = total_frames
    return max(1, ((resampled - 1) // VAE_TEMPORAL_FACTOR) * VAE_TEMPORAL_FACTOR + 1)
