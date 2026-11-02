import random
import re
from pathlib import Path


import numpy as np
from einops import rearrange
import torch
import torchvision.io
from src.config import VAE_TEMPORAL_FACTOR


def seed_everything(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_video(tensor: torch.Tensor, save_path: str, fps: int = 30) -> None:
    """Save (T, 3, H, W) pixels as video, including one-frame clips."""
    if tensor.ndim != 4 or tensor.shape[1] != 3 or min(tensor.shape) <= 0:
        raise ValueError(f"Video must have shape (T, 3, H, W), got {tuple(tensor.shape)}.")
    if fps <= 0:
        raise ValueError("Video FPS must be positive.")
    if tensor.is_floating_point():
        tensor = tensor.mul(255).clamp(0, 255).to(torch.uint8)
    elif tensor.dtype != torch.uint8:
        raise ValueError("Video pixels must be uint8 or floating-point values in [0, 1].")
    video_tensor = rearrange(tensor.cpu(), "t c h w -> t h w c")
    torchvision.io.write_video(save_path, video_tensor, fps=fps, options={"crf": "8"})


def get_from_dict(data: dict, path: str):
    """Look up a dot-separated key path."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary, got {type(data).__name__}")

    keys = path.split(".")
    current_level = data

    for key in keys:
        current_level = current_level[key]

    return current_level


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None

    step_pattern = re.compile(r"global_step(\d+)")
    latest_step = -1
    latest_ckpt = None

    for subdir in ckpt_dir.iterdir():
        if subdir.is_dir():
            match = step_pattern.fullmatch(subdir.name)
            if match:
                step = int(match.group(1))
                checkpoint = subdir / f"step_{step}.pth"
                if step > latest_step and checkpoint.is_file() and checkpoint.stat().st_size > 0:
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
