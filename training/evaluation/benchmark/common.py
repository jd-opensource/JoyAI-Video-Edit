"""Shared data loading and cached inference for the two video-editing benchmarks."""

import argparse
import csv
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm

from src.config import DEFAULT_VIDEO_RESOLUTION, VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR, load_config_class_from_pyfile
from src.distributed.parallel_states import clean_dist_env, get_parallel_state, init_distributed_environment_and_sequence_parallel
from src.models import load_dit, load_pipeline
from src.utils import _dynamic_resize_from_bucket, _resolve_full_length_target_frames, extract_video_tensor, save_video, seed_everything


RESOLUTIONS = {480: (480, 832), 720: DEFAULT_VIDEO_RESOLUTION, 736: (736, 1280)}


@dataclass(frozen=True)
class BenchmarkItem:
    category: str
    prompt: str
    video_path: Path


def build_parser(description: str, num_frames: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Explicit inference configuration; checkpoint-side configs are not loaded.")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=tuple(RESOLUTIONS), default=720)
    parser.add_argument("--num-frames", type=int, default=num_frames, help="Maximum output frames, rounded down to 8n+1.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Defaults to the supplied config's inference setting.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Defaults to the supplied config's inference setting.")
    parser.add_argument("--source-guidance-scale", type=float, default=1.0)
    parser.add_argument("--neg-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--use-relative-rope", action="store_true")
    parser.add_argument("--max-temporal-ids", type=int, default=8, help="Temporal RoPE ceiling when relative RoPE is enabled.")
    parser.add_argument(
        "--store-clean-only-self", action=argparse.BooleanOptionalAction, default=True,
        help="Store clean history chunks with isolated self-attention (default: enabled).",
    )
    return parser


def read_metadata(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as metadata_file:
        if path.suffix.lower() == ".csv":
            yield from csv.DictReader(metadata_file)
        elif path.suffix.lower() == ".jsonl":
            for line in metadata_file:
                if line.strip():
                    yield json.loads(line)
        else:
            raise ValueError(f"Expected CSV or JSONL metadata, got {path}.")


def load_items(metadata_path: Path, dataset_root: Path, categories: str = "all", prompt_overrides=None) -> list[BenchmarkItem]:
    selected = None if categories.strip().lower() == "all" else {category.strip() for category in categories.split(",") if category.strip()}
    items = []
    skipped_prompts = 0
    for row_index, row in enumerate(read_metadata(metadata_path)):
        category = str(row.get("edited_type", row.get("segment", ""))).strip()
        if selected is not None and category not in selected:
            continue
        if not category or Path(category).name != category or category in {".", ".."}:
            raise ValueError(f"Invalid category in metadata row {row_index}: {category!r}.")
        source = str(row.get("original_video", row.get("video", ""))).strip()
        if not source:
            raise ValueError(f"Missing video path in metadata row {row_index}.")
        prompt = str(row.get("prompt", "")).strip()
        if prompt_overrides is not None:
            prompt = prompt_overrides.get(source, prompt)
        if not prompt:
            skipped_prompts += 1
            continue
        video_path = Path(source)
        if not video_path.is_absolute():
            if video_path.parts[0] == dataset_root.name:
                video_path = Path(*video_path.parts[1:])
            video_path = dataset_root / video_path
        items.append(BenchmarkItem(category, prompt, video_path))
    if skipped_prompts:
        logging.warning("Skipped %d metadata rows with empty prompts.", skipped_prompts)
    return items


def _build_uniform_frame_indices(num_frames: int, sampling_n_frame: int) -> list[int]:
    if num_frames < sampling_n_frame:
        raise ValueError(
            f"Reference video has only {num_frames} frames, but bucket requires {sampling_n_frame}."
        )

    bin_edges = torch.linspace(0, num_frames, steps=sampling_n_frame + 1, dtype=torch.float64)
    frame_idx = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        start_idx = int(math.floor(start.item()))
        end_idx = max(int(math.ceil(end.item())) - 1, start_idx)
        frame_idx.append(min((start_idx + end_idx) // 2, num_frames - 1))
    return frame_idx


def _build_deterministic_frame_indices(
    num_frames: int,
    sampling_n_frame: int,
    source_fps: float | None = None,
    target_fps: float | None = None,
) -> list[int]:
    if source_fps is None or source_fps <= 0 or target_fps is None or target_fps <= 0:
        return _build_uniform_frame_indices(num_frames, sampling_n_frame)

    max_source_index = num_frames - 1
    target_duration = (sampling_n_frame - 1) / target_fps
    source_duration = max_source_index / source_fps

    if source_duration + 1e-6 < target_duration:
        return _build_uniform_frame_indices(num_frames, sampling_n_frame)

    target_times = torch.arange(sampling_n_frame, dtype=torch.float64) / target_fps
    frame_idx = torch.round(target_times * source_fps).to(torch.int64)
    frame_idx = torch.clamp(frame_idx, min=0, max=max_source_index)
    return frame_idx.tolist()


def _get_video_fps(video_reader) -> float | None:
    try:
        fps = float(video_reader.get_avg_fps())
    except Exception:
        return None
    if math.isfinite(fps) and fps > 0:
        return fps
    return None


def _get_video_batch(video_reader: Any, frame_idx: list[int]) -> torch.Tensor:
    if hasattr(video_reader, "get_batch"):
        pixel = video_reader.get_batch(frame_idx)
    else:
        pixel = video_reader.get_frames_at(frame_idx).data

    if pixel.ndim != 4:
        raise ValueError(f"Unexpected video batch shape: {tuple(pixel.shape)}")
    if pixel.shape[1] in (1, 3):
        return pixel
    if pixel.shape[-1] in (1, 3):
        return pixel.permute(0, 3, 1, 2)
    raise ValueError(f"Unexpected video channel layout: {tuple(pixel.shape)}")


def load_reference_video(video_path: Path, cfg, resolution: int, num_frames: int):
    import decord

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if num_frames <= 0:
        raise ValueError("`num_frames` must be positive.")
    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(str(video_path))
    total_frames = len(reader)
    source_fps = _get_video_fps(reader)
    target_fps = cfg.fps if cfg.fps > 0 else None
    available_frames = _resolve_full_length_target_frames(total_frames, source_fps, target_fps)
    sampling_frames = min(num_frames, available_frames)
    sampling_frames = (sampling_frames - 1) // VAE_TEMPORAL_FACTOR * VAE_TEMPORAL_FACTOR + 1
    resize_kwargs = dict(
        bucket_configs=None,
        vid_basesizes=[RESOLUTIONS[resolution]],
        num_frames=sampling_frames,
        prioritize_frame_matching=cfg.prioritize_frame_matching,
        multiple_vides=True,
        spatial_multiple=VAE_SPATIAL_FACTOR,
    )
    first_frame = _get_video_batch(reader, [0])
    _, bucket = _dynamic_resize_from_bucket(first_frame, return_bucket=True, **resize_kwargs)
    frame_indices = _build_deterministic_frame_indices(total_frames, bucket[2], source_fps, target_fps)
    pixels = _get_video_batch(reader, frame_indices)
    pixels = _dynamic_resize_from_bucket(pixels, **resize_kwargs).clamp(0, 255).to(torch.uint8).cpu()
    normalized = pixels.to(torch.float32).div(127.5).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    return normalized, pixels, bucket


def build_pipeline_kwargs(item, args, cfg, generator):
    reference, pixels, bucket = load_reference_video(item.video_path, cfg, args.resolution, args.num_frames)
    if cfg.use_vit:
        prompt = f"<|im_start|>user\n<image>\n{item.prompt}<|im_end|>\n"
        negative_prompt = f"<|im_start|>user\n<image>\n{args.neg_prompt}<|im_end|>\n"
        images = [Image.fromarray(pixels[0].permute(1, 2, 0).numpy())]
    else:
        prompt, negative_prompt, images = item.prompt, args.neg_prompt, None
    kwargs = dict(
        prompt=[prompt],
        negative_prompt=[negative_prompt],
        images=images,
        reference_visual_content=reference.unsqueeze(1),
        height=bucket[-2],
        width=bucket[-1],
        num_frames=bucket[2],
        num_inference_steps=cfg.num_inference_steps if args.num_inference_steps is None else args.num_inference_steps,
        guidance_scale=cfg.guidance_scale if args.guidance_scale is None else args.guidance_scale,
        source_guidance_scale=args.source_guidance_scale,
        generator=generator,
        output_type="pt",
        return_dict=False,
        enable_denormalization=cfg.enable_denormalization,
        max_sequence_length=cfg.text_token_max_length,
        use_vit=cfg.use_vit,
        use_relative_temporal_ids=args.use_relative_rope,
        store_clean_only_self=args.store_clean_only_self,
    )
    if args.use_relative_rope:
        kwargs["max_temporal_ids"] = args.max_temporal_ids
    return kwargs, pixels


def run_inference(args, items: list[BenchmarkItem]) -> None:
    if args.sp_size <= 0 or args.num_frames <= 0:
        raise ValueError("Sequence parallel size and frame count must be positive.")
    if args.max_items is not None:
        if args.max_items < 0:
            raise ValueError("`max_items` must be non-negative.")
        items = items[:args.max_items]
    cfg = load_config_class_from_pyfile(args.config)()
    cfg.dit_ckpt = args.ckpt_path
    cfg.hsdp_shard_dim = 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment_and_sequence_parallel(args.sp_size)
    try:
        state = get_parallel_state()
        data_rank = state.global_rank // args.sp_size
        data_size = state.world_size // args.sp_size
        is_writer = state.global_rank % args.sp_size == 0
        seed_everything(args.seed)
        model = load_dit(cfg, device=device)
        model.config.use_inference_kv_cache = True
        model.requires_grad_(False)
        model.eval()
        pipeline = load_pipeline(cfg, model, device)
        pipeline.set_progress_bar_config(disable=True)
        assigned = items[data_rank::data_size]
        for item in tqdm(assigned, desc=f"rank {data_rank}", disable=not is_writer):
            output_dir = args.save_path / "fullset" / item.category
            result_path = output_dir / f"{item.video_path.stem}.mp4"
            skip = [result_path.exists() if is_writer else None]
            if args.sp_size > 1:
                dist.broadcast_object_list(skip, src=data_rank * args.sp_size, group=state.sp_group)
            if skip[0]:
                continue
            generator = torch.Generator(device=device).manual_seed(args.seed)
            pipeline_kwargs, source_pixels = build_pipeline_kwargs(item, args, cfg, generator)
            with torch.inference_mode():
                result = pipeline(**pipeline_kwargs)
            if is_writer:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_video(extract_video_tensor(result[0, -1]), str(result_path), fps=cfg.fps)
                save_video(extract_video_tensor(source_pixels), str(output_dir / f"{item.video_path.stem}_src.mp4"), fps=cfg.fps)
                (output_dir / f"{item.video_path.stem}_prompt.txt").write_text(item.prompt, encoding="utf-8")
        dist.barrier()
    finally:
        clean_dist_env()
