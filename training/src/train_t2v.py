"""Teacher forcing and resampling forcing for video-editing SFT."""

import os
import time
import gc
import math
from pathlib import Path
from collections import deque
from datetime import datetime
import random
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from diffusers import get_scheduler

from src.dataset.webdataset import build_webdataset_dataloader
from src.models import (
    load_dit, load_pipeline
)
from src.optimizers import load_optimizer
from src.distributed.parallel_states import (
    init_distributed_environment_and_sequence_parallel,
    get_parallel_state, sp_enabled, clean_dist_env
)
from src.distributed.communications import broadcast_within_sp_group, broadcast_item
from src.config import ExpConfig, load_config_class_from_pyfile
from src.utils import (
    seed_everything,
    find_latest_checkpoint,
)
from src.utils.logging import setup_logger
from src.utils.activation_checkpointing import apply_activation_checkpointing
from src.utils.checkpoint import load_checkpoint, save_checkpoint


SUPPORTED_FORCING_STRATEGIES = ("teacher_forcing", "resampling_forcing")
LOSS_TASK_TYPES = ("t2i", "i2i", "t2v", "v2v")


def _validate_forcing_strategy(cfg: ExpConfig) -> str:
    strategy = getattr(cfg, "causal_forcing_strategy", None)
    if strategy not in SUPPORTED_FORCING_STRATEGIES:
        raise ValueError(
            f"Unsupported causal_forcing_strategy={strategy!r}; "
            f"stage-1 training supports only {SUPPORTED_FORCING_STRATEGIES}."
        )
    return strategy


def build_multiple_images_cfg_prompt(num_condition_images: int) -> str:
    image_tokens = "<image>\n" * num_condition_images
    return f"<|im_start|>user\n{image_tokens}<|im_end|>\n"


def infer_task_type(num_items: int, num_frames: int) -> str:
    if num_frames == 1:
        return "t2i" if num_items == 1 else "i2i"
    return "t2v" if num_items == 1 else "v2v"


def build_multi_item_prompt_images(pixel_values: torch.Tensor) -> torch.Tensor:
    prompt_images = (pixel_values[:, :-1, :, 0] + 1) * 127.5
    prompt_images = rearrange(prompt_images, "b n c h w -> (b n) c h w")
    return prompt_images


def _sample_sigmas_scalar(
    shape: tuple,
    latents: torch.Tensor,
    cfg: ExpConfig,
    noise_generator_cpu: torch.Generator,
) -> torch.Tensor:
    if cfg.weighting_scheme == "lognorm":
        assert cfg.train_flow_shift is not None, "flow_shift must be specified for lognorm weighting scheme."
        sigmas = torch.normal(
            mean=0.0, std=1.0,
            size=shape,
            device="cpu",
            generator=noise_generator_cpu
        )
        sigmas = torch.nn.functional.sigmoid(sigmas)
        if cfg.enable_denormalization:
            sigmas = (cfg.train_flow_shift * sigmas) / \
                (1 + (cfg.train_flow_shift - 1) * sigmas)
        else:
            train_flow_rescale = max(math.sqrt(math.prod(latents.shape[-3:])/getattr(cfg, 'train_flow_base')), 1)
            sigmas = (cfg.train_flow_shift * train_flow_rescale * sigmas) / \
                (1 + (cfg.train_flow_shift * train_flow_rescale - 1) * sigmas)
    else:
        raise NotImplementedError(
            f"Unsupported weighting scheme: {cfg.weighting_scheme}")
    return sigmas


def sample_noise_and_timestep(
    latents: torch.Tensor,
    cfg: ExpConfig,
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
    model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn(
        latents.shape,
        generator=noise_generator_cuda,
        dtype=latents.dtype,
        device=latents.device
    )

    use_chunkwise = (
        model is not None
        and getattr(cfg, "causal_forcing_strategy", None) in SUPPORTED_FORCING_STRATEGIES
        and getattr(model.config, "causal", False)
    )

    if use_chunkwise:
        chunk_size = getattr(model.config, "chunk_size", None)
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(
                f"`chunk_size` must be positive when causal forcing is enabled, got {chunk_size}."
            )
        total_latent_frames = latents.shape[-3]
        num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size
        chunk_sigmas = _sample_sigmas_scalar(
            (latents.shape[0], num_chunks), latents, cfg, noise_generator_cpu
        )
        sigmas = chunk_sigmas.to(device=latents.device)
        if sp_enabled():
            broadcast_within_sp_group(sigmas)
        chunk_ids = torch.div(
            torch.arange(total_latent_frames, device=latents.device),
            chunk_size,
            rounding_mode="floor",
        )
        sigmas = sigmas[:, chunk_ids]
    else:
        sigmas = _sample_sigmas_scalar(
            (latents.shape[0],), latents, cfg, noise_generator_cpu
        )
        sigmas = sigmas.to(device=latents.device)
        if sp_enabled():
            broadcast_within_sp_group(sigmas)

    timesteps = (sigmas * 1000)

    return noise, timesteps, sigmas


def _build_teacher_forcing_history(
    target_clean: torch.Tensor,
    cfg: ExpConfig,
    model: torch.nn.Module,
    noise_generator_cpu: torch.Generator,
    noise_generator_cuda: torch.Generator,
) -> torch.Tensor:
    """Ground-truth history with the configured per-chunk Gaussian augmentation."""
    max_sigma = float(getattr(cfg, "teacher_forcing_noise_max_sigma", 0.0))
    if max_sigma <= 0:
        return target_clean
    min_sigma = float(getattr(cfg, "teacher_forcing_noise_min_sigma", 0.0))
    min_sigma = max(0.0, min(min_sigma, max_sigma))

    batch_size, _, total_latent_frames, _, _ = target_clean.shape
    chunk_size = int(getattr(model.config, "chunk_size", 1) or 1)
    num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size

    sig_chunk = torch.rand(
        (batch_size, num_chunks), generator=noise_generator_cpu,
    ).to(device=target_clean.device, dtype=target_clean.dtype)
    sig_chunk = min_sigma + sig_chunk * (max_sigma - min_sigma)
    if sp_enabled():
        broadcast_within_sp_group(sig_chunk)

    chunk_ids = torch.div(
        torch.arange(total_latent_frames, device=target_clean.device),
        chunk_size,
        rounding_mode="floor",
    )
    sigma = sig_chunk[:, chunk_ids].view(batch_size, 1, total_latent_frames, 1, 1)

    aug_noise = torch.randn(
        target_clean.shape,
        generator=noise_generator_cuda,
        device=target_clean.device,
        dtype=target_clean.dtype,
    )
    return (1 - sigma) * target_clean + sigma * aug_noise


def _sample_resampling_forcing_sigmas(
    batch_size: int,
    num_chunks: int,
    cfg: ExpConfig,
    device: torch.device,
    dtype: torch.dtype,
    noise_generator_cpu: torch.Generator,
) -> torch.Tensor:
    """Draw shifted logit-normal noise levels independently for each history chunk."""
    shift = float(getattr(cfg, "resampling_forcing_shift_s", 0.6))
    normal = torch.normal(
        mean=0.0, std=1.0, size=(batch_size, num_chunks),
        device="cpu", generator=noise_generator_cpu,
    )
    sigmas = torch.sigmoid(normal)
    if shift != 1.0:
        sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)
    sigmas = sigmas.to(device=device, dtype=dtype)
    if sp_enabled():
        broadcast_within_sp_group(sigmas)
    return sigmas


@torch.no_grad()
def _resampling_forcing_rollout(
    clean_target: torch.Tensor,
    cfg: ExpConfig,
    model: torch.nn.Module,
    pipeline: Any,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    ref_video_latent: torch.Tensor | None,
    ref_image_latent: Any,
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
) -> torch.Tensor:
    """Generate detached history from noisy ground-truth chunks using one Euler step per chunk."""
    device, dtype = clean_target.device, clean_target.dtype
    batch_size, _, total_frames, _, _ = clean_target.shape
    chunk_size = int(getattr(model.config, "chunk_size", 0) or 0)
    if not getattr(model.config, "causal", False) or chunk_size <= 0:
        raise ValueError("Resampling forcing requires a causal DiT with chunk_size > 0.")
    chunk_windows = pipeline._get_chunk_windows(
        total_latent_frames=total_frames,
        chunk_size=chunk_size,
        window_size=int(getattr(model.config, "local_window_size", 1)),
        global_sink_chunk=bool(getattr(model.config, "global_sink_chunk", False)),
    )
    chunk_sigmas = _sample_resampling_forcing_sigmas(
        batch_size, len(chunk_windows), cfg, device, dtype, noise_generator_cpu,
    )
    noise = torch.randn(clean_target.shape, generator=noise_generator_cuda, device=device, dtype=dtype)
    model_dtype = model.dtype
    prompt_embeds = prompt_embeds.to(dtype=model_dtype)
    if isinstance(ref_image_latent, list):
        ref_image_latent = [latent.to(dtype=model_dtype) if latent is not None else None for latent in ref_image_latent]
    elif ref_image_latent is not None:
        ref_image_latent = ref_image_latent.to(dtype=model_dtype)
    generated_history = clean_target.clone()

    patch_size = model.config.patch_size
    spatial_tokens = (clean_target.shape[-2] // patch_size[1]) * (clean_target.shape[-1] // patch_size[2])
    for window in chunk_windows:
        chunk_index = window["chunk_idx"]
        chunk_start, chunk_end = window["chunk_start"], window["chunk_end"]
        selected_ids = window["selected_chunk_ids"]
        active_frames = chunk_end - chunk_start
        temporal_ids = pipeline._gather_window_temporal_ids(
            selected_ids, chunk_size, total_frames, device, relative=False,
        ).unsqueeze(0).expand(batch_size, -1)
        sigma = chunk_sigmas[:, chunk_index]
        sigma_view = sigma.view(batch_size, 1, 1, 1, 1)
        target_chunk = clean_target[:, :, chunk_start:chunk_end]
        noisy_chunk = (1.0 - sigma_view) * target_chunk + sigma_view * noise[:, :, chunk_start:chunk_end]
        history = pipeline._gather_window_tensor(
            generated_history, selected_ids, chunk_size, total_frames, temporal_dim=2,
        ).to(dtype=model_dtype)
        hidden_states = history.clone()
        hidden_states[:, :, -active_frames:] = noisy_chunk.to(dtype=model_dtype)
        frame_timesteps = sigma.new_zeros((batch_size, history.shape[2]))
        frame_timesteps[:, -active_frames:] = sigma[:, None] * 1000.0
        token_timesteps = frame_timesteps[:, ::patch_size[0]].repeat_interleave(spatial_tokens, dim=1)
        model_kwargs = dict(
            hidden_states=hidden_states,
            clean_video_latent=history,
            timestep=token_timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            noisy_temporal_ids=temporal_ids,
            ref_image_latent=ref_image_latent,
            return_dict=False,
        )
        if ref_video_latent is not None:
            model_kwargs["ref_video_latent"] = pipeline._gather_window_tensor(
                ref_video_latent, selected_ids, chunk_size, total_frames, temporal_dim=2,
            ).to(dtype=model_dtype)
        with model.cache_context("cond"):
            prediction = model(**model_kwargs)[0][:, :, -active_frames:]
        clean_chunk = (noisy_chunk - sigma_view * prediction.to(dtype=dtype)).detach()
        generated_history[:, :, chunk_start:chunk_end] = clean_chunk

    generated_history = generated_history.detach()
    if sp_enabled():
        broadcast_within_sp_group(generated_history)
    return generated_history


def prepare_forcing_inputs(
    cfg: ExpConfig,
    model: torch.nn.Module,
    pipeline: Any,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    additional_inputs: dict[str, Any],
    *,
    step: int,
    rng: random.Random,
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build target-only flow matching inputs and the selected causal history."""
    strategy = _validate_forcing_strategy(cfg)
    noise, timesteps, sigmas = sample_noise_and_timestep(
        latents, cfg, noise_generator_cuda, noise_generator_cpu, model=model,
    )
    target_clean, target_noise = latents[:, -1], noise[:, -1]
    if sigmas.ndim == 1:
        sigma_view = sigmas.view(-1, 1, 1, 1, 1)
    elif sigmas.ndim == 2:
        sigma_view = sigmas[:, None, :, None, None]
    else:
        raise ValueError(f"Unsupported sigma shape: {tuple(sigmas.shape)}")
    noisy_target = (1 - sigma_view) * target_clean + sigma_view * target_noise
    target = target_noise - target_clean
    forcing_inputs = dict(additional_inputs)
    if latents.shape[1] > 1:
        forcing_inputs["ref_video_latent"] = rearrange(latents[:, :-1], "b n c t h w -> b c (n t) h w")
        reference_drop_rate = float(getattr(cfg, "ref_cfg_rate", 0.0))
        if reference_drop_rate > 0.0 and rng.random() < reference_drop_rate:
            forcing_inputs["ref_video_latent"] = None
    else:
        forcing_inputs["ref_video_latent"] = None
    forcing_inputs["ref_image_latent"] = additional_inputs.get("ref_image_latents")
    if getattr(model.config, "causal", False) and additional_inputs["task_type"] in ("t2v", "v2v"):
        warmup_steps = int(getattr(cfg, "resampling_forcing_warmup_steps", 0))
        if strategy == "teacher_forcing" or step <= warmup_steps:
            forcing_inputs["clean_video_latent"] = _build_teacher_forcing_history(
                target_clean, cfg, model, noise_generator_cpu, noise_generator_cuda,
            )
        else:
            forcing_inputs["clean_video_latent"] = _resampling_forcing_rollout(
                target_clean, cfg, model, pipeline, prompt_embeds, prompt_embeds_mask,
                forcing_inputs["ref_video_latent"], forcing_inputs["ref_image_latent"],
                noise_generator_cuda, noise_generator_cpu,
            )
    return noisy_target, timesteps, target, forcing_inputs


@torch.no_grad()
def prepare_inputs(
    pixel_values: torch.Tensor,
    caption: list[str],
    pipeline: Any,
    cfg: ExpConfig,
    rng: random.Random,
    generator: torch.Generator = None,
    ref_image_pixels: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    device = pipeline.vae.device
    dtype = pipeline.vae.dtype
    batch_size, num_items, _, num_frames, _, _ = pixel_values.shape

    pixel_values = pixel_values.to(dtype=dtype, device=device)
    pixel_values_fused = rearrange(
        pixel_values, "b n c t h w -> (b n) c t h w")

    is_causal = getattr(pipeline.transformer.config, "causal", False)
    dit_chunk_size = getattr(pipeline.transformer.config, "chunk_size", None)
    total_pixel_frames = pixel_values_fused.shape[2]
    if (total_pixel_frames - 1) % pipeline.vae.ffactor_temporal:
        raise ValueError(f"Training videos must contain {pipeline.vae.ffactor_temporal}n+1 frames, got {total_pixel_frames}.")

    use_chunkwise_encode = (
        is_causal
        and dit_chunk_size is not None
        and dit_chunk_size > 0
        and total_pixel_frames > 1
    )

    if use_chunkwise_encode:
        ffactor_t = pipeline.vae.ffactor_temporal
        window_pixels = dit_chunk_size * ffactor_t
        window_frames = 1 + window_pixels
        stride = ffactor_t
        T = total_pixel_frames
        num_latents = (T - 1) // stride + 1

        lat_list = []
        for k in range(num_latents):
            if k == 0:
                window = pixel_values_fused[:, :, :1]
            else:
                end_frame = k * stride
                start_frame = max(0, end_frame - window_pixels)
                window = pixel_values_fused[:, :, start_frame:end_frame + 1]
                pad_needed = window_frames - window.shape[2]
                if pad_needed > 0:
                    pad = pixel_values_fused[:, :, :1].expand(-1, -1, pad_needed, -1, -1)
                    window = torch.cat([pad, window], dim=2)

            if cfg.enable_denormalization:
                h = pipeline.vae.encode(window).latent_dist.sample(generator)
                h = pipeline.normalize_latents(h)
            else:
                h = pipeline.vae.encode(window)
            lat_list.append(h[:, :, -1:])

        latents = torch.cat(lat_list, dim=2)
    else:
        if cfg.enable_denormalization:
            latents = pipeline.vae.encode(
                pixel_values_fused).latent_dist.sample(generator)
            latents = pipeline.normalize_latents(latents)
        else:
            latents = pipeline.vae.encode(pixel_values_fused)

    latents = rearrange(latents, "(b n) c t h w -> b n c t h w", b=batch_size)

    task_type = infer_task_type(num_items=num_items, num_frames=num_frames)
    is_image = task_type == "t2i"
    is_video = task_type == "t2v"
    is_multi_item = task_type in {"i2i", "v2v"}

    reference_latents = None
    ref_image_latents = None

    if is_multi_item:
        loss_mask = torch.ones_like(latents[:, -1])
        reference_latents = latents[:, :-1]
        if ref_image_pixels is not None and any(p is not None for p in ref_image_pixels):
            ref_image_latents = []
            for idx, pixel in enumerate(ref_image_pixels):
                if pixel is None:
                    ref_image_latents.append(None)
                    continue
                if pixel.dim() != 4:
                    raise ValueError(
                        f"Expected ref_image_pixel shape [c, t, h, w], got {tuple(pixel.shape)}."
                    )
                ref_img = pixel.unsqueeze(0).to(device=device, dtype=dtype)
                if cfg.enable_denormalization:
                    ref_lat = pipeline.vae.encode(ref_img).latent_dist.sample()
                    ref_lat = pipeline.normalize_latents(ref_lat)
                else:
                    ref_lat = pipeline.vae.encode(ref_img)
                ref_image_latents.append(ref_lat[:, :, :1])
    else:
        loss_mask = torch.ones_like(latents[:, 0] if latents.ndim == 6 else latents)

    if is_image or is_video:
        template_type = 'image' if is_image else 'video'
        caption = ["" if rng.random() < cfg.cfg_rate else c for c in caption]
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=caption,
            device=device,
            template_type=template_type,
            max_sequence_length=cfg.text_token_max_length,
        )
    elif is_multi_item:
        if cfg.use_vit and task_type in ('i2i', 'v2v'):
            cfg_prompt = build_multiple_images_cfg_prompt(1)
            caption = [
                cfg_prompt if rng.random() < cfg.cfg_rate
                else f"<|im_start|>user\n<image>\n{c}<|im_end|>\n"
                for c in caption
            ]

            if task_type == 'v2v':
                prompt_images = (pixel_values[:, 0, :, 0] + 1) * 127.5
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt_multiple_images(
                    prompt=caption,
                    images=prompt_images,
                    device=device,
                    max_sequence_length=cfg.text_token_max_length,
                )
            else:
                prompt_images = build_multi_item_prompt_images(pixel_values)
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt_multiple_images(
                    prompt=caption,
                    images=prompt_images,
                    device=device,
                    max_sequence_length=cfg.text_token_max_length,
                )
        else:
            template_type = 'image' if task_type == 'i2i' else 'video'
            caption = ["" if rng.random() < cfg.cfg_rate else c for c in caption]
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=caption,
                device=device,
                template_type=template_type,
                max_sequence_length=cfg.text_token_max_length,
            )
    else:
        raise ValueError("Invalid input data format.")

    additional_inputs = {
        "loss_mask": loss_mask,
        "task_type": task_type,
        "reference_latents": reference_latents,
        "ref_image_latents": ref_image_latents,
    }
    return latents, prompt_embeds, prompt_embeds_mask, additional_inputs


def forward_and_compute_loss(
    cfg: ExpConfig,
    model: torch.nn.Module,
    noisy_model_input: torch.Tensor,
    prompt_embeds,
    prompt_embeds_mask,
    timesteps: torch.Tensor,
    target: torch.Tensor,
    additional_inputs: dict[str, torch.Tensor | str],
) -> torch.Tensor:
    """Masked target flow-matching loss with ground-truth or resampled history."""
    noisy_model_input = noisy_model_input.to(
        dtype=model.dtype, device=model.device)

    clean_video_latent = additional_inputs.get("clean_video_latent")
    if clean_video_latent is not None:
        clean_video_latent = clean_video_latent.to(dtype=model.dtype, device=model.device)
    ref_video_latent = additional_inputs.get("ref_video_latent")
    if ref_video_latent is not None:
        ref_video_latent = ref_video_latent.to(dtype=model.dtype, device=model.device)
    ref_image_latent = additional_inputs.get("ref_image_latent")
    if ref_image_latent is not None:
        if isinstance(ref_image_latent, list):
            ref_image_latent = [
                latent.to(dtype=model.dtype, device=model.device) if latent is not None else None
                for latent in ref_image_latent
            ]
        else:
            ref_image_latent = ref_image_latent.to(dtype=model.dtype, device=model.device)

    model_kwargs = dict(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        return_dict=False,
    )

    if timesteps.ndim == 2:
        patch_size = getattr(model.config, "patch_size", [1, 2, 2])
        _, _, t_latent, h_latent, w_latent = noisy_model_input.shape
        t_patch = t_latent // patch_size[0]
        h_patch = h_latent // patch_size[1]
        w_patch = w_latent // patch_size[2]
        spatial_tokens = h_patch * w_patch
        sampled_ts = timesteps[:, ::patch_size[0]]
        token_timesteps = sampled_ts.unsqueeze(-1).expand(-1, -1, spatial_tokens).reshape(
            timesteps.shape[0], t_patch * spatial_tokens
        )
        model_kwargs["timestep"] = token_timesteps
    if clean_video_latent is not None:
        model_kwargs["clean_video_latent"] = clean_video_latent
    if ref_video_latent is not None:
        model_kwargs["ref_video_latent"] = ref_video_latent
    if ref_image_latent is not None:
        model_kwargs["ref_image_latent"] = ref_image_latent

    model_pred = model(**model_kwargs)

    loss_mask = additional_inputs['loss_mask']
    loss = F.mse_loss(model_pred[0].float(), target.float(),
                      reduction="none")
    loss = loss * loss_mask
    return loss.sum() / loss_mask.sum().clamp_min(1)


def setup_distributed_training(cfg: ExpConfig) -> tuple[int, int, int, torch.device]:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment_and_sequence_parallel(sp_size=cfg.sp_size)

    return local_rank, global_rank, world_size, device


def setup_model_and_optimizer(cfg: ExpConfig, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
    dit = load_dit(cfg, device=device)
    dit.requires_grad_(True)
    dit.train()
    if cfg.enable_activation_checkpointing:
        dit = apply_activation_checkpointing(
            dit,
            skip_interval=cfg.activation_checkpointing_skip_interval,
        )

    optimizer = load_optimizer(cfg.optimizer_name, dit, cfg)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
        last_epoch=-1,
    )

    return dit, optimizer, lr_scheduler


def main(cfg: ExpConfig):
    _validate_forcing_strategy(cfg)

    _local_rank, global_rank, world_size, device = setup_distributed_training(cfg)

    assert cfg.seed is not None, "Seed must be specified in the configuration."
    seed_everything(cfg.seed)

    assert cfg.output_dir is not None, "Output directory must be specified in the configuration."
    exp_dir = log_dir = ckpt_dir = None

    if global_rank <= 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        exp_dir = Path(cfg.output_dir) / \
            f"{cfg.exp_name}_sp{cfg.sp_size}_world{world_size}"
        log_dir = exp_dir / "logs"
        ckpt_dir = exp_dir / "checkpoints"

        for directory in [exp_dir, log_dir, ckpt_dir]:
            os.makedirs(directory, exist_ok=True)

    dist.barrier()
    exp_dir = broadcast_item(exp_dir, src=0)
    log_dir = broadcast_item(log_dir, src=0)
    ckpt_dir = broadcast_item(ckpt_dir, src=0)

    logger = setup_logger(log_dir)
    tb_writer = None
    if global_rank <= 0:
        tb_writer = SummaryWriter(log_dir=log_dir)
        with open(os.path.join(exp_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            f.write(cfg.to_json_string())

    train_dataloader = build_webdataset_dataloader(
        image_data_files=cfg.train_image_data_files,
        multiple_images_data_files=cfg.train_multiple_images_data_files,
        multiple_videos_data_files=cfg.train_multiple_videos_data_files,
        video_data_files=cfg.train_video_data_files,
        bucket_configs=cfg.bucket_configs,
        bucket_configs_options=cfg.bucket_configs_options,
        bucket_configs_options_prob=cfg.bucket_configs_options_prob,
        prioritize_frame_matching=cfg.prioritize_frame_matching,
        image_caption_keys=cfg.train_image_caption_keys,
        image_caption_sampling_prob=cfg.train_image_caption_sampling_prob,
        multiple_images_caption_keys=cfg.train_multiple_images_caption_keys,
        multiple_images_caption_sampling_prob=cfg.train_multiple_images_caption_sampling_prob,
        multiple_videos_caption_keys=cfg.train_multiple_videos_caption_keys,
        multiple_videos_caption_sampling_prob=cfg.train_multiple_videos_caption_sampling_prob,
        video_caption_keys=cfg.train_video_caption_keys,
        video_caption_sampling_prob=cfg.train_video_caption_sampling_prob,
        image_sampling_prob=cfg.image_sampling_prob,
        multiple_images_sampling_prob=cfg.multiple_images_sampling_prob,
        multiple_videos_sampling_prob=cfg.multiple_videos_sampling_prob,
        video_sampling_prob=cfg.video_sampling_prob,
        ensure_divisible_shards=cfg.ensure_divisible_shards,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        fps=cfg.fps,
        seed=cfg.seed,
        tar_files_shuffle_seed=cfg.tar_files_shuffle_seed,
        rec_aug_rate=cfg.rec_aug_rate,
        ref_image_basesize=cfg.ref_image_basesize,
        ref_image_bucket_configs=cfg.ref_image_bucket_configs,
    )

    dit, optimizer, lr_scheduler = setup_model_and_optimizer(cfg, device)

    init_step, epoch = 0, 0
    if cfg.auto_resume:
        latest_ckpt_dir = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt_dir is not None:
            cfg.resume_from_checkpoint = str(latest_ckpt_dir)
            if not cfg.resume_optimizer:
                logger.warning(
                    "auto resume_optimizer was not True — automatically setting to True.")
                cfg.resume_optimizer = True
            if not cfg.resume_dataloader:
                logger.warning(
                    "auto resume_dataloader was not True — automatically setting to True.")
                cfg.resume_dataloader = True
            logger.info(
                f"Auto-resume: found latest checkpoint folder {cfg.resume_from_checkpoint}, "
                f"resume_optimizer={cfg.resume_optimizer}, resume_dataloader={cfg.resume_dataloader}"
            )
        else:
            logger.info("No checkpoint folder found.")
    if cfg.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        init_step, epoch = load_checkpoint(
            model=dit,
            path=cfg.resume_from_checkpoint,
            device=device,
            optimizer=optimizer if cfg.resume_optimizer else None,
            scheduler=lr_scheduler if cfg.resume_optimizer else None,
            dataloader=train_dataloader if cfg.resume_dataloader else None,
        )

    pipeline = load_pipeline(cfg, dit, device)

    total_batch_size = cfg.micro_batch_size * \
        (world_size / cfg.sp_size) * cfg.gradient_accumulation_steps
    total_params = sum(p.numel()
                       for p in dit.parameters() if p.requires_grad) / 1e9
    logger.info(
        f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    logger.info(f"Micro batch size: {cfg.micro_batch_size}")
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Total training steps: {cfg.max_train_steps}")
    logger.info(f"Total trainable parameters = {total_params:.2f} B")
    logger.info(
        f"Enable activation checkpointing: {cfg.enable_activation_checkpointing}")

    noise_seed = cfg.seed + \
        (get_parallel_state().sp_group_id if sp_enabled() else global_rank)
    noise_generator_cpu = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise_generator_cuda = torch.Generator(
        device="cuda").manual_seed(noise_seed)
    rng = random.Random(noise_seed)

    step_times: deque[float] = deque(maxlen=100)
    data_times: deque[float] = deque(maxlen=100)
    preprocess_times: deque[float] = deque(maxlen=100)

    gc.disable()

    train_data_iterator = iter(train_dataloader)
    for step in range(init_step + 1, cfg.max_train_steps + 1):
        record_loss = {'Loss/train_loss': 0.0}
        log_step = step % cfg.log_interval == 0
        task_loss_stats = torch.zeros((len(LOSS_TASK_TYPES), 2), device=device) if log_step else None
        optimizer.zero_grad()
        step_start_time = time.perf_counter()
        for accumulation_step in range(cfg.gradient_accumulation_steps):
            data_start_time = time.perf_counter()
            batch = next(train_data_iterator)
            data_time = time.perf_counter() - data_start_time
            data_times.append(data_time)

            preprocess_start_time = time.perf_counter()
            latents, prompt_embeds, prompt_embeds_mask, additional_inputs = prepare_inputs(
                pixel_values=batch["pixel"],
                caption=batch["caption"],
                pipeline=pipeline,
                cfg=cfg,
                rng=rng,
                generator=noise_generator_cuda,
                ref_image_pixels=batch.get("ref_image_pixel"),
            )
            preprocess_time = time.perf_counter() - preprocess_start_time
            preprocess_times.append(preprocess_time)

            noisy_model_input, timesteps, target, additional_inputs = prepare_forcing_inputs(
                cfg, dit, pipeline, latents, prompt_embeds, prompt_embeds_mask, additional_inputs,
                step=step,
                rng=rng,
                noise_generator_cuda=noise_generator_cuda,
                noise_generator_cpu=noise_generator_cpu,
            )
            loss = forward_and_compute_loss(
                cfg,
                model=dit,
                noisy_model_input=noisy_model_input,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                timesteps=timesteps,
                target=target,
                additional_inputs=additional_inputs,
            )
            if task_loss_stats is not None:
                task_id = LOSS_TASK_TYPES.index(additional_inputs["task_type"])
                task_loss_stats[task_id, 0] += loss.detach()
                task_loss_stats[task_id, 1] += 1
            loss = loss / cfg.gradient_accumulation_steps
            assert not torch.isnan(loss).any(
            ), f"Loss contains NaN values: {loss}"
            assert not torch.isinf(loss).any(
            ), f"Loss contains Inf values: {loss}"
            loss.backward()

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            record_loss['Loss/train_loss'] += avg_loss.item()

        if step == 1 or step % cfg.grad_check_interval == 0:
            for param in dit.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(
                    ), "NaN detected in gradient."
                    assert not torch.isinf(param.grad).any(
                    ), "Inf detected in gradient."

        grad_norm = 0.0
        if cfg.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                dit.parameters(), max_norm=cfg.max_grad_norm)
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            grad_norm = grad_norm.item()

        optimizer.step()
        lr_scheduler.step()
        step_time = time.perf_counter() - step_start_time
        step_times.append(step_time)

        avg_preprocess_time = sum(preprocess_times) / len(preprocess_times)
        avg_data_time = sum(data_times) / len(data_times)
        avg_step_time = sum(step_times) / len(step_times)
        if log_step:
            dist.all_reduce(task_loss_stats, op=dist.ReduceOp.SUM)
        if global_rank <= 0 and log_step:
            for task_type, (loss_sum, sample_count) in zip(LOSS_TASK_TYPES, task_loss_stats.tolist()):
                if sample_count > 0:
                    record_loss[f"Loss/{task_type}"] = loss_sum / sample_count
            progress_info = {
                "grad_norm": grad_norm,
                "step_time": f"{step_time:.2f}s",
                "data_time": f"{data_time:.2f}s",
                "preprocess_time": f"{preprocess_time:.2f}s",
            }
            progress_info.update(record_loss)
            logger.info(f"Step {step}/{cfg.max_train_steps}: {progress_info}")

            tb_log_info = {
                "Grad/grad_norm": grad_norm,
                "LR": lr_scheduler.get_last_lr()[0],
                "Time/step_time": step_time,
                "Time/data_time": data_time,
                "Time/preprocess_time": preprocess_time,
                "Time/avg_step_time": avg_step_time,
                "Time/avg_data_time": avg_data_time,
                "Time/avg_preprocess_time": avg_preprocess_time,
            }
            tb_log_info.update(record_loss)
            for k, v in tb_log_info.items():
                tb_writer.add_scalar(k, v, step)

        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(
                model=dit,
                save_dir=ckpt_dir,
                step=step,
                global_rank=global_rank,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                dataloader=train_dataloader,
            )

        if step % cfg.gc_interval == 0:
            gc.collect()

    if tb_writer is not None:
        tb_writer.close()
    clean_dist_env()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video editing")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()

    config_class = load_config_class_from_pyfile(args.config)
    cfg = config_class()
    main(cfg)
