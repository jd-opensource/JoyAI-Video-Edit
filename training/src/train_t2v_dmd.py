"""Resilient-DMD training with long-horizon autoregressive rollouts."""

import gc
import glob
import inspect
import os
import random
import sys
import time
from collections import deque
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import get_scheduler
from einops import rearrange
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter

from src.config import ExpConfig, load_config_class_from_pyfile
from src.dataset.webdataset import build_webdataset_dataloader
from src.distributed.communications import broadcast_item
from src.distributed.parallel_states import (
    clean_dist_env,
    get_parallel_state,
    init_distributed_environment_and_sequence_parallel,
    sp_enabled,
)
from src.models import load_dit, load_pipeline
from src.train_t2v import (
    LOSS_TASK_TYPES,
    build_multi_item_prompt_images,
    prepare_inputs,
    sample_noise_and_timestep,
)
from src.utils import find_latest_checkpoint, seed_everything
from src.utils.activation_checkpointing import apply_activation_checkpointing
from src.utils.constants import PRECISION_TO_TYPE
from src.utils.fsdp_load import maybe_load_fsdp_model, pt_weights_iterator, safetensors_weights_iterator
from src.utils.logging import setup_logger
from src.utils.utils import build_from_config, get_obj_from_str


_SAVE_OPTIONS = StateDictOptions(full_state_dict=True, cpu_offload=True)
_LOAD_OPTIONS = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)


_LOAD_OPTIONS_PARTIAL = StateDictOptions(
    full_state_dict=True, broadcast_from_rank0=True, strict=False,
)


_UNSET = object()

DEFAULT_JOYOMNI_LORA_TARGET_MODULES = [
    "img_attn_qkv",
    "img_attn_proj",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_attn_qkv",
    "txt_attn_proj",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def _set_cfg_attr(cfg: ExpConfig, name: str, value: Any) -> Any:
    old_value = getattr(cfg, name)
    setattr(cfg, name, value)
    return old_value


def dmd_uses_lora(cfg: ExpConfig) -> bool:
    return bool(getattr(cfg, "dmd_use_lora", False))


def dmd_generator_adapter_name(cfg: ExpConfig) -> str:
    return str(getattr(cfg, "dmd_generator_lora_adapter_name", "generator"))


def dmd_fake_score_adapter_name(cfg: ExpConfig) -> str:
    return str(getattr(cfg, "dmd_fake_score_lora_adapter_name", "fake_score"))


def get_module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _uses_pretrained_loading(config: dict | None) -> bool:
    if config is None or config.get("pretrained") is None:
        return False
    target = str(config.get("target", ""))
    try:
        cls = get_obj_from_str(target)
    except Exception:
        return target.startswith("diffusers.")
    return hasattr(cls, "from_pretrained")


def _strip_common_state_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("model.", "module.", "transformer."):
        stripped_state_dict = {}
        matched = 0
        for key, value in state_dict.items():
            if key.startswith(prefix):
                stripped_state_dict[key[len(prefix):]] = value
                matched += 1
            else:
                stripped_state_dict[key] = value
        if matched > 0:
            state_dict = stripped_state_dict
    return state_dict


def _load_pt_or_safetensor_state_dict(ckpt_path: str, ckpt_type: str) -> dict[str, torch.Tensor]:
    if ckpt_type == "safetensor":
        safetensors_files = glob.glob(os.path.join(str(ckpt_path), "*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {ckpt_path}")
        state_dict = dict(safetensors_weights_iterator(safetensors_files))
    elif ckpt_type == "pt":
        state_dict = dict(pt_weights_iterator([ckpt_path]))
        if "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        raise ValueError(f"Unknown ckpt_type={ckpt_type!r}; expected 'pt' or 'safetensor'.")
    return _strip_common_state_prefixes(state_dict)


def _build_base_dit(cfg: ExpConfig, device: torch.device) -> torch.nn.Module:
    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    if _uses_pretrained_loading(cfg.dit_arch_config):
        model_kwargs = {"torch_dtype": dtype, "device": device}
    else:
        model_kwargs = {"dtype": dtype, "device": device, "args": cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    if hasattr(model, "materialize_meta_modules"):
        model.materialize_meta_modules()
    if not dist.is_initialized() or dist.get_world_size() == 1:
        model.to(device=device)

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        model = model.to(dtype)
    return model


def load_stage1_dit(
    cfg: ExpConfig,
    device: torch.device,
    ckpt_path: str | None,
    *,
    trainable: bool,
    role: str,
) -> torch.nn.Module:
    if not ckpt_path:
        raise ValueError(
            f"`{role}` checkpoint is required. Set `dmd_stage1_ckpt` or `dmd_{role}_ckpt` in the config."
        )

    old_dit_ckpt = _set_cfg_attr(cfg, "dit_ckpt", ckpt_path)
    old_dit_ckpt_type = _set_cfg_attr(cfg, "dit_ckpt_type", getattr(cfg, f"dmd_{role}_ckpt_type", cfg.dit_ckpt_type))
    try:
        model = load_dit(cfg, device=device)
    finally:
        setattr(cfg, "dit_ckpt", old_dit_ckpt)
        setattr(cfg, "dit_ckpt_type", old_dit_ckpt_type)

    if trainable:
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def _adapter_name_fragment(adapter_name: str) -> str:
    return f".{adapter_name}."


def iter_adapter_parameters(model: torch.nn.Module, adapter_name: str) -> list[torch.nn.Parameter]:
    fragment = _adapter_name_fragment(adapter_name)
    return [param for name, param in model.named_parameters() if fragment in name]


def set_trainable_adapter(model: torch.nn.Module, adapter_name: str | None) -> None:
    for name, param in model.named_parameters():
        is_lora_param = "lora_" in name
        if adapter_name is None:
            param.requires_grad_(False)
        elif is_lora_param:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def make_optimizer(
    cfg: ExpConfig,
    model: torch.nn.Module,
    lr: float,
    adapter_name: str | None = None,
) -> torch.optim.Optimizer:
    params = iter_adapter_parameters(model, adapter_name) if adapter_name is not None else [
        param for param in model.parameters() if param.requires_grad
    ]
    if not params:
        suffix = f" for adapter {adapter_name!r}" if adapter_name is not None else ""
        raise ValueError(f"No trainable parameters found when building optimizer{suffix}.")
    if cfg.optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer {cfg.optimizer_name!r}; DMD currently expects adamw.")
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )


def resolve_lora_target_modules(
    model: torch.nn.Module,
    target_modules: Sequence[str],
) -> list[str]:
    named_modules = dict(model.named_modules())
    if not target_modules:
        raise ValueError("`dmd_lora_target_modules`/`lora_target_modules` must not be empty.")

    resolved: list[str] = []
    seen: set[str] = set()
    for target in target_modules:
        if not target:
            continue
        if target in named_modules:
            matches = [target]
        else:
            matches = [name for name in named_modules if name.endswith(f".{target}")]
        if not matches:
            raise ValueError(f"Could not find any Joyomni transformer modules matching LoRA target `{target}`.")
        for match in matches:
            if match not in seen:
                resolved.append(match)
                seen.add(match)

    if not resolved:
        raise ValueError("No valid LoRA target modules were resolved for the Joyomni transformer.")
    return resolved


def build_lora_config(cfg: ExpConfig, model: torch.nn.Module):
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for DMD LoRA training. Install it in the training environment, "
            "for example `pip install peft==0.18.0`."
        ) from exc

    lora_rank = int(getattr(cfg, "dmd_lora_rank", getattr(cfg, "lora_rank", 64)))
    lora_alpha = int(getattr(cfg, "dmd_lora_alpha", getattr(cfg, "lora_alpha", lora_rank)))
    lora_dropout = float(getattr(cfg, "dmd_lora_dropout", getattr(cfg, "lora_dropout", 0.0)))
    lora_bias = str(getattr(cfg, "dmd_lora_bias", getattr(cfg, "lora_bias", "none")))
    target_modules = resolve_lora_target_modules(
        model,
        list(getattr(cfg, "dmd_lora_target_modules", getattr(cfg, "lora_target_modules", DEFAULT_JOYOMNI_LORA_TARGET_MODULES))),
    )
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules,
    )


def activate_lora_adapter(model: torch.nn.Module, adapter_name: str | None) -> None:
    if adapter_name is None:
        return
    if not hasattr(model, "set_adapter"):
        raise ValueError(f"Model does not support PEFT adapters, cannot activate adapter {adapter_name!r}.")
    if hasattr(model, "enable_adapters"):
        model.enable_adapters()
    model.set_adapter(adapter_name)


def disable_lora_adapters(model: torch.nn.Module) -> None:
    if hasattr(model, "disable_adapters"):
        model.disable_adapters()


def _active_adapters(model: torch.nn.Module) -> list[str] | None:
    if not hasattr(model, "active_adapters"):
        return None
    try:
        active = model.active_adapters()
    except Exception:
        return None
    if active is None:
        return None
    if isinstance(active, str):
        return [active]
    return list(active)


@contextmanager
def lora_adapter_context(
    model: torch.nn.Module,
    *,
    adapter_name: str | None = None,
    disable_adapters: bool = False,
):
    if adapter_name is None and not disable_adapters:
        yield
        return

    previous_adapters = _active_adapters(model)
    if disable_adapters:
        disable_lora_adapters(model)
    else:
        activate_lora_adapter(model, adapter_name)
    try:
        yield
    finally:
        if previous_adapters:
            activate_lora_adapter(model, previous_adapters)
        elif not disable_adapters and hasattr(model, "enable_adapters"):
            model.enable_adapters()


@contextmanager
def model_training_mode(model: torch.nn.Module, training: bool):
    previous_training = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(previous_training)


def load_lora_base_dit(
    cfg: ExpConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, str, str]:
    base_ckpt = (
        getattr(cfg, "dmd_base_ckpt", None)
        or getattr(cfg, "dmd_real_score_ckpt", None)
        or getattr(cfg, "dmd_stage1_ckpt", None)
    )
    base_ckpt_type = getattr(
        cfg,
        "dmd_base_ckpt_type",
        getattr(cfg, "dmd_real_score_ckpt_type", getattr(cfg, "dit_ckpt_type", "pt")),
    )

    model = _build_base_dit(cfg, device)
    if base_ckpt:
        state_dict = _load_pt_or_safetensor_state_dict(base_ckpt, base_ckpt_type)
        load_state_dict = {}
        for key, value in state_dict.items():
            if (
                key == "img_in.weight"
                and hasattr(model, "img_in")
                and model.img_in.weight.shape != value.shape
            ):
                value_new = value.new_zeros(model.img_in.weight.shape)
                value = value.reshape_as(value_new)
            load_state_dict[key] = value
        strict = not _uses_pretrained_loading(cfg.dit_arch_config)
        missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=strict)
        del state_dict, load_state_dict
        if missing_keys or unexpected_keys:
            from src.utils.logging import get_logger

            active_logger = get_logger()
            if missing_keys:
                active_logger.warning(f"Missing keys when loading shared DMD LoRA base: {missing_keys[:20]}")
            if unexpected_keys:
                active_logger.warning(f"Unexpected keys when loading shared DMD LoRA base: {unexpected_keys[:20]}")
    else:
        from src.utils.logging import get_logger
        get_logger().info("No dmd_stage1_ckpt/dmd_base_ckpt specified; using pretrained weights as LoRA base.")

    model.requires_grad_(False)
    lora_config = build_lora_config(cfg, model)
    generator_adapter_name = dmd_generator_adapter_name(cfg)
    fake_score_adapter_name = dmd_fake_score_adapter_name(cfg)
    if generator_adapter_name == fake_score_adapter_name:
        raise ValueError("Generator and fake-score LoRA adapter names must be different.")

    model.add_adapter(lora_config, adapter_name=generator_adapter_name)
    model.add_adapter(lora_config, adapter_name=fake_score_adapter_name)
    activate_lora_adapter(model, generator_adapter_name)

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    for name, param in model.named_parameters():
        if "lora_" in name and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)

    if cfg.enable_activation_checkpointing:
        model = apply_activation_checkpointing(
            model,
            skip_interval=cfg.activation_checkpointing_skip_interval,
        )

    model = maybe_load_fsdp_model(
        model=model,
        hsdp_shard_dim=cfg.hsdp_shard_dim,
        reshard_after_forward=cfg.reshard_after_forward,
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cpu_offload=cfg.cpu_offload,
        pin_cpu_memory=cfg.pin_cpu_memory,
    )
    set_trainable_adapter(model, generator_adapter_name)
    return model.eval(), generator_adapter_name, fake_score_adapter_name


def setup_distributed_training(cfg: ExpConfig) -> tuple[int, int, int, torch.device]:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment_and_sequence_parallel(sp_size=cfg.sp_size)
    return local_rank, global_rank, world_size, device


def setup_dmd_models_and_optimizers(
    cfg: ExpConfig,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    Any,
    Any,
]:
    if dmd_uses_lora(cfg):
        shared_model, generator_adapter_name, fake_score_adapter_name = load_lora_base_dit(cfg, device)
        generator_optimizer = make_optimizer(
            cfg,
            shared_model,
            lr=getattr(cfg, "dmd_generator_learning_rate", cfg.learning_rate),
            adapter_name=generator_adapter_name,
        )
        fake_score_optimizer = make_optimizer(
            cfg,
            shared_model,
            lr=getattr(cfg, "dmd_fake_score_learning_rate", getattr(cfg, "learning_rate_critic", cfg.learning_rate)),
            adapter_name=fake_score_adapter_name,
        )
        generator_lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=generator_optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=cfg.max_train_steps,
            last_epoch=-1,
        )
        fake_score_lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=fake_score_optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=cfg.max_train_steps,
            last_epoch=-1,
        )
        return (
            shared_model,
            shared_model,
            shared_model,
            generator_optimizer,
            fake_score_optimizer,
            generator_lr_scheduler,
            fake_score_lr_scheduler,
        )

    stage1_ckpt = getattr(cfg, "dmd_stage1_ckpt", None)
    generator_ckpt = getattr(cfg, "dmd_generator_ckpt", None) or stage1_ckpt
    fake_score_ckpt = getattr(cfg, "dmd_fake_score_ckpt", None) or stage1_ckpt
    real_score_ckpt = getattr(cfg, "dmd_real_score_ckpt", None) or stage1_ckpt

    generator = load_stage1_dit(cfg, device, generator_ckpt, trainable=True, role="generator")
    fake_score = load_stage1_dit(cfg, device, fake_score_ckpt, trainable=True, role="fake_score")
    real_score = load_stage1_dit(cfg, device, real_score_ckpt, trainable=False, role="real_score")

    if cfg.enable_activation_checkpointing:
        for model in (generator, fake_score):
            apply_activation_checkpointing(
                model,
                skip_interval=cfg.activation_checkpointing_skip_interval,
            )

    generator_optimizer = make_optimizer(cfg, generator, lr=getattr(cfg, "dmd_generator_learning_rate", cfg.learning_rate))
    fake_score_optimizer = make_optimizer(
        cfg,
        fake_score,
        lr=getattr(cfg, "dmd_fake_score_learning_rate", getattr(cfg, "learning_rate_critic", cfg.learning_rate)),
    )

    generator_lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=generator_optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
        last_epoch=-1,
    )
    fake_score_lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=fake_score_optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
        last_epoch=-1,
    )
    return (
        generator,
        fake_score,
        real_score,
        generator_optimizer,
        fake_score_optimizer,
        generator_lr_scheduler,
        fake_score_lr_scheduler,
    )


def _sigma_view(sigmas: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    if sigmas.ndim == 1:
        return sigmas.view(-1, 1, 1, 1, 1)
    if sigmas.ndim == 2:
        return sigmas[:, None, :, None, None]
    raise ValueError(f"Unsupported sigma shape: {tuple(sigmas.shape)}")


def flow_to_x0(noisy_latents: torch.Tensor, flow_pred: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    sigma = _sigma_view(sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype), noisy_latents)
    return noisy_latents - sigma * flow_pred.to(dtype=noisy_latents.dtype)


def build_noisy_inputs_and_targets_from_sigmas(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = _sigma_view(sigmas.to(device=latents.device, dtype=latents.dtype), latents)
    noisy_model_input = (1 - sigma) * latents + sigma * noise
    target = noise - latents
    return noisy_model_input, target


def _has_ref_image_latent(ref_image_latent: Any) -> bool:
    """Return whether the batch contains reference-image latents."""
    if torch.is_tensor(ref_image_latent):
        return True
    if isinstance(ref_image_latent, list):
        return any(item is not None for item in ref_image_latent)
    return False


class ResilientDMDController:
    """Compute rollout-frozen source-guidance weights from attention statistics."""

    def __init__(self, cfg: ExpConfig, device: torch.device) -> None:
        self.device = device
        self.enabled = True
        self._disable_reason: str | None = None
        self.kappa = float(getattr(cfg, "dmd_attn_rho_kappa", 2.0))
        self.w_floor = 0.5
        self.w_max = 1.0
        self.ratio_margin = max(0.0, float(getattr(cfg, "dmd_attn_rho_ratio_margin", 0.05)))
        self.min_cell_count = 24.0
        self.query_stride = 1
        self.chunk_stride = 1
        self.smooth_chunks = 2
        self.beta = float(getattr(cfg, "dmd_source_scale_ema_beta", 0.99))
        self.num_sigma = 2
        self.num_depth = 3
        self._ctx: dict[str, Any] | None = None

        architecture = getattr(cfg, "dit_arch_config", None) or {}
        parameters = architecture.get("params", architecture)
        depth = parameters.get("mm_double_blocks_depth")
        self.num_layers = int(depth or 0)
        if depth is None:
            self._disable("dit_arch_config has no mm_double_blocks_depth to verify attention calls")
        if sp_enabled():
            self._disable("source-attention control requires sequence parallel size 1")
        self.ctrl_layers = tuple(range(self.num_layers))
        self._control_indices = {layer: index for index, layer in enumerate(self.ctrl_layers)}
        shape = (len(self.ctrl_layers), self.num_sigma, self.num_depth)
        self.ema = torch.zeros(shape, device=device)
        self.ema_cnt = torch.zeros_like(self.ema)
        self._reset_rollout_state()
        self._layer_index = 0
        self._control_observations: list[tuple[int, torch.Tensor]] = []
        self._install_patch(cfg)

    def _install_patch(self, cfg: ExpConfig) -> None:
        if not self.enabled:
            return
        target = (getattr(cfg, "dit_arch_config", None) or {}).get("target", "")
        try:
            model_class = get_obj_from_str(target)
            module = sys.modules[model_class.__module__]
            original_attention = module._run_flex_or_sdpa
        except Exception as exc:
            self._disable(f"cannot resolve the attention function on {target!r}: {exc}")
            return

        def controlled_attention(query, key, value, block_mask, attention_mask, lazy_dense_mask_fn=None):
            output = original_attention(query, key, value, block_mask, attention_mask, lazy_dense_mask_fn)
            if self._ctx is not None:
                self._on_attention(query, key)
            return output

        self._module = module
        self._original_attention = original_attention
        self._controlled_attention = controlled_attention
        module._run_flex_or_sdpa = controlled_attention

    def close(self) -> None:
        """Restore the wrapped attention function without replacing later hooks."""
        module = getattr(self, "_module", None)
        if module is not None and module._run_flex_or_sdpa is self._controlled_attention:
            module._run_flex_or_sdpa = self._original_attention
        self._ctx = None

    def _disable(self, reason: str) -> None:
        if not self.enabled and self._disable_reason is not None:
            return
        self.enabled = False
        self._disable_reason = reason
        self._ctx = None
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[WARN][rank{rank}] Resilient-DMD source controller disabled: {reason}")

    def _reset_rollout_state(self) -> None:
        self._roll_sum = torch.zeros_like(self.ema)
        self._roll_cnt = torch.zeros_like(self.ema_cnt)
        self._chunk_w: dict[int, torch.Tensor] = {}
        self._ratio_deque: deque[tuple[int, torch.Tensor]] = deque(maxlen=self.smooth_chunks)

    def build_layout(
        self,
        *,
        window_latents: torch.Tensor,
        model: torch.nn.Module,
        local_chunk_start: int,
        local_chunk_len: int,
        chunk_idx: int,
        num_chunks: int,
    ) -> dict[str, int] | None:
        """Locate active target queries and visible source/history key segments."""
        if not self.enabled:
            return None
        patch_time, patch_height, patch_width = map(int, getattr(model.config, "patch_size", [1, 2, 2]))
        _, _, window_frames, height, width = window_latents.shape
        if window_frames % patch_time or height % patch_height or width % patch_width or local_chunk_start % patch_time:
            self._disable("patch size does not divide the rollout window and active-chunk boundary")
            return None
        if local_chunk_len % patch_time:
            self._disable("active source frames are not divisible by the temporal patch size")
            return None
        spatial_tokens = (height // patch_height) * (width // patch_width)
        return {
            "win_tok": (window_frames // patch_time) * spatial_tokens,
            "active_start_tok": (local_chunk_start // patch_time) * spatial_tokens,
            "ref_tok": (local_chunk_len // patch_time) * spatial_tokens,
            "depth_bucket": min(self.num_depth - 1, chunk_idx * self.num_depth // max(num_chunks, 1)),
            "chunk_idx": chunk_idx,
        }

    @contextmanager
    def armed(self, *, layout: dict[str, Any], sigma_bucket: int):
        """Measure one generator exit forward and store its detached source weight."""
        if not self.enabled:
            yield
            return
        self._ctx = {**layout, "sigma_bucket": min(max(sigma_bucket, 0), self.num_sigma - 1)}
        self._layer_index = 0
        self._control_observations = []
        try:
            yield
        finally:
            context = self._ctx
            self._ctx = None
            if self.enabled and self._layer_index != self.num_layers:
                self._disable(f"forward made {self._layer_index} attention calls, expected {self.num_layers}")
            elif self.enabled and self._control_observations:
                sigma_index, depth_index = context["sigma_bucket"], context["depth_bucket"]
                ratios = []
                for control_index, rho in self._control_observations:
                    if float(self.ema_cnt[control_index, sigma_index, depth_index]) < self.min_cell_count:
                        continue
                    ratios.append(self.ema[control_index, sigma_index, depth_index] / rho.detach().clamp_min(1e-4))
                chunk_index = context["chunk_idx"]
                while self._ratio_deque and self._ratio_deque[0][0] <= chunk_index - self.smooth_chunks:
                    self._ratio_deque.popleft()
                if ratios:
                    self._ratio_deque.append((chunk_index, torch.stack(ratios, dim=0)))
                weight = self._control_w()
                if weight is not None:
                    self._chunk_w[context["chunk_idx"]] = weight
            self._control_observations = []

    @torch.no_grad()
    def _segment_lse(self, scaled_query: torch.Tensor, key: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Exact blockwise logsumexp, without materializing the full attention matrix."""
        query_parts = []
        for query_start in range(0, scaled_query.shape[1], 512):
            query_block = scaled_query[:, query_start:query_start + 512]
            segment_lse = None
            for key_start in range(start, end, 2048):
                key_block = key[:, key_start:min(key_start + 2048, end)].float()
                logits = torch.einsum("bqhd,bkhd->bhqk", query_block, key_block)
                part = torch.logsumexp(logits, dim=-1)
                segment_lse = part if segment_lse is None else torch.logaddexp(segment_lse, part)
            query_parts.append(segment_lse)
        return torch.cat(query_parts, dim=2)

    @torch.no_grad()
    def _on_attention(self, query: torch.Tensor, key: torch.Tensor) -> None:
        context = self._ctx
        if context is None or not self.enabled:
            return
        layer_index = self._layer_index
        self._layer_index += 1
        if layer_index >= self.num_layers or layer_index not in self._control_indices:
            return
        if query.ndim != 4 or key.ndim != 4 or key.shape[:2] != query.shape[:2]:
            self._disable(f"unexpected attention shapes query={tuple(query.shape)}, key={tuple(key.shape)}")
            return
        window_tokens, source_tokens = context["win_tok"], context["ref_tok"]
        active_start = context["active_start_tok"]
        if not (0 < active_start < window_tokens and source_tokens > 0 and query.shape[1] >= window_tokens + source_tokens):
            self._disable("source/history token accounting does not match the attention sequence")
            return
        scaled_query = query[:, active_start:window_tokens:self.query_stride].float() * (float(query.shape[-1]) ** -0.5)
        history_lse = self._segment_lse(scaled_query, key, 0, active_start)
        source_lse = self._segment_lse(scaled_query, key, window_tokens, window_tokens + source_tokens)
        rho = torch.sigmoid(source_lse - history_lse).mean(dim=(1, 2))
        control_index = self._control_indices[layer_index]
        sigma_index, depth_index = context["sigma_bucket"], context["depth_bucket"]
        self._roll_sum[control_index, sigma_index, depth_index] += rho.sum()
        self._roll_cnt[control_index, sigma_index, depth_index] += rho.shape[0]
        self._control_observations.append((control_index, rho))

    def _control_w(self) -> torch.Tensor | None:
        if not self._ratio_deque:
            return None
        ratio = torch.cat([ratios for _, ratios in self._ratio_deque], dim=0).mean(dim=0)
        weight = self.w_floor * (ratio.clamp_min(0.0) / (1.0 + self.ratio_margin)) ** self.kappa
        return weight.clamp(self.w_floor, self.w_max)

    def group_anchor_w(
        self,
        chunk_ids: Sequence[int],
        chunk_frames: Sequence[int],
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return per-frame weights; unmeasured/cold groups retain static guidance."""
        if not self.enabled or not any(chunk_id in self._chunk_w for chunk_id in chunk_ids):
            return None
        parts = []
        for chunk_id, frames in zip(chunk_ids, chunk_frames):
            weight = self._chunk_w.get(chunk_id)
            if weight is None:
                weight = torch.full((batch,), self.w_floor, device=device)
            parts.append(weight.to(device=device).view(batch, 1, 1, 1, 1).expand(batch, 1, frames, 1, 1))
        return torch.cat(parts, dim=2).float()

    def finish_rollout(self) -> None:
        """Update matched EMAs only after the rollout; synchronize failure across ranks."""
        packed = torch.cat([
            self._roll_sum.flatten(), self._roll_cnt.flatten(),
            torch.tensor([0.0 if self.enabled else 1.0], device=self.device),
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed)
        cell_count = self._roll_sum.numel()
        rollout_sum = packed[:cell_count].view_as(self._roll_sum)
        rollout_count = packed[cell_count:2 * cell_count].view_as(self._roll_cnt)
        self._reset_rollout_state()
        if float(packed[-1]) > 0:
            if self.enabled:
                self._disable("source controller disabled on another rank")
            return
        seen = rollout_count > 0
        rollout_mean = rollout_sum / rollout_count.clamp_min(1.0)
        self.ema[seen] = torch.where(
            self.ema_cnt[seen] > 0,
            self.beta * self.ema[seen] + (1.0 - self.beta) * rollout_mean[seen],
            rollout_mean[seen],
        )
        self.ema_cnt[seen] += rollout_count[seen]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "ctrl_layers": torch.tensor(self.ctrl_layers, dtype=torch.long),
            "ema": self.ema.detach().cpu(),
            "ema_cnt": self.ema_cnt.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore per-layer EMA statistics from a checkpoint."""
        saved_ema, saved_count = state["ema"], state["ema_cnt"]
        if saved_ema.shape != saved_count.shape or saved_ema.shape[1:] != self.ema.shape[1:]:
            raise ValueError("Source-controller checkpoint has incompatible sigma/depth bins or counts.")
        saved_layers = state.get("ctrl_layers")
        if saved_layers is None:
            if saved_ema.shape[0] != self.num_layers:
                raise ValueError("Legacy attention checkpoint does not match the model layer count.")
            indices = list(self.ctrl_layers)
        else:
            saved_layers = torch.as_tensor(saved_layers).tolist()
            if len(saved_layers) != saved_ema.shape[0] or any(layer not in saved_layers for layer in self.ctrl_layers):
                raise ValueError("Source-controller checkpoint is missing configured control layers.")
            indices = [saved_layers.index(layer) for layer in self.ctrl_layers]
        self.ema.copy_(saved_ema[indices].to(self.device))
        self.ema_cnt.copy_(saved_count[indices].to(self.device))


def _build_token_timesteps(
    timesteps: torch.Tensor,
    noisy_model_input: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """Expand frame timesteps to match the target token sequence."""
    if timesteps.ndim != 2:
        return timesteps
    patch_size = getattr(model.config, "patch_size", [1, 2, 2])
    _, _, t_latent, h_latent, w_latent = noisy_model_input.shape
    t_patch = t_latent // patch_size[0]
    h_patch = h_latent // patch_size[1]
    w_patch = w_latent // patch_size[2]
    spatial_tokens = h_patch * w_patch
    sampled_ts = timesteps[:, :: patch_size[0]]
    return sampled_ts.unsqueeze(-1).expand(-1, -1, spatial_tokens).reshape(
        timesteps.shape[0], t_patch * spatial_tokens
    )


def _flatten_multi_item_latents(
    latents: torch.Tensor,
    additional_inputs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split (B, N, C, T, H, W) latents into a target and source conditioning."""
    if latents.ndim != 6:
        target_latents = latents
        ref_video_latent_5d = None
    else:
        num_items = latents.shape[1]
        if num_items > 1:
            ref_video_latent_5d = rearrange(latents[:, :-1], "b n c t h w -> b c (n t) h w")
        else:
            ref_video_latent_5d = None
        target_latents = latents[:, -1]

    additional_inputs["ref_video_latent"] = ref_video_latent_5d
    additional_inputs["ref_image_latent"] = additional_inputs.get("ref_image_latents")
    if additional_inputs.get("loss_mask") is not None:
        loss_mask = additional_inputs["loss_mask"]
        if loss_mask.ndim == 6:
            additional_inputs["loss_mask"] = loss_mask[:, -1]
    return target_latents, ref_video_latent_5d


def _build_negative_prompt_images(
    cfg: ExpConfig,
    pixel_values: torch.Tensor,
    task_type: str,
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
    """Match negative-prompt visual tokens to the conditional branch."""
    if task_type == "v2v":
        return (pixel_values[:, 0, :, 0] + 1) * 127.5

    return build_multi_item_prompt_images(pixel_values)


def encode_negative_prompt(
    cfg: ExpConfig,
    pipeline: Any,
    batch_size: int,
    *,
    task_type: str,
    pixel_values: torch.Tensor | None = None,
    ref_image_pixels: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the negative prompt with the same visual-token layout as the conditional prompt."""
    device = get_module_device(pipeline.transformer)
    negative_prompt = getattr(cfg, "dmd_negative_prompt", "")
    is_image = task_type in ("t2i", "i2i")
    template_type = "image" if is_image else "video"

    if cfg.use_vit and task_type in ("i2i", "v2v"):
        if pixel_values is None:
            raise ValueError(
                "encode_negative_prompt: `pixel_values` is required for use_vit "
                "+ i2i/v2v so the negative prompt can be encoded with the same "
                "multimodal layout as the conditional prompt."
            )

        neg_prompt = f"<|im_start|>user\n<image>\n{negative_prompt}<|im_end|>\n"
        prompts = [neg_prompt] * batch_size
        prompt_images = _build_negative_prompt_images(
            cfg, pixel_values, task_type, device,
        )
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt_multiple_images(
            prompt=prompts,
            images=prompt_images,
            device=device,
            max_sequence_length=cfg.text_token_max_length,
        )
    else:
        prompts = [negative_prompt] * batch_size
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=prompts,
            device=device,
            max_sequence_length=cfg.text_token_max_length,
            template_type=template_type,
        )

    transformer_dtype = pipeline.transformer.dtype
    return prompt_embeds.to(dtype=transformer_dtype), prompt_embeds_mask


def _cfg_combine_dmd(
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    if uncond is None or guidance_scale == 0.0:
        return cond
    return cond + guidance_scale * (cond - uncond)


def _prepare_score_additional_inputs(
    additional_inputs: dict[str, Any],
    generated_latents: torch.Tensor,
) -> dict[str, Any]:
    """Score generated samples with their detached generated history."""
    score_inputs = dict(additional_inputs)
    clean_latents = generated_latents.detach()
    history_latents = score_inputs.pop("score_history_latent", None)
    if history_latents is not None:
        clean_latents = torch.cat([history_latents.detach(), clean_latents], dim=2)
    score_inputs["clean_video_latent"] = clean_latents
    if score_inputs.get("ref_video_latent") is None:
        score_inputs.pop("ref_video_latent", None)
    if score_inputs.get("ref_image_latent") is None:
        ref_image = score_inputs.get("ref_image_latents")
        if ref_image is not None:
            score_inputs["ref_image_latent"] = ref_image
        else:
            score_inputs.pop("ref_image_latent", None)
    return score_inputs


def score_flow_prediction(
    cfg: ExpConfig,
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    timesteps: torch.Tensor,
    additional_inputs: dict[str, Any],
) -> torch.Tensor:
    """Score all target chunks in one forward with the SFT mask."""
    noisy_latents = noisy_latents.to(dtype=model.dtype)
    prompt_embeds = prompt_embeds.to(dtype=model.dtype)
    if timesteps.ndim == 1:
        timesteps = timesteps[:, None].expand(-1, noisy_latents.shape[2])

    history_frames = 0
    clean_video_latent = additional_inputs.get("clean_video_latent")
    if clean_video_latent is not None:
        clean_video_latent = clean_video_latent.to(dtype=model.dtype)
        history_frames = clean_video_latent.shape[2] - noisy_latents.shape[2]
        if history_frames < 0:
            raise ValueError("Score clean history must include every target frame.")
        if history_frames:
            noisy_latents = torch.cat([clean_video_latent[:, :, :history_frames], noisy_latents], dim=2)
            timesteps = F.pad(timesteps, (history_frames, 0), value=0)
    ref_video_latent = additional_inputs.get("ref_video_latent")
    if ref_video_latent is not None:
        ref_video_latent = ref_video_latent.to(dtype=model.dtype)
    ref_image_latent = additional_inputs.get("ref_image_latent")
    if ref_image_latent is not None:
        if isinstance(ref_image_latent, list):
            ref_image_latent = [
                latent.to(dtype=model.dtype) if latent is not None else None
                for latent in ref_image_latent
            ]
        else:
            ref_image_latent = ref_image_latent.to(dtype=model.dtype)

    token_timesteps = _build_token_timesteps(timesteps, noisy_latents, model)

    model_kwargs = dict(
        hidden_states=noisy_latents,
        timestep=token_timesteps,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        return_dict=False,
    )
    if clean_video_latent is not None:
        model_kwargs["clean_video_latent"] = clean_video_latent
    if additional_inputs.get("noisy_temporal_ids") is not None:
        model_kwargs["noisy_temporal_ids"] = additional_inputs["noisy_temporal_ids"]
    if ref_video_latent is not None:
        model_kwargs["ref_video_latent"] = ref_video_latent
    if ref_image_latent is not None:
        model_kwargs["ref_image_latent"] = ref_image_latent

    return model(**model_kwargs)[0][:, :, history_frames:]


def _sample_score_inputs(
    cfg: ExpConfig,
    model: torch.nn.Module,
    pipeline: Any,
    generated_latents: torch.Tensor,
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noise, timesteps, sigmas = sample_noise_and_timestep(
        generated_latents.detach(), cfg, noise_generator_cuda, noise_generator_cpu, model=model,
    )
    min_sigma = getattr(cfg, "dmd_min_sigma", 0.02)
    max_sigma = getattr(cfg, "dmd_max_sigma", 0.98)
    if min_sigma is not None or max_sigma is not None:
        sigmas = sigmas.clamp(
            min=min_sigma if min_sigma is not None else 0.0,
            max=max_sigma if max_sigma is not None else 1.0,
        )
        timesteps = (sigmas * pipeline.scheduler.config.num_train_timesteps).round()
    noisy_latents, target = build_noisy_inputs_and_targets_from_sigmas(generated_latents, noise, sigmas)
    return noisy_latents, target, timesteps, sigmas


def compute_dmd_gradient(
    cfg: ExpConfig,
    fake_score: torch.nn.Module,
    real_score: torch.nn.Module,
    pipeline: Any,
    generated_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_mask: torch.Tensor | None,
    additional_inputs: dict[str, Any],
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
) -> torch.Tensor:
    """Resilient-DMD gradient: source guidance changes only the real-score teacher."""
    noisy_latents, _, timesteps, sigmas = _sample_score_inputs(
        cfg, fake_score, pipeline, generated_latents, noise_generator_cuda, noise_generator_cpu,
    )
    score_inputs = _prepare_score_additional_inputs(additional_inputs, generated_latents)
    use_lora = dmd_uses_lora(cfg)
    fake_adapter_name = dmd_fake_score_adapter_name(cfg) if use_lora else None
    fake_mode_context = model_training_mode(fake_score, False) if use_lora else nullcontext()
    real_mode_context = model_training_mode(real_score, False) if use_lora else nullcontext()

    with torch.no_grad():
        with fake_mode_context, lora_adapter_context(fake_score, adapter_name=fake_adapter_name):
            fake_flow = score_flow_prediction(
                cfg, fake_score, noisy_latents, prompt_embeds, prompt_embeds_mask, timesteps, score_inputs,
            )
        with real_mode_context, lora_adapter_context(real_score, disable_adapters=use_lora):
            real_flow_cond = score_flow_prediction(
                cfg, real_score, noisy_latents, prompt_embeds, prompt_embeds_mask, timesteps, score_inputs,
            )
            real_flow_uncond = score_flow_prediction(
                cfg, real_score, noisy_latents, negative_prompt_embeds, negative_prompt_embeds_mask, timesteps, score_inputs,
            )
            source_weight = float(getattr(cfg, "dmd_real_source_guidance_scale", 0.5))
            anchor_weights = additional_inputs.get("source_anchor_w_frames")
            real_flow_noref = None
            if (source_weight != 0.0 or anchor_weights is not None) and score_inputs.get("ref_video_latent") is not None:
                source_free_inputs = dict(score_inputs)
                source_free_inputs.pop("ref_video_latent")
                real_flow_noref = score_flow_prediction(
                    cfg, real_score, noisy_latents, prompt_embeds, prompt_embeds_mask, timesteps, source_free_inputs,
                )
        real_flow = _cfg_combine_dmd(
            real_flow_cond, real_flow_uncond, float(getattr(cfg, "dmd_real_guidance_scale", 2.5)),
        )
        if real_flow_noref is not None:
            source_direction = real_flow_cond - real_flow_noref
            if anchor_weights is not None:
                if anchor_weights.shape[2] != noisy_latents.shape[2]:
                    raise ValueError("Source-anchor weights must align with the scored latent frames.")
                source_weight = anchor_weights.detach().to(device=source_direction.device, dtype=source_direction.dtype)
            real_flow = real_flow + source_weight * source_direction

        pred_fake_x0 = flow_to_x0(noisy_latents, fake_flow, sigmas)
        pred_real_x0 = flow_to_x0(noisy_latents, real_flow, sigmas)
        gradient = pred_fake_x0 - pred_real_x0
        if getattr(cfg, "dmd_normalize_gradient", True):
            residual = generated_latents - pred_real_x0
            chunk_size = getattr(cfg, "dmd_generator_chunk_size", None)
            if chunk_size is None and hasattr(fake_score, "config"):
                chunk_size = getattr(fake_score.config, "chunk_size", None)
            if chunk_size is not None and chunk_size > 0 and residual.shape[2] > chunk_size:
                batch_size, channels, frames, height, width = residual.shape
                full_chunks = frames // chunk_size
                full_frames = full_chunks * chunk_size
                full_residual = residual[:, :, :full_frames].view(batch_size, channels, full_chunks, chunk_size, height, width)
                chunk_normalizer = full_residual.abs().mean(dim=(1, 3, 4, 5), keepdim=True)
                normalizer = chunk_normalizer.expand_as(full_residual).reshape(batch_size, channels, full_frames, height, width)
                if full_frames < frames:
                    remainder = residual[:, :, full_frames:]
                    remainder_normalizer = remainder.abs().mean(dim=(1, 2, 3, 4), keepdim=True)
                    normalizer = torch.cat((normalizer, remainder_normalizer.expand_as(remainder)), dim=2)
            else:
                normalizer = residual.abs().mean(dim=(1, 2, 3, 4), keepdim=True)
            gradient = gradient / normalizer.clamp_min(getattr(cfg, "dmd_normalizer_eps", 1e-6))
        return torch.nan_to_num(gradient)


def _masked_mean_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    normalizer: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Use a shared valid-element count when reducing rollout segments."""
    mask = mask.expand_as(loss)
    if normalizer is None:
        normalizer = mask.sum(dtype=loss.dtype)
    else:
        normalizer = torch.as_tensor(normalizer, device=loss.device, dtype=loss.dtype)
    return (loss * mask).sum() / normalizer.clamp_min(1)


def compute_generator_dmd_loss(
    cfg: ExpConfig,
    fake_score: torch.nn.Module,
    real_score: torch.nn.Module,
    pipeline: Any,
    generated_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_mask: torch.Tensor | None,
    additional_inputs: dict[str, Any],
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
    gradient_stats: torch.Tensor | None = None,
) -> torch.Tensor:
    gradient = compute_dmd_gradient(
        cfg, fake_score, real_score, pipeline, generated_latents, prompt_embeds,
        prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask,
        additional_inputs, noise_generator_cuda, noise_generator_cpu,
    )
    target = (generated_latents.float() - gradient.float()).detach()
    loss = 0.5 * F.mse_loss(generated_latents.float(), target, reduction="none")
    total_loss = _masked_mean_loss(loss, additional_inputs["loss_mask"], additional_inputs.get("loss_normalizer"))
    if gradient_stats is not None:
        with torch.no_grad():
            gradient_values = gradient.detach().float()
            mask = additional_inputs["loss_mask"].detach().expand_as(gradient_values)
            moments = torch.stack((
                (gradient_values.abs() * mask).sum(dtype=torch.float64),
                (gradient_values.square() * mask).sum(dtype=torch.float64),
                mask.sum(dtype=torch.float64),
            ))
            gradient_stats.add_(torch.where(torch.isfinite(total_loss.detach()), moments, 0.0))
    return total_loss


def compute_fake_score_loss(
    cfg: ExpConfig,
    fake_score: torch.nn.Module,
    pipeline: Any,
    generated_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    additional_inputs: dict[str, Any],
    noise_generator_cuda: torch.Generator,
    noise_generator_cpu: torch.Generator,
) -> torch.Tensor:
    """Flow-match detached rollouts; keep the fake-score adapter active through backward."""
    noisy_latents, target, timesteps, _ = _sample_score_inputs(
        cfg, fake_score, pipeline, generated_latents.detach(), noise_generator_cuda, noise_generator_cpu,
    )
    score_inputs = _prepare_score_additional_inputs(additional_inputs, generated_latents)
    fake_flow = score_flow_prediction(
        cfg, fake_score, noisy_latents, prompt_embeds, prompt_embeds_mask, timesteps, score_inputs,
    )
    loss = F.mse_loss(fake_flow.float(), target.float(), reduction="none")
    return _masked_mean_loss(loss, additional_inputs["loss_mask"], additional_inputs.get("loss_normalizer"))


def _sample_exit_index(num_timesteps: int) -> int:
    if num_timesteps <= 0:
        raise ValueError("Resilient-DMD requires at least one denoising timestep.")
    if num_timesteps == 1:
        return 0
    return int(torch.randint(0, num_timesteps, (1,)).item())


def _expand_per_token_timestep(
    timestep_value: torch.Tensor,
    window_latents: torch.Tensor,
    model: torch.nn.Module,
    local_chunk_start: int = 0,
) -> torch.Tensor:
    """Set history-token time to zero and active-token time to timestep_value."""
    patch_size = getattr(model.config, "patch_size", [1, 2, 2])
    batch_size, _, t_latent, h_latent, w_latent = window_latents.shape
    t_patch = t_latent // patch_size[0]
    h_patch = h_latent // patch_size[1]
    w_patch = w_latent // patch_size[2]
    spatial_tokens = h_patch * w_patch

    timestep_value = timestep_value.to(window_latents.device)
    if local_chunk_start <= 0:
        per_token = timestep_value.expand(t_patch * spatial_tokens)
    else:
        local_chunk_start_patch = local_chunk_start // patch_size[0]
        history_token_count = local_chunk_start_patch * spatial_tokens
        active_token_count = t_patch * spatial_tokens - history_token_count
        zero = torch.zeros(history_token_count, device=window_latents.device, dtype=timestep_value.dtype)
        active = timestep_value.expand(active_token_count)
        per_token = torch.cat([zero, active], dim=0)
    return per_token.unsqueeze(0).expand(batch_size, -1)


def _run_streaming_model(
    model: torch.nn.Module,
    *,
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    temporal_ids: torch.Tensor,
    ref_video_latent: torch.Tensor | None,
    ref_image_latent: torch.Tensor | list[torch.Tensor | None] | None,
) -> torch.Tensor:
    """Conditional, cache-free rollout with zero-time history and active-only source/text."""
    model_kwargs = dict(
        hidden_states=latent_model_input.to(dtype=model.dtype),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds.to(dtype=model.dtype),
        encoder_hidden_states_mask=prompt_embeds_mask,
        noisy_temporal_ids=temporal_ids.unsqueeze(0).expand(latent_model_input.shape[0], -1),
        active_chunk_only_conditioning=True,
        return_dict=False,
    )
    if ref_video_latent is not None:
        model_kwargs["ref_video_latent"] = ref_video_latent.to(dtype=model.dtype)
    if ref_image_latent is not None:
        if isinstance(ref_image_latent, list):
            model_kwargs["ref_image_latent"] = [
                latent.to(dtype=model.dtype) if latent is not None else None for latent in ref_image_latent
            ]
        else:
            model_kwargs["ref_image_latent"] = ref_image_latent.to(dtype=model.dtype)
    with model.cache_context("cond"):
        return model(**model_kwargs)[0]


def _resolve_rollout_num_chunks(cfg: ExpConfig, global_rank: int, spec: Any = _UNSET) -> int:
    """Resolve one shared rollout length so FSDP ranks make identical forward counts."""
    if spec is _UNSET:
        spec = getattr(cfg, "dmd_rollout_num_chunks", -1)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("The rollout length range must be non-empty.")
        lower, upper = int(spec[0]), int(spec[-1])
        if lower <= 0 or lower > upper:
            raise ValueError("A rollout length range must satisfy 0 < lower <= upper.")
        sampled = int(torch.randint(lower, upper + 1, (1,)).item()) if global_rank <= 0 else 0
        return int(broadcast_item(sampled, src=0))
    return int(spec)


def _ref_content_chunk_id(chunk_id: int, num_ref_chunks: int) -> int:
    """Index forward/reverse source traversal without repeating endpoint frames."""
    if num_ref_chunks <= 0:
        raise ValueError("`num_ref_chunks` must be positive.")
    if num_ref_chunks == 1:
        return 0
    period = 2 * (num_ref_chunks - 1)
    position = chunk_id % period
    return position if position < num_ref_chunks else period - position


def _gather_ref_chunks(
    ref_video_latent: torch.Tensor | None,
    selected_chunk_ids: Sequence[int],
    chunk_size: int,
    num_ref_chunks: int,
) -> torch.Tensor | None:
    """Gather ping-pong source content while leaving rollout/RoPE chunk IDs unchanged."""
    if ref_video_latent is None:
        return None
    chunks = []
    for chunk_id in selected_chunk_ids:
        source_chunk_id = _ref_content_chunk_id(chunk_id, num_ref_chunks)
        start = source_chunk_id * chunk_size
        end = min(ref_video_latent.shape[2], start + chunk_size)
        chunks.append(ref_video_latent[:, :, start:end])
    return torch.cat(chunks, dim=2)


def generate_dmd_latents(
    cfg: ExpConfig,
    pipeline: Any,
    generator_model: torch.nn.Module,
    latent_shape_like: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    additional_inputs: dict[str, Any],
    noise_generator_cuda: torch.Generator,
    *,
    requires_grad: bool = True,
    rollout_num_chunks_override: int | None = None,
    group_backward_fn: Callable[[torch.Tensor, dict[str, Any]], None] | None = None,
    controller: ResilientDMDController | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None, bool]:
    """Generate random-exit rollouts; return latents, score inputs, and whether backward ran."""
    generator_adapter = dmd_generator_adapter_name(cfg) if dmd_uses_lora(cfg) else None
    activate_lora_adapter(generator_model, generator_adapter)
    device = latent_shape_like.device
    num_inference_steps = int(getattr(cfg, "dmd_generator_num_inference_steps", 2))
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    input_frames = latent_shape_like.shape[2]
    chunk_size = pipeline._resolve_streaming_chunk_size(getattr(cfg, "dmd_generator_chunk_size", None), input_frames)
    rollout_chunks = rollout_num_chunks_override
    if rollout_chunks is None:
        rollout_chunks = getattr(cfg, "dmd_rollout_num_chunks", -1)
        if isinstance(rollout_chunks, (list, tuple)):
            raise ValueError("Resolve rollout length ranges across ranks before calling generate_dmd_latents.")
    rollout_chunks = int(rollout_chunks)
    if additional_inputs.get("task_type") in ("t2i", "i2i"):
        rollout_chunks = -1
    use_long_rollout = rollout_chunks > 0
    total_frames = rollout_chunks * chunk_size if use_long_rollout else input_frames
    context_shape = list(latent_shape_like.shape)
    context_shape[2] = total_frames
    context_latents = torch.randn(context_shape, generator=noise_generator_cuda, device=device, dtype=torch.float32)

    source_latents = additional_inputs.get("ref_video_latent")
    num_source_chunks = None
    if source_latents is not None:
        source_latents = source_latents.to(device=device, dtype=torch.float32)
        num_source_chunks = (source_latents.shape[2] + chunk_size - 1) // chunk_size
    reference_latents = additional_inputs.get("ref_image_latent")
    if torch.is_tensor(reference_latents):
        reference_latents = reference_latents.to(device=device, dtype=torch.float32)
    elif isinstance(reference_latents, list):
        reference_latents = [latent.to(device=device, dtype=torch.float32) if latent is not None else None for latent in reference_latents]

    window_size = getattr(cfg, "dmd_generator_window_size", None)
    if window_size is None:
        window_size = getattr(generator_model.config, "local_window_size", 1)
    global_sink = pipeline._resolve_global_sink_chunk(getattr(cfg, "dmd_generator_global_sink_chunk", None), generator_model)
    chunk_windows = pipeline._get_chunk_windows(
        total_latent_frames=total_frames, chunk_size=chunk_size, window_size=int(window_size), global_sink_chunk=global_sink,
    )
    group_size = getattr(cfg, "dmd_grad_window_chunks", None)
    grouped_backward = use_long_rollout and group_backward_fn is not None and group_size is not None and int(group_size) > 0
    group_size = int(group_size) if grouped_backward else 0
    group_chunks: list[torch.Tensor] = []
    group_ids: list[int] = []
    generated_chunks: list[torch.Tensor] = []
    last_group: torch.Tensor | None = None

    def score_overrides(
        latents: torch.Tensor, chunk_ids: list[int], chunk_frames: list[int],
    ) -> dict[str, Any]:
        history_ids = chunk_windows[chunk_ids[0]]["selected_chunk_ids"][:-1]
        score_chunk_ids = history_ids + chunk_ids
        overrides: dict[str, Any] = {
            "noisy_temporal_ids": pipeline._gather_window_temporal_ids(
                score_chunk_ids, chunk_size, total_frames, device, relative=False,
            ),
        }
        if history_ids:
            overrides["score_history_latent"] = pipeline._gather_window_tensor(
                context_latents, history_ids, chunk_size, total_frames, temporal_dim=2,
            ).detach()
        if use_long_rollout:
            overrides["loss_mask"] = torch.ones_like(latents)
            overrides["loss_normalizer"] = context_latents.numel()
        if source_latents is not None:
            overrides["ref_video_latent"] = _gather_ref_chunks(source_latents, score_chunk_ids, chunk_size, num_source_chunks)
        if controller is not None:
            weights = controller.group_anchor_w(
                chunk_ids, chunk_frames, batch=latents.shape[0], device=device,
            )
            if weights is not None:
                overrides["source_anchor_w_frames"] = weights
        return overrides

    def flush_group() -> torch.Tensor:
        group_latents = torch.cat(group_chunks, dim=2)
        overrides = score_overrides(group_latents, group_ids, [chunk.shape[2] for chunk in group_chunks])
        group_backward_fn(group_latents, overrides)
        return group_latents.detach()

    relative_positions = bool(getattr(cfg, "dmd_rollout_use_relative_temporal_ids", False))
    temporal_kwargs: dict[str, Any] = {"relative": relative_positions}
    max_temporal_ids = getattr(cfg, "max_temporal_ids", None)
    if relative_positions and max_temporal_ids is not None and "max_temporal_ids" in inspect.signature(pipeline._gather_window_temporal_ids).parameters:
        temporal_kwargs["max_temporal_ids"] = max_temporal_ids
    if hasattr(generator_model, "reset_inference_kv_cache"):
        generator_model.reset_inference_kv_cache()

    for chunk_window in chunk_windows:
        chunk_index = chunk_window["chunk_idx"]
        chunk_start, chunk_end = chunk_window["chunk_start"], chunk_window["chunk_end"]
        selected_ids = chunk_window["selected_chunk_ids"]
        active_frames = chunk_end - chunk_start
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        scheduler_sigmas = pipeline.scheduler.sigmas
        exit_index = _sample_exit_index(len(timesteps))
        window_latents = pipeline._gather_window_tensor(context_latents, selected_ids, chunk_size, total_frames, temporal_dim=2).clone()
        temporal_ids = pipeline._gather_window_temporal_ids(selected_ids, chunk_size, total_frames, device, **temporal_kwargs)
        local_start = window_latents.shape[2] - active_frames
        window_source = None
        if source_latents is not None:
            if use_long_rollout:
                window_source = _gather_ref_chunks(source_latents, selected_ids, chunk_size, num_source_chunks)
            else:
                window_source = pipeline._gather_window_tensor(source_latents, selected_ids, chunk_size, total_frames, temporal_dim=2)
        layout = None
        if (
            controller is not None and controller.enabled and window_source is not None
            and not _has_ref_image_latent(reference_latents) and local_start > 0
            and chunk_index % controller.chunk_stride == 0
        ):
            layout = controller.build_layout(
                window_latents=window_latents, model=generator_model,
                local_chunk_start=local_start, local_chunk_len=active_frames,
                chunk_idx=chunk_index, num_chunks=len(chunk_windows),
            )
        active_output = None
        for step_index, timestep in enumerate(timesteps):
            at_exit = step_index == exit_index
            grad_context = torch.enable_grad() if requires_grad and at_exit else torch.no_grad()
            token_timesteps = _expand_per_token_timestep(timestep, window_latents, generator_model, local_chunk_start=local_start)
            with grad_context:
                control_context = controller.armed(layout=layout, sigma_bucket=0 if step_index == 0 else 1) if layout is not None and at_exit else nullcontext()
                with control_context:
                    flow = _run_streaming_model(
                        generator_model, latent_model_input=window_latents, timestep=token_timesteps,
                        prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
                        temporal_ids=temporal_ids, ref_video_latent=window_source, ref_image_latent=reference_latents,
                    )
                active_latents = window_latents[:, :, local_start:]
                active_flow = flow[:, :, local_start:]
                if not at_exit:
                    next_latents = pipeline.scheduler.step(active_flow, timestep, active_latents.clone(), return_dict=False)[0]
                    window_latents[:, :, local_start:] = next_latents
                    continue
                sigma = scheduler_sigmas[step_index].to(device=device, dtype=active_latents.dtype).view(1, 1, 1, 1, 1)
                active_output = active_latents - sigma * active_flow.to(dtype=active_latents.dtype)
                break
        if active_output is None:
            raise RuntimeError(f"DMD rollout produced no output for chunk {chunk_index}.")
        context_latents[:, :, chunk_start:chunk_end] = active_output.detach()
        if not grouped_backward:
            generated_chunks.append(active_output)
            continue
        group_chunks.append(active_output)
        group_ids.append(chunk_index)
        if len(group_chunks) >= group_size:
            last_group = flush_group()
            group_chunks = []
            group_ids = []

    if grouped_backward:
        if group_chunks:
            last_group = flush_group()
        if last_group is None:
            raise RuntimeError("Grouped DMD rollout produced no training segments.")
        generated_latents, overrides = last_group, None
    else:
        generated_latents = torch.cat(generated_chunks, dim=2)
        chunk_ids = list(range(len(chunk_windows)))
        overrides = score_overrides(generated_latents, chunk_ids, [chunk.shape[2] for chunk in generated_chunks])
    if controller is not None:
        controller.finish_rollout()
    return generated_latents, overrides, grouped_backward


def save_dmd_checkpoint(
    generator: torch.nn.Module,
    fake_score: torch.nn.Module,
    save_dir: str | Path,
    step: int,
    global_rank: int,
    cfg: ExpConfig | None = None,
    epoch: int = 0,
    generator_optimizer: torch.optim.Optimizer | None = None,
    fake_score_optimizer: torch.optim.Optimizer | None = None,
    generator_scheduler: Any | None = None,
    fake_score_scheduler: Any | None = None,
    dataloader: Any | None = None,
) -> None:
    save_path = Path(save_dir) / f"global_step{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    if cfg is not None and dmd_uses_lora(cfg):
        try:
            from peft.utils import get_peft_model_state_dict
        except ImportError as exc:
            raise ImportError("PEFT is required to save DMD LoRA checkpoints.") from exc

        generator_adapter_name = dmd_generator_adapter_name(cfg)
        fake_score_adapter_name = dmd_fake_score_adapter_name(cfg)
        activate_lora_adapter(generator, generator_adapter_name)
        model_state_dict = get_model_state_dict(generator, options=_SAVE_OPTIONS)
        generator_lora_state_dict = get_peft_model_state_dict(
            generator,
            state_dict=model_state_dict,
            adapter_name=generator_adapter_name,
        )
        activate_lora_adapter(fake_score, fake_score_adapter_name)
        fake_score_lora_state_dict = get_peft_model_state_dict(
            fake_score,
            state_dict=model_state_dict,
            adapter_name=fake_score_adapter_name,
        )
        activate_lora_adapter(generator, generator_adapter_name)
        adapter_config = generator.peft_config[generator_adapter_name]
        target_modules = adapter_config.target_modules
        state_dict = {
            "step": step,
            "epoch": epoch,
            "dmd_use_lora": True,
            "generator_adapter_name": generator_adapter_name,
            "fake_score_adapter_name": fake_score_adapter_name,
            "generator_lora": generator_lora_state_dict,
            "generator_lora_config": {
                "r": adapter_config.r,
                "lora_alpha": adapter_config.lora_alpha,
                "target_modules": target_modules if isinstance(target_modules, str) else sorted(target_modules),
                "bias": adapter_config.bias,
            },
            "fake_score_lora": fake_score_lora_state_dict,
        }
    else:
        generator_state_dict = get_model_state_dict(generator, options=_SAVE_OPTIONS)
        state_dict = {
            "step": step,
            "epoch": epoch,

            "model": generator_state_dict,
            "fake_score": get_model_state_dict(fake_score, options=_SAVE_OPTIONS),
        }

    if generator_optimizer is not None:
        state_dict["generator_optimizer"] = get_optimizer_state_dict(
            generator, generator_optimizer, options=_SAVE_OPTIONS
        )
    if fake_score_optimizer is not None:
        state_dict["fake_score_optimizer"] = get_optimizer_state_dict(
            fake_score, fake_score_optimizer, options=_SAVE_OPTIONS
        )
    if generator_scheduler is not None:
        state_dict["generator_scheduler"] = generator_scheduler.state_dict()
    if fake_score_scheduler is not None:
        state_dict["fake_score_scheduler"] = fake_score_scheduler.state_dict()

    if global_rank <= 0:
        torch.save(state_dict, save_path / f"step_{step}.pth")

    if dataloader is not None:
        dataloader_dir = save_path / "dataloader"
        dataloader_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"dataloader": dataloader.state_dict()},
            dataloader_dir / f"dataloader_step{step}_rank{global_rank}.pth",
        )


def load_dmd_checkpoint(
    generator: torch.nn.Module,
    fake_score: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    cfg: ExpConfig | None = None,
    generator_optimizer: torch.optim.Optimizer | None = None,
    fake_score_optimizer: torch.optim.Optimizer | None = None,
    generator_scheduler: Any | None = None,
    fake_score_scheduler: Any | None = None,
    dataloader: Any | None = None,
) -> tuple[int, int]:
    path = Path(path)
    if path.name.endswith(".pth"):
        checkpoint_path = path
        path = path.parent
    else:
        ckpt_paths = sorted(path.glob("*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(f"Checkpoint {path} does not contain a .pth file.")
        checkpoint_path = ckpt_paths[0]

    state_dict = torch.load(checkpoint_path, map_location=device)
    if cfg is not None and dmd_uses_lora(cfg):
        try:
            from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
        except ImportError as exc:
            raise ImportError("PEFT is required to load DMD LoRA checkpoints.") from exc

        generator_adapter_name = dmd_generator_adapter_name(cfg)
        fake_score_adapter_name = dmd_fake_score_adapter_name(cfg)
        generator_lora_state_dict = state_dict.get("generator_lora")
        fake_score_lora_state_dict = state_dict.get("fake_score_lora")
        if generator_lora_state_dict is None or fake_score_lora_state_dict is None:
            raise KeyError("DMD LoRA checkpoint must contain `generator_lora` and `fake_score_lora` state dicts.")

        protection_keys = (
            "noise_type_embed",
            "ref_image_type_embed",
            "ref_video_type_embed",
            "img_in_ref.weight",
            "img_in_ref.bias",
        )

        def _snapshot_protection(model: torch.nn.Module) -> dict[str, torch.Tensor]:
            full_state = get_model_state_dict(model, options=_SAVE_OPTIONS)
            return {k: full_state[k] for k in protection_keys if k in full_state}

        gen_state_with_names = _insert_adapter_name_into_state_dict(
            dict(generator_lora_state_dict),
            adapter_name=generator_adapter_name,
            parameter_prefix="lora_",
        )
        fake_state_with_names = _insert_adapter_name_into_state_dict(
            dict(fake_score_lora_state_dict),
            adapter_name=fake_score_adapter_name,
            parameter_prefix="lora_",
        )

        protection_state = _snapshot_protection(generator)

        combined_state_dict = {}
        combined_state_dict.update(protection_state)
        combined_state_dict.update(gen_state_with_names)
        combined_state_dict.update(fake_state_with_names)

        load_result = set_model_state_dict(
            generator, combined_state_dict, options=_LOAD_OPTIONS_PARTIAL,
        )

        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if unexpected:
            from src.utils.logging import get_logger
            get_logger().error(
                f"LoRA resume produced {len(unexpected)} unexpected key(s) — these "
                f"keys exist in the ckpt but have no target in the model. "
                f"(showing up to 20): {unexpected[:20]}"
            )
            raise RuntimeError(
                f"DMD LoRA resume failed: {len(unexpected)} unexpected key(s). "
                "Check that `dmd_lora_target_modules` / `dmd_lora_rank` / adapter "
                "names match between save time and resume time."
            )

        attempted_keys = set(combined_state_dict.keys())
        missing = list(getattr(load_result, "missing_keys", []) or [])
        attempted_missing = [k for k in missing if k in attempted_keys]
        if attempted_missing:
            from src.utils.logging import get_logger
            get_logger().error(
                f"LoRA resume: {len(attempted_missing)} key(s) we tried to load "
                f"are reported missing — they were not applied to the model. "
                f"(showing up to 20): {attempted_missing[:20]}"
            )
            raise RuntimeError(
                "DMD LoRA resume failed: keys we attempted to load were not "
                "applied. PEFT key transformation may be inconsistent with model."
            )

        from src.utils.logging import get_logger
        get_logger().info(
            f"LoRA resume loaded {len(combined_state_dict)} key(s) into model "
            f"(protection={len(protection_state)}, "
            f"generator={len(gen_state_with_names)}, "
            f"fake_score={len(fake_state_with_names)})."
        )

        activate_lora_adapter(generator, generator_adapter_name)
    else:
        generator_state_dict = state_dict.get("generator", state_dict.get("model"))
        if generator_state_dict is None:
            raise KeyError("DMD checkpoint must contain either `generator` or `model` state dict.")
        set_model_state_dict(generator, generator_state_dict, options=_LOAD_OPTIONS)
        set_model_state_dict(fake_score, state_dict["fake_score"], options=_LOAD_OPTIONS)

    if generator_optimizer is not None and "generator_optimizer" in state_dict:
        set_optimizer_state_dict(generator, generator_optimizer, state_dict["generator_optimizer"], options=_LOAD_OPTIONS)
    if fake_score_optimizer is not None and "fake_score_optimizer" in state_dict:
        set_optimizer_state_dict(fake_score, fake_score_optimizer, state_dict["fake_score_optimizer"], options=_LOAD_OPTIONS)
    if generator_scheduler is not None and "generator_scheduler" in state_dict:
        generator_scheduler.load_state_dict(state_dict["generator_scheduler"])
    if fake_score_scheduler is not None and "fake_score_scheduler" in state_dict:
        fake_score_scheduler.load_state_dict(state_dict["fake_score_scheduler"])

    dataloader_path = path / "dataloader"
    if dataloader is not None and dataloader_path.exists():
        global_rank = get_parallel_state().global_rank
        step = path.name.split("step")[-1]
        dataloader_rank_path = dataloader_path / f"dataloader_step{step}_rank{global_rank}.pth"
        if dataloader_rank_path.exists():
            dataloader_state = torch.load(dataloader_rank_path, map_location=device)
            dataloader.load_state_dict(dataloader_state["dataloader"])

    return state_dict.get("step", 0), state_dict.get("epoch", 0)


def _reduce_log_tensor(value: torch.Tensor | float | int, device: torch.device) -> float:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=device)

    value = value.detach().to(device=device, dtype=torch.float32, copy=True)
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


def main(cfg: ExpConfig):
    _local_rank, global_rank, world_size, device = setup_distributed_training(cfg)

    assert cfg.seed is not None, "Seed must be specified in the configuration."
    seed_everything(cfg.seed)
    if getattr(cfg, "cfg_rate", 0.0) != 0.0 and global_rank <= 0:
        print("DMD training usually expects cfg_rate=0.0; current config may blank some prompts.")

    if global_rank <= 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        exp_dir = Path(cfg.output_dir) / f"{cfg.exp_name}_sp{cfg.sp_size}_world{world_size}"
        log_dir = exp_dir / "logs"
        ckpt_dir = exp_dir / "checkpoints"
        for directory in [exp_dir, log_dir, ckpt_dir]:
            os.makedirs(directory, exist_ok=True)
    else:
        exp_dir = log_dir = ckpt_dir = None

    dist.barrier()
    exp_dir = broadcast_item(exp_dir, src=0)
    log_dir = broadcast_item(log_dir, src=0)
    ckpt_dir = broadcast_item(ckpt_dir, src=0)

    logger = setup_logger(log_dir)
    if global_rank <= 0:
        tb_writer = SummaryWriter(log_dir=log_dir)
        with open(exp_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as config_file:
            config_file.write(cfg.to_json_string())
    else:
        tb_writer = None

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

    (
        generator_model,
        fake_score,
        real_score,
        generator_optimizer,
        fake_score_optimizer,
        generator_lr_scheduler,
        fake_score_lr_scheduler,
    ) = setup_dmd_models_and_optimizers(cfg, device)

    pipeline = load_pipeline(cfg, generator_model, device)

    controller = ResilientDMDController(cfg, device)
    init_step, epoch = 0, 0
    if cfg.auto_resume:
        latest_ckpt_dir = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt_dir is not None:
            cfg.resume_from_checkpoint = str(latest_ckpt_dir)
            cfg.resume_optimizer = True
            cfg.resume_dataloader = True
            logger.info(f"Auto-resume: found latest checkpoint folder {cfg.resume_from_checkpoint}")
        else:
            logger.info("No checkpoint folder found.")

    if cfg.resume_from_checkpoint is not None:
        logger.info(f"Resuming DMD checkpoint from: {cfg.resume_from_checkpoint}")
        init_step, epoch = load_dmd_checkpoint(
            generator_model,
            fake_score,
            cfg.resume_from_checkpoint,
            device,
            cfg=cfg,
            generator_optimizer=generator_optimizer if cfg.resume_optimizer else None,
            fake_score_optimizer=fake_score_optimizer if cfg.resume_optimizer else None,
            generator_scheduler=generator_lr_scheduler if cfg.resume_optimizer else None,
            fake_score_scheduler=fake_score_lr_scheduler if cfg.resume_optimizer else None,
            dataloader=train_dataloader if cfg.resume_dataloader else None,
        )
        resume_path = Path(cfg.resume_from_checkpoint)
        resume_directory = resume_path if resume_path.is_dir() else resume_path.parent
        controller_path = resume_directory / "resilient_dmd_controller.pt"
        if not controller_path.exists():
            controller_path = resume_directory / "attn_rho_probe.pt"
        if controller_path.exists():
            controller.load_state_dict(torch.load(controller_path, map_location="cpu", weights_only=True))
            logger.info(f"Resumed Resilient-DMD source controller from {controller_path}")
        else:
            logger.info("No source-controller checkpoint found; starting with static source guidance.")

    total_batch_size = cfg.micro_batch_size * (world_size / cfg.sp_size) * cfg.gradient_accumulation_steps
    logger.info(f"Generator update ratio: {getattr(cfg, 'dmd_fake_gen_update_ratio', 5)}")
    logger.info(f"DMD generator steps: {getattr(cfg, 'dmd_generator_num_inference_steps', None)}")
    logger.info(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    logger.info(f"Micro batch size: {cfg.micro_batch_size}")
    logger.info(f"Total batch size per update: {total_batch_size}")
    if dmd_uses_lora(cfg):
        logger.info("DMD LoRA mode: shared base transformer with separate generator/fake-score adapters.")
        logger.info(f"Generator LoRA adapter: {dmd_generator_adapter_name(cfg)}")
        logger.info(f"Fake-score LoRA adapter: {dmd_fake_score_adapter_name(cfg)}")

    noise_seed = cfg.seed + (get_parallel_state().sp_group_id if sp_enabled() else global_rank)
    noise_generator_cpu = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise_generator_cuda = torch.Generator(device="cuda").manual_seed(noise_seed)
    rng = random.Random(noise_seed)

    step_times: deque[float] = deque(maxlen=100)
    data_times: deque[float] = deque(maxlen=100)
    preprocess_times: deque[float] = deque(maxlen=100)
    gc.disable()

    train_data_iterator = iter(train_dataloader)
    for step in range(init_step + 1, cfg.max_train_steps + 1):
        train_generator = step % getattr(cfg, "dmd_fake_gen_update_ratio", 5) == 0
        log_step = step % cfg.log_interval == 0
        task_loss_stats = torch.zeros((2, len(LOSS_TASK_TYPES), 2), device=device, dtype=torch.float64) if log_step else None
        dmd_grad_stats = torch.zeros(3, device=device, dtype=torch.float64) if log_step and train_generator else None

        step_rollout_num_chunks = _resolve_rollout_num_chunks(cfg, global_rank)

        step_critic_rollout_num_chunks = _resolve_rollout_num_chunks(
            cfg,
            global_rank,
            spec=getattr(cfg, "dmd_critic_rollout_num_chunks", -1),
        )
        record_loss: dict[str, float] = {}
        step_start_time = time.perf_counter()
        last_data_time = 0.0
        last_preprocess_time = 0.0

        if train_generator:
            generator_optimizer.zero_grad(set_to_none=True)
            if dmd_uses_lora(cfg):
                generator_model.train()
                activate_lora_adapter(generator_model, dmd_generator_adapter_name(cfg))
            else:
                generator_model.train()
                fake_score.eval()
                real_score.eval()
            for accumulation_step in range(cfg.gradient_accumulation_steps):
                data_start_time = time.perf_counter()
                batch = next(train_data_iterator)
                last_data_time = time.perf_counter() - data_start_time
                data_times.append(last_data_time)

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
                target_latents, _ = _flatten_multi_item_latents(latents, additional_inputs)
                negative_prompt_embeds, negative_prompt_embeds_mask = encode_negative_prompt(
                    cfg,
                    pipeline,
                    batch_size=target_latents.shape[0],
                    task_type=additional_inputs["task_type"],
                    pixel_values=batch["pixel"],
                    ref_image_pixels=batch.get("ref_image_pixel"),
                )
                last_preprocess_time = time.perf_counter() - preprocess_start_time
                preprocess_times.append(last_preprocess_time)

                grouped_dmd_loss_sum = 0.0

                def _generator_group_backward(group_latents, group_overrides):
                    nonlocal grouped_dmd_loss_sum
                    group_inputs = {**additional_inputs, **group_overrides}
                    group_loss = compute_generator_dmd_loss(
                        cfg,
                        fake_score,
                        real_score,
                        pipeline,
                        group_latents,
                        prompt_embeds,
                        prompt_embeds_mask,
                        negative_prompt_embeds,
                        negative_prompt_embeds_mask,
                        group_inputs,
                        noise_generator_cuda,
                        noise_generator_cpu,
                        gradient_stats=dmd_grad_stats,
                    )
                    group_loss = group_loss / cfg.gradient_accumulation_steps
                    if not torch.isfinite(group_loss):
                        if global_rank == 0:
                            logger.warning(
                                f"Step {step} accumulation {accumulation_step}: "
                                f"non-finite generator loss {group_loss.item()}; skipping group."
                            )
                        return
                    group_loss.backward()
                    grouped_dmd_loss_sum += group_loss.detach() * cfg.gradient_accumulation_steps

                generated_latents, score_overrides, grouped_backward = generate_dmd_latents(
                    cfg,
                    pipeline,
                    generator_model,
                    target_latents,
                    prompt_embeds,
                    prompt_embeds_mask,
                    additional_inputs,
                    noise_generator_cuda,
                    rollout_num_chunks_override=step_rollout_num_chunks,
                    group_backward_fn=_generator_group_backward,
                    controller=controller,
                )

                if grouped_backward:
                    loss_value = grouped_dmd_loss_sum
                else:
                    score_inputs = {**additional_inputs, **(score_overrides or {})}
                    loss = compute_generator_dmd_loss(
                        cfg,
                        fake_score,
                        real_score,
                        pipeline,
                        generated_latents,
                        prompt_embeds,
                        prompt_embeds_mask,
                        negative_prompt_embeds,
                        negative_prompt_embeds_mask,
                        score_inputs,
                        noise_generator_cuda,
                        noise_generator_cpu,
                        gradient_stats=dmd_grad_stats,
                    ) / cfg.gradient_accumulation_steps
                    if not torch.isfinite(loss):
                        if global_rank == 0:
                            logger.warning(
                                f"Step {step} accumulation {accumulation_step}: "
                                f"non-finite generator loss {loss.item()}; skipping micro-batch."
                            )
                        generator_optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    loss_value = loss.detach() * cfg.gradient_accumulation_steps
                if task_loss_stats is not None and torch.is_tensor(loss_value):
                    task_id = LOSS_TASK_TYPES.index(additional_inputs["task_type"])
                    task_loss_stats[0, task_id, 0] += loss_value.detach()
                    task_loss_stats[0, task_id, 1] += 1
                record_loss["Loss/generator_dmd_loss"] = record_loss.get("Loss/generator_dmd_loss", 0.0) + (
                    _reduce_log_tensor(loss_value, device) / cfg.gradient_accumulation_steps
                )

            if step == 1 or step % cfg.grad_check_interval == 0:
                for param in generator_model.parameters():
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any(), "NaN detected in generator gradient."
                        assert not torch.isinf(param.grad).any(), "Inf detected in generator gradient."

            if cfg.max_grad_norm is not None:
                generator_clip_params = (
                    iter_adapter_parameters(generator_model, dmd_generator_adapter_name(cfg))
                    if dmd_uses_lora(cfg)
                    else generator_model.parameters()
                )
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(generator_clip_params, max_norm=cfg.max_grad_norm)
                if log_step:
                    record_loss["Grad/generator_norm"] = _reduce_log_tensor(generator_grad_norm, device)
            generator_optimizer.step()
            generator_lr_scheduler.step()
            generator_optimizer.zero_grad(set_to_none=True)

        fake_score_optimizer.zero_grad(set_to_none=True)
        if dmd_uses_lora(cfg):
            fake_score.train()
            activate_lora_adapter(fake_score, dmd_fake_score_adapter_name(cfg))
        else:
            generator_model.eval()
            fake_score.train()
            real_score.eval()
        for accumulation_step in range(cfg.gradient_accumulation_steps):
            data_start_time = time.perf_counter()
            batch = next(train_data_iterator)
            last_data_time = time.perf_counter() - data_start_time
            data_times.append(last_data_time)

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
            target_latents, _ = _flatten_multi_item_latents(latents, additional_inputs)
            last_preprocess_time = time.perf_counter() - preprocess_start_time
            preprocess_times.append(last_preprocess_time)

            grouped_fake_loss_sum = 0.0

            def _critic_group_backward(group_latents, group_overrides):
                nonlocal grouped_fake_loss_sum
                group_inputs = {**additional_inputs, **group_overrides}
                adapter_context = (
                    lora_adapter_context(fake_score, adapter_name=dmd_fake_score_adapter_name(cfg))
                    if dmd_uses_lora(cfg)
                    else nullcontext()
                )
                with adapter_context:
                    group_loss = compute_fake_score_loss(
                        cfg,
                        fake_score,
                        pipeline,
                        group_latents,
                        prompt_embeds,
                        prompt_embeds_mask,
                        group_inputs,
                        noise_generator_cuda,
                        noise_generator_cpu,
                    )
                    group_loss = group_loss / cfg.gradient_accumulation_steps
                    if not torch.isfinite(group_loss):
                        if global_rank == 0:
                            logger.warning(
                                f"Step {step} accumulation {accumulation_step}: "
                                f"non-finite fake-score loss {group_loss.item()}; skipping group."
                            )
                        return
                    group_loss.backward()
                    grouped_fake_loss_sum += group_loss.detach() * cfg.gradient_accumulation_steps

            generated_latents, score_overrides, grouped_backward = generate_dmd_latents(
                cfg,
                pipeline,
                generator_model,
                target_latents,
                prompt_embeds,
                prompt_embeds_mask,
                additional_inputs,
                noise_generator_cuda,
                requires_grad=False,
                rollout_num_chunks_override=step_critic_rollout_num_chunks,
                group_backward_fn=_critic_group_backward,
            )

            torch.cuda.empty_cache()
            if grouped_backward:
                loss_value = grouped_fake_loss_sum
            else:
                score_inputs = {**additional_inputs, **(score_overrides or {})}
                if dmd_uses_lora(cfg):
                    activate_lora_adapter(fake_score, dmd_fake_score_adapter_name(cfg))
                fake_loss = compute_fake_score_loss(
                    cfg,
                    fake_score,
                    pipeline,
                    generated_latents,
                    prompt_embeds,
                    prompt_embeds_mask,
                    score_inputs,
                    noise_generator_cuda,
                    noise_generator_cpu,
                ) / cfg.gradient_accumulation_steps
                if not torch.isfinite(fake_loss):
                    if global_rank == 0:
                        logger.warning(
                            f"Step {step} accumulation {accumulation_step}: "
                            f"non-finite fake-score loss {fake_loss.item()}; skipping micro-batch."
                        )
                    fake_score_optimizer.zero_grad(set_to_none=True)
                    continue
                fake_loss.backward()
                loss_value = fake_loss.detach() * cfg.gradient_accumulation_steps
            if task_loss_stats is not None and torch.is_tensor(loss_value):
                task_id = LOSS_TASK_TYPES.index(additional_inputs["task_type"])
                task_loss_stats[1, task_id, 0] += loss_value.detach()
                task_loss_stats[1, task_id, 1] += 1
            record_loss["Loss/fake_score_loss"] = record_loss.get("Loss/fake_score_loss", 0.0) + (
                _reduce_log_tensor(loss_value, device) / cfg.gradient_accumulation_steps
            )

        if step == 1 or step % cfg.grad_check_interval == 0:
            for param in fake_score.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), "NaN detected in fake-score gradient."
                    assert not torch.isinf(param.grad).any(), "Inf detected in fake-score gradient."

        if cfg.max_grad_norm is not None:
            fake_clip_params = (
                iter_adapter_parameters(fake_score, dmd_fake_score_adapter_name(cfg))
                if dmd_uses_lora(cfg)
                else fake_score.parameters()
            )
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(fake_clip_params, max_norm=cfg.max_grad_norm)
            if log_step:
                record_loss["Grad/fake_score_norm"] = _reduce_log_tensor(fake_grad_norm, device)
        fake_score_optimizer.step()
        fake_score_lr_scheduler.step()
        fake_score_optimizer.zero_grad(set_to_none=True)

        step_time = time.perf_counter() - step_start_time
        step_times.append(step_time)
        avg_preprocess_time = sum(preprocess_times) / len(preprocess_times)
        avg_data_time = sum(data_times) / len(data_times)
        avg_step_time = sum(step_times) / len(step_times)
        if task_loss_stats is not None:
            dist.all_reduce(task_loss_stats, op=dist.ReduceOp.SUM)
        if dmd_grad_stats is not None:
            dist.all_reduce(dmd_grad_stats, op=dist.ReduceOp.SUM)
        if global_rank <= 0 and log_step:
            for loss_name, task_stats in zip(("generator_dmd_loss", "fake_score_loss"), task_loss_stats.tolist()):
                for task_type, (loss_sum, batch_count) in zip(LOSS_TASK_TYPES, task_stats):
                    if batch_count > 0:
                        record_loss[f"Loss/{loss_name}/{task_type}"] = loss_sum / batch_count
            if dmd_grad_stats is not None:
                grad_abs_sum, grad_squared_sum, grad_count = dmd_grad_stats.tolist()
                if grad_count > 0:
                    record_loss["DMD/grad_abs_mean"] = grad_abs_sum / grad_count
                    record_loss["DMD/grad_rms"] = (grad_squared_sum / grad_count) ** 0.5
            progress_info = {
                "step_time": f"{step_time:.2f}s",
                "data_time": f"{last_data_time:.2f}s",
                "preprocess_time": f"{last_preprocess_time:.2f}s",
                "train_generator": train_generator,
                "rollout_num_chunks": step_rollout_num_chunks,
            }
            progress_info.update(record_loss)
            logger.info(f"Step {step}/{cfg.max_train_steps}: {progress_info}")

            tb_log_info = {
                "LR/generator": generator_lr_scheduler.get_last_lr()[0],
                "LR/fake_score": fake_score_lr_scheduler.get_last_lr()[0],
                "Time/step_time": step_time,
                "Time/data_time": last_data_time,
                "Time/preprocess_time": last_preprocess_time,
                "Time/avg_step_time": avg_step_time,
                "Time/avg_data_time": avg_data_time,
                "Time/avg_preprocess_time": avg_preprocess_time,
                "DMD/rollout_num_chunks": step_rollout_num_chunks,
            }
            tb_log_info.update(record_loss)
            for key, value in tb_log_info.items():
                tb_writer.add_scalar(key, value, step)

        if step % cfg.checkpoint_interval == 0:
            save_dmd_checkpoint(
                generator_model,
                fake_score,
                ckpt_dir,
                step,
                global_rank,
                cfg=cfg,
                epoch=epoch,
                generator_optimizer=generator_optimizer,
                fake_score_optimizer=fake_score_optimizer,
                generator_scheduler=generator_lr_scheduler,
                fake_score_scheduler=fake_score_lr_scheduler,
                dataloader=train_dataloader,
            )
            if global_rank == 0:
                torch.save(
                    controller.state_dict(),
                    Path(ckpt_dir) / f"global_step{step}" / "resilient_dmd_controller.pt",
                )

        if step % cfg.gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

    controller.close()
    if tb_writer is not None:
        tb_writer.close()
    clean_dist_env()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Joyomni MMDiT DMD distillation training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config_class = load_config_class_from_pyfile(args.config)
    cfg = config_class()
    main(cfg)
