import os
import glob
import torch
import torch.distributed as dist
from collections.abc import Sequence

from src.utils.fsdp_load import maybe_load_fsdp_model, pt_weights_iterator, safetensors_weights_iterator
from src.utils.logging import get_logger
from src.utils.constants import PRECISION_TO_TYPE
from src.utils.utils import build_from_config, get_obj_from_str


def _uses_pretrained_loading(config: dict | None) -> bool:
    if config is None or config.get("pretrained") is None:
        return False

    target = str(config.get("target", ""))
    try:
        cls = get_obj_from_str(target)
    except Exception:
        return target.startswith("diffusers.")

    return hasattr(cls, "from_pretrained")


def _normalize_train_modules(train_modules: str | Sequence[str] | None) -> list[str] | None:
    if train_modules is None:
        return None
    if isinstance(train_modules, str):
        train_modules = [train_modules]

    normalized = [module_name.strip() for module_name in train_modules if module_name and module_name.strip()]
    return normalized or None


def _configure_dit_trainable_modules(model: torch.nn.Module, cfg) -> None:
    logger = get_logger()
    train_modules = _normalize_train_modules(getattr(cfg, "dit_train_modules", None))

    if train_modules is None:
        model.requires_grad_(True)
        return

    model.requires_grad_(False)
    named_modules = dict(model.named_modules())
    missing_modules = [module_name for module_name in train_modules if module_name not in named_modules]
    if missing_modules:
        available_modules = [module_name for module_name in named_modules.keys() if module_name]
        preview = ", ".join(available_modules[:20])
        if len(available_modules) > 20:
            preview = f"{preview}, ..."
        raise ValueError(
            f"Unknown dit_train_modules: {missing_modules}. "
            f"Available DiT modules include: {preview}"
        )

    for module_name in train_modules:
        named_modules[module_name].requires_grad_(True)

    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_param_names:
        raise ValueError(f"No trainable DiT parameters were enabled for dit_train_modules={train_modules}.")

    logger.info(f"Training {len(trainable_param_names)} parameter tensors.")


def load_pipeline(cfg, dit, device: torch.device):
    """Build the frozen VAE, text encoder, scheduler, and editing pipeline."""
    vae = build_from_config(
        cfg.vae_arch_config, torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision], device=device,
    )
    tokenizer, text_encoder = build_from_config(
        cfg.text_encoder_arch_config, torch_dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision], device=device,
    )
    pipeline = build_from_config(
        cfg.pipeline_arch_config or {"target": "src.models.common.diffusion.pipelines.Pipeline"},
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=build_from_config(cfg.scheduler_arch_config),
        args=cfg,
    )
    return pipeline.to(device)


def _read_dit_checkpoint(path, checkpoint_type):
    if checkpoint_type == "pt":
        state = dict(pt_weights_iterator([path]))
    elif checkpoint_type == "safetensor":
        files = [str(path)] if os.path.isfile(path) else sorted(glob.glob(os.path.join(str(path), "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No safetensors checkpoints found at {path}.")
        state = dict(safetensors_weights_iterator(files))
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type!r}.")
    return state.get("model", state)


def _merge_generator_lora(model, checkpoint, cfg):
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    weights = checkpoint["generator_lora"]
    adapter_config = checkpoint.get("generator_lora_config")
    if adapter_config is None:
        ranks = {weight.shape[0] for name, weight in weights.items() if ".lora_A." in name}
        if len(ranks) != 1:
            raise ValueError("The generator checkpoint must contain a uniform-rank LoRA adapter.")
        rank = ranks.pop()
        adapter_config = {
            "r": rank,
            "lora_alpha": getattr(cfg, "dmd_lora_alpha", rank),
            "target_modules": sorted({name.split(".lora_")[0] for name in weights if ".lora_" in name}),
            "bias": getattr(cfg, "dmd_lora_bias", "none"),
        }
    adapter_name = checkpoint.get("generator_adapter_name", "generator")
    model.add_adapter(LoraConfig(**adapter_config), adapter_name=adapter_name)
    result = set_peft_model_state_dict(model, weights, adapter_name=adapter_name)
    missing_adapter_keys = [name for name in result.missing_keys if ".lora_" in name]
    if missing_adapter_keys or result.unexpected_keys:
        raise ValueError("Generator LoRA weights do not match the configured model.")
    model.fuse_lora(adapter_names=[adapter_name], safe_fusing=True)
    model.unload_lora()


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    """Load DiT model with FSDP support."""
    logger = get_logger()

    state_dict = _read_dit_checkpoint(cfg.dit_ckpt, cfg.dit_ckpt_type) if cfg.dit_ckpt is not None else None
    adapter_checkpoint = None
    if state_dict is not None and "generator_lora" in state_dict:
        adapter_checkpoint = state_dict
        base_path = getattr(cfg, "dmd_base_ckpt", None) or getattr(cfg, "dmd_stage1_ckpt", None)
        if not base_path or os.path.abspath(base_path) == os.path.abspath(cfg.dit_ckpt):
            raise ValueError("Set JOYAI_STAGE1_CKPT or dmd_base_ckpt to the SFT base for generator LoRA inference.")
        state_dict = _read_dit_checkpoint(base_path, getattr(cfg, "dmd_base_ckpt_type", "pt"))

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    use_pretrained_loading = _uses_pretrained_loading(cfg.dit_arch_config)
    if use_pretrained_loading:
        model_kwargs = {'torch_dtype': dtype, 'device': device}
    else:
        model_kwargs = {'dtype': dtype, 'device': device, 'args': cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    if hasattr(model, "materialize_meta_modules"):
        model.materialize_meta_modules()
    if not dist.is_initialized() or dist.get_world_size() == 1:
        model.to(device=device)

    if state_dict is not None:
        for prefix in ("model.", "module.", "transformer."):
            stripped_state_dict = {}
            matched = 0
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    stripped_state_dict[k[len(prefix):]] = v
                    matched += 1
                else:
                    stripped_state_dict[k] = v
            if matched > 0:
                state_dict = stripped_state_dict

        load_state_dict = {}
        for k, v in state_dict.items():
            if (
                k == "img_in.weight"
                and hasattr(model, "img_in")
                and model.img_in.weight.shape != v.shape
            ):
                v_new = v.new_zeros(model.img_in.weight.shape)
                v = v.reshape_as(v_new)
            if (
                k == "target_type_embed"
                and hasattr(model, "noise_type_embed")
            ):
                load_state_dict["noise_type_embed"] = v
            if (
                k == "cond_type_embed"
                and hasattr(model, "ref_video_type_embed")
            ):
                load_state_dict["ref_video_type_embed"] = v

            load_state_dict[k] = v
        strict = not use_pretrained_loading
        missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=strict)
        if missing_keys:
            logger.warning(f"Missing keys when loading DiT: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected_keys[:20]}")

    if adapter_checkpoint is not None:
        _merge_generator_lora(model, adapter_checkpoint, cfg)
    _configure_dit_trainable_modules(model, cfg)

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

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(
            f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
]
