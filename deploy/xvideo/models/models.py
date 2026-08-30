import gc
import os

import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from loguru import logger

from xvideo.models.dit import Transformer3DModel
from xvideo.models.vae import XVAEChunkCausal
from xvideo.models.scheduler import FlowMatchDiscreteScheduler
from xvideo.models.pipeline import Pipeline, PRECISION_TO_TYPE


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_ckpt,
        dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    if cpu_offload:
        # The 7B text encoder runs only at session start / KV reset, so it stays
        # in CPU RAM and streams through the GPU during encode_prompt (~0 resident).
        model = model.eval().requires_grad_(False)
        from accelerate import cpu_offload as _accelerate_cpu_offload
        from xvideo.lowvram import te_pin_memory_enabled

        state_dict = None
        if te_pin_memory_enabled():
            try:
                state_dict = {
                    name: (tensor.pin_memory() if tensor.device.type == "cpu" else tensor)
                    for name, tensor in model.state_dict().items()
                }
                logger.info("Text encoder offload: pinned host copy for faster H2D staging")
            except RuntimeError as exc:
                logger.warning(f"Text encoder offload: pin_memory failed ({exc!r}); using pageable RAM")
                state_dict = None
        _accelerate_cpu_offload(
            model,
            execution_device=torch.device(device),
            state_dict=state_dict,
        )
        logger.info(f"Text encoder: sequential CPU offload enabled (execution device {device})")
    else:
        model = model.to(device).eval().requires_grad_(False)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        text_encoder_ckpt,
        local_files_only=True,
    )
    return tokenizer, model


def _arch_params(arch_config) -> dict:
    return dict(arch_config.get("params", {})) if isinstance(arch_config, dict) else {}


def build_vae(cfg, device: torch.device):
    pretrained = cfg.vae_arch_config["pretrained"]
    vae = XVAEChunkCausal.from_pretrained(
        pretrained,
        torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision],
        device=device,
        **_arch_params(cfg.vae_arch_config),
    )
    return vae


def load_pipeline(cfg, dit, device: torch.device):
    from xvideo.lowvram import te_cpu_offload_enabled

    vae = build_vae(cfg, device)

    te_offload = te_cpu_offload_enabled()
    tokenizer, text_encoder = load_text_encoder(
        torch_dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision],
        device=device,
        cpu_offload=te_offload,
        **_arch_params(cfg.text_encoder_arch_config),
    )

    scheduler = FlowMatchDiscreteScheduler(**_arch_params(cfg.scheduler_arch_config))

    pipeline = Pipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
        **_arch_params(cfg.pipeline_arch_config),
    )
    if te_offload:
        # DiffusionPipeline.to() refuses to move pipelines that contain a
        # sequentially-offloaded model; place the remaining modules manually
        # (the DiT is already on `device` from load_dit).
        pipeline.vae.to(device)
        return pipeline
    return pipeline.to(device)


def _move_dit_to_device_lowvram(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Stage a CPU-built DiT to `device` block-by-block, quantizing to FP8 (and
    releasing bf16) as each block lands, so the ~31GiB bf16 model is never
    resident on the GPU (peak ~= FP8 footprint + one bf16 block)."""
    import time

    from xvideo.models.dit.dit import _fp8_stream_wanted, _maybe_install_fp8_stream

    fp8_streams = [s for s in ("img", "txt") if _fp8_stream_wanted(s)]
    if len(fp8_streams) < 2:
        logger.warning(
            "Low-VRAM DiT load without full FP8 quantization "
            f"(enabled streams: {fp8_streams or 'none'}): non-quantized blocks stay "
            "bf16 and a 32GB card will likely OOM. Set JOYOMNI_FP8_IMG=1 and "
            "JOYOMNI_FP8_TXT=1."
        )

    for name, child in model.named_children():
        if name == "double_blocks":
            continue
        child.to(device)

    started = time.time()
    num_blocks = len(model.double_blocks)
    for i, block in enumerate(model.double_blocks):
        block.to(device)
        for stream in fp8_streams:
            _maybe_install_fp8_stream(block, stream)
        if (i + 1) % 10 == 0 or i + 1 == num_blocks:
            logger.info(
                f"Low-VRAM DiT load: {i + 1}/{num_blocks} blocks staged to {device} "
                f"({time.time() - started:.1f}s)"
            )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(device)
        logger.info(
            f"Low-VRAM DiT load done in {time.time() - started:.1f}s; "
            f"device used {(total_b - free_b) / 2**30:.1f}/{total_b / 2**30:.1f} GiB"
        )
    return model


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    from xvideo.lowvram import low_vram_enabled

    device_obj = torch.device(device)
    low_vram = low_vram_enabled() and device_obj.type == "cuda"

    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(f"Loading DiT checkpoint: {cfg.dit_ckpt}")
        # No mmap on HF Spaces: /data is FUSE, page faults crawl (hours for 30 GiB).
        mmap_ok = "SPACE_ID" not in os.environ
        state_dict = torch.load(cfg.dit_ckpt, map_location="cpu", weights_only=True, mmap=mmap_ok)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    # Low-VRAM: build on meta (no host RAM), assign mmap tensors in place, then
    # stage blocks to the GPU as FP8. A CPU-built bf16 copy plus the 32 GiB pth
    # mmap exceeds 32 GB free host RAM on this box.
    build_device = torch.device("meta") if low_vram else device
    with torch.device(build_device):
        model = Transformer3DModel(
            dtype=dtype, device=build_device, **_arch_params(cfg.dit_arch_config)
        )
    if not low_vram:
        model.to(device=build_device)

    if state_dict is not None:
        for prefix in ("model.", "module.", "transformer."):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }

        load_state_dict = {}
        for k, v in state_dict.items():
            if (
                k == "img_in.weight" and
                hasattr(model, "img_in") and
                model.img_in.weight.shape != v.shape
            ):
                logger.info(f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v = v.reshape_as(v.new_zeros(model.img_in.weight.shape))
            load_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(
            load_state_dict, strict=True, assign=low_vram
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading DiT: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected_keys[:20]}")
        del state_dict, load_state_dict
        gc.collect()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    if low_vram:
        model = _move_dit_to_device_lowvram(model, device_obj)

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
    "build_vae",
]
