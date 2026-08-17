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
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_ckpt,
        dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
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
    vae = build_vae(cfg, device)

    tokenizer, text_encoder = load_text_encoder(
        torch_dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision],
        device=device,
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
    return pipeline.to(device)


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(f"Loading DiT checkpoint: {cfg.dit_ckpt}")
        state_dict = torch.load(cfg.dit_ckpt, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    model = Transformer3DModel(
        dtype=dtype, device=device, **_arch_params(cfg.dit_arch_config)
    )
    model.to(device=device)

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

        missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys when loading DiT: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected_keys[:20]}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
    "build_vae",
]
