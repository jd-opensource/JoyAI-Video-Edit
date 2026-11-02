"""Load an offline MiMo decode package; never compile during a PE request."""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("joyomni.pe")
_prepared = False
_bundle = None
_SIGNATURE_FIELDS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "vocab_size", "rms_norm_eps", "hidden_act",
)


def text_signature(model):
    config = model.config.get_text_config()
    return {name: getattr(config, name) for name in _SIGNATURE_FIELDS}


def prepare_mimo_aoti():
    """CPU-only startup work, before ZeroGPU forks or user requests."""
    global _prepared, _bundle
    if _prepared:
        return
    _prepared = True
    if os.environ.get("JOYOMNI_PE_AOTI", "1").lower() not in {"1", "true", "yes", "on"}:
        return
    root = Path(os.environ.get(
        "JOYOMNI_PE_AOTI_DIR",
        str(Path(__file__).resolve().parents[2] / "deps" / "aoti" / "mimo_decode2048"),
    ))
    package = root / "package.pt2"
    if not package.is_file():
        return
    try:
        from spaces.zero.torch.aoti import LazyAOTIModel
        metadata = json.loads((root / "manifest.json").read_text())
        if metadata["format"] != "mimo_decode_v1":
            raise ValueError("Unknown MiMo AOTI package format")
        # Constructor probes CPU ISA only; __call__ loads the CUDA package later.
        LazyAOTIModel(str(package))
        _bundle = (package, metadata, LazyAOTIModel)
        logger.info("MiMo AOTI CPU preparation complete: %s", package)
    except Exception:
        logger.exception("MiMo AOTI preparation failed; keeping eager PE")


class _Decoder:
    def __init__(self, model):
        import torch

        package, metadata, loader = _bundle
        device = model.lm_head.weight.device
        if device.type != "cuda" or model.lm_head.weight.dtype != torch.bfloat16:
            raise ValueError("MiMo AOTI requires a resident CUDA BF16 encoder")
        if torch.__version__.split("+")[0] != metadata["torch_version"]:
            raise ValueError("MiMo AOTI PyTorch version mismatch")
        if list(torch.cuda.get_device_capability(device)) != metadata["cuda_capability"]:
            raise ValueError("MiMo AOTI GPU architecture mismatch")
        if text_signature(model) != metadata["text_config"]:
            raise ValueError("MiMo AOTI model configuration mismatch")
        self.capacity = metadata["capacity"]
        language = model.model.language_model
        weights = {
            "language_model." + name: tensor
            for name, tensor in list(language.named_parameters()) + list(language.named_buffers())
        }
        weights["lm_head.weight"] = model.lm_head.weight
        weights["slots"] = torch.arange(self.capacity, device=device).view(1, 1, 1, -1)
        self.runner = loader(str(package)).with_weights(weights)

    def generate(self, model, inputs, max_tokens, should_stop):
        import torch

        output = model(**inputs, use_cache=True, logits_to_keep=1)
        length = inputs.input_ids.shape[1]
        token = output.logits[:, -1].argmax(-1).view(1, 1)
        position = torch.tensor([length], device=token.device)
        rope = (position + model.model.rope_deltas).view(1, 1, 1).expand(3, 1, 1).clone()
        keys, values = [], []
        for layer in output.past_key_values.layers:
            shape = (*layer.keys.shape[:2], self.capacity, layer.keys.shape[-1])
            key = torch.zeros(shape, device=layer.keys.device, dtype=layer.keys.dtype)
            value = torch.zeros_like(key)
            key[:, :, :length].copy_(layer.keys)
            value[:, :, :length].copy_(layer.values)
            keys.append(key)
            values.append(value)
        del output
        tokens = [token.item()]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        while tokens[-1] not in eos and len(tokens) < max_tokens:
            if should_stop():
                raise TimeoutError("MiMo AOTI generation cancelled")
            token = self.runner(token, position, rope, keys, values)
            tokens.append(token.item())
            position.add_(1)
            rope.add_(1)
        return torch.tensor(tokens, dtype=torch.long)


def generate_mimo_aoti(model, inputs, max_tokens, should_stop):
    """Return CPU output tokens, or None when this request should use eager."""
    if _bundle is None:
        return None
    if inputs.input_ids.shape[0] != 1:
        return None
    capacity = _bundle[1]["capacity"]
    if inputs.input_ids.shape[1] + max_tokens > capacity:
        return None
    if not bool(inputs.attention_mask.all()):
        return None
    decoder = getattr(model, "_pe_aoti_decoder", None)
    if decoder is False:
        return None
    if decoder is None:
        try:
            decoder = _Decoder(model)
        except Exception:
            logger.exception("MiMo AOTI unavailable for this encoder; keeping eager PE")
            model._pe_aoti_decoder = False
            return None
        model._pe_aoti_decoder = decoder
    if should_stop():
        raise TimeoutError("MiMo AOTI cancelled before prefill")
    try:
        return decoder.generate(model, inputs, max_tokens, should_stop)
    except TimeoutError:
        raise
    except Exception:
        # Avoid repeating a broken package load on every session. The caller
        # drains this request and falls back to the raw prompt as usual.
        model._pe_aoti_decoder = False
        logger.exception("MiMo AOTI generation failed; disabling it for this encoder")
        raise
