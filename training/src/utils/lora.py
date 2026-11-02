"""Validate and gather adapter weights without materializing the frozen backbone."""

from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def validate_lora_state_dict(model, weights, adapter_name: str) -> None:
    from peft.utils import get_peft_model_state_dict

    if not isinstance(weights, Mapping):
        raise ValueError(f"LoRA adapter {adapter_name!r} must contain a state dictionary.")
    expected = get_peft_model_state_dict(model, adapter_name=adapter_name, save_embedding_layers=False)
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    mismatched = [
        name for name in expected.keys() & weights.keys()
        if not isinstance(weights[name], torch.Tensor) or weights[name].shape != expected[name].shape
    ]
    if not expected or missing or unexpected or mismatched:
        raise ValueError(
            f"LoRA adapter {adapter_name!r} does not match the model: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={sorted(mismatched)[:20]}."
        )


def gather_lora_state_dict(model, adapter_name: str) -> dict[str, torch.Tensor]:
    from peft.utils import get_peft_model_state_dict

    # FSDP2 state_dict() returns DTensors without gathering the full model.
    adapter_state = get_peft_model_state_dict(model, adapter_name=adapter_name, save_embedding_layers=False)
    is_writer = not dist.is_initialized() or dist.get_rank() == 0
    gathered = {}
    for name, value in adapter_state.items():
        if isinstance(value, DTensor):
            value = value.full_tensor()
        if is_writer:
            gathered[name] = value.detach().cpu()
    return gathered
