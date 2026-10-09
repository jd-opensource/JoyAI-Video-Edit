import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

TRANSFORMER_BLOCK_NAMES = ["double_blocks", "single_blocks"]


def apply_activation_checkpointing(
    module: torch.nn.Module,
    skip_interval: int = 1
) -> torch.nn.Module:
    """Apply activation checkpointing to transformer blocks."""
    return _checkpoint_blocks_with_skip(module, skip_interval)


def _checkpoint_blocks_with_skip(
    module: torch.nn.Module,
    skip_interval: int
) -> torch.nn.Module:
    """Apply checkpointing to transformer blocks with specified skip interval."""
    if skip_interval < 1:
        raise ValueError(
            f"skip_interval must be >= 1, but got {skip_interval}")

    for block_name in TRANSFORMER_BLOCK_NAMES:
        blocks = getattr(module, block_name, None)
        if blocks is None:
            continue

        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if index % skip_interval == 0:
                checkpointed_block = checkpoint_wrapper(
                    block, preserve_rng_state=False)
                blocks.register_module(layer_id, checkpointed_block)

    return module
