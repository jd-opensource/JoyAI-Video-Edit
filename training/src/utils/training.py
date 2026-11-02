"""Shared loss and optimizer handling for distributed training."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FSDPModule

from src.distributed.communications import all_ranks_finite


def synchronize_batch_frames(batch: dict, device: torch.device) -> dict:
    """Keep chunk-by-chunk forward counts equal when ranks draw different buckets."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return batch
    frames = torch.tensor(batch["pixel"].shape[3], dtype=torch.int64, device=device)
    dist.all_reduce(frames, op=dist.ReduceOp.MIN)
    return {**batch, "pixel": batch["pixel"][:, :, :, :int(frames.item())]}


def backward_if_finite(loss: torch.Tensor, logger, context: str, *, model=None) -> bool:
    if not all_ranks_finite(loss):
        # A skipped backward leaves FSDP's root parameters unsharded. Restore
        # the optimizer's sharded parameter views without dropping prior grads.
        if model is not None:
            for module in reversed(list(model.modules())):
                if isinstance(module, FSDPModule):
                    module.reshard()
        logger.warning(f"{context}: non-finite loss on at least one rank; skipping backward on all ranks.")
        return False
    loss.backward()
    return True


def step_optimizer(optimizer, scheduler, max_grad_norm, successful_backwards: int, logger):
    """Skip empty/non-finite updates and advance the scheduler only after a step."""
    if successful_backwards == 0:
        optimizer.zero_grad(set_to_none=True)
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_norm=float("inf") if max_grad_norm is None else max_grad_norm,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    if all_ranks_finite(grad_norm):
        optimizer.step()
        scheduler.step()
    else:
        logger.warning("Non-finite gradient norm on at least one rank; skipping optimizer and scheduler on all ranks.")
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def log_bucket_batch_sizes(cfg, world_size: int, logger) -> None:
    buckets = list(cfg.bucket_configs or [])
    sizes = sorted({bucket[0] for bucket in buckets})
    logger.info(f"Per-rank bucket batch sizes: {sizes}")
    logger.info(
        "Effective batch size per update = sum of rank-local bucket batch sizes "
        f"across {world_size} ranks and {cfg.gradient_accumulation_steps} accumulation steps."
    )
