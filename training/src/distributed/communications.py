# Adapted from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/distributed/communication_op.py

import torch
import torch.distributed as dist


def broadcast_item(item, src: int = 0):
    if not dist.is_initialized():
        return item

    item_list = [item]
    dist.broadcast_object_list(item_list, src=src)
    return item_list[0]


def all_ranks_finite(value: torch.Tensor) -> bool:
    """Make the same finite-value decision on every rank before backward/step."""
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        value = value.to_local()
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())
