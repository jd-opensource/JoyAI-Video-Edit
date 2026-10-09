# Adapted from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/distributed/communication_op.py

import torch
import torch.distributed as dist
from src.distributed.parallel_states import get_parallel_state


def broadcast_within_sp_group(input_: torch.Tensor):
    src = get_parallel_state().sp_group_id * get_parallel_state().sp_size
    dist.broadcast(input_, src=src, group=get_parallel_state().sp_group)


def broadcast_item(item, src: int = 0):
    if not dist.is_initialized():
        return item

    item_list = [item]
    dist.broadcast_object_list(item_list, src=src)
    return item_list[0]
