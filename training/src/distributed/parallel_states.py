"""Distributed process setup for data-parallel and FSDP training."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def print_rank0(message: str) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        print(message)


def init_distributed_environment(backend: str = "nccl") -> None:
    if dist.is_initialized():
        return
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print_rank0(f"Initializing distributed environment with world_size={world_size}")
    dist.init_process_group(
        backend=backend, world_size=world_size, rank=global_rank,
        timeout=timedelta(minutes=30),
    )


def setup_distributed_training() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment()
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def clean_dist_env() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
