import os
from datetime import timedelta

import torch.distributed as dist


class ParallelState:
    """Manages distributed training and sequence parallel state."""

    def __init__(self, global_rank: int = 0, world_size: int = 1):
        self.global_rank = global_rank
        self.world_size = world_size
        self.sp_group = None
        self.sp_size = None
        self.rank_within_sp_group = None
        self.sp_group_id = None

    @property
    def sp_enabled(self) -> bool:
        return self.sp_size is not None and self.sp_size > 1


_parallel_state: ParallelState = None


def print_rank0(message: str) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        print(message)


def init_distributed_environment_and_sequence_parallel(sp_size: int) -> None:
    """Initialize distributed environment and sequence parallel groups."""
    global _parallel_state

    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    _parallel_state = ParallelState(global_rank, world_size)

    print_rank0(
        f"Initializing distributed environment with world_size={world_size}")
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=global_rank,
        timeout=timedelta(minutes=30),
    )

    if sp_size <= 1:
        print_rank0(
            f"Skipping sequence parallel initialization, sp_size={sp_size}")
        return

    print_rank0(f"Initializing sequence parallel state with sp_size={sp_size}")
    _initialize_sequence_parallel_group(sp_size)


def _initialize_sequence_parallel_group(sp_size: int) -> None:
    """Initialize sequence parallel groups."""
    global _parallel_state

    assert _parallel_state.world_size % sp_size == 0, (
        f"world_size must be divisible by sp_size, "
        f"but got world_size: {_parallel_state.world_size}, sp_size: {sp_size}"
    )

    _parallel_state.sp_size = sp_size
    num_sp_groups = _parallel_state.world_size // sp_size

    for group_id in range(num_sp_groups):
        start_rank = group_id * sp_size
        end_rank = start_rank + sp_size
        ranks = list(range(start_rank, end_rank))
        group = dist.new_group(ranks)

        if _parallel_state.global_rank in ranks:
            _parallel_state.sp_group = group
            _parallel_state.rank_within_sp_group = _parallel_state.global_rank - start_rank
            _parallel_state.sp_group_id = group_id


def clean_dist_env() -> None:
    """Clean up distributed environment and parallel state."""
    global _parallel_state

    if _parallel_state is not None:
        _parallel_state = None

    if dist.is_initialized():
        dist.destroy_process_group()


def get_parallel_state() -> ParallelState:
    """Get the global parallel state instance."""
    if _parallel_state is None:
        raise RuntimeError(
            "Parallel state is not initialized. "
            "Please call init_distributed_environment_and_sequence_parallel first."
        )
    return _parallel_state


def sp_enabled() -> bool:
    return _parallel_state is not None and _parallel_state.sp_enabled
