# Adapted from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/models/loader/

from collections.abc import Callable, Generator, Sequence

from tqdm import tqdm
import torch
from torch import nn
from torch.distributed import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from safetensors.torch import safe_open
from src.utils.logging import get_logger


_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def safetensors_weights_iterator(
    hf_weights_files: Sequence[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield `(name, tensor)` pairs from safetensors checkpoint shards."""
    enable_tqdm = not torch.distributed.is_initialized(
    ) or torch.distributed.get_rank() == 0
    device = "cpu"
    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="pt", device=device) as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param


def pt_weights_iterator(
    hf_weights_files: Sequence[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield `(name, tensor)` pairs from `.pt` checkpoint shards."""
    device = "cpu"
    enable_tqdm = not torch.distributed.is_initialized(
    ) or torch.distributed.get_rank() == 0
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(bin_file, map_location=device, weights_only=True)
        yield from state.items()
        del state


def maybe_load_fsdp_model(
    model: nn.Module,
    hsdp_shard_dim: int,
    reshard_after_forward: bool,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    output_dtype: torch.dtype | None = None,
    pin_cpu_memory: bool = True,
) -> nn.Module:
    """Apply FSDP when the distributed world size exceeds one."""
    mp_policy = MixedPrecisionPolicy(param_dtype,
                                     reduce_dtype,
                                     output_dtype,
                                     cast_forward_inputs=False)

    if not torch.distributed.is_initialized():
        return model

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return model

    if world_size % hsdp_shard_dim != 0:
        raise AssertionError(
            f"world_size {world_size} must be divisible by hsdp_shard_dim {hsdp_shard_dim}"
        )
    hsdp_replicate_dim = world_size // hsdp_shard_dim

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
        mesh_dim_names=("replicate", "shard"),
    )
    shard_model(model,
                cpu_offload=cpu_offload,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                mesh=device_mesh,
                fsdp_shard_conditions=getattr(model, "_fsdp_shard_conditions", None),
                pin_cpu_memory=pin_cpu_memory)

    return model


def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa: B008
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] | None = None,
    pin_cpu_memory: bool = True,
) -> None:
    """Shard child modules before their parents with FSDP2."""

    if fsdp_shard_conditions is None or len(fsdp_shard_conditions) == 0:
        logger = get_logger()
        logger.warning(
            "The FSDP shard condition list is empty or None. No modules will be sharded in %s",
            type(model).__name__)
        return

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(
            pin_memory=pin_cpu_memory)

    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any([
                shard_condition(n, m)
                for shard_condition in fsdp_shard_conditions
        ]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    fully_shard(model, **fsdp_kwargs)
