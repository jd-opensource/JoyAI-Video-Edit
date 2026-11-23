from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     set_model_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.fsdp import FSDPModule
from torchdata.stateful_dataloader import StatefulDataLoader
from src.utils.logging import get_logger

_SAVE_OPTIONS = StateDictOptions(full_state_dict=True, cpu_offload=True)
_LOAD_OPTIONS = StateDictOptions(
    full_state_dict=True, broadcast_from_rank0=True)


def save_rng_state(path: Path, rngs: dict) -> None:
    numpy_state = np.random.get_state()
    torch.save({
        "rngs": {name: rng.get_state() if isinstance(rng, torch.Generator) else rng.getstate()
                 for name, rng in rngs.items()},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }, path)


def load_rng_state(path: Path, rngs: dict) -> None:
    if not path.is_file():
        get_logger().warning(f"RNG state is missing; random sequences will restart: {path}")
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, rng in rngs.items():
        if isinstance(rng, torch.Generator):
            rng.set_state(state["rngs"][name])
        else:
            rng.setstate(state["rngs"][name])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    match = re.fullmatch(r"global_step(\d+)", path.name)
    if match:
        checkpoint = path / f"step_{int(match.group(1))}.pth"
        if checkpoint.is_file():
            return checkpoint
    candidates = sorted(path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint found in {path}.")
    if len(candidates) != 1:
        raise ValueError(f"Multiple checkpoints found in {path}; specify the checkpoint file.")
    return candidates[0]


def load_dataloader_state(dataloader, checkpoint_path: Path, step: int) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = checkpoint_path.parent / "dataloader" / f"dataloader_step{step}_rank{rank}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Requested dataloader resume state is missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    dataloader.load_state_dict(state["dataloader"])


def save_checkpoint(
    model: torch.nn.Module | FSDPModule,
    save_dir: str | Path,
    step: int,
    global_rank: int,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    dataloader: StatefulDataLoader = None,
    rngs: dict | None = None,
) -> None:
    logger = get_logger()

    save_path = Path(save_dir) / f"global_step{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "step": step,
        "model": get_model_state_dict(model, options=_SAVE_OPTIONS)
    }

    if optimizer is not None:
        state_dict["optimizer"] = get_optimizer_state_dict(
            model, optimizer, options=_SAVE_OPTIONS)

    if scheduler is not None:
        state_dict["scheduler"] = scheduler.state_dict()

    if global_rank <= 0:
        ckpt_path = save_path / f"step_{step}.pth"
        logger.info(f"Saving checkpoint at step {step} to {ckpt_path}")
        torch.save(state_dict, ckpt_path)

    if dataloader is not None:
        dataloader_dir = save_path / 'dataloader'
        dataloader_dir.mkdir(parents=True, exist_ok=True)
        dataloader_path = dataloader_dir / \
            f"dataloader_step{step}_rank{global_rank}.pth"
        state_dict = {"dataloader": dataloader.state_dict()}
        torch.save(state_dict, dataloader_path)
    if rngs is not None:
        save_rng_state(save_path / f"rng_rank{global_rank}.pth", rngs)


def load_checkpoint(
    model: torch.nn.Module | FSDPModule,
    path: str | Path,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    dataloader: torch.utils.data.DataLoader = None,
) -> int:
    logger = get_logger()

    checkpoint_path = resolve_checkpoint_path(path)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, component in (("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None and key not in state_dict:
            raise KeyError(f"Requested resume state {key!r} is missing from {checkpoint_path}.")

    set_model_state_dict(model, state_dict["model"], options=_LOAD_OPTIONS)

    if optimizer is not None:
        set_optimizer_state_dict(
            model, optimizer, state_dict["optimizer"], options=_LOAD_OPTIONS)

    if scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler"])

    if dataloader is not None:
        load_dataloader_state(dataloader, checkpoint_path, state_dict.get("step", 0))

    return state_dict.get("step", 0)
