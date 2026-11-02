from pathlib import Path

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


def save_checkpoint(
    model: torch.nn.Module | FSDPModule,
    save_dir: str | Path,
    step: int,
    global_rank: int,
    epoch: int = 0,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    dataloader: StatefulDataLoader = None
) -> None:
    logger = get_logger()

    save_path = Path(save_dir) / f"global_step{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "step": step,
        "epoch": epoch,
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
        save_path = save_path / 'dataloader'
        save_path.mkdir(parents=True, exist_ok=True)
        dataloader_path = save_path / \
            f"dataloader_step{step}_rank{global_rank}.pth"
        state_dict = {"dataloader": dataloader.state_dict()}
        torch.save(state_dict, dataloader_path)


def load_checkpoint(
    model: torch.nn.Module | FSDPModule,
    path: str | Path,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    dataloader: torch.utils.data.DataLoader = None,
) -> tuple[int, int]:
    logger = get_logger()

    if isinstance(path, str):
        path = Path(path)

    if path.suffix == '.pth':
        checkpoint_path = path
        path = path.parent
    else:
        ckpt_paths = sorted(path.glob("*.pth"))
        if len(ckpt_paths) < 1:
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")
        checkpoint_path = ckpt_paths[0]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    set_model_state_dict(model, state_dict["model"], options=_LOAD_OPTIONS)

    if optimizer is not None and "optimizer" in state_dict:
        set_optimizer_state_dict(
            model, optimizer, state_dict["optimizer"], options=_LOAD_OPTIONS)

    if scheduler is not None and "scheduler" in state_dict:
        scheduler.load_state_dict(state_dict["scheduler"])

    dataloader_path = Path(path / "dataloader")
    if dataloader is not None and dataloader_path.exists():
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        step = state_dict.get("step", 0)
        dataloader_step_rank_path = dataloader_path / \
            f"dataloader_step{step}_rank{global_rank}.pth"
        _state_dict = torch.load(
            dataloader_step_rank_path, map_location="cpu", weights_only=False)
        dataloader_rank = _state_dict["dataloader"]
        dataloader.load_state_dict(dataloader_rank)

    return state_dict.get("step", 0), state_dict.get("epoch", 0)
