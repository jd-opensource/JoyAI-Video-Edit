import torch

def load_optimizer(optimizer_name, model, cfg):
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found when building optimizer.")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )
    raise ValueError(
        f"Unknown optimizer name: {optimizer_name}, must be 'adamw'"
    )


__all__ = [
    "load_optimizer",
]
