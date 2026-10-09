import torch
import torch.nn as nn


def load_modulation(
        modulate_type: str,
        hidden_size: int,
        factor: int,
        act_layer=nn.SiLU,
        dtype=None,
        device=None):
    factory_kwargs = {"dtype": dtype, "device": device}
    if modulate_type == 'wanx':
        return ModulateWan(hidden_size, factor, **factory_kwargs)
    elif modulate_type == 'adaLN':
        return ModulateDiT(hidden_size, factor, act_layer, **factory_kwargs)
    elif modulate_type == 'jdx':
        return ModulateX(hidden_size, factor, **factory_kwargs)
    else:
        raise ValueError(
            f"Unknown modulation type: {modulate_type}.")


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: nn.SiLU,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.factor = factor
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x)).chunk(self.factor, dim=-1)


class ModulateWan(nn.Module):
    """Modulation layer for WanX."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size,
                        dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


class ModulateX(nn.Module):
    """Modulation layer for WanX."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factor = factor
        _ = hidden_size, dtype, device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in x.chunk(self.factor, dim=1)]


def modulate(x, shift=None, scale=None):
    """Apply shift and scale to the hidden states."""
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    """Apply an element-wise gate to the residual branch output."""
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)
