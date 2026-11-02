"""
Reference code
[FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
[WANX] https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
"""

from dataclasses import dataclass

from einops import rearrange

import torch

from torch import Tensor, nn

import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor

class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, logvar = parameters.chunk(2, dim=1)
        self.std = (0.5 * logvar.clamp(-30.0, 20.0)).exp()

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean

@dataclass
class EncoderOutput:
    latent_dist: DiagonalGaussianDistribution = None

@dataclass
class DecoderOutput:
    sample: torch.FloatTensor = None
    posterior: DiagonalGaussianDistribution | None = None

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = False, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim, )

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma + self.bias

class AttnBlock(nn.Module):
    """Frame-level self-attention"""

    def __init__(self, in_channels: int):
        super().__init__()

        self.norm = RMSNorm(in_channels, channel_first=True, images=False)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: Tensor) -> Tensor:
        b, _, t, h, w = x.shape

        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, "b c t h w -> (b t) 1 (h w) c")
        k = rearrange(k, "b c t h w -> (b t) 1 (h w) c")
        v = rearrange(v, "b c t h w -> (b t) 1 (h w) c")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "(b t) 1 (h w) c -> b c t h w", b=b, t=t, h=h, w=w)

        x = self.proj_out(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attention(x)
