import math

from einops import rearrange
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from src.models.mmdit.vae.layers import RMSNorm, swish, AttnBlock, DecoderOutput, EncoderOutput, DiagonalGaussianDistribution

CACHE_T = 1


class ChunkCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        chunk_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.chunk_size = chunk_size
        # Padding order: left, right, top, bottom, past, future.
        assert self.padding[0] == 1, "Causal padding only supports padding of 1 in temporal dimension."
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1], 0, 0)
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        padding = list(self._padding)
        if cache_x is not None:
            padding_front = cache_x.to(x.device)
        else:
            assert x.shape[
                2] == 1, f"Input temporal dimension is expected to be 1 when cache_x is None, got {x.shape[2]}."
            padding_front = x
        x = torch.cat([padding_front, x, x[:, :, -1:, :, :]], dim=2)
        x = F.pad(x, padding)
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, chunk_size: int, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = RMSNorm(in_channels, channel_first=True, images=False)
        self.conv1 = ChunkCausalConv3d(
            chunk_size, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = RMSNorm(out_channels, channel_first=True, images=False)
        self.conv2 = ChunkCausalConv3d(
            chunk_size, out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ChunkCausalConv3d(
                chunk_size, in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, feat_cache: Tensor | None = None, feat_idx: int | None = None):
        shortcut = x

        x = self.norm1(x)
        x = swish(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = swish(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        if self.in_channels != self.out_channels:
            shortcut = self.nin_shortcut(shortcut)

        return x + shortcut


class DownsampleBlock(nn.Module):
    def __init__(self, chunk_size: int, in_channels: int, out_channels: int, temporal_downsample: bool):
        super().__init__()
        factor = 2 * 2 * 2 if temporal_downsample else 1 * 2 * 2
        self.conv = ChunkCausalConv3d(
            chunk_size, in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.temporal_downsample = temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor, feat_cache: Tensor | None = None, feat_idx: int | None = None, first_chunk: bool = False) -> Tensor:
        r1 = 2 if self.temporal_downsample else 1

        if self.temporal_downsample and first_chunk:
            shortcut = torch.cat([x[:, :, :1, :, :], x], dim=2)
        else:
            shortcut = x
        shortcut = rearrange(
            shortcut, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv(x)

        if self.temporal_downsample and first_chunk:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)
        else:
            x = x
        x = rearrange(
            x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(
            B, x.shape[1], self.group_size, T, H, W).mean(dim=2)
        return x + shortcut


class UpsampleBlock(nn.Module):
    def __init__(self, chunk_size: int, in_channels: int, out_channels: int, temporal_upsample: bool):
        super().__init__()
        factor = 2 * 2 * 2 if temporal_upsample else 1 * 2 * 2
        self.conv = ChunkCausalConv3d(
            chunk_size, in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.temporal_upsample = temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor, feat_cache: Tensor | None = None, feat_idx: int | None = None, first_chunk: bool = False) -> Tensor:
        r1 = 2 if self.temporal_upsample else 1

        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(
            shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv(x)

        x = rearrange(
            x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)

        x += shortcut
        if self.temporal_upsample and first_chunk:
            x = x[:, :, 1:, :, :]

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
        chunk_size: int,
    ):
        super().__init__()

        self.z_channels = z_channels
        self.block_in_channels = block_in_channels
        self.num_res_blocks = num_res_blocks

        cur_chunk_size = chunk_size
        self.conv_in = ChunkCausalConv3d(
            cur_chunk_size, in_channels, block_in_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(ResidualBlock(
                    cur_chunk_size, in_channels=block_in, out_channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.down_blocks.append(DownsampleBlock(
                    cur_chunk_size, block_in, block_out, temporal_downsample[i_level]))
                if temporal_downsample[i_level]:
                    cur_chunk_size //= 2

        self.mid_blocks = nn.ModuleList([
            ResidualBlock(cur_chunk_size, in_channels=block_in,
                          out_channels=block_in),
            AttnBlock(block_in),
            ResidualBlock(cur_chunk_size, in_channels=block_in,
                          out_channels=block_in),
        ])

        self.norm_out = RMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = ChunkCausalConv3d(
            cur_chunk_size, block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, feat_cache: Tensor | None = None, feat_idx: int | None = None, first_chunk: bool = False) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for block in self.down_blocks:
            if isinstance(block, DownsampleBlock):
                x = block(x, feat_cache=feat_cache,
                          feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        x = self.norm_out(x)
        x = swish(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_upsample: tuple[bool, ...],
        chunk_size: int,
    ):
        super().__init__()

        self.z_channels = z_channels
        self.block_in_channels = block_in_channels
        self.num_res_blocks = num_res_blocks

        cur_chunk_size = chunk_size // (2 ** sum(temporal_upsample[:len(block_in_channels) - 1]))
        block_in = block_in_channels[0]
        self.conv_in = ChunkCausalConv3d(
            cur_chunk_size, z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid_blocks = nn.ModuleList([
            ResidualBlock(cur_chunk_size, in_channels=block_in,
                          out_channels=block_in),
            AttnBlock(block_in),
            ResidualBlock(cur_chunk_size, in_channels=block_in,
                          out_channels=block_in),
        ])

        self.up_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(self.num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(
                    cur_chunk_size, in_channels=block_in, out_channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.up_blocks.append(UpsampleBlock(
                    cur_chunk_size, block_in, block_out, temporal_upsample[i_level]))
                if temporal_upsample[i_level]:
                    cur_chunk_size *= 2

        self.norm_out = RMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = ChunkCausalConv3d(
            cur_chunk_size, block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, feat_cache: Tensor | None = None, feat_idx: int | None = None, first_chunk: bool = False) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        for block in self.up_blocks:
            if isinstance(block, UpsampleBlock):
                x = block(x, feat_cache=feat_cache,
                          feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        x = self.norm_out(x)
        x = swish(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class Stem(nn.Module):
    def __init__(self, channels: int, stride: int = 3, group: int = 2):
        super().__init__()
        self.stride = stride
        self.group = group
        self.proj = nn.Conv2d(
            channels * stride * stride,
            channels * group * group,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, channels, num_frames, height, width = inputs.shape
        if height % self.stride != 0 or width % self.stride != 0:
            return inputs
        output_height = height * self.group // self.stride
        output_width = width * self.group // self.stride
        features = inputs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        features = F.pixel_unshuffle(features, self.stride)
        features = self.proj(features)
        features = F.pixel_shuffle(features, self.group)
        return features.reshape(
            batch_size, num_frames, channels, output_height, output_width
        ).permute(0, 2, 1, 3, 4)


class HeadResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.pw(self.act(self.dw(inputs)))


class Head(nn.Module):
    def __init__(
        self,
        channels: int,
        scale: float = 1.5,
        hidden: int = 32,
        num_blocks: int = 4,
        mid_channels: int = 12,
    ):
        super().__init__()
        self.scale = float(scale)
        self.conv_in = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.blocks = nn.Sequential(*[HeadResBlock(hidden) for _ in range(num_blocks)])
        self.reduce = nn.Conv2d(hidden, mid_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, channels, num_frames, height, width = inputs.shape
        output_height = round(height * self.scale)
        output_width = round(width * self.scale)
        frames = inputs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        features = self.act(self.conv_in(frames))
        features = self.blocks(features)
        features = self.reduce(features)
        features = F.interpolate(
            features, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        residual = self.conv_out(features)
        base = F.interpolate(
            frames, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        output = (base + residual).clamp(-1.0, 1.0)
        return output.reshape(
            batch_size, num_frames, channels, output_height, output_width
        ).permute(0, 2, 1, 3, 4)


class XVAEChunkCausal(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        latent_channels: int,
        layers_per_block: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
        chunk_size: int,
        latents_mean: tuple[float, ...] = None,
        latents_std: tuple[float, ...] = None,
        enable_slicing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.ffactor_temporal = 2 ** sum(temporal_downsample[:len(block_in_channels) - 1])
        self.chunk_size = chunk_size
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        self.stem = Stem(in_channels)
        encoder_spatial_factor = patch_size * 2 ** (len(block_in_channels) - 1)
        if encoder_spatial_factor % self.stem.group:
            raise ValueError("Encoder spatial compression must be divisible by the stem output group.")
        self.ffactor_spatial = encoder_spatial_factor * self.stem.stride // self.stem.group
        if chunk_size <= 0 or chunk_size % self.ffactor_temporal:
            raise ValueError(f"VAE chunk_size must be a positive multiple of {self.ffactor_temporal}.")

        self.encoder = Encoder(
            in_channels=in_channels * (patch_size**2),
            z_channels=latent_channels,
            num_res_blocks=layers_per_block,
            block_in_channels=block_in_channels,
            temporal_downsample=temporal_downsample,
            chunk_size=chunk_size,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels * (patch_size**2),
            num_res_blocks=layers_per_block,
            block_in_channels=tuple(reversed(block_in_channels)),
            temporal_upsample=temporal_downsample,
            chunk_size=chunk_size,
        )

        self.head = Head(out_channels, scale=self.stem.stride / self.stem.group)

        self.use_slicing = enable_slicing

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    @staticmethod
    def patchify(x, patch_size: int) -> Tensor:
        if patch_size == 1:
            return x
        x = rearrange(x, "b c t (h r1) (w r2) -> b (c r1 r2) t h w",
                      r1=patch_size, r2=patch_size)
        return x

    @staticmethod
    def unpatchify(x, patch_size: int) -> Tensor:
        if patch_size == 1:
            return x
        x = rearrange(x, "b (r1 r2 c) t h w -> b c t (h r1) (w r2)",
                      r1=patch_size, r2=patch_size)
        return x

    def clear_cache(self):
        if not hasattr(self, "_enc_conv_num") or not hasattr(self, "_dec_conv_num"):
            self._enc_conv_num = sum(isinstance(m, ChunkCausalConv3d)
                                     for m in self.encoder.modules())
            self._dec_conv_num = sum(isinstance(m, ChunkCausalConv3d)
                                     for m in self.decoder.modules())
        self._enc_conv_idx = [0]
        self._dec_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        self._dec_feat_map = [None] * self._dec_conv_num

    def _validate_input(self, inputs: Tensor, first_chunk: bool):
        if inputs.ndim != 5:
            raise ValueError(f"VAE input must have shape (batch, channels, frames, height, width), got {inputs.shape}.")
        num_frames, height, width = inputs.shape[2:]
        temporal_offset = int(first_chunk)
        if num_frames <= 0 or (num_frames - temporal_offset) % self.ffactor_temporal:
            raise ValueError(f"Temporal dimension must be {self.ffactor_temporal}n+{temporal_offset}, got {num_frames}.")
        if height <= 0 or width <= 0 or height % self.ffactor_spatial or width % self.ffactor_spatial:
            raise ValueError(f"Spatial dimensions must be positive multiples of {self.ffactor_spatial}, got {(height, width)}.")

    def _encode(self, x: Tensor):
        self._validate_input(x, first_chunk=True)
        x = self.stem(x)
        x = self.patchify(x, self.patch_size)

        out = []
        self.clear_cache()
        iter_ = 1 + math.ceil((x.shape[2] - 1) / self.chunk_size)
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                h = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    first_chunk=True
                )
            else:
                h = self.encoder(
                    x[:, :, 1 + (i - 1) * self.chunk_size: 1 +
                      i * self.chunk_size, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    first_chunk=False
                )
            out.append(h)
        out = torch.cat(out, dim=2)
        self.clear_cache()
        return out

    def encode(self, x: Tensor, return_dict: bool = True):
        assert len(
            x.shape) == 5, f"Input tensor must be 5D (b, c, t, h, w), got {x.shape}."

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)

        return EncoderOutput(latent_dist=posterior)

    def encode_init(self):
        self.clear_cache()

    def encode_step(self, x_chunk: Tensor, first_chunk: bool = False) -> Tensor:
        self._validate_input(x_chunk, first_chunk=first_chunk)
        x_chunk = self.stem(x_chunk)
        x_chunk = self.patchify(x_chunk, self.patch_size)
        self._enc_conv_idx = [0]
        return self.encoder(
            x_chunk,
            feat_cache=self._enc_feat_map,
            feat_idx=self._enc_conv_idx,
            first_chunk=first_chunk
        )

    def save_enc_cache(self):
        import copy
        return copy.deepcopy(self._enc_feat_map)

    def restore_enc_cache(self, saved):
        import copy
        self._enc_feat_map = copy.deepcopy(saved)

    def decode_init(self):
        self.clear_cache()

    def decode_step(self, z_chunk: Tensor, first_chunk: bool = False) -> Tensor:
        self._dec_conv_idx = [0]
        h = self.decoder(
            z_chunk,
            feat_cache=self._dec_feat_map,
            feat_idx=self._dec_conv_idx,
            first_chunk=first_chunk
        )
        return self.head(self.unpatchify(h, self.patch_size))

    def save_dec_cache(self):
        import copy
        return copy.deepcopy(self._dec_feat_map)

    def restore_dec_cache(self, saved):
        import copy
        self._dec_feat_map = copy.deepcopy(saved)

    def _decode(self, z: Tensor):
        latent_chunk_size = self.chunk_size // self.ffactor_temporal

        self.clear_cache()
        decoded = []
        iter_ = 1 + math.ceil((z.shape[2] - 1) / latent_chunk_size)
        for i in range(iter_):
            self._dec_conv_idx = [0]
            if i == 0:
                h = self.decoder(
                    z[:, :, :1, :, :],
                    feat_cache=self._dec_feat_map,
                    feat_idx=self._dec_conv_idx,
                    first_chunk=True
                )
            else:
                h = self.decoder(
                    z[:, :, 1 + (i - 1) * latent_chunk_size: 1 +
                      i * latent_chunk_size, :, :],
                    feat_cache=self._dec_feat_map,
                    feat_idx=self._dec_conv_idx,
                    first_chunk=False
                )
            decoded.append(h)
        decoded = torch.cat(decoded, dim=2)
        self.clear_cache()

        decoded = self.unpatchify(decoded, self.patch_size)
        decoded = self.head(decoded)
        return decoded

    def decode(self, z: Tensor, return_dict: bool = True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)
