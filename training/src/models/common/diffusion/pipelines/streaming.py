# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from dataclasses import dataclass
from einops import rearrange
from PIL import Image

from transformers import Qwen2Tokenizer, PreTrainedModel, AutoProcessor

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from src.models.common.diffusion.schedulers import FlowMatchDiscreteScheduler
from src.models.mmdit.dit import Transformer3DModel
from src.models.mmdit.vae import XVAEChunkCausal
from src.utils.constants import PRECISION_TO_TYPE

def apg_delta(
    delta: torch.Tensor,
    ref: torch.Tensor,
    parallel_scale: float = 0.2,
    orthogonal_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project guidance into parallel and orthogonal components."""
    batch_size = delta.shape[0]
    delta_f = delta.reshape(batch_size, -1).float()
    ref_f = ref.reshape(batch_size, -1).float()
    ref_norm_sq = (ref_f * ref_f).sum(dim=1, keepdim=True).clamp_min(eps)
    proj_coeff = (delta_f * ref_f).sum(dim=1, keepdim=True) / ref_norm_sq
    delta_parallel = proj_coeff * ref_f
    delta_orthogonal = delta_f - delta_parallel
    out = parallel_scale * delta_parallel + orthogonal_scale * delta_orthogonal
    return out.reshape_as(delta).to(delta.dtype)


@dataclass
class PipelineOutput(BaseOutput):
    videos: torch.Tensor


class Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: XVAEChunkCausal,
        text_encoder: PreTrainedModel,
        tokenizer: Qwen2Tokenizer,
        transformer: Transformer3DModel,
        scheduler: FlowMatchDiscreteScheduler,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = self.vae.ffactor_spatial
        self.vae_scale_factor_temporal = self.vae.ffactor_temporal


        text_encoder_ckpt = args.text_encoder_arch_config["params"]["text_encoder_ckpt"]
        self.qwen_processor = AutoProcessor.from_pretrained(text_encoder_ckpt, local_files_only=True)

        self.prompt_template_encode = {
            'image': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            'multiple_images': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
            'video': "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        }
        self.prompt_template_encode_start_idx = {
            'image': 34,
            'multiple_images': 34,
            'video': 91,
        }

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        template_type: str = 'image',
        device: Optional[torch.device] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=max_sequence_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        mask = txt_tokens.attention_mask.bool()
        lengths = mask.sum(dim=1)
        hidden = torch.split(encoder_hidden_states.last_hidden_state[mask], lengths.tolist())
        hidden = [sample[drop_idx:] for sample in hidden]
        prompt_embeds = torch.nn.utils.rnn.pad_sequence(hidden, batch_first=True)
        lengths = (lengths - drop_idx).clamp_min(0)
        prompt_mask = torch.arange(prompt_embeds.shape[1], device=device)[None, :] < lengths[:, None]
        return prompt_embeds, prompt_mask.to(txt_tokens.attention_mask.dtype)


    def encode_prompt_multiple_images(
        self,
        prompt: List[str],
        device: Optional[torch.device] = None,
        images: Optional[List[Image.Image]] = None,
        max_sequence_length: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        template = self.prompt_template_encode['multiple_images']
        drop_idx = self.prompt_template_encode_start_idx['multiple_images']
        prompt = [p.replace(
            '<image>\n', '<|vision_start|><|image_pad|><|vision_end|>') for p in prompt]
        prompt = [template.format(p) for p in prompt]

        if images is not None:
            resized = []
            for image in images:
                scale = (512 ** 2 / (image.height * image.width)) ** 0.5
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                resized.append(image.resize(size, Image.BILINEAR))
            images = resized

        inputs = self.qwen_processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_states = encoder_hidden_states.last_hidden_state
        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs['attention_mask'][:, drop_idx:]
        if prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]
        return prompt_embeds, prompt_embeds_mask


    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        images: Optional[List[Image.Image]] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        template_type: str = 'image',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text or reference-image prompts and their attention masks."""
        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if images is not None:
                prompt_embeds, prompt_embeds_mask = self.encode_prompt_multiple_images(
                    prompt=prompt,
                    images=images,
                    device=device,
                    max_sequence_length=max_sequence_length,
                )
            else:
                prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                    prompt,
                    template_type,
                    device,
                    max_sequence_length=max_sequence_length,
                )
        return (
            prompt_embeds[:, :max_sequence_length].repeat_interleave(num_videos_per_prompt, dim=0),
            prompt_embeds_mask[:, :max_sequence_length].repeat_interleave(num_videos_per_prompt, dim=0),
        )


    def check_inputs(
        self,
        prompt,
        height,
        width,
        num_frames,
        images=None,
        reference_visual_content=None,
        negative_prompt=None,
        num_videos_per_prompt=1,
        output_type="pt",
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
    ):
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                f"`height` and `width` must be integers, but got {type(height)} and {type(width)}."
            )
        if height <= 0 or width <= 0:
            raise ValueError(
                f"`height` and `width` must be positive, but got {height} and {width}."
            )
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, but got {height} and {width}."
            )
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError(
                f"`num_frames` must be a positive integer, but got {num_frames}."
            )
        if (num_frames - 1) % self.vae_scale_factor_temporal:
            raise ValueError(f"`num_frames` must be {self.vae_scale_factor_temporal}n+1, but got {num_frames}.")
        if not isinstance(num_videos_per_prompt, int) or num_videos_per_prompt <= 0:
            raise ValueError(
                f"`num_videos_per_prompt` must be a positive integer, but got {num_videos_per_prompt}."
            )
        if output_type not in {"pt", "latent"}:
            raise ValueError(
                f"Unsupported output_type: {output_type}. Supported values are 'pt' and 'latent'."
            )
        if images is not None:
            if not isinstance(images, (list, tuple)):
                raise ValueError(
                    f"`images` must be a list or tuple of PIL images, but got {type(images)}."
                )
            if not all(isinstance(image, Image.Image) for image in images):
                raise ValueError("`images` must contain only `PIL.Image.Image` items.")
        if reference_visual_content is not None:
            if not torch.is_tensor(reference_visual_content):
                raise ValueError(
                    "`reference_visual_content` must be a torch.Tensor, "
                    f"but got {type(reference_visual_content)}."
                )
            if reference_visual_content.ndim not in {5, 6, 7}:
                raise ValueError(
                    "`reference_visual_content` must have shape `(num_refs, c, t, h, w)`, "
                    "`(batch, num_refs, c, t, h, w)`, or `(batch, num_refs, c, 1, t, h, w)`."
                )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif isinstance(prompt, list) and not all(isinstance(p, str) for p in prompt):
            raise ValueError("`prompt` must be a string or a list of strings.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
        if isinstance(negative_prompt, list) and not all(isinstance(p, str) for p in negative_prompt):
            raise ValueError("`negative_prompt` must be a string or a list of strings.")
        if isinstance(prompt, list) and isinstance(negative_prompt, list) and len(prompt) != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt` must have the same batch size as `prompt`, but got {len(negative_prompt)} and {len(prompt)}."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )
        if prompt_embeds is not None and prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(
                "`prompt_embeds_mask` batch size must match `prompt_embeds` batch size."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask.shape[0] != negative_prompt_embeds.shape[0]:
            raise ValueError(
                "`negative_prompt_embeds_mask` batch size must match `negative_prompt_embeds` batch size."
            )

    def normalize_latents(self, latent: torch.Tensor):
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(
                1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            latent = (latent - latents_mean) / latents_std
        else:
            latent = latent * self.vae.config.scaling_factor
        return latent

    def denormalize_latents(self, latent: torch.Tensor):
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(
                1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            latent = latent * latents_std + latents_mean
        else:
            latent = latent / self.vae.config.scaling_factor
        return latent

    def _sample_vae_latents(
        self,
        inputs: torch.Tensor,
        *,
        enable_denormalization: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        dit_chunk_size = self.transformer.config.chunk_size
        total_t = inputs.shape[2]
        if (total_t - 1) % self.vae.ffactor_temporal:
            raise ValueError(f"Reference videos must contain {self.vae.ffactor_temporal}n+1 frames, got {total_t}.")

        if total_t > 1:
            ffactor_t = self.vae.ffactor_temporal
            window_pixels = dit_chunk_size * ffactor_t
            window_frames = 1 + window_pixels
            stride = ffactor_t
            num_latents = (total_t - 1) // stride + 1

            lat_list = []
            for k in range(num_latents):
                if k == 0:
                    window = inputs[:, :, :1]
                else:
                    end_frame = k * stride
                    start_frame = max(0, end_frame - window_pixels)
                    window = inputs[:, :, start_frame:end_frame + 1]
                    pad_needed = window_frames - window.shape[2]
                    if pad_needed > 0:
                        pad = inputs[:, :, :1].expand(-1, -1, pad_needed, -1, -1)
                        window = torch.cat([pad, window], dim=2)

                h = self._encode_vae_single(window, enable_denormalization=enable_denormalization, generator=generator)
                lat_list.append(h[:, :, -1:])

            return torch.cat(lat_list, dim=2)

        return self._encode_vae_single(inputs, enable_denormalization=enable_denormalization, generator=generator)

    def _encode_vae_single(
        self,
        inputs: torch.Tensor,
        *,
        enable_denormalization: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        original_device = inputs.device
        inputs = inputs.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(inputs).latent_dist.sample(generator)
        if enable_denormalization:
            latents = self.normalize_latents(latents)
        return latents.to(original_device)


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        reference_visual_content=None,
        enable_denormalization=True,
    ):
        latent_temporal = (video_length - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            batch_size,
            1,
            num_channels_latents,
            latent_temporal,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        reference_latents = None
        if reference_visual_content is not None:
            if reference_visual_content.ndim == 5:
                reference_visual_content = reference_visual_content.unsqueeze(0)
            elif reference_visual_content.ndim == 7:
                if reference_visual_content.shape[3] != 1:
                    raise ValueError("7D `reference_visual_content` must have a singleton frame-group dimension.")
                reference_visual_content = reference_visual_content.squeeze(3)
            if reference_visual_content.shape[0] != batch_size or reference_visual_content.shape[1] <= 0:
                raise ValueError(f"Reference videos must have batch size {batch_size} and at least one reference.")
            ref_video = rearrange(
                reference_visual_content.to(device=device, dtype=dtype), "b n c t h w -> (b n) c t h w",
            )
            reference_latents = self._sample_vae_latents(
                ref_video, enable_denormalization=enable_denormalization,
            )
            reference_latents = rearrange(
                reference_latents, "(b n) c t h w -> b n c t h w", b=batch_size,
            ).cpu()
            if reference_latents.shape[2:] != shape[2:]:
                raise ValueError(f"Reference latent shape must be {shape[2:]}, got {tuple(reference_latents.shape[2:])}.")
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"Target latent shape must be {shape}, got {tuple(latents.shape)}.")
        else:
            latents = latents.clone()
        return latents.cpu(), reference_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps


    _KV_CACHE_ID_REF_IMAGE = -1

    @staticmethod
    def _kv_cache_memory_id(kind: str, chunk_id: Optional[int] = None) -> int:
        if kind == "clean" and chunk_id is not None:
            return int(chunk_id)
        if kind == "ref_image":
            return Pipeline._KV_CACHE_ID_REF_IMAGE
        raise ValueError(f"Unsupported cache kind or missing chunk id: {kind!r}.")

    @staticmethod
    def _get_chunk_windows(
        total_latent_frames: int,
        chunk_size: int,
        window_size: int,
        global_sink_chunk: bool,
    ) -> List[Dict[str, Any]]:
        """Select the first-chunk sink and the recent causal history."""
        if total_latent_frames <= 0 or chunk_size <= 0 or window_size <= 0:
            raise ValueError("Frame count, chunk size, and window size must be positive.")
        windows = []
        num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size
        for chunk_index in range(num_chunks):
            if global_sink_chunk and chunk_index > 0:
                tail_size = max(window_size - 1, 1)
                tail_start = max(1, chunk_index - tail_size + 1)
                selected_chunk_ids = [0] + list(range(tail_start, chunk_index + 1))
            else:
                window_start = max(0, chunk_index - window_size + 1)
                selected_chunk_ids = list(range(window_start, chunk_index + 1))
            windows.append({
                "chunk_idx": chunk_index,
                "chunk_start": chunk_index * chunk_size,
                "chunk_end": min(total_latent_frames, (chunk_index + 1) * chunk_size),
                "selected_chunk_ids": selected_chunk_ids,
            })
        return windows

    @staticmethod
    def _chunk_frame_bounds(chunk_id: int, chunk_size: int, total_latent_frames: int) -> Tuple[int, int]:
        chunk_start = chunk_id * chunk_size
        chunk_end = min(total_latent_frames, chunk_start + chunk_size)
        return chunk_start, chunk_end

    @classmethod
    def _gather_window_temporal_ids(
        cls,
        selected_chunk_ids: List[int],
        chunk_size: int,
        total_latent_frames: int,
        device: torch.device,
        relative: bool = False,
        max_temporal_ids: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply a ceiling independently of relative window numbering."""
        temporal_ids = []
        for cid in selected_chunk_ids:
            frame_start, frame_end = cls._chunk_frame_bounds(cid, chunk_size, total_latent_frames)
            temporal_ids.append(torch.arange(frame_start, frame_end, device=device, dtype=torch.long))
        temporal_ids = torch.cat(temporal_ids, dim=0)
        if max_temporal_ids is not None:
            if max_temporal_ids < 2 * chunk_size:
                raise ValueError("`max_temporal_ids` must be at least twice `chunk_size` to keep sink/history/active chunks distinct.")
            shift = (temporal_ids.max() - max_temporal_ids).clamp_min(0)
            return (temporal_ids - shift).clamp_min(0)
        return torch.arange(temporal_ids.numel(), device=device) if relative else temporal_ids

    @classmethod
    def _gather_window_tensor(
        cls,
        tensor: torch.Tensor,
        selected_chunk_ids: List[int],
        chunk_size: int,
        total_latent_frames: int,
        temporal_dim: int = 2,
    ) -> torch.Tensor:
        if tensor is None:
            return None
        chunk_slices = []
        for cid in selected_chunk_ids:
            frame_start, frame_end = cls._chunk_frame_bounds(cid, chunk_size, total_latent_frames)
            slicing = [slice(None)] * tensor.ndim
            slicing[temporal_dim] = slice(frame_start, frame_end)
            chunk_slices.append(tensor[tuple(slicing)])
        return torch.cat(chunk_slices, dim=temporal_dim)

    def _prefill_static_reference_kv_cache(
        self,
        model,
        *,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        reference_image_latents: torch.Tensor,
        transformer_dtype: torch.dtype,
    ) -> None:
        """Cache optional reference images; source video and text remain live."""
        reference = reference_image_latents.to(device=model.device, dtype=transformer_dtype)
        for scope, embeddings, mask in (
            ("cond", prompt_embeds, prompt_embeds_mask),
            ("uncond", negative_prompt_embeds, negative_prompt_embeds_mask),
        ):
            if embeddings is None:
                continue
            with model.cache_context(scope):
                model(
                    hidden_states=reference,
                    timestep=reference.new_zeros(reference.shape[0]),
                    encoder_hidden_states=embeddings,
                    encoder_hidden_states_mask=mask,
                    kv_cache_mode="store",
                    kv_cache_scope=scope,
                    kv_cache_chunk_id=self._kv_cache_memory_id("ref_image"),
                    kv_cache_selected_chunk_ids=[],
                    self_attn_input_mode="ref_image_cache",
                    skip_text_stream=True,
                    return_dict=False,
                )


    def _store_clean_chunk_kv_cache(
        self,
        model,
        *,
        clean_chunk_latents: torch.Tensor,
        chunk_temporal_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        active_chunk_id: int,
        history_chunk_ids: Optional[List[int]],
        pre_rope: bool = False,
        cached_temporal_ids: Optional[torch.Tensor] = None,
    ) -> None:
        zero_timestep = clean_chunk_latents.new_zeros(clean_chunk_latents.shape[0])
        selected_chunk_ids = [self._kv_cache_memory_id("clean", cid) for cid in (history_chunk_ids or [])]
        for scope_name, enc_states, enc_mask in (
            ("cond", prompt_embeds, prompt_embeds_mask),
            ("uncond", negative_prompt_embeds, negative_prompt_embeds_mask),
        ):
            if enc_states is None:
                continue
            with model.cache_context(scope_name):
                model(
                    hidden_states=clean_chunk_latents,
                    timestep=zero_timestep,
                    encoder_hidden_states=enc_states,
                    encoder_hidden_states_mask=enc_mask,
                    noisy_temporal_ids=chunk_temporal_ids,
                    cached_temporal_ids=cached_temporal_ids,
                    kv_cache_mode="reuse_store" if selected_chunk_ids else "store",
                    kv_cache_scope=scope_name,
                    kv_cache_chunk_id=self._kv_cache_memory_id("clean", active_chunk_id),
                    kv_cache_selected_chunk_ids=selected_chunk_ids,
                    kv_cache_pre_rope=pre_rope,
                    skip_text_stream=True,
                    return_dict=False,
                )

    def _resolve_streaming_chunk_size(self, explicit_chunk_size: Optional[int]) -> int:
        if explicit_chunk_size is not None:
            chunk_size = explicit_chunk_size
        else:
            chunk_size = self.transformer.config.chunk_size
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
        return chunk_size

    @staticmethod
    def _resolve_global_sink_chunk(
        explicit_global_sink_chunk: Optional[bool],
        transformer,
    ) -> bool:
        if explicit_global_sink_chunk is not None:
            return explicit_global_sink_chunk
        return transformer.config.global_sink_chunk

    @staticmethod
    def _autocast(device: torch.device, dtype: torch.dtype):
        if device.type not in {"cuda", "cpu"}:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)

    def _predict_cached_noise(
        self,
        latents,
        timestep,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        reference_video,
        reference_images,
        temporal_ids,
        cached_temporal_ids,
        active_chunk_id,
        cache_memory_ids,
        dynamic_temporal_ids,
        source_guidance_scale,
    ):
        def predict(scope, embeddings, mask, source):
            with self.transformer.cache_context(scope):
                prediction = self.transformer(
                    hidden_states=latents,
                    timestep=timestep.repeat(latents.shape[0]),
                    encoder_hidden_states=embeddings,
                    encoder_hidden_states_mask=mask,
                    ref_video_latent=source,
                    ref_image_latent=reference_images,
                    noisy_temporal_ids=temporal_ids,
                    cached_temporal_ids=cached_temporal_ids,
                    kv_cache_mode="reuse",
                    kv_cache_scope=scope,
                    kv_cache_chunk_id=active_chunk_id,
                    kv_cache_selected_chunk_ids=cache_memory_ids,
                    kv_cache_pre_rope=dynamic_temporal_ids,
                    return_dict=False,
                )[0]
            return prediction.unsqueeze(1) if prediction.ndim == 5 else prediction

        full_prediction = predict("cond", prompt_embeds, prompt_embeds_mask, reference_video)
        source_delta = None
        if source_guidance_scale != 1.0:
            source_delta = full_prediction - predict("cond", prompt_embeds, prompt_embeds_mask, None)
        prediction = full_prediction
        if self.do_classifier_free_guidance:
            unconditional = predict("uncond", negative_prompt_embeds, negative_prompt_embeds_mask, reference_video)
            prediction = prediction + (self.guidance_scale - 1.0) * apg_delta(full_prediction - unconditional, full_prediction)
        if source_delta is not None:
            prediction = prediction + (source_guidance_scale - 1.0) * apg_delta(source_delta, full_prediction)
        return prediction

    def _decode_chunk(self, latents, previous_frame, enable_denormalization):
        """Hand off one boundary frame through its re-encoded pseudo latent."""
        batch_size = latents.shape[0]
        latents = rearrange(latents, "b n c t h w -> (b n) c t h w")
        if enable_denormalization:
            latents = self.denormalize_latents(latents)
        latents = latents.to(self.vae.device)
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        decoded = []
        for latent_index in range(latents.shape[2]):
            decode_input = latents[:, :, latent_index:latent_index + 1]
            if previous_frame is not None:
                boundary = previous_frame.to(device=self.vae.device, dtype=self.vae.dtype)
                anchor = self.vae.encode(boundary).latent_dist.sample()
                decode_input = torch.cat([anchor, decode_input], dim=2)
            with self._autocast(self.vae.device, vae_dtype):
                pixels = self.vae.decode(decode_input, return_dict=False)[0]
            if previous_frame is not None:
                pixels = pixels[:, :, -self.vae_scale_factor_temporal:]
            previous_frame = pixels[:, :, -1:].clone()
            decoded.append(rearrange(pixels, "(b n) c t h w -> b n c t h w", b=batch_size).cpu())
        return torch.cat(decoded, dim=3), previous_frame

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int,
        images: Optional[List[Image.Image]] = None,
        reference_visual_content: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        source_guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: str = "pt",
        return_dict: bool = True,
        max_sequence_length: int = 4096,
        enable_denormalization: bool = True,
        use_vit: bool = True,
        ref_image_latent: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        chunk_size: Optional[int] = None,
        window_size: int = 1,
        global_sink_chunk: Optional[bool] = None,
        use_relative_temporal_ids: bool = False,
        max_temporal_ids: Optional[int] = None,
        store_clean_only_self: Optional[bool] = None,
    ):
        """Generate pixels or latents with a bounded KV cache and boundary-frame VAE handoff."""
        if store_clean_only_self is None:
            store_clean_only_self = getattr(self.args, "store_clean_only_self", False)
        self.check_inputs(
            prompt, height, width, num_frames,
            images=images,
            reference_visual_content=reference_visual_content,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("`num_inference_steps` must be a positive integer.")
        if not self.transformer.config.causal:
            raise ValueError("Streaming inference requires a chunk-causal transformer.")
        if max_temporal_ids is not None and (type(max_temporal_ids) is not int or max_temporal_ids < 0):
            raise ValueError("`max_temporal_ids` must be a non-negative integer or None.")
        dynamic_temporal_ids = use_relative_temporal_ids or max_temporal_ids is not None
        self._guidance_scale = guidance_scale
        device = self.transformer.device
        target_dtype = PRECISION_TO_TYPE[self.args.dit_precision]
        if isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        template_type = "image" if num_frames == 1 else "video"
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            images=images if use_vit else None,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            template_type=template_type,
        )
        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                image_tokens = "<image>\n" * len(images) if images is not None and use_vit else ""
                negative_prompt = [f"<|im_start|>user\n{image_tokens}<|im_end|>\n"] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                images=images if use_vit else None,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                template_type=template_type,
            )
        else:
            negative_prompt_embeds = negative_prompt_embeds_mask = None

        if reference_visual_content is not None:
            if reference_visual_content.ndim == 5:
                reference_visual_content = reference_visual_content.unsqueeze(0)
            reference_visual_content = reference_visual_content.repeat_interleave(num_videos_per_prompt, dim=0)
        latents, reference_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, self.vae.config.latent_channels,
            height, width, num_frames, prompt_embeds.dtype, device, generator, latents,
            reference_visual_content=reference_visual_content,
            enable_denormalization=enable_denormalization,
        )
        total_latent_frames = latents.shape[3]
        chunk_size = self._resolve_streaming_chunk_size(chunk_size)
        chunk_windows = self._get_chunk_windows(
            total_latent_frames, chunk_size,
            getattr(self.transformer.config, "local_window_size", window_size),
            self._resolve_global_sink_chunk(global_sink_chunk, self.transformer),
        )
        self._num_timesteps = num_inference_steps * len(chunk_windows)
        reference_video = None
        if reference_latents is not None:
            reference_video = rearrange(reference_latents, "b n c t h w -> b c (n t) h w").to(dtype=target_dtype)
        reference_images = ref_image_latent
        if torch.is_tensor(reference_images):
            reference_images = list(reference_images.split(1))
        if reference_images is not None:
            reference_images = [reference.to(dtype=target_dtype) if reference is not None else None for reference in reference_images]
            reference_images = [reference for reference in reference_images for _ in range(num_videos_per_prompt)]
        prefill_reference = None
        if reference_images and all(reference is not None for reference in reference_images) and len({tuple(reference.shape[1:]) for reference in reference_images}) == 1:
            stacked = torch.cat(reference_images, dim=0)
            if stacked.shape[0] == latents.shape[0]:
                prefill_reference = stacked
        static_cache_ids = {self._kv_cache_memory_id("ref_image")} if prefill_reference is not None else set()
        decoded_chunks = []
        previous_frame = None
        self.transformer.reset_inference_kv_cache()
        try:
            if prefill_reference is not None:
                self._prefill_static_reference_kv_cache(
                    self.transformer,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                    reference_image_latents=prefill_reference,
                    transformer_dtype=target_dtype,
                )
            with self.progress_bar(total=self.num_timesteps) as progress_bar:
                for chunk_index, window in enumerate(chunk_windows):
                    chunk_start, chunk_end = window["chunk_start"], window["chunk_end"]
                    selected_ids = window["selected_chunk_ids"]
                    history_ids, active_id = selected_ids[:-1], selected_ids[-1]
                    current_latents = latents[:, :, :, chunk_start:chunk_end].to(device).clone()
                    if dynamic_temporal_ids:
                        window_ids = self._gather_window_temporal_ids(
                            selected_ids, chunk_size, total_latent_frames, device,
                            relative=use_relative_temporal_ids, max_temporal_ids=max_temporal_ids,
                        )
                        active_length = chunk_end - chunk_start
                        temporal_ids = window_ids[-active_length:]
                        history_temporal_ids = window_ids[:-active_length]
                        cached_ids = history_temporal_ids.unsqueeze(0).expand(current_latents.shape[0], -1)
                    else:
                        temporal_ids = self._gather_window_temporal_ids([active_id], chunk_size, total_latent_frames, device)
                        cached_ids = None
                    temporal_ids = temporal_ids.unsqueeze(0).expand(current_latents.shape[0], -1)
                    active_reference = None
                    if reference_video is not None:
                        active_reference = self._gather_window_tensor(
                            reference_video, [active_id], chunk_size, total_latent_frames,
                        ).to(device=device, dtype=target_dtype)
                    cache_ids = [self._kv_cache_memory_id("clean", chunk_id) for chunk_id in history_ids]
                    cache_ids.extend(sorted(static_cache_ids))
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    for timestep in self.scheduler.timesteps:
                        with self._autocast(device, target_dtype):
                            prediction = self._predict_cached_noise(
                                current_latents.to(target_dtype), timestep,
                                prompt_embeds, prompt_embeds_mask,
                                negative_prompt_embeds, negative_prompt_embeds_mask,
                                active_reference, None if prefill_reference is not None else reference_images,
                                temporal_ids, cached_ids, active_id, cache_ids,
                                dynamic_temporal_ids, source_guidance_scale,
                            )
                            current_latents = self.scheduler.step(prediction, timestep, current_latents.clone(), return_dict=False)[0]
                        latents[:, :, :, chunk_start:chunk_end] = current_latents.cpu()
                        progress_bar.update()
                    self.transformer.evict_kv_cache_chunks(set(cache_ids))
                    if output_type == "pt":
                        decoded, previous_frame = self._decode_chunk(current_latents, previous_frame, enable_denormalization)
                        decoded_chunks.append(decoded)
                    store_cached_ids = cached_ids if history_ids and not store_clean_only_self else None
                    self._store_clean_chunk_kv_cache(
                        self.transformer,
                        clean_chunk_latents=current_latents.to(target_dtype),
                        chunk_temporal_ids=temporal_ids,
                        prompt_embeds=prompt_embeds,
                        prompt_embeds_mask=prompt_embeds_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                        active_chunk_id=active_id,
                        history_chunk_ids=[] if store_clean_only_self else history_ids,
                        pre_rope=dynamic_temporal_ids,
                        cached_temporal_ids=store_cached_ids,
                    )
                    next_ids = chunk_windows[chunk_index + 1]["selected_chunk_ids"][:-1] if chunk_index + 1 < len(chunk_windows) else []
                    self.transformer.evict_kv_cache_chunks(static_cache_ids | set(next_ids))
            if output_type == "pt":
                output = torch.cat(decoded_chunks, dim=3)
                output = (output / 2 + 0.5).clamp(0, 1).float().permute(0, 1, 3, 2, 4, 5)
            else:
                output = torch.cat([reference_latents, latents], dim=1) if reference_latents is not None else latents
            return PipelineOutput(videos=output) if return_dict else output
        finally:
            self.transformer.reset_inference_kv_cache()
            self.maybe_free_model_hooks()
