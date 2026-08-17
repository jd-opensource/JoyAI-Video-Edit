from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from PIL import Image

from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xvideo.models.dit import Transformer3DModel, SELF_ATTN_MODE_REF_IMAGE_CACHE

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


class Pipeline(DiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: Transformer3DModel,
        scheduler: KarrasDiffusionSchedulers,
        args=None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = self.vae.ffactor_spatial
        self.vae_scale_factor_temporal = self.vae.ffactor_temporal

        text_encoder_ckpt = dict(args.text_encoder_arch_config.get("params", {}))['text_encoder_ckpt']
        self.qwen_processor = AutoProcessor.from_pretrained(text_encoder_ckpt, use_fast=True)

        self.prompt_template_encode = {
            'image': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            'multiple_images': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
            'video': "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        }

        _user_id = self.tokenizer.convert_tokens_to_ids("user")
        _im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        _img_prefix = self.tokenizer(self.prompt_template_encode["image"].split("{}")[0]).input_ids
        _vid_prefix = self.tokenizer(self.prompt_template_encode["video"].split("{}")[0]).input_ids
        _image_idx = _img_prefix.index(_user_id)
        self.prompt_template_encode_start_idx = {
            "image": _image_idx,
            "multiple_images": _image_idx,
            "video": max(i for i, x in enumerate(_vid_prefix) if x == _im_start),
        }

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

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
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(
            e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = min(
            max_sequence_length,
            max(u.size(0) for u in split_hidden_states),
        )
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
             for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
             for u in attn_mask_list]
        )

        return prompt_embeds, encoder_attention_mask

    def encode_prompt_multiple_images(
        self,
        prompt: List[str],
        device: Optional[torch.device] = None,
        images: Optional[torch.Tensor | List[Image.Image] | List[torch.Tensor]] = None,
        template_type: str = 'multiple_images',
        max_sequence_length: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        if template_type != 'multiple_images':
            raise ValueError(
                f"template_type must be 'multiple_images' for image-conditioned prompts, got {template_type!r}"
            )
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        prompt = [p.replace(
            '<image>\n', '<|vision_start|><|image_pad|><|vision_end|>') for p in prompt]
        prompt = [template.format(p) for p in prompt]

        VIT_FIXED_SIZE = 512
        target_area = VIT_FIXED_SIZE * VIT_FIXED_SIZE

        def _vit_hw(h: int, w: int) -> Tuple[int, int]:
            scale = (target_area / max(h * w, 1)) ** 0.5
            return max(1, int(round(h * scale))), max(1, int(round(w * scale)))

        if images is not None:
            import torch.nn.functional as _F

            def _resize_tensor(t: torch.Tensor) -> torch.Tensor:
                squeeze = t.dim() == 3
                if squeeze:
                    t = t.unsqueeze(0)
                new_h, new_w = _vit_hw(t.shape[-2], t.shape[-1])
                t = _F.interpolate(
                    t.float(), size=(new_h, new_w),
                    mode="bilinear", align_corners=False,
                ).to(t.dtype)
                if squeeze:
                    t = t.squeeze(0)
                return t

            if isinstance(images, torch.Tensor):
                images = _resize_tensor(images)
            elif isinstance(images, list):
                resized = []
                for img in images:
                    if isinstance(img, Image.Image):
                        new_h, new_w = _vit_hw(img.height, img.width)
                        resized.append(img.resize((new_w, new_h), Image.BILINEAR))
                    elif isinstance(img, torch.Tensor):
                        resized.append(_resize_tensor(img))
                    else:
                        resized.append(img)
                images = resized

        inputs = self.qwen_processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            **inputs,
            output_hidden_states=True,
        )
        last_hidden_states = encoder_hidden_states.hidden_states[-1]
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
        if images is not None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt_multiple_images(
                prompt=prompt,
                images=images,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        else:
            device = device or self._execution_device

            prompt = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

            if prompt_embeds is None:
                prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                    prompt,
                    template_type,
                    device,
                    max_sequence_length=max_sequence_length,
                )
            prompt_embeds = prompt_embeds[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1)
            prompt_embeds_mask = prompt_embeds_mask.repeat(
                1, num_videos_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(
                batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

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
    ) -> torch.Tensor:
        is_causal = getattr(self.transformer.config, "causal", False)
        dit_chunk_size = getattr(self.transformer.config, "chunk_size", None)
        total_t = inputs.shape[2]

        use_chunkwise = (
            is_causal and
            dit_chunk_size is not None and
            dit_chunk_size > 0 and
            total_t > 1
        )

        if use_chunkwise:
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

                h = self._encode_vae_single(window, enable_denormalization=enable_denormalization)
                lat_list.append(h[:, :, -1:])

            return torch.cat(lat_list, dim=2)

        return self._encode_vae_single(inputs, enable_denormalization=enable_denormalization)

    def _encode_vae_single(
        self,
        inputs: torch.Tensor,
        *,
        enable_denormalization: bool,
    ) -> torch.Tensor:
        original_device = inputs.device
        inputs = inputs.to(self.vae.device)

        from xvideo.models.vae import vae_compile as _vc
        inputs = _vc.prep_input(inputs)

        latents = self.vae.encode(inputs).latent_dist.sample()
        if enable_denormalization:
            latents = self.normalize_latents(latents)
        return latents.to(original_device)

    _KV_CACHE_ID_REF_IMAGE = -1

    @staticmethod
    def _kv_cache_memory_id(kind: str, chunk_id: Optional[int] = None) -> int:
        if kind == "clean":
            if chunk_id is None:
                raise ValueError("`chunk_id` is required for clean cache ids.")
            return int(chunk_id)
        if kind == "ref_image":
            return Pipeline._KV_CACHE_ID_REF_IMAGE
        raise ValueError(f"Unsupported cache kind: {kind!r}")

    @staticmethod
    def _get_chunk_windows(
        total_latent_frames: int,
        chunk_size: int,
        window_size: int,
        global_sink_chunk: bool,
    ) -> List[Dict[str, Any]]:
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}.")

        windows = []
        num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(total_latent_frames, chunk_start + chunk_size)
            if global_sink_chunk and chunk_idx > 0:
                tail_window_size = max(window_size - 1, 1)
                tail_chunk_start = max(1, chunk_idx - tail_window_size + 1)
                selected_chunk_ids = [0] + list(range(tail_chunk_start, chunk_idx + 1))
            else:
                window_chunk_start = max(0, chunk_idx - window_size + 1)
                selected_chunk_ids = list(range(window_chunk_start, chunk_idx + 1))

            windows.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "selected_chunk_ids": selected_chunk_ids,
                }
            )
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
        max_temporal_ids: Optional[int] = None,
    ) -> torch.Tensor:
        if max_temporal_ids is not None:
            abs_ids = []
            for cid in selected_chunk_ids:
                frame_start, frame_end = cls._chunk_frame_bounds(cid, chunk_size, total_latent_frames)
                abs_ids.append(torch.arange(frame_start, frame_end, device=device, dtype=torch.long))
            abs_ids = torch.cat(abs_ids, dim=0)
            shift = (abs_ids.max() - int(max_temporal_ids)).clamp_min(0)
            return (abs_ids - shift).clamp_min(0)
        else:
            temporal_ids = []
            offset = 0
            for cid in selected_chunk_ids:
                frame_start, frame_end = cls._chunk_frame_bounds(cid, chunk_size, total_latent_frames)
                chunk_len = frame_end - frame_start
                temporal_ids.append(torch.arange(offset, offset + chunk_len, device=device, dtype=torch.long))
                offset += chunk_len
            return torch.cat(temporal_ids, dim=0)

    def _prefill_static_reference_kv_cache(
        self,
        model,
        *,
        prompt_embeds: torch.Tensor,
        reference_image_latents: torch.Tensor,
        transformer_dtype: torch.dtype,
    ) -> None:
        ref_img = reference_image_latents.to(device=self._execution_device, dtype=transformer_dtype)
        ref_frames = ref_img.shape[2]
        with model.cache_context("cond"):
            model(
                hidden_states=ref_img,
                timestep=torch.zeros((ref_img.shape[0],), device=ref_img.device, dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds,
                current_temporal_ids=torch.zeros((ref_img.shape[0], ref_frames), device=ref_img.device, dtype=torch.long),
                kv_cache_mode="store",
                kv_cache_scope="cond",
                kv_cache_chunk_id=self._kv_cache_memory_id("ref_image"),
                kv_cache_selected_chunk_ids=[],
                self_attn_input_mode=SELF_ATTN_MODE_REF_IMAGE_CACHE,
                skip_text_stream=True,
            )

    def _store_clean_chunk_kv_cache(
        self,
        model,
        *,
        clean_chunk_latents: torch.Tensor,
        chunk_temporal_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        active_chunk_id: int,
        history_chunk_ids: Optional[List[int]],
        pre_rope: bool = False,
        cached_temporal_ids: Optional[torch.Tensor] = None,
        store_mode: str = "store",
    ) -> None:
        if store_mode not in ("store", "reuse_store"):
            raise ValueError(f"`store_mode` must be 'store' or 'reuse_store', got {store_mode!r}.")
        selected_chunk_ids = (
            [self._kv_cache_memory_id("clean", cid) for cid in (history_chunk_ids or [])]
            if store_mode == "reuse_store"
            else []
        )
        with model.cache_context("cond"):
            model(
                hidden_states=clean_chunk_latents,
                timestep=torch.zeros(
                    (clean_chunk_latents.shape[0],),
                    device=clean_chunk_latents.device,
                    dtype=clean_chunk_latents.dtype,
                ),
                encoder_hidden_states=prompt_embeds,
                current_temporal_ids=chunk_temporal_ids,
                cached_temporal_ids=cached_temporal_ids,
                kv_cache_mode=store_mode,
                kv_cache_scope="cond",
                kv_cache_chunk_id=self._kv_cache_memory_id("clean", active_chunk_id),
                kv_cache_selected_chunk_ids=selected_chunk_ids,
                kv_cache_pre_rope=pre_rope,
                skip_text_stream=True,
            )

    def _resolve_streaming_chunk_size(self, explicit_chunk_size: Optional[int], total_latent_frames: int) -> int:
        if explicit_chunk_size is not None:
            chunk_size = explicit_chunk_size
        else:
            if not getattr(self.transformer.config, "causal", False):
                chunk_size = total_latent_frames
            else:
                chunk_size = getattr(self.transformer.config, "chunk_size", None)
                if chunk_size is None:
                    chunk_size = total_latent_frames
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
        if transformer is None:
            return False
        return bool(getattr(transformer.config, "global_sink_chunk", False))
