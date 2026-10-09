import os
import io
import html
import math
import json
import ast
import random
from typing import Any
import re

import numpy as np
import regex
from PIL import Image
import decord
import datasets
from datasets.distributed import split_dataset_by_node
from einops import rearrange
import torch
import torchvision.transforms.functional as TF
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import IterableDataset
from diffusers.utils import is_ftfy_available

from src.dataset.bucket_util import BucketGroup
from src.distributed.parallel_states import get_parallel_state, sp_enabled
from src.utils import get_evenly_divisible_files, get_from_dict
from src.utils.logging import get_logger
from src.config import DEFAULT_VIDEO_RESOLUTION, generate_video_image_bucket

if is_ftfy_available():
    import ftfy

decord.bridge.set_bridge("torch")
datasets.disable_caching()


def prompt_clean(text: str) -> str:
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return regex.sub(r"\s+", " ", text).strip()


def parse_json_field(json_data: Any) -> Any:
    """Decode JSON metadata supplied as a mapping, string, or bytes."""
    if not isinstance(json_data, (str, bytes, bytearray)):
        return json_data
    try:
        return json.loads(json_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = json_data.decode("utf-8", "replace") if isinstance(
            json_data, (bytes, bytearray)) else json_data
        return ast.literal_eval(text)


def conversation_to_prompt(
    conversation: list[dict[str, Any]],
    drop_last_assistant: bool = False,
) -> str:
    """Convert conversation dict to prompt string."""
    prompt = ""
    if drop_last_assistant and len(conversation) > 0:
        message = conversation[-1]
        if message.get('role', '') == 'assistant' or message.get('from', '') == 'gpt':
            conversation = conversation[:-1]
    for message in conversation:
        if 'role' in message:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif 'from' in message:
            role = message["from"]
            content = message["value"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "human":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "gpt":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return prompt


def to_rgb_on_white(image: Image.Image) -> Image.Image:
    """Convert to RGB, compositing transparency over white."""
    has_alpha = (
        image.mode in ("RGBA", "LA")
        or (image.mode == "P" and "transparency" in image.info)
    )
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def resize_crop_normalize(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    h, w = x.shape[-2:]
    bh, bw = target_size
    scale = max(bh / h, bw / w)
    resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

    x = TF.resize(x, (resize_h, resize_w),
                  interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    x = TF.center_crop(x, target_size)
    return x / 127.5 - 1.0


class ImageVideoWebDataset(IterableDataset):
    def __init__(
        self,
        image_data_files: str | list[str] | None,
        multiple_images_data_files: str | list[str] | None,
        multiple_videos_data_files: str | list[str] | None,
        video_data_files: str | list[str] | None,
        bucket_configs: list[tuple[int, int, int, int, int]] | None = None,
        bucket_configs_options: list[tuple[int, int, int, int, int]] | None = None,
        bucket_configs_options_prob: float = 0.5,
        prioritize_frame_matching: bool = True,
        image_caption_keys: list[str] | None = None,
        image_caption_sampling_prob: list[float] | None = None,
        multiple_images_caption_keys: list[str] | None = None,
        multiple_images_caption_sampling_prob: list[float] | None = None,
        multiple_videos_caption_keys: list[str] | None = None,
        multiple_videos_caption_sampling_prob: list[float] | None = None,
        video_caption_keys: list[str] | None = None,
        video_caption_sampling_prob: list[float] | None = None,
        image_sampling_prob: float = 0.3,
        multiple_images_sampling_prob: float = 0.3,
        multiple_videos_sampling_prob: float = 0.0,
        video_sampling_prob: float = 0.4,
        ensure_divisible_shards: bool = True,
        shuffle: bool = True,
        fps: int = -1,
        seed: int = 42,
        tar_files_shuffle_seed: int | None = None,
        buffer_size: int = 1000,
        rec_aug_rate: float = 0.0,
        ref_image_basesize: int = DEFAULT_VIDEO_RESOLUTION[0],
        ref_image_bucket_configs: list[tuple[int, int, int, int, int]] | None = None,
    ):
        if image_data_files is None and not math.isclose(image_sampling_prob, 0.0, abs_tol=1e-8):
            raise ValueError(
                "image_sampling_prob must be 0 when image_data_files is None"
            )
        if image_data_files is not None:
            assert image_caption_keys is not None and image_caption_sampling_prob is not None
            assert len(image_caption_keys) == len(image_caption_sampling_prob)
        if multiple_images_data_files is None and not math.isclose(multiple_images_sampling_prob, 0.0, abs_tol=1e-8):
            raise ValueError(
                "multiple_images_sampling_prob must be 0 when multiple_images_data_files is None"
            )
        if multiple_images_data_files is not None:
            assert multiple_images_caption_keys is not None and multiple_images_caption_sampling_prob is not None
            assert len(multiple_images_caption_keys) == len(multiple_images_caption_sampling_prob)
        if multiple_videos_data_files is None and not math.isclose(multiple_videos_sampling_prob, 0.0, abs_tol=1e-8):
            raise ValueError(
                "multiple_videos_sampling_prob must be 0 when multiple_videos_data_files is None"
            )
        if multiple_videos_data_files is not None:
            assert multiple_videos_caption_keys is not None and multiple_videos_caption_sampling_prob is not None
            assert len(multiple_videos_caption_keys) == len(multiple_videos_caption_sampling_prob)
        if video_data_files is None and not math.isclose(video_sampling_prob, 0.0, abs_tol=1e-8):
            raise ValueError(
                "video_sampling_prob must be 0 when video_data_files is None"
            )
        if video_data_files is not None:
            assert video_caption_keys is not None and video_caption_sampling_prob is not None
            assert len(video_caption_keys) == len(video_caption_sampling_prob)
        logger = get_logger()

        self.image_caption_keys = image_caption_keys
        self.image_caption_sampling_prob = image_caption_sampling_prob
        self.multiple_images_caption_keys = multiple_images_caption_keys
        self.multiple_images_caption_sampling_prob = multiple_images_caption_sampling_prob
        self.multiple_videos_caption_keys = multiple_videos_caption_keys
        self.multiple_videos_caption_sampling_prob = multiple_videos_caption_sampling_prob
        self.video_caption_keys = video_caption_keys
        self.video_caption_sampling_prob = video_caption_sampling_prob
        self.image_sampling_prob = image_sampling_prob
        self.multiple_images_sampling_prob = multiple_images_sampling_prob
        self.multiple_videos_sampling_prob = multiple_videos_sampling_prob
        self.video_sampling_prob = video_sampling_prob
        self.ensure_divisible_shards = ensure_divisible_shards
        self.shuffle = shuffle
        self.seed = seed
        self.tar_files_shuffle_seed = tar_files_shuffle_seed
        self.buffer_size = buffer_size
        self.fps = fps
        self.rec_aug_rate = rec_aug_rate

        _, rank = self._get_distributed_info()
        self.rng = random.Random(seed + rank)
        # Share task choices across ranks; advance only after yielding.
        self.task_rng = random.Random(seed)

        self.bucket_group = BucketGroup(
            bucket_configs, prioritize_frame_matching=prioritize_frame_matching)
        self.current_buckets = {b: []
                                for b in self.bucket_group.bucket_configs}

        if ref_image_bucket_configs is None:
            ref_image_bucket_configs = generate_video_image_bucket(
                img_basesize=ref_image_basesize,
                bs_img=1,
                bs_vid=0,
                bs_mimg=0,
                bs_mvid=0,
            )
        self.ref_image_bucket_group = BucketGroup(
            ref_image_bucket_configs,
            prioritize_frame_matching=prioritize_frame_matching,
        )

        self.bucket_group_option = None
        if bucket_configs_options is not None:
            self.bucket_configs_options_prob = bucket_configs_options_prob
            self.bucket_group_option = BucketGroup(
                bucket_configs_options, prioritize_frame_matching=prioritize_frame_matching)
            temp_current_buckest = {b: []
                                    for b in self.bucket_group_option.bucket_configs}
            self.current_buckets.update(temp_current_buckest)

        self._image_dataset, self._image_iterator = None, None
        if image_data_files is not None:
            self._image_dataset, self._image_iterator = self._init_dataset(
                image_data_files, logger)

        self._multiple_images_dataset, self._multiple_images_iterator = None, None
        if multiple_images_data_files is not None:
            # Declare both image extensions so shard schemas retain their images.
            multiple_images_features = datasets.Features({
                "json": datasets.Value("string"),
                "__key__": datasets.Value("string"),
                "__url__": datasets.Value("string"),
                "0.jpg": datasets.Image(),
                "1.jpg": datasets.Image(),
                "2.jpg": datasets.Image(),
                "0.png": datasets.Image(),
                "1.png": datasets.Image(),
                "2.png": datasets.Image(),
            })
            self._multiple_images_dataset, self._multiple_images_iterator = self._init_dataset(
                multiple_images_data_files, logger, features=multiple_images_features)

        self._multiple_videos_dataset, self._multiple_videos_iterator = None, None
        if multiple_videos_data_files is not None:
            multiple_videos_features = datasets.Features({
                "json": datasets.Value("string"),
                "__key__": datasets.Value("string"),
                "__url__": datasets.Value("string"),
                "0.mp4": datasets.Video(),
                "1.mp4": datasets.Video(),
                "0.jpg": datasets.Image(),
            })
            self._multiple_videos_dataset, self._multiple_videos_iterator = self._init_dataset(
                multiple_videos_data_files, logger, features=multiple_videos_features)

        self._video_dataset, self._video_iterator = None, None
        if video_data_files is not None:
            self._video_dataset, self._video_iterator = self._init_dataset(
                video_data_files, logger)

    def _init_dataset(self, data_files: str | list[str], logger, features=None):
        world_size, rank = self._get_distributed_info()
        data_files = [data_files] if isinstance(data_files, str) else list(data_files)

        flags = [f.lower().endswith(".json") for f in data_files]
        if all(flags):
            all_json = True
        elif not any(flags):
            all_json = False
        else:
            raise ValueError(
                "All files must be JSON or none of them should be JSON")

        if all_json:
            output_data_files = []
            for data_file in data_files:
                with open(data_file, "r") as f:
                    data_file = json.load(f)["resolved_files"]
                output_data_files.extend(data_file)
            data_files = output_data_files

            if self.tar_files_shuffle_seed is not None:
                tar_rng = random.Random(self.tar_files_shuffle_seed)
                tar_rng.shuffle(data_files)

        if self.ensure_divisible_shards:
            data_files = get_evenly_divisible_files(
                data_files, world_size, logger=logger, is_skip_glob=all_json)

        dataset = datasets.load_dataset(
            "webdataset", data_files=data_files, streaming=True, split="train",
            features=features)
        dataset = split_dataset_by_node(
            dataset, rank=rank, world_size=world_size)

        if self.shuffle:
            seed = self.seed + dataset.epoch
            dataset = dataset.shuffle(seed=seed, buffer_size=self.buffer_size)

        iterator = iter(dataset)
        return dataset, iterator

    def _get_distributed_info(self):
        if sp_enabled():
            ps = get_parallel_state()
            return ps.world_size // ps.sp_size, ps.sp_group_id
        return int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))

    def _sample_bucket_group(self):
        if self.bucket_group_option is None:
            return self.bucket_group

        if not 0 <= self.bucket_configs_options_prob <= 1:
            raise ValueError(
                f"Illegal bucket_configs_options_prob: {self.bucket_configs_options_prob}"
            )
        return self.rng.choices(
            [self.bucket_group, self.bucket_group_option],
            weights=[
                1 - self.bucket_configs_options_prob,
                self.bucket_configs_options_prob,
            ],
            k=1,
        )[0]

    def _open_video_reader(self, video: Any):
        if isinstance(video, dict):
            path = video.get("path")
            video_bytes = video.get("bytes")
            if video_bytes is not None:
                video = video_bytes
            elif path is not None:
                return decord.VideoReader(path)
            else:
                raise ValueError("Invalid video dict: both 'path' and 'bytes' are None.")

        if isinstance(video, (bytes, bytearray)):
            return decord.VideoReader(io.BytesIO(video))

        if hasattr(video, "get_frames_at") and hasattr(video, "metadata"):
            return video

        raise TypeError(f"Unsupported video type: {type(video)}")

    def _get_video_hw(self, video_reader: Any) -> tuple[int, int]:
        metadata = getattr(video_reader, "metadata", None)
        height = getattr(metadata, "height", None)
        width = getattr(metadata, "width", None)
        if height is not None and width is not None:
            return int(height), int(width)

        first_frame = video_reader[0]
        if first_frame.ndim != 3:
            raise ValueError(
                f"Unexpected frame shape: {tuple(first_frame.shape)}"
            )
        if first_frame.shape[0] in (1, 3):
            return int(first_frame.shape[1]), int(first_frame.shape[2])
        return int(first_frame.shape[0]), int(first_frame.shape[1])

    def _get_video_batch(self, video_reader: Any, frame_idx: list[int]) -> torch.Tensor:
        if hasattr(video_reader, "get_batch"):
            pixel = video_reader.get_batch(frame_idx)
        else:
            pixel = video_reader.get_frames_at(frame_idx).data

        if pixel.ndim != 4:
            raise ValueError(f"Unexpected video batch shape: {tuple(pixel.shape)}")
        if pixel.shape[1] in (1, 3):
            return pixel
        if pixel.shape[-1] in (1, 3):
            return rearrange(pixel, "t h w c -> t c h w")
        raise ValueError(f"Unexpected channel layout in video batch: {tuple(pixel.shape)}")

    def _process_video(self, video: Any):
        video_reader = self._open_video_reader(video)
        n_frame = len(video_reader)
        if n_frame <= 0:
            raise ValueError("Empty video encountered.")

        height, width = self._get_video_hw(video_reader)
        fps_ratio = self._get_video_fps_ratio(video_reader)
        bucket_group = self._sample_bucket_group()
        bucket = bucket_group.find_best_bucket((1, int(n_frame // fps_ratio), height, width))
        pixel = self._sample_video_with_bucket(video_reader, bucket)
        pixel = pixel.unsqueeze(0)
        return pixel, bucket

    def _get_video_fps_ratio(self, video_reader: Any) -> float:
        if hasattr(video_reader, "get_avg_fps"):
            cur_fps = video_reader.get_avg_fps()
        else:
            metadata = getattr(video_reader, "metadata", None)
            cur_fps = getattr(metadata, "average_fps", None)
        if cur_fps is None:
            return 1.0
        if self.fps > 0 and cur_fps > self.fps:
            return cur_fps / self.fps
        return 1.0

    def _sample_video_with_bucket(
        self,
        video_reader: Any,
        bucket: tuple[int, int, int, int, int],
        start_ratio: float | None = None,
        frame_rand: np.ndarray | None = None,
    ) -> torch.Tensor:
        n_frame = len(video_reader)
        if n_frame <= 0:
            raise ValueError("Empty video encountered.")

        _, _, sampling_n_frame, target_height, target_width = bucket
        fps_ratio = self._get_video_fps_ratio(video_reader)
        scaled_n_frame = math.ceil(sampling_n_frame * fps_ratio)
        clip_len = min(n_frame, max(scaled_n_frame, sampling_n_frame))

        max_start = n_frame - clip_len
        if start_ratio is None:
            start_idx = self.rng.randint(0, max_start) if max_start > 0 else 0
        else:
            start_idx = int(round(max_start * start_ratio)) if max_start > 0 else 0

        bin_edges = np.linspace(0, clip_len, num=sampling_n_frame + 1, dtype=np.int64)
        bin_starts = bin_edges[:-1]
        bin_ends = np.maximum(bin_edges[1:], bin_starts + 1)
        if frame_rand is None:
            frame_rand = np.fromiter((self.rng.random() for _ in range(sampling_n_frame)), dtype=np.float32)
        sample_idx = bin_starts + ((bin_ends - bin_starts) * frame_rand).astype(np.int64)
        sample_idx = np.clip(sample_idx, 0, clip_len - 1)
        frame_idx = (start_idx + sample_idx).tolist()

        pixel = self._get_video_batch(video_reader, frame_idx)
        pixel = resize_crop_normalize(pixel, (target_height, target_width))
        return rearrange(pixel, "t c h w -> c t h w")

    def _process_multiple_videos(self, videos: list[Any]):
        if len(videos) == 0:
            raise ValueError("No videos provided for processing.")

        video_readers = [self._open_video_reader(video) for video in videos]
        target_reader = video_readers[-1]
        if len(target_reader) <= 0:
            raise ValueError("Target video is empty.")

        target_height, target_width = self._get_video_hw(target_reader)
        target_fps_ratio = self._get_video_fps_ratio(target_reader)
        bucket_group = self._sample_bucket_group()
        bucket = bucket_group.find_best_bucket(
            (len(videos), int(len(target_reader) // target_fps_ratio), target_height, target_width)
        )

        start_ratio = self.rng.random()
        _, _, sampling_n_frame, _, _ = bucket
        frame_rand = np.fromiter((self.rng.random() for _ in range(sampling_n_frame)), dtype=np.float32)
        pixel = [
            self._sample_video_with_bucket(
                video_reader,
                bucket,
                start_ratio=start_ratio,
                frame_rand=frame_rand,
            )
            for video_reader in video_readers
        ]
        pixel = torch.stack(pixel, dim=0)
        return pixel, bucket

    def _process_ref_image(self, image: Image.Image):
        pixel = torch.from_numpy(np.array(to_rgb_on_white(image)))
        h, w = pixel.shape[:2]
        bucket = self.ref_image_bucket_group.find_best_bucket((1, 1, h, w))
        _, _, _, target_height, target_width = bucket
        pixel = rearrange(pixel, "h w c -> 1 c h w")
        pixel = resize_crop_normalize(pixel, (target_height, target_width))
        return rearrange(pixel, "1 c h w -> c 1 h w")

    def _detect_multi_image_ext(self, item) -> str | None:
        """Find the image extension used by a multi-image sample."""
        for ext in ("jpg", "png"):
            if item.get(f"0.{ext}") is not None and item.get(f"1.{ext}") is not None:
                return ext
        return None

    def _collect_sequential_media(self, item, suffix: str):
        media = []
        index = 0
        # Explicit schemas retain unused extension keys as None.
        while item.get(f"{index}.{suffix}") is not None:
            media.append(item[f"{index}.{suffix}"])
            index += 1
        return media

    def _process_multiple_images(self, images: list[Image.Image], use_rec_aug: bool):
        if len(images) == 0:
            raise ValueError("No images provided for processing.")

        bucket_source = images[0] if use_rec_aug else images[-1]
        w, h = bucket_source.size
        bucket_group = self._sample_bucket_group()
        num_items = 2 if use_rec_aug else len(images)
        bucket = bucket_group.find_best_bucket((num_items, 1, h, w))
        _, _, _, target_height, target_width = bucket

        pixel = []
        for img in images:
            img = torch.from_numpy(np.array(to_rgb_on_white(img)))
            img = rearrange(img, "h w c -> 1 c h w")
            img = resize_crop_normalize(img, (target_height, target_width))
            img = rearrange(img, "1 c h w -> c 1 h w")
            pixel.append(img)

        if use_rec_aug:
            recon_image = pixel[0]
            pixel = [recon_image, recon_image]

        pixel = torch.stack(pixel, dim=0)
        return pixel, bucket

    def _process_image(self, image: Image.Image):
        pixel = torch.from_numpy(np.array(to_rgb_on_white(image)))
        h, w = pixel.shape[:2]

        pixel = rearrange(pixel, "h w c -> 1 c h w")
        bucket_group = self._sample_bucket_group()
        bucket = bucket_group.find_best_bucket((1, 1, h, w))
        _, _, _, target_height, target_width = bucket
        pixel = resize_crop_normalize(pixel, (target_height, target_width))
        pixel = rearrange(pixel, "1 c h w -> c 1 h w")
        pixel = pixel.unsqueeze(0)
        return pixel, bucket

    def _get_caption(self, item, json_data):
        logger = get_logger()

        available_captions = []
        weights = []

        is_multiple_images = self._detect_multi_image_ext(item) is not None
        is_multiple_videos = bool("0.mp4" in item)
        is_image = bool("jpg" in item) and not is_multiple_images
        is_video = bool("mp4" in item) and not is_multiple_videos
        if is_multiple_videos:
            caption_keys = self.multiple_videos_caption_keys
            caption_sampling_prob = self.multiple_videos_caption_sampling_prob
        elif is_multiple_images:
            caption_keys = self.multiple_images_caption_keys
            caption_sampling_prob = self.multiple_images_caption_sampling_prob
        elif is_image:
            caption_keys = self.image_caption_keys
            caption_sampling_prob = self.image_caption_sampling_prob
        elif is_video:
            caption_keys = self.video_caption_keys
            caption_sampling_prob = self.video_caption_sampling_prob
        else:
            raise ValueError(f"Invalid item: {item.keys()}")

        for key, prob in zip(caption_keys, caption_sampling_prob):
            try:
                value = get_from_dict(json_data, key)
            except (KeyError, TypeError):
                continue

            if value is None:
                continue

            try:
                if isinstance(value, list):
                    caption = conversation_to_prompt(
                        value, drop_last_assistant=True)
                else:
                    caption = prompt_clean(str(value))
            except Exception:
                logger.warning("Skipped an invalid caption.")
                continue

            if not caption.strip():
                continue

            available_captions.append(caption)
            weights.append(prob)

        if not available_captions:
            msg = "No valid captions found."
            logger.warning(msg)
            raise ValueError(msg)

        caption = self.rng.choices(available_captions, weights=weights, k=1)[0]
        return caption

    def _process_item(self, item):
        json_data = item.get("json")
        if json_data is None:
            raise ValueError("Missing 'json' field in item")
        try:
            json_data = parse_json_field(json_data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, SyntaxError) as error:
            raise ValueError("Invalid JSON format in item['json']") from error

        image_ext = self._detect_multi_image_ext(item)
        is_multiple_images = image_ext is not None
        is_multiple_videos = bool("0.mp4" in item)
        is_image = bool("jpg" in item) and not is_multiple_images
        is_video = bool("mp4" in item) and not is_multiple_videos

        caption = self._get_caption(item, json_data)
        ref_image_pixel = None
        if is_multiple_videos:
            pixel, bucket_id = self._process_multiple_videos(
                self._collect_sequential_media(item, "mp4")
            )
            ref_image = item.get("0.jpg")
            caption_only = isinstance(json_data, dict) and set(json_data) == {"caption"}
            metadata = json_data.get("metadata") if isinstance(json_data, dict) else None
            if not caption_only and not (isinstance(metadata, dict) and metadata.get("ref_image", None) is not None):
                ref_image = None
            if ref_image is not None:
                ref_image_pixel = self._process_ref_image(ref_image)
        elif is_multiple_images:
            images = self._collect_sequential_media(item, image_ext)
            # Three-item samples contain reference, source, and target images.
            if len(images) == 3:
                ref_image_pixel = self._process_ref_image(images[0])
                pixel, bucket_id = self._process_multiple_images(
                    images[1:], use_rec_aug=False)
            else:
                if self.rng.random() < self.rec_aug_rate:
                    use_rec_aug = True

                    def replace_caption_text(caption: np.str_, new_text: str):
                        caption_str = str(caption)
                        replaced = re.sub(
                            r"(<image>\n)(.*?)(<\|im_end\|>)",
                            rf"\1{new_text}\3",
                            caption_str,
                            flags=re.DOTALL
                        )
                        return np.str_(replaced)
                    caption = replace_caption_text(
                        caption=caption, new_text="Reconstruct this image.")
                else:
                    use_rec_aug = False
                pixel, bucket_id = self._process_multiple_images(images, use_rec_aug)
        elif is_image:
            pixel, bucket_id = self._process_image(item["jpg"])
        elif is_video:
            pixel, bucket_id = self._process_video(item["mp4"])
        else:
            raise ValueError(f"Invalid item: {item.keys()}")

        return pixel, caption, bucket_id, ref_image_pixel

    def _sample_task_mode(self):
        """Sample a task mode using the cross-rank synchronized RNG."""
        return self.task_rng.choices(
            ['image', 'multiple_images', 'multiple_videos', 'video'],
            weights=[
                self.image_sampling_prob,
                self.multiple_images_sampling_prob,
                self.multiple_videos_sampling_prob,
                self.video_sampling_prob],
            k=1
        )[0]

    def _get_dataset_and_iterator(self, mode: str):
        if mode == 'image':
            return self._image_dataset, self._image_iterator
        elif mode == 'multiple_images':
            return self._multiple_images_dataset, self._multiple_images_iterator
        elif mode == 'multiple_videos':
            return self._multiple_videos_dataset, self._multiple_videos_iterator
        elif mode == 'video':
            return self._video_dataset, self._video_iterator
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_next_item(self, mode: str):
        dataset, iterator = self._get_dataset_and_iterator(mode)

        try:
            item = next(iterator, None)
            if item is None:
                dataset.set_epoch(dataset.epoch + 1)
                new_iterator = iter(dataset)
                setattr(self, f"_{mode}_iterator", new_iterator)
                item = next(new_iterator)
            return item
        except Exception:
            logger = get_logger()
            logger.warning(f"Failed to read a {mode} sample.")
            raise

    def __iter__(self):
        logger = get_logger()
        consecutive_read_failures = 0
        consecutive_process_failures = 0

        current_mode = self._sample_task_mode()

        while True:
            try:
                item = self._get_next_item(current_mode)
                consecutive_read_failures = 0
            except Exception:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 50:
                    logger.error(
                        "Exceeded dataset read failure limit. "
                        f"consecutive_read_failures={consecutive_read_failures}"
                    )
                    raise RuntimeError(
                        f"Dataset reader entered a failure loop for {consecutive_read_failures} consecutive attempts."
                    )
                continue

            url = item.get("__url__", "N/A")
            key = item.get("__key__", "N/A")
            try:
                pixel, caption, bucket_id, ref_image_pixel = self._process_item(item)
                consecutive_process_failures = 0
            except Exception as e:
                consecutive_process_failures += 1
                logger.warning("Skipped a sample that could not be processed.")
                if consecutive_process_failures >= 50:
                    logger.error(
                        "Exceeded item processing failure limit. "
                        f"consecutive_process_failures={consecutive_process_failures}"
                    )
                    raise RuntimeError(
                        "Dataset processor entered a failure loop for "
                        f"{consecutive_process_failures} consecutive items."
                    ) from e
                continue

            bucket = self.current_buckets[bucket_id]
            bucket.append({"pixel": pixel, "caption": caption, "ref_image_pixel": ref_image_pixel,
                          "key": key, "url": url, "bucket_id": bucket_id})

            if len(bucket) == bucket_id[0]:
                yield bucket[:]
                bucket.clear()
                current_mode = self._sample_task_mode()

    def state_dict(self):
        return {
            "image": self._image_dataset.state_dict() if self._image_dataset is not None else None,
            "multiple_images": self._multiple_images_dataset.state_dict() if self._multiple_images_dataset is not None else None,
            "multiple_videos": self._multiple_videos_dataset.state_dict() if self._multiple_videos_dataset is not None else None,
            "video": self._video_dataset.state_dict() if self._video_dataset is not None else None,
            "rng_state": self.rng.getstate(),
            "task_rng_state": self.task_rng.getstate(),
            "current_buckets": self.current_buckets,
        }

    def load_state_dict(self, state_dict):
        if self._image_dataset is not None and state_dict["image"] is not None:
            self._image_dataset.load_state_dict(state_dict["image"])
            self._image_iterator = iter(self._image_dataset)
        if self._multiple_images_dataset is not None and state_dict["multiple_images"] is not None:
            self._multiple_images_dataset.load_state_dict(
                state_dict["multiple_images"])
            self._multiple_images_iterator = iter(
                self._multiple_images_dataset)
        if self._multiple_videos_dataset is not None and state_dict["multiple_videos"] is not None:
            self._multiple_videos_dataset.load_state_dict(
                state_dict["multiple_videos"])
            self._multiple_videos_iterator = iter(
                self._multiple_videos_dataset)
        if self._video_dataset is not None and state_dict["video"] is not None:
            self._video_dataset.load_state_dict(state_dict["video"])
            self._video_iterator = iter(self._video_dataset)
        if "rng_state" in state_dict and state_dict["rng_state"] is not None:
            self.rng.setstate(state_dict["rng_state"])
        if "task_rng_state" in state_dict and state_dict["task_rng_state"] is not None:
            self.task_rng.setstate(state_dict["task_rng_state"])
        if "current_buckets" in state_dict and state_dict["current_buckets"] is not None:
            self.current_buckets = state_dict["current_buckets"]


def collate_batch(batch):
    pixels = torch.stack([item["pixel"] for item in batch])
    captions = [item["caption"] for item in batch]
    ref_image_pixels = [item.get("ref_image_pixel") for item in batch]
    keys = [item["key"] for item in batch]
    urls = [item["url"] for item in batch]
    bucket_ids = [item["bucket_id"] for item in batch]
    return {
        "pixel": pixels,
        "caption": captions,
        "ref_image_pixel": ref_image_pixels,
        "key": keys,
        "url": urls,
        "bucket_id": bucket_ids,
    }


def build_webdataset_dataloader(
    image_data_files: str | list[str] | None = None,
    multiple_images_data_files: str | list[str] | None = None,
    multiple_videos_data_files: str | list[str] | None = None,
    video_data_files: str | list[str] | None = None,
    bucket_configs: list[tuple[int, int, int, int, int]] | None = None,
    bucket_configs_options: list[tuple[int, int, int, int, int]] | None = None,
    bucket_configs_options_prob: float = 0.5,
    prioritize_frame_matching: bool = True,
    image_caption_keys: list[str] | None = None,
    image_caption_sampling_prob: list[float] | None = None,
    multiple_images_caption_keys: list[str] | None = None,
    multiple_images_caption_sampling_prob: list[float] | None = None,
    multiple_videos_caption_keys: list[str] | None = None,
    multiple_videos_caption_sampling_prob: list[float] | None = None,
    video_caption_keys: list[str] | None = None,
    video_caption_sampling_prob: list[float] | None = None,
    image_sampling_prob: float = 0.3,
    multiple_images_sampling_prob: float = 0.3,
    multiple_videos_sampling_prob: float = 0.0,
    video_sampling_prob: float = 0.4,
    shuffle: bool = True,
    ensure_divisible_shards: bool = True,
    num_workers: int = 1,
    fps: int = -1,
    seed: int = 42,
    tar_files_shuffle_seed: int | None = None,
    buffer_size: int = 1000,
    rec_aug_rate: float = 0.0,
    ref_image_basesize: int = DEFAULT_VIDEO_RESOLUTION[0],
    ref_image_bucket_configs: list[tuple[int, int, int, int, int]] | None = None,
):
    dataset = ImageVideoWebDataset(
        image_data_files=image_data_files,
        multiple_images_data_files=multiple_images_data_files,
        multiple_videos_data_files=multiple_videos_data_files,
        video_data_files=video_data_files,
        bucket_configs=bucket_configs,
        bucket_configs_options=bucket_configs_options,
        bucket_configs_options_prob=bucket_configs_options_prob,
        prioritize_frame_matching=prioritize_frame_matching,
        image_caption_keys=image_caption_keys,
        image_caption_sampling_prob=image_caption_sampling_prob,
        multiple_images_caption_keys=multiple_images_caption_keys,
        multiple_images_caption_sampling_prob=multiple_images_caption_sampling_prob,
        multiple_videos_caption_keys=multiple_videos_caption_keys,
        multiple_videos_caption_sampling_prob=multiple_videos_caption_sampling_prob,
        video_caption_keys=video_caption_keys,
        video_caption_sampling_prob=video_caption_sampling_prob,
        image_sampling_prob=image_sampling_prob,
        multiple_images_sampling_prob=multiple_images_sampling_prob,
        multiple_videos_sampling_prob=multiple_videos_sampling_prob,
        video_sampling_prob=video_sampling_prob,
        ensure_divisible_shards=ensure_divisible_shards,
        shuffle=shuffle,
        fps=fps,
        seed=seed,
        tar_files_shuffle_seed=tar_files_shuffle_seed,
        buffer_size=buffer_size,
        rec_aug_rate=rec_aug_rate,
        ref_image_basesize=ref_image_basesize,
        ref_image_bucket_configs=ref_image_bucket_configs,
    )
    return StatefulDataLoader(dataset, batch_size=None, num_workers=num_workers, collate_fn=collate_batch)
