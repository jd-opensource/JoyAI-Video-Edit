# Scoring prompts from OpenVE-3M (https://arxiv.org/abs/2512.07826).
GLOBAL_STYLE = """
You are a data rater specializing in grading video style transfer edits. You will be given an input video, a reference style (image or video), and the styled result video. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:

Instruction Compliance
1. Target style absent or clearly wrong.
2. Style shows in a few areas/frames only, or mixed with unrelated styles.
3. Key traits (palette, brushwork, texture) present but patchy or inconsistent across frames.
4. Style reproduced well across almost the whole video; only small local or brief temporal mismatches.
5. Full, faithful transfer: colour, texture, brushwork, and lighting all match the exemplar consistently over the entire duration and space of the video.

Consistency & Detail Fidelity
1. Major objects, layout, or overall motion lost/distorted; original scene barely recognisable.
2. Main subject recognisable, but its size, perspective, motion, or key parts are clearly wrong/missing.
3. Overall structure and motion correct; some local warping, minor omissions, or slight motion jerkiness.
4. Nearly all geometry and motion intact; only slight, non-distracting deformation.
5. All objects, spatial relations, and motion are perfectly kept; only stylistic, harmless distortion.

Visual Quality & Stability
1. Extreme flickering or “boiling” effects; the style is completely unstable frame-to-frame, making the video unwatchable.
2. Significant and distracting flickering or temporal inconsistency in style application.
3. Noticeable but tolerable flicker or texture “boiling”, especially during motion.
4. Largely stable with only minor, subtle flickering visible in areas of complex motion or fine texture.
5. Perfectly stable and temporally coherent; the style appears “stuck” to the scene with no flickering.

Note: The scores for Consistency & Detail Fidelity and Visual Quality & Stability should not be higher than the Instruction Compliance score.

Example Response Format
Brief reasoning: A short explanation of the scores based on the criteria above, no more than 30 words.
Instruction Compliance: A number from 1 to 5.
Consistency & Detail Fidelity: A number from 1 to 5.
Visual Quality & Stability: A number from 1 to 5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


BACKGROUND_CHANGE = """
You are a data rater specializing in grading video background editing. You will be given two videos (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:

Instruction Compliance
1. No change, or background unrelated to prompt, or foreground also replaced/distorted.
2. Background partly replaced or wrong style/content; foreground noticeably altered.
3. Main background replaced but elements missing/extra, or faint spill onto subject edges.
4. Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).
5. Background exactly matches prompt (content, style, placement); all foreground pixels untouched.

Consistency & Detail Fidelity
1. Large tearing, posterisation, or significant temporal artifacts like flickering, jittering edges; edit area obvious at a glance.
2. Clear cut-out halos, colour-resolution gap, or obvious edge instability over time.
3. Blend acceptable but visible on closer look: slight edge blur, or minor temporal instability.
4. Nearly invisible seams; edges are stable across motion, textures aligned, only minor issues when zoomed in.
5. Indistinguishable composite: edges, textures, resolution and colour grading are perfectly continuous and stable throughout the video.

Visual Quality & Stability
1. Severe mismatch: wrong horizon, conflicting light, floating subject, or static background during camera movement.
2. Noticeable inconsistencies in light or scale; incorrect perspective shifts during motion.
3. Overall believable; small errors in shadow, perspective, or minor motion tracking flaws.
4. Lighting, scale, and depth well matched; background tracks convincingly with camera motion.
5. Physically flawless: coherent light, shadows, perspective, and depth throughout.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_CHANGE = """
You are a data rater specializing in grading video replacement edits. You will be given two videos (before and after editing) and the editing instructions.

Instruction Compliance
1. Target not replaced or unrelated edit.
2. Partial replacement or wrong class.
3. Largely replaced but with visible remnants or incorrect count/position.
4. Correct replacement with minor attribute errors.
5. Perfect replacement matching class, number, position, scale, pose, motion, and detail.

Consistency & Detail Fidelity
1. Video heavily broken or object flickers uncontrollably.
2. Obvious seams, colour mismatch, or unstable background.
3. Mostly correct but noticeable flicker or lighting inconsistency.
4. Nearly seamless; only tiny temporal artefacts.
5. Completely seamless and temporally stable.

Visual Quality & Stability
1. Severe tracking, lighting, or perspective errors.
2. Missing shadows, poor occlusion, or mismatched motion.
3. Mostly correct with minor inconsistencies.
4. Well-tracked with realistic interactions.
5. Physically flawless integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_REMOVE = """
You are a data rater specializing in grading video object removal editing.

Instruction Compliance
1. No edit or completely wrong.
2. Wrong object removed or partial removal.
3. Correct object removed with major errors or ghosting.
4. Correct object removed with minor fragments.
5. Perfect removal with everything else untouched.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious erase marks or jitter.
3. Noticeable temporal inconsistency.
4. Minor edge issues only on close inspection.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Background badly reconstructed or static.
2. Background shifts or jitters over time.
3. Mostly correct with small flaws.
4. Clean and stable reconstruction.
5. Background perfectly matches original motion and detail.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_ADD = """
You are a data rater specializing in grading video object addition editing.

Instruction Compliance
1. No edit or wrong object added.
2. Partial or wrong addition.
3. Correct object added with major attribute errors.
4. Correct object with minor inaccuracies.
5. Perfect addition with all attributes correct.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious paste marks or jitter.
3. Noticeable lighting or colour mismatch.
4. Minor edge or temporal artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical errors or occlusion issues.
2. Poor contact, occlusion, or motion.
3. Mostly correct with minor flaws.
4. Realistic shadows, reflections, and motion.
5. Perfect physical and temporal integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


SUBTITLES_EDIT = """
You are a data rater specializing in grading instruction-following subtitle edits.

Instruction Compliance
1. Wrong subtitle or no edit.
2. Right action but wrong content or partial edit.
3. Mostly correct with significant errors.
4. Correct with minor inaccuracies.
5. Perfect subtitle edit with zero unintended changes.

Visual Quality & Stability
1. Attributes completely wrong or unreadable.
2. Major deviation from requested attributes.
3. Acceptable but inconsistent placement or style.
4. Minor inaccuracies only.
5. Perfect attribute matching or professional default choice.

Consistency & Detail Fidelity
1. Major video corruption or subtitle damage.
2. Noticeable artifacts or unintended subtitle changes.
3. Minor unintended effects.
4. Almost perfect preservation.
5. Perfect isolation of the edit.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CAMERA_MULTI_SHOT_EDIT = """
You are a data rater specializing in grading camera shot type alteration edits.

Instruction Compliance
1. Shot type unchanged or wrong.
2. Direction correct but degree wrong.
3. Generally correct but poorly framed.
4. Correct shot with minor framing issues.
5. Perfect shot type and framing.

Visual Quality & Stability
1. Severe distortion or glitches.
2. Distracting jitter or warping.
3. Minor visual flaws.
4. Very stable with tiny artefacts.
5. Perfectly stable and clear.

Consistency & Detail Fidelity
1. Completely different scene.
2. Major illogical changes.
3. Noticeable continuity errors.
4. Highly consistent with minor discrepancies.
5. Perfect consistency and continuity.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CREATIVE_EDIT = """
You are a data rater specializing in grading instruction-following creative video edits.

Instruction Compliance
1. Instruction ignored.
2. Attempted but fundamentally failed.
3. Generally follows instruction with major deviations.
4. Successful with minor inaccuracies.
5. Perfect creative execution throughout.

Visual Quality & Stability
1. Unwatchable due to flicker or artefacts.
2. Obvious temporal inconsistency or seams.
3. Mostly stable with noticeable boiling.
4. Very stable with subtle artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical inconsistencies.
2. Major lighting or motion errors.
3. Mostly believable with minor flaws.
4. Realistic interaction and preserved details.
5. Indistinguishable from real footage.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


import argparse
import base64
import csv
import concurrent.futures
import json
import mimetypes
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from tqdm import tqdm

from src.config import VAE_SPATIAL_FACTOR, _generate_video_hw_buckets_from_ratios


API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://modelservice.jdcloud.com/v1")
MODEL_ID = "Gemini-2.5-pro"
REQUEST_TIMEOUT = float(os.environ.get("GEMINI_REQUEST_TIMEOUT", 120))
MAX_ATTEMPTS = 100
RETRY_SLEEP = 0.0
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
EVAL_DOWNSAMPLE = True
EVAL_FPS = 4
EVAL_BUCKET_BASE_SIZE = (480, 832)
EVAL_CRF = 32
EVAL_MAX_SECONDS = 60


def get_eval_bucket(video_path: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation",
            "-of", "json", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {video_path}")
    stream = streams[0]
    height, width = int(stream["height"]), int(stream["width"])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid video dimensions: {video_path}")
    try:
        pixel_ratio = float(Fraction(stream.get("sample_aspect_ratio", "1:1").replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        pixel_ratio = 1.0
    aspect_ratio = height / (width * pixel_ratio) if pixel_ratio > 0 else height / width
    rotation = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = side_data["rotation"]
    if round(float(rotation)) % 180 == 90:
        aspect_ratio = 1 / aspect_ratio
    buckets = _generate_video_hw_buckets_from_ratios(*EVAL_BUCKET_BASE_SIZE, align=VAE_SPATIAL_FACTOR)
    return min(buckets, key=lambda shape: abs(shape[0] / shape[1] - aspect_ratio))


def get_preprocessing_tag() -> str:
    if not EVAL_DOWNSAMPLE:
        return "origvideo"
    height, width = EVAL_BUCKET_BASE_SIZE
    return f"fps{EVAL_FPS}_bucket{height}x{width}_a{VAE_SPATIAL_FACTOR}_crf{EVAL_CRF}_t{EVAL_MAX_SECONDS}"


def downsample_video_for_eval(video_path: str, bucket_size: tuple[int, int] | None = None) -> str:
    if not EVAL_DOWNSAMPLE:
        return video_path
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    height, width = bucket_size or get_eval_bucket(video_path)
    cache_dir = src.parent / ".gemini_eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{get_preprocessing_tag()}_h{height}w{width}"
    dst = cache_dir / f"{src.name}.{suffix}.mp4"
    if dst.is_file() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns and dst.stat().st_size > 0:
        return str(dst)
    with NamedTemporaryFile(dir=cache_dir, suffix=".tmp.mp4", delete=False) as temporary_file:
        tmp = Path(temporary_file.name)
    cmd = [
        FFMPEG_BIN,
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", "1", "-filter_threads", "1",
        "-i",
        str(src),
        "-t",
        str(EVAL_MAX_SECONDS),
        "-map", "0:v:0",
        "-vf",
        f"fps={EVAL_FPS},scale=trunc(iw*sar/2)*2:ih,setsar=1,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2:flags=bilinear,"
        f"crop={width}:{height},setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-threads", "1",
        "-preset",
        "veryfast",
        "-crf",
        str(EVAL_CRF),
        "-pix_fmt", "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"Empty evaluation video: {src}")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return str(dst)

def avg_score_by_edited_type(jsonl_path):
    score_sum = defaultdict(int)
    score_count = defaultdict(int)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                edited_type = record.get("edited_type")
                scores = record.get("scores", [])
                if edited_type and scores:
                    score_sum[edited_type] += sum(scores)
                    score_count[edited_type] += len(scores)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
    avg_scores = {
        edited_type: score_sum[edited_type] / score_count[edited_type]
        for edited_type in score_sum
        if score_count[edited_type] > 0
    }
    print(score_count)
    return avg_scores

prompt_type = {
    'global_style': GLOBAL_STYLE,
    'local_change': LOCAL_CHANGE,
    'background_change': BACKGROUND_CHANGE,
    'local_remove': LOCAL_REMOVE,
    'local_add': LOCAL_ADD,
}

TASK_ALIASES = {
    'style_transfer': 'global_style',
    'ps_human': 'local_change',
}


def parse_task_types(task_types: str):
    if task_types.strip().lower() == "all":
        return None
    parsed = {task.strip() for task in task_types.split(",") if task.strip()}
    unsupported = parsed - set(prompt_type.keys())
    if unsupported:
        raise ValueError(
            f"Unsupported task types for current public-board evaluation logic: {sorted(unsupported)}. "
            f"Supported task types are: {sorted(prompt_type.keys())}"
        )
    return parsed


def normalize_edited_type(edited_type: str) -> str:
    return TASK_ALIASES.get(edited_type, edited_type)


def build_edited_video_path(save_dir: Path, edited_type: str, original_video: str) -> Path:
    original_path = Path(original_video)
    stem = original_path.stem
    plain_stem = stem.removesuffix("_rife_3x_slowmo")
    candidates = [
        save_dir / "fullset" / edited_type / original_path.name,
        save_dir / "fullset" / edited_type / f"{stem}.mp4",
        save_dir / "fullset" / edited_type / f"{plain_stem}.mp4",
        save_dir / edited_type / original_path.name,
        save_dir / edited_type / f"{stem}.mp4",
        save_dir / edited_type / f"{plain_stem}.mp4",
        save_dir / original_path.name,
        save_dir / f"{stem}.mp4",
        save_dir / f"{plain_stem}.mp4",
        save_dir / f"{stem}-0_regular.mp4",
        save_dir / f"{stem}-0_ema.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_source_path(original_video: str, dataset_root: Path) -> Path:
    path = Path(original_video)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        path = Path(*path.parts[1:])
    return dataset_root / path


def build_source_video_path(
    save_dir: Path, edited_type: str, original_video: str, dataset_root: Path = Path("."), source_from_dataset: bool = False,
) -> Path:
    if source_from_dataset:
        return resolve_source_path(original_video, dataset_root)
    original_path = Path(original_video)
    stem = Path(original_video).stem
    plain_stem = stem.removesuffix("_rife_3x_slowmo")
    base = save_dir / "fullset" / edited_type
    candidates = [
        base / (stem + "_src.mp4"),
        base / (plain_stem + "_src.mp4"),
        base / (stem + ".original.mp4"),
        base / (plain_stem + ".original.mp4"),
        save_dir / edited_type / (stem + "_src.mp4"),
        save_dir / edited_type / (plain_stem + "_src.mp4"),
        save_dir / edited_type / (stem + ".original.mp4"),
        save_dir / edited_type / (plain_stem + ".original.mp4"),
        resolve_source_path(original_video, dataset_root),
        original_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return resolve_source_path(original_video, dataset_root)


def get_score_filename(model_id: str, task_types: str, source_from_dataset: bool = False) -> str:
    eval_tag = get_preprocessing_tag()
    if source_from_dataset:
        eval_tag += "_dataset_source"
    if task_types.strip().lower() == "all":
        return f"longbench_{model_id}_{eval_tag}_score.jsonl"
    task_tag = task_types.replace(",", "_")
    return f"longbench_{model_id}_{eval_tag}_{task_tag}_score.jsonl"


def build_manifest_edited_video_path(result_root: Path, row: dict[str, str]) -> Path:
    category = normalize_edited_type(row.get('category') or row.get('edited_type', ''))
    video_id = row['video_id']
    candidates = [
        result_root / "fullset" / category / f"{video_id}.mp4",
        result_root / category / f"{video_id}.mp4",
        result_root / f"{video_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_manifest_source_video_path(row: dict[str, str], dataset_root: Path = Path(".")) -> Path:
    for key in ['copied_original_video', 'original_video', 'source_original_path', 'source_video']:
        value = row.get(key, '')
        if value:
            path = resolve_source_path(value, dataset_root)
            if path.is_file():
                return path
    value = row.get('copied_original_video') or row.get('original_video') or row.get('source_original_path') or row.get('source_video') or ''
    return resolve_source_path(value, dataset_root)


def video_to_data_uri(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if mime_type is None:
        mime_type = "video/mp4"
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{video_base64}"


def run_two_videos_gemini(video_path_1, video_path_2, sys_prompt, meta_info, idx):
    try:
        from openai import OpenAI

        if not API_KEY:
            raise ValueError("Set GEMINI_API_KEY before scoring.")
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)
        bucket_size = get_eval_bucket(video_path_2) if EVAL_DOWNSAMPLE else None
        eval_video_path_1 = downsample_video_for_eval(video_path_1, bucket_size)
        eval_video_path_2 = downsample_video_for_eval(video_path_2, bucket_size)
        video_1_data_uri = video_to_data_uri(eval_video_path_1)
        video_2_data_uri = video_to_data_uri(eval_video_path_2)
    except Exception as e:
        print(f"Init failed: {e}")
        return None

    for attempt in range(MAX_ATTEMPTS):
        try:
            api_start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_ID,
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": sys_prompt},
                            {"type": "video_url", "video_url": {"url": video_1_data_uri}},
                            {"type": "video_url", "video_url": {"url": video_2_data_uri}},
                        ],
                    }
                ]
            )
            api_latency_sec = time.perf_counter() - api_start_time

            result = response.choices[0].message.content or ""
            result_text = normalize_response_text(result)
            scores = check_format(result_text)

            if scores:
                print(f"Task {idx} Success on attempt {attempt+1}, api_latency={api_latency_sec:.3f}s")
                timing_info = {
                    "api_latency_sec": api_latency_sec,
                    "api_attempt": attempt + 1,
                    "eval_video_1_bytes": os.path.getsize(eval_video_path_1),
                    "eval_video_2_bytes": os.path.getsize(eval_video_path_2),
                }
                return scores, result_text, meta_info, idx, timing_info

        except Exception as error:
            print(f"Task {idx} Attempt {attempt+1} failed: {error}")
        if RETRY_SLEEP > 0 and attempt + 1 < MAX_ATTEMPTS:
            time.sleep(RETRY_SLEEP)

    return None


def normalize_response_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        if text_parts:
            return "\n".join(text_parts)
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def check_format(out):
    labels = ("Instruction Compliance", "Visual Quality & Stability", "Consistency & Detail Fidelity")
    scores = {}
    for line in normalize_response_text(out).splitlines():
        label, separator, value = line.replace("**", "").strip().partition(":")
        label = label.strip()
        if separator and label in labels:
            try:
                score = float(value.strip())
            except ValueError:
                return False
            if label in scores or not score.is_integer() or not 1 <= score <= 5:
                return False
            scores[label] = int(score)
    return [scores[label] for label in labels] if len(scores) == len(labels) else False


def load_scored_indices(score_path: Path) -> set[int]:
    scored: set[int] = set()
    if not score_path.exists():
        return scored
    with score_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in record:
                scored.add(int(record["idx"]))
    return scored


def append_process_videos(tasks, score_path: Path, max_workers=20) -> int:
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor, score_path.open('a', encoding='utf-8') as f:
        futures = [executor.submit(run_two_videos_gemini, *task) for task in tasks]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="OpenVE scoring",
        ):
            item = future.result()
            if item:
                if len(item) == 5:
                    scores, result, meta_info, idx, timing_info = item
                else:
                    scores, result, meta_info, idx = item
                    timing_info = {}
                record = {"idx": idx, "scores": scores, "result": result}
                record.update(meta_info)
                record.update(timing_info)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                completed += 1
    return completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=MODEL_ID,
                        help="Gemini model to use for evaluation")
    parser.add_argument('--video_paths', type=str, nargs='+',
                        default=['outputs_eval/longv2vbench'],
                        help="Result video roots to evaluate.")
    parser.add_argument('--dataset-root', type=str, default='data/LongV2VBench',
                        help="LongBench dataset root.")
    parser.add_argument('--csv-path', type=str, default=None,
                        help="Metadata CSV; defaults to DATASET_ROOT/benchmark_videos.csv.")
    parser.add_argument('--manifest-path', type=str, default=None,
                        help="Optional manifest to use instead of the metadata CSV.")
    parser.add_argument('--source-from-dataset', action='store_true',
                        help="Read CSV source paths from the dataset instead of saved result-side source videos.")
    parser.add_argument('--task-types', type=str, default='all',
                        help="Comma-separated task types to evaluate, or 'all'.")
    parser.add_argument('--max-workers', type=int, default=int(os.environ.get('GEMINI_MAX_WORKERS', 16)))
    parser.add_argument('--max-attempts', type=int, default=int(os.environ.get('GEMINI_MAX_ATTEMPTS', 100)))
    parser.add_argument('--retry-sleep', type=float, default=float(os.environ.get('GEMINI_RETRY_SLEEP', 0)))
    parser.add_argument('--max-items', type=int, default=None,
                        help="Maximum number of unscored rows per result directory.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Only build and count evaluation tasks; do not call Gemini or write scores.")
    parser.add_argument('--eval-downsample', action=argparse.BooleanOptionalAction, default=True,
                        help="Resize to aspect-ratio buckets and sample frames before scoring.")
    parser.add_argument('--eval-fps', type=int, default=int(os.environ.get('GEMINI_EVAL_FPS', 4)))
    parser.add_argument('--eval-bucket-base-size', type=int, nargs=2, default=EVAL_BUCKET_BASE_SIZE,
                        metavar=('HEIGHT', 'WIDTH'), help="Base size for evaluation buckets; defaults to 480 832.")
    parser.add_argument('--eval-crf', type=int, default=int(os.environ.get('GEMINI_EVAL_CRF', 32)))
    parser.add_argument('--eval-max-seconds', type=int, default=int(os.environ.get('GEMINI_EVAL_MAX_SECONDS', 60)))
    args = parser.parse_args()
    if min(args.eval_bucket_base_size) <= 0 or args.eval_fps <= 0 or args.eval_max_seconds <= 0:
        parser.error("Evaluation dimensions, FPS, and duration must be positive.")
    if not 0 <= args.eval_crf <= 51:
        parser.error("Evaluation CRF must be between 0 and 51.")
    if args.max_workers <= 0 or args.max_attempts <= 0 or args.retry_sleep < 0:
        parser.error("Workers and attempts must be positive; retry sleep must be non-negative.")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive.")
    if not args.dry_run and not API_KEY:
        parser.error("Set GEMINI_API_KEY before scoring.")
    MODEL_ID = args.model_id
    MAX_ATTEMPTS = args.max_attempts
    RETRY_SLEEP = args.retry_sleep
    EVAL_DOWNSAMPLE = args.eval_downsample
    EVAL_FPS = args.eval_fps
    EVAL_BUCKET_BASE_SIZE = tuple(args.eval_bucket_base_size)
    EVAL_CRF = args.eval_crf
    EVAL_MAX_SECONDS = args.eval_max_seconds
    video_paths = args.video_paths
    dataset_root = Path(args.dataset_root)
    csv_path = Path(args.csv_path) if args.csv_path else dataset_root / "benchmark_videos.csv"
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    input_path = manifest_path or csv_path
    if not input_path.is_file():
        parser.error(f"Metadata file not found: {input_path}")
    selected_task_types = parse_task_types(args.task_types)
    score_filename = get_score_filename(MODEL_ID, args.task_types, args.source_from_dataset)
    for save_dir in video_paths:
        save_dir = Path(save_dir)
        score_path = save_dir / score_filename
        scored_indices = load_scored_indices(score_path)
        if scored_indices:
            print(f"Found existing score file {score_path}, resume with {len(scored_indices)} scored records.")
        tasks = []
        print(f"Using input metadata: {input_path}")
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = row.get('instruction') or row.get('original_prompt') or row.get('prompt', '')
                if not prompt.strip():
                    print(f"Missing prompt, skip idx={idx}")
                    continue
                edited_type = normalize_edited_type(row.get('category') or row.get('edited_type', ''))
                if selected_task_types is not None and edited_type not in selected_task_types:
                    continue
                if idx in scored_indices:
                    continue
                if edited_type not in prompt_type:
                    print(f"Unsupported edited_type={edited_type!r}, skip idx={idx}")
                    continue
                if input_path == manifest_path:
                    edited_video_path = build_manifest_edited_video_path(save_dir, row)
                    video_path = build_manifest_source_video_path(row, dataset_root)
                    video_id = row.get('video_id') or Path(video_path).stem
                else:
                    edited_video_path = build_edited_video_path(save_dir, edited_type, row['original_video'])
                    video_path = build_source_video_path(
                        save_dir, edited_type, row['original_video'], dataset_root, args.source_from_dataset,
                    )
                    video_id = Path(row['original_video']).stem
                if not edited_video_path.is_file():
                    print(f"Missing edited video, skip: {edited_video_path}")
                    continue
                if not video_path.is_file():
                    print(f"Missing source video, skip: {video_path}")
                    continue
                sys_prompt = prompt_type[edited_type].format(edit_prompt=prompt)
                meta_info = {
                    "video_id": video_id,
                    "prompt": prompt,
                    "edited_type": edited_type,
                    "source_video": str(video_path),
                    "edited_video": str(edited_video_path),
                }
                tasks.append((str(video_path), str(edited_video_path), sys_prompt, meta_info, idx))
                if args.max_items is not None and len(tasks) >= args.max_items:
                    break

            print(f"Scoring {len(tasks)} videos for {save_dir}")
            print(f"Score file will be saved to {score_path}")
            if args.dry_run:
                type_counts = defaultdict(int)
                for task in tasks:
                    type_counts[task[3]['edited_type']] += 1
                print("Dry run task counts:")
                for edited_type, count in sorted(type_counts.items()):
                    print(f"  {edited_type}: {count}")
                continue
            score_path.parent.mkdir(parents=True, exist_ok=True)
            completed = append_process_videos(tasks, score_path, max_workers=args.max_workers)
            print(f"Appended {completed} scored records to {score_path}")
            if completed < len(tasks):
                print(f"{len(tasks) - completed} tasks failed; rerun to retry the missing scores.")

    if args.dry_run:
        raise SystemExit(0)

    for save_dir in video_paths:
        file_path = Path(save_dir) / score_filename
        if not file_path.exists():
            print(f"Score file not found, skip summary: {file_path}")
            continue
        averages = avg_score_by_edited_type(file_path)
        if not averages:
            print(f"No valid scores found, skip summary: {file_path}")
            continue
        all_scores = sum(averages.values()) / len(averages)
        summary_lines = [f"All scores: {all_scores:.2f}"]
        print(summary_lines[0])
        for edited_type, avg in averages.items():
            line = f"  {edited_type}: {avg:.2f}"
            summary_lines.append(line)
            print(line)

        summary_path = file_path.with_suffix(".txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"Summary saved to {summary_path}")
