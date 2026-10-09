 # Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 
import argparse
import csv
import os
from tqdm import tqdm
import time
import re
import json
import base64
import logging
import mimetypes
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI


API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://modelservice.jdcloud.com/v1")
MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "Gemini-2.5-pro")
MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "5"))
RETRY_SLEEP = float(os.environ.get("GEMINI_RETRY_SLEEP", "60"))

Global_Style = """\nYou are a data rater specializing in grading video style transfer edits. You will be given an input video, a reference style (image or video), and the styled result video. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:\n\nStyle Fidelity\n1. Target style absent or clearly wrong.\n2. Style shows in a few areas/frames only, or mixed with unrelated styles.\n3. Key traits (palette, brushwork, texture) present but patchy or inconsistent across frames.\n4. Style reproduced well across almost the whole video; only small local or brief temporal mismatches.\n5. Full, faithful transfer: colour, texture, brushwork, and lighting all match the exemplar consistently over the entire duration and space of the video.\n\nContent Preservation\n1. Major objects, layout, or overall motion lost/distorted; original scene barely recognisable.\n2. Main subject recognisable, but its size, perspective, motion, or key parts are clearly wrong/missing.\n3. Overall structure and motion correct; some local warping, minor omissions, or slight motion jerkiness.\n4. Nearly all geometry and motion intact; only slight, non-distracting deformation.\n5. All objects, spatial relations, and motion are perfectly kept; only stylistic, harmless distortion.\n\nTemporal Stability\n1. Extreme flickering or "boiling" effects; the style is completely unstable frame-to-frame, making the video unwatchable.\n2. Significant and distracting flickering or temporal inconsistency in style application.\n3. Noticeable but tolerable flicker or texture "boiling", especially during motion.\n4. Largely stable with only minor, subtle flickering visible in areas of complex motion or fine texture.\n5. Perfectly stable and temporally coherent; the style appears "stuck" to the scene with no flickering.\n\nThe scores for Content Preservation and Temporal Stability should not be higher than the Style Fidelity score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the scores based on the criteria above, no more than 30 words.\nStyle Fidelity: A number from 1 to 5.\nContent Preservation: A number from 1 to 5.\nTemporal Stability: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""
Local_Change = """\nYou are a data rater specializing in grading video replacement edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives, paying close attention to temporal consistency (how the edit holds up over time and with motion).\n\nPrompt Compliance\n1 Target not replaced, or an unrelated object/part of the video edited.\n2 Only part of the target replaced (e.g., in only a few frames), or wrong class/description used.\n3 Target largely replaced but other objects altered, remnants visible across frames, or count/position clearly wrong.\n4 Correct object fully replaced for the entire duration; only minor attribute errors (colour, size, etc.).\n5 Perfect replacement: all and only the specified objects replaced for the entire duration; new objects’ class, number, position, scale, pose, motion and detail exactly match the prompt.\n\nVisual Naturalness & Temporal Stability\n1 Video heavily broken or new object deformed / flickers uncontrollably / jitters erratically.\n2 Obvious seams/edges that flicker or move unnaturally; strong mismatch in resolution or colour that is inconsistent across frames; background not restored or is unstable.\n3 Basic style similar, but lighting or palette clashes are inconsistent as the video plays; fuzzy edges, noise or minor flickering/jittering are noticeable.\n4 Style almost uniform and stable; tiny temporal artefacts (e.g., edge shimmer) visible only on close, frame-by-frame inspection; casual viewers see no edit.\n5 Completely seamless and temporally stable; new objects blend fully with the scene in every frame, edit area undetectable.\n\nPhysical & Motion Integrity\n1 Floating or sliding unnaturally (poor motion tracking), severe perspective/light errors inconsistent with camera/object movement; background heavily warped or unstable.\n2 Missing or static shadows/reflections that do not move with the object/light; poor occlusion; new object’s motion clearly mismatches scene motion.\n3 Lighting, perspective and interactions mostly correct but with minor inconsistencies over time; motion tracking has small, tolerable drifts.\n4 New object's motion is well-tracked and it interacts realistically with the scene (shadows, reflections) and preserves existing details throughout the video.\n5 Physically and dynamically flawless: motion, perspective, shadows, and reflections are perfectly integrated and move correctly with the scene and camera in every frame; background untouched and stable.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Naturalness & Temporal Stability: A number from 1 to 5.\nPhysical & Motion Integrity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""
Background_Change = """\nYou are a data rater specializing in grading video background editing. You will be given two videos (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:\n\nInstruction Compliance\n1 No change, or background unrelated to prompt, or foreground also replaced/distorted.\n2 Background partly replaced or wrong style/content; foreground noticeably altered.\n3 Main background replaced but elements missing/extra, or faint spill onto subject edges.\n4 Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).\n5 Background exactly matches prompt (content, style, placement); all foreground pixels untouched.\n\nVisual & Temporal Seamlessness (Edge, Blend & Stability)\n1 Large tearing, posterisation, or significant temporal artifacts like flickering, jittering edges; edit area obvious at a glance.\n2 Clear cut-out halos, colour-resolution gap, or obvious edge 'boiling' (instability) over time.\n3 Blend acceptable but visible on closer look: slight edge blur, or minor temporal instability (e.g., shimmer).\n4 Nearly invisible seams; edges are stable across motion, textures aligned, only minor issues when zoomed in.\n5 Indistinguishable composite: edges, textures, resolution and colour grading are perfectly continuous and stable throughout the video's duration.\n\nPhysical Consistency (Lighting, Perspective, Motion & Depth)\n1 Severe mismatch: wrong horizon, conflicting light, floating subject, or background remains static during camera movement (no parallax).\n2 Noticeable inconsistencies in light or scale; incorrect perspective shifts during motion.\n3 Overall believable; small errors in shadow, perspective, or minor motion tracking flaws.\n4 Lighting, scale, and depth well matched; background perspective and scale track convincingly with camera motion.\n5 Physically flawless: foreground and new background share coherent light, shadows, perspective, and atmospheric depth throughout all subject and camera motion, enhancing overall realism.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nInstruction Compliance: A number from 1 to 5.\nVisual & Temporal Seamlessness: A number from 1 to 5.\nPhysical Consistency: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""
Local_Remove = """\nYou are a data rater specializing in grading video object removal editing. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the edit quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 No edit performed, the video is corrupted, or the edit is completely wrong.\n2 Wrong object/class removed, or target only partially removed, or an unrelated object is also removed.\n3 Correct object removed, but with significant errors: unintended objects are also removed, OR significant fragments/ghosting of the target remain.\n4 The correct object is removed; only minor issues like a few tiny fragments remaining or tiny, unintended background items being affected.\n5 Perfect: All and only the requested objects are removed as instructed; every other element is untouched.\n\nVisual & Temporal Naturalness\n1 Video is badly broken, full of artefacts, or shows severe flickering/jittering throughout.\n2 Obvious erase marks or "smudges"; the inpainted background's style, resolution, or palette strongly mismatches; the edited region jitters or appears static against a moving background.\n3 General style is similar, but the inpainted background's lighting/colours clearly clash or are inconsistent across frames; noticeable temporal disharmony.\n4 Style is almost uniform; minor edge issues around the removed area or slight temporal instability (e.g., minor flicker) visible only on close inspection.\n5 Perfectly seamless; the removal is temporally stable and visually indistinguishable from a clean background.\n\nPhysical & Detail Coherence\n1 Key original elements are blocked by poor inpainting; the background is heavily distorted or hallucinates incorrect structures; motion is completely wrong (e.g., a static patch in a moving scene).\n2 The inpainted background visibly shifts, jitters, or is poorly reconstructed over time, failing to match the original scene's motion.\n3 Background reconstruction is mostly correct and consistent; remaining flaws are small and acceptable; background changes are localized and stable.\n4 No loss of original detail around the removed area; background reconstruction is clean, stable, and respects the scene's geometry and motion.\n5 The background is essentially untouched and stable; the inpainted area perfectly matches the surrounding content's motion, texture, and detail over time.\n\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual & Temporal Naturalness: A number from 1 to 5.\nPhysical & Detail Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:"""
Local_Add = """\nYou are a data rater specializing in grading video object addition editing. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the edit quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 No edit performed, the video is corrupted, or the edit is completely wrong.\n2 Wrong object/class added, or target only partially added, or an unrelated object is also added.\n3 Correct object added, but with significant errors: key attributes (e.g., position, colour, count, size) are wrong.\n4 The correct object is added with main attributes correct; only minor details are off (e.g., slight colour mismatch, minor position error).\n5 Perfect: All and only the requested objects are added as instructed; every other element is untouched.\n\nVisual & Temporal Naturalness\n1 Video is badly broken, full of artefacts, or shows severe flickering/jittering throughout.\n2 Obvious paste marks; style, resolution, or palette of the added object strongly mismatches; the added region jitters or appears static against a moving background.\n3 General style is similar, but lighting/colours on the added object clearly clash or are inconsistent across frames; noticeable temporal disharmony.\n4 Style is almost uniform; minor edge issues around the added object or slight temporal instability (e.g., minor flicker) visible only on close inspection.\n5 Perfectly seamless; the edit is temporally stable and visually indistinguishable from the original video's content and motion.\n\nPhysical & Detail Coherence\n1 Severe physical errors (e.g., the added object floats, has wrong perspective/lighting); key original elements are blocked; motion of the added object is completely wrong.\n2 Contact with surfaces, occlusion by other objects, or motion of the added object is handled poorly.\n3 Lighting, perspective, and motion of the added object are mostly correct and consistent with the scene; remaining flaws are small and acceptable.\n4 Shadows, reflections, and material response from the added object are believable and move correctly with the scene; no loss of original detail.\n5 Edit enhances overall realism: the added object has precise highlights, shadows, and motion effects that are temporally coherent and perfectly integrated.\n\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual & Temporal Naturalness: A number from 1 to 5.\nPhysical & Detail Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:"""


TASK_PROMPTS = {
    "global_style": Global_Style,
    "local_change": Local_Change,
    "background_change": Background_Change,
    "local_remove": Local_Remove,
    "local_add": Local_Add,
}


def extract_scores_and_average(entry: str, edited_type: str) -> float | None:
    labels = re.findall(r"(?:^|\n)([^\n:]+): A number from 1 to 5\.", TASK_PROMPTS[edited_type])
    scores = {}
    for line in entry.replace("\\n", "\n").splitlines():
        label, separator, value = line.replace("**", "").strip().partition(":")
        label = label.strip()
        if separator and label in labels:
            try:
                score = float(value.strip())
            except ValueError:
                return None
            if label in scores or not score.is_integer() or not 1 <= score <= 5:
                return None
            scores[label] = score
    return sum(scores.values()) / len(labels) if len(labels) == 3 and len(scores) == 3 else None


def video_to_data_uri(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if mime_type is None:
        mime_type = "video/mp4"
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{video_base64}"


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


def call_gemini_model(original_video_path, edited_video_path, prompt):
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        original_video_uri = video_to_data_uri(original_video_path) if os.path.exists(original_video_path) else original_video_path
        edited_video_uri = video_to_data_uri(edited_video_path) if os.path.exists(edited_video_path) else edited_video_path
    except Exception as e:
        return f"ERROR: Init Gemini client or encode videos failed: {e}"

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {"type": "video_url", "video_url": {"url": original_video_uri}},
                            {"type": "video_url", "video_url": {"url": edited_video_uri}},
                        ],
                    }
                ],
            )
            result = response.choices[0].message.content or ""
            if attempt > 0:
                logging.info(f"The Gemini call succeeded after {attempt} retries.")
            return normalize_response_text(result)
        except Exception as e:
            error_msg = f"An error occurred while calling the Gemini model.: {e}"
            logging.warning(f"Retry for {attempt + 1}th time: {error_msg}")
            if RETRY_SLEEP > 0:
                time.sleep(RETRY_SLEEP)

    return error_msg


def load_completed_rows(output_csv_path):
    completed = {}
    if not os.path.exists(output_csv_path):
        return completed, None
    with open(output_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return completed, None
        for row in reader:
            key = (row.get('original_video', ''), row.get('edited_result_path', ''), row.get('prompt', ''), row.get('edited_type', ''))
            result = row.get('results', '')
            average = extract_scores_and_average(result, key[3]) if key[3] in TASK_PROMPTS else None
            if key[0] and key[1] and average is not None:
                row['average'] = average
                completed[key] = row
    return completed, fieldnames


def evaluate_row(row_idx, row, header, root_path, edited_video_path):
    try:
        edited_type = row.get('edited_type', '')
        prompt = row.get('prompt', '')
        original_video = row.get('original_video', '')
        edited_result_path = row.get(edited_video_path, '')
        original_video_abs = os.path.join(root_path, original_video)

        if not os.path.exists(edited_result_path):
            print(f"Warning: The edited video file does not exist.: {edited_result_path}")
            original_row = [row.get(col, '') for col in header]
            return original_row + [f"ERROR: Video file not found: {edited_result_path}", "ERROR"], None, edited_type

        if not os.path.exists(original_video_abs):
            print(f"Warning: The original video file does not exist.: {original_video_abs}")
            original_row = [row.get(col, '') for col in header]
            return original_row + [f"ERROR: Video file not found: {original_video_abs}", "ERROR"], None, edited_type

        system_prompt = TASK_PROMPTS[edited_type]

        full_system_prompt = system_prompt.replace('<edit_prompt>', prompt)
        print(f"Scoring row {row_idx + 1}")
        response = call_gemini_model(original_video_abs, edited_result_path, full_system_prompt)
        formatted_response = response.replace('\n', '\\n')
        average_score = extract_scores_and_average(response, edited_type)
        original_row = [row.get(col, '') for col in header]
        return original_row + [formatted_response, average_score], average_score, edited_type
    except Exception as e:
        print(f"Scoring row {row_idx + 1} failed: {e}")
        original_row = [row.get(col, '') for col in header]
        return original_row + [f"ERROR: {e}", "ERROR"], None, row.get('edited_type', '')


def process_csv(input_csv_path, output_csv_path, root_path, edited_video_path="edited_result_path", max_workers=1, resume=True):
    start = time.time()
    all_scores_by_type = {}
    all_scores = []
    completed_rows, _ = load_completed_rows(output_csv_path) if resume else ({}, None)

    with open(input_csv_path, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        header = reader.fieldnames
        output_header = header + ['results', 'average']
        rows = list(reader)

    pending = []
    output_rows = []
    for row_idx, row in enumerate(rows):
        key = (row.get('original_video', ''), row.get(edited_video_path, ''), row.get('prompt', ''), row.get('edited_type', ''))
        if key in completed_rows:
            output_rows.append([completed_rows[key].get(col, '') for col in output_header])
            try:
                avg = float(completed_rows[key].get('average', ''))
                edited_type = row.get('edited_type', '')
                all_scores.append(avg)
                all_scores_by_type.setdefault(edited_type, []).append(avg)
            except Exception:
                pass
        else:
            output_rows.append(None)
            pending.append((row_idx, row))

    print(f"Resume enabled: {len(rows) - len(pending)} completed, {len(pending)} pending. max_workers={max_workers}")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def write_all():
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(output_header)
            for item in output_rows:
                if item is not None:
                    writer.writerow(item)

    if max_workers <= 1:
        for row_idx, row in tqdm(pending, desc=f"Processing {os.path.basename(input_csv_path)}"):
            result_row, average_score, edited_type = evaluate_row(row_idx, row, header, root_path, edited_video_path)
            output_rows[row_idx] = result_row
            if average_score is not None:
                all_scores.append(average_score)
                all_scores_by_type.setdefault(edited_type, []).append(average_score)
            write_all()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_row, row_idx, row, header, root_path, edited_video_path): row_idx for row_idx, row in pending}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(input_csv_path)}"):
                row_idx = futures[future]
                result_row, average_score, edited_type = future.result()
                output_rows[row_idx] = result_row
                if average_score is not None:
                    all_scores.append(average_score)
                    all_scores_by_type.setdefault(edited_type, []).append(average_score)
                write_all()

    type_averages = {}
    for edited_type, scores in all_scores_by_type.items():
        valid_scores = [score for score in scores if 1 <= score <= 5]
        type_averages[edited_type] = sum(valid_scores) / len(valid_scores) if valid_scores else None

    valid_all_scores = [score for score in all_scores if 1 <= score <= 5]
    category_averages = [score for score in type_averages.values() if score is not None]
    overall_average = round(sum(category_averages) / len(category_averages), 2) if category_averages else None
    type_averages = {category: round(score, 2) if score is not None else None for category, score in type_averages.items()}
    write_all()

    stats_output_path = output_csv_path.replace('.csv', '_stats.json')
    stats_data = {
        "overall_average": overall_average,
        "type_averages": type_averages,
        "total_processed": len(rows),
        "total_valid_scores": len(valid_all_scores),
        "failed_count": len(rows) - len(valid_all_scores),
        "breakdown_by_type": {}
    }
    for k, v in all_scores_by_type.items():
        valid_scores_for_type = [score for score in v if 1 <= score <= 5]
        stats_data["breakdown_by_type"][k] = {
            "count": len(valid_scores_for_type),
            "average": round(sum(valid_scores_for_type) / len(valid_scores_for_type), 2) if valid_scores_for_type else None,
            "original_count": len(v),
            "invalid_count": len(v) - len(valid_scores_for_type)
        }
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete! Results saved to: {output_csv_path}")
    print(f"Statistical results saved to: {stats_output_path}")
    print(f"Average score for each editing type: {type_averages}")
    print(f"Overall average score: {overall_average}")
    print(f'Time for total: {time.time()-start} seconds')

def build_original_eval_csv(csv_path, save_dir, dataset_root, output_csv_path, task_types="all"):
    selected = None if task_types.strip().lower() == "all" else {x.strip() for x in task_types.split(',') if x.strip()}
    if selected is not None and not selected.issubset(TASK_PROMPTS):
        raise ValueError(f"Unsupported editing categories: {sorted(selected - TASK_PROMPTS.keys())}")
    with open(csv_path, 'r', encoding='utf-8-sig') as infile, \
        open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or []) + ["edited_result_path"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            edited_type = row.get('edited_type', '')
            if selected is not None and edited_type not in selected:
                continue
            if edited_type not in TASK_PROMPTS:
                continue
            original_name = Path(row.get('original_video', '')).stem + ".mp4"
            edited_path = Path(save_dir) / "fullset" / edited_type / original_name
            source_path = edited_path.with_name(edited_path.stem + "_src.mp4")
            if not edited_path.exists() or not source_path.exists():
                continue
            row = dict(row)
            row['original_video'] = str(source_path.resolve())
            row['edited_result_path'] = str(edited_path.resolve())
            prompt_path = edited_path.with_name(edited_path.stem + "_prompt.txt")
            if prompt_path.is_file():
                row['prompt'] = prompt_path.read_text(encoding="utf-8").strip()
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use the Gemini 2.5 Pro model to process video and prompt CSV files.")
    parser.add_argument("--input_csv", type=str, default="benchmark/your_model_out.csv", help="The path to the input CSV file.")
    parser.add_argument("--root_path", type=str, default="yours_OpenVE-Bench_path", help="The path to the OpenVE-Bench videos.")
    parser.add_argument("--output_csv", type=str, default="benchmark/your_model_out_gemini.csv", help="The path to the output CSV file.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Gemini model to use for evaluation.")
    parser.add_argument("--video_paths", type=str, nargs='+', default=None, help="Output video directories to evaluate, OpenVE-style.")
    parser.add_argument("--dataset-root", type=str, default=None, help="OpenVE-Bench dataset root, e.g. /path/OpenVE-Bench.")
    parser.add_argument("--csv-path", type=str, default=None, help="OpenVE-Bench benchmark CSV path.")
    parser.add_argument("--task-types", type=str, default="all", help="Comma-separated task types, or all.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("GEMINI_MAX_WORKERS", "1")), help="Parallel Gemini workers for original evaluator.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite completed rows.")
    args = parser.parse_args()
    if not API_KEY:
        parser.error("Set GEMINI_API_KEY before scoring.")
    MODEL_ID = args.model_id

    if args.video_paths:
        if not args.dataset_root or not args.csv_path:
            raise ValueError("--video_paths mode requires --dataset-root and --csv-path")
        for save_dir in args.video_paths:
            save_dir = Path(save_dir)
            input_csv = save_dir / "openve_original_eval_input.csv"
            output_csv = save_dir / f"openve_original_{MODEL_ID}_score.csv"
            build_original_eval_csv(args.csv_path, save_dir, args.dataset_root, input_csv, args.task_types)
            process_csv(str(input_csv), str(output_csv), args.dataset_root, edited_video_path="edited_result_path", max_workers=args.max_workers, resume=not args.no_resume)
    else:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        process_csv(args.input_csv, args.output_csv, args.root_path, edited_video_path="edited_result_path", max_workers=args.max_workers, resume=not args.no_resume)
