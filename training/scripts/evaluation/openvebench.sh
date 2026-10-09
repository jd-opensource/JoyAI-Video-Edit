#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/stage2_distillation.py}"
CKPT="${2:-${JOYAI_CHECKPOINT:-}}"
: "${CKPT:?Pass a checkpoint as the second argument or set JOYAI_CHECKPOINT.}"
DATASET_ROOT="${DATASET_ROOT:-data/OpenVE-Bench}"
CSV_PATH="${CSV_PATH:-${DATASET_ROOT}/benchmark_videos.csv}"
TASK_TYPES="${TASK_TYPES:-global_style,background_change,local_change,local_remove,local_add}"
SAVE="${SAVE:-outputs_eval/openvebench}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
NEG_PROMPT="An abstract, computer-generated scene with distorted and blurry visuals. A deformed, disfigured figure without specific features, depicted as an illustration. The background is a collage of grainy textures and striped patterns, lacking clear visual content. The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, grainy effects, and includes logos and text elements. The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition."

EXTRA_ARGS=()
if [[ "${USE_RELATIVE_ROPE:-True}" == "True" ]]; then
    EXTRA_ARGS+=(--use-relative-rope)
fi
if [[ "${USE_PE:-False}" == "True" ]]; then
    EXTRA_ARGS+=(--pe-csv-path "${PE_CSV_PATH:-${DATASET_ROOT}/pe_benchmark_videos.csv}")
fi
if [[ -n "${INFERENCE_STEP:-}" ]]; then
    EXTRA_ARGS+=(--num-inference-steps "$INFERENCE_STEP")
fi
if [[ -n "${CFG:-}" ]]; then
    EXTRA_ARGS+=(--guidance-scale "$CFG")
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

torchrun \
    --nnodes "${NNODES:-${WORLD_SIZE:-1}}" \
    --nproc_per_node "${GPUS_PER_NODE:-8}" \
    --node_rank "$NODE_RANK" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-6000}" \
    --module evaluation.benchmark.infer_openvebench_joyomni \
    --config "$CONFIG" \
    --ckpt-path "$CKPT" \
    --save-path "$SAVE" \
    --dataset-root "$DATASET_ROOT" \
    --csv-path "$CSV_PATH" \
    --task-types "$TASK_TYPES" \
    --source-guidance-scale "${SRC_CFG:-1.0}" \
    --neg-prompt "$NEG_PROMPT" \
    --resolution "${RESOLUTION:-720}" \
    --max-temporal-ids "${MAX_TEMPORAL_IDS:-8}" \
    --store-clean-only-self \
    --seed 42 \
    "${EXTRA_ARGS[@]}" "${@:3}"

if [[ "${RUN_EVAL:-False}" == "True" && "$NODE_RANK" == "0" ]]; then
    : "${GEMINI_API_KEY:?Set GEMINI_API_KEY to run evaluation.}"
    python evaluation/OpenVE-Bench/eval_openve_gemini.py \
        --video_paths "$SAVE" \
        --dataset-root "$DATASET_ROOT" \
        --csv-path "$CSV_PATH" \
        --task-types "$TASK_TYPES" \
        --model_id "${MODEL_ID:-Gemini-2.5-pro}" \
        --max-workers "${MAX_WORKERS:-32}"
fi
