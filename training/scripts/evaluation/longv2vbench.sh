#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/stage2_distillation.py}"
CKPT="${2:-${JOYAI_CHECKPOINT:-}}"
: "${CKPT:?Pass a checkpoint as the second argument or set JOYAI_CHECKPOINT.}"
DATASET_ROOT="${DATASET_ROOT:-data/LongV2VBench}"
METADATA_PATH="${METADATA_PATH:-${DATASET_ROOT}/benchmark_videos.csv}"
SEGMENTS="${SEGMENTS:-all}"
SAVE="${SAVE:-outputs_eval/longv2vbench}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
NEG_PROMPT="An abstract, computer-generated scene with distorted and blurry visuals. A deformed, disfigured figure without specific features, depicted as an illustration. The background is a collage of grainy textures and striped patterns, lacking clear visual content. The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, grainy effects, and includes logos and text elements. The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition."

EXTRA_ARGS=()
if [[ "${USE_RELATIVE_ROPE:-True}" == "True" ]]; then
    EXTRA_ARGS+=(--use-relative-rope)
fi
if [[ "${STORE_CLEAN_ONLY_SELF:-True}" == "True" ]]; then
    EXTRA_ARGS+=(--store-clean-only-self)
else
    EXTRA_ARGS+=(--no-store-clean-only-self)
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
    --module evaluation.benchmark.infer_longv2vbench_joyomni \
    --config "$CONFIG" \
    --ckpt-path "$CKPT" \
    --save-path "$SAVE" \
    --dataset-root "$DATASET_ROOT" \
    --metadata-path "$METADATA_PATH" \
    --segments "$SEGMENTS" \
    --source-guidance-scale "${SRC_CFG:-1.0}" \
    --neg-prompt "$NEG_PROMPT" \
    --resolution "${RESOLUTION:-720}" \
    --max-temporal-ids "${MAX_TEMPORAL_IDS:-8}" \
    --seed 42 \
    "${EXTRA_ARGS[@]}" "${@:3}"

if [[ "${RUN_EVAL:-False}" == "True" && "$NODE_RANK" == "0" ]]; then
    : "${GEMINI_API_KEY:?Set GEMINI_API_KEY to run evaluation.}"
    python evaluation/longv2vbench/eval_longbench_gemini.py \
        --video_paths "$SAVE" \
        --dataset-root "$DATASET_ROOT" \
        --csv-path "$METADATA_PATH" \
        --task-types "$SEGMENTS" \
        --source-from-dataset \
        --model_id "${MODEL_ID:-Gemini-2.5-pro}" \
        --max-workers "${MAX_WORKERS:-32}" \
        --max-attempts 20 --retry-sleep 5 \
        --eval-bucket-base-size 480 832 --eval-fps 4
fi
