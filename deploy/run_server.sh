#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$HERE/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env.local"
  set +a
fi

JOYOMNI_CONDA_SH="${JOYOMNI_CONDA_SH:-}"
JOYOMNI_CONDA_ENV="${JOYOMNI_CONDA_ENV:-}"
if [ -n "$JOYOMNI_CONDA_ENV" ]; then
  : "${NVCC_PREPEND_FLAGS:=}" "${NVCC_APPEND_FLAGS:=}"
  export NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
  if [ -n "$JOYOMNI_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$JOYOMNI_CONDA_SH"
  fi
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
  conda activate "$JOYOMNI_CONDA_ENV"
  hash -r
fi

cd "$HERE"

export TORCHINDUCTOR_CACHE_DIR="$HERE/deps/cache/torchinductor"
export TRITON_CACHE_DIR="$HERE/deps/cache/triton"
export CUDA_CACHE_PATH="$HERE/deps/cache/nv_compute"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE"

export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"
export JOYOMNI_CUDA_GRAPH="${JOYOMNI_CUDA_GRAPH:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-1}"
export JOYOMNI_TXT_PARALLEL="${JOYOMNI_TXT_PARALLEL:-1}"

RECORD_DIR="${JOYOMNI_RECORD_DIR:-$HERE/recordings}"

CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$HERE/deps/checkpoints}"
DIT_CKPT="${JOYOMNI_DIT_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth}"
VAE_CKPT="${JOYOMNI_VAE_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/vae}"
TE_CKPT="${JOYOMNI_TEXT_ENCODER_CKPT:-$CKPT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${JOYOMNI_FACE_ONNX:-$CKPT_ROOT/face_detection_yunet_2023mar.onnx}"
PERSON_ONNX="${JOYOMNI_PERSON_ONNX:-$CKPT_ROOT/yolov8n.onnx}"

DEVICE="${JOYOMNI_DEVICE:-cuda:0}"
HOST="${JOYOMNI_HOST:-0.0.0.0}"
PORT="${JOYOMNI_PORT:-8080}"

python xvideo/serving/serve_joyomni_streaming.py \
  --dit-ckpt          "$DIT_CKPT" \
  --vae-ckpt          "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx   "$FACE_ONNX" \
  --person-detector-onnx "$PERSON_ONNX" \
  --record-dir "$RECORD_DIR" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --width "${JOYOMNI_WIDTH:-840}" --height "${JOYOMNI_HEIGHT:-480}" \
  --fps "${JOYOMNI_FPS:-24}" \
  --host "$HOST" --port "$PORT" \
  "$@"
