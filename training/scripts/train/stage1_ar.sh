#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/stage1_ar.py}"
GPUS_PER_NODE="${2:-${GPUS_PER_NODE:-1}}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

exec torchrun \
    --nnodes "${NNODES:-1}" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --node_rank "${NODE_RANK:-0}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29500}" \
    --module src.train_t2v --config "$CONFIG" "${@:3}"
