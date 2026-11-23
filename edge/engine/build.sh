#!/usr/bin/env bash
# Build the joyedge CPU engine (libjoyedge + joyedge-quantize + joyedge-bench).
#   edge/engine/build.sh            # release build into edge/engine/build
#   GGML_DIR=/path/to/ggml edge/engine/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GGML_DIR="${GGML_DIR:-$HERE/third_party/ggml}"
GGML_COMMIT="${GGML_COMMIT:-d9fe7bff3d2bf4d077e25c654486d02c321734aa}"   # commit the engine was developed and tested against

if [ ! -f "$GGML_DIR/CMakeLists.txt" ]; then
  mkdir -p "$(dirname "$GGML_DIR")"
  git clone https://github.com/ggml-org/ggml "$GGML_DIR"
  git -C "$GGML_DIR" checkout "$GGML_COMMIT"
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DGGML_DIR="$GGML_DIR" "$@"
cmake --build "$HERE/build" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
echo "built: $HERE/build/libjoyedge.* $HERE/build/joyedge-quantize $HERE/build/joyedge-bench"
