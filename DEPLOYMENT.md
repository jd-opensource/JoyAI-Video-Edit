# Deployment Guide

This guide takes JoyAI-Video-Edit streaming v2v editing from a fresh clone to a
fully working server: environment, weights, configuration, and launch.

JoyAI-Video-Edit is a real-time, streaming **video-to-video editing** service. A
client streams source frames over a WebSocket; the server runs a streaming DiT +
xVAE pipeline (with face/person presence gating) and streams edited frames back.

- **Serving stack:** FastAPI + WebSocket, served by `uvicorn`.
- **Entry point:** [`xvideo/serving/serve_joyomni_streaming.py`](deploy/xvideo/serving/serve_joyomni_streaming.py)
- **Launcher:** [`run_server.sh`](deploy/run_server.sh) (binds `0.0.0.0:8080` by default).
- **Web UI:** [`static/index.html`](deploy/static/index.html), served at `GET /`.

---

## 1. Layout

```
deploy/
├── run_server.sh          # launcher (env-driven; sources .env.local if present)
├── requirements.txt       # pinned Python deps (SageAttention built separately)
├── sageattention-cudagraph-stream.patch  # stream fix for SageAttention (see §2)
├── .env.example.fa4       # env template: B200 (FA4 -> cuDNN)
├── .env.example.sage      # env template: RTX PRO 6000 / RTX 5090 (sage -> cuDNN)
├── joyomni_ops/           # in-tree CUDA op library (FP8 GEMM + fused kernels); pip install
├── xvideo/                # service code
│   ├── config.py          # runtime/model config defaults
│   ├── utils.py           # resize buckets, seeding helpers
│   ├── inductor_autotune_fix.py  # torch 2.9+ compile-cache fix (see §5)
│   ├── models/            # dit/, vae/, pipeline, flow-match scheduler, loaders
│   └── serving/           # FastAPI app, streaming runtime, CUDA-graph runner, prompt-enhancement
├── static/index.html      # browser client
├── rv2v_reference/        # reference images for the UI
└── deps/                  # weights + compile cache — NOT in git
    ├── checkpoints/       # DiT / xVAE / MiMo-VL / onnx detectors (~51G)
    └── cache/             # torchinductor / triton / nv_compute caches
```

> **`deploy/deps/` is git-ignored.** It must exist on disk for the server to
> start, but it is not tracked by this repo — you populate it in step 3.

---

## 2. Prepare the environment

> All commands in this guide run from the **repo root** (the directory
> containing `deploy/`), unless a step explicitly `cd`s elsewhere.

SageAttention and `flash-attn-4` are **not** on PyPI and are not in
`requirements.txt`; build them separately after the base deps. The FP8 kernels
are provided by the in-tree `joyomni_ops` library, built in the next step.

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r deploy/requirements.txt
```

Both CUDA builds below (SageAttention and joyomni_ops) need **`nvcc` ≥ 12.8 in
the env** — Blackwell (`sm_100`/`sm_120`) support landed in 12.8; an older nvcc
either refuses the arch or emits no valid kernel. Install it into the conda env
(self-contained) and verify:

```bash
conda install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
                              libcublas-dev libcusparse-dev libcusolver-dev
nvcc --version | grep release                    # -> release 12.8
nvcc --list-gpu-code | grep -E "sm_100|sm_120"   # Blackwell target present
```

(`libcublas-dev` / `libcusparse-dev` / `libcusolver-dev` supply the
`cublas_v2.h` / `cusparse.h` / `cusolverDn.h` headers that PyTorch's CUDA
headers include during the build.)

Then install the attention and kernel dependencies:

- **SageAttention 2.2.0** (*required on sage machines — RTX PRO 6000 / RTX 5090;
  skip on B200*) — INT8/FP8 quantized attention used for the DiT's long-kv
  denoise attention (short-kv passes use PyTorch SDPA/cuDNN). Build from source
  with the bundled CUDA-graph stream fix:

  ```bash
  # from the repo root
  git clone https://github.com/thu-ml/SageAttention.git deploy/tmp/SageAttention
  cd deploy/tmp/SageAttention
  git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
  git apply ../../sageattention-cudagraph-stream.patch
  export CUDA_HOME=$CONDA_PREFIX
  # compile only for this machine's GPU architecture (e.g. B200 -> "10.0",
  # RTX 5090 / PRO 6000 -> "12.0"); auto-detected from the driver:
  export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))")
  echo "building for arch $TORCH_CUDA_ARCH_LIST"
  EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 python setup.py install
  cd -
  ```

  The patch routes every kernel launch through `at::cuda::getCurrentCUDAStream()`
  instead of the default stream — without it, upstream SageAttention records
  empty CUDA graphs (kernels escape capture) and the server's graph path
  produces noise. If sageattention is absent or `JOYOMNI_SAGE_ATTN=0`, attention
  falls back to SDPA (cuDNN).
- **flash-attn-4** (`4.0.0b13`, *required on FA4 machines — B200; skip on
  RTX PRO 6000 / 5090, its JIT does not support sm_120*) — provides
  `flash_attn.cute`; kernels JIT at runtime, no build step. Deps must be
  pinned exactly:

  ```bash
  python -m pip install flash-attn-4==4.0.0b13 \
    nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 apache-tvm-ffi==0.1.12
  ```

  If it can't be imported or its kernel fails at runtime, the DiT
  automatically falls back to cuDNN.
- **joyomni_ops** — in-tree CUDA op library ([`deploy/joyomni_ops/`](deploy/joyomni_ops/))
  providing the FP8 GEMM + fused norm/rope kernels the DiT uses (extracted from
  sgl-kernel, Apache-2.0, no `sgl_kernel`/`sglang` runtime dependency).

  The FP8 GEMM is built with [cutlass](https://github.com/NVIDIA/cutlass)
  (nvcc ≥ 12.8 for Blackwell — installed above). Build against a pinned
  cutlass checkout:

  ```bash
  git clone https://github.com/NVIDIA/cutlass.git deploy/tmp/cutlass
  git -C deploy/tmp/cutlass checkout dcf215af
  # like SageAttention above: build only this machine's arch (default is a
  # 5-arch fat binary — sm_80..120a — which multiplies compile time ~5x)
  export JOYOMNI_OPS_CUDA_ARCHS=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}a' if cc[0] >= 10 else f'{cc[0]}{cc[1]}')")
  echo "building joyomni_ops for sm_$JOYOMNI_OPS_CUDA_ARCHS"
  JOYOMNI_OPS_CUTLASS_DIR=$(pwd)/deploy/tmp/cutlass \
    python -m pip install --no-build-isolation ./deploy/joyomni_ops
  ```

  (`--no-build-isolation` reuses the env's existing `setuptools`/`torch` instead
  of pip fetching them into an isolated build env — required behind a restricted
  index/mirror, and it ensures the extension builds against the installed torch.)

> If you can't provide CUDA ≥ 12.8 (or cutlass), build the light variant
> (`JOYOMNI_OPS_NO_FP8=1 python -m pip install --no-build-isolation ./deploy/joyomni_ops`) and disable
> **both** FP8 paths — `JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0` — so nothing calls the
> FP8 kernel; the DiT then runs those Linears in bf16. SageAttention is
> independent of this — if absent, attention uses the SDPA/cuDNN path.

Verify the key runtime imports:

```bash
python - <<'PY'
import torch, cv2, av, transformers, diffusers
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("cv2", cv2.__version__, "| transformers", transformers.__version__)
try:
    import sageattention; print("sageattention: OK (default attention path)")
except Exception as e:
    print("sageattention: absent -> DiT uses SDPA/cuDNN fallback (slower)")
try:
    import flash_attn.cute; print("flash_attn.cute: OK (FA4 importable; kernels JIT at first use)")
except Exception as e:
    print("flash_attn: absent (optional) -> sage/SDPA path")
try:
    import joyomni_ops; print("joyomni_ops: OK | has_fp8 =", joyomni_ops.has_fp8())
except Exception as e:
    print("joyomni_ops: MISSING ->", e, "(build deploy/joyomni_ops; set JOYOMNI_FP8_IMG=0 to skip FP8)")
PY
```

---

## 3. Fetch the weights

All weights live under `deploy/deps/checkpoints/` (~51 GB total). Create it and
download each dependency.

```bash
mkdir -p deploy/deps/checkpoints
cd deploy/deps/checkpoints
```

**3a. DiT + xVAE** — the released JoyAI-Video-Edit weight repo:

```bash
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir JoyAI-Video-Edit \
  --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"
```

> The repo also ships the older `dit/joyai_video_edit_dit_0804.pth` (~32.5 GB);
> `--include` skips it — the server uses **0811**.

This should produce:

```
JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth
JoyAI-Video-Edit/vae/config.json
JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

**3b. Text/vision encoder** — MiMo-VL:

```bash
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir MiMo-VL-7B-RL-2508
```

**3c. ONNX detectors** (*optional*) — from `deploy/deps/checkpoints/`:

```bash
# YuNet face detector (OpenCV Zoo, git-LFS — use the media.githubusercontent URL)
curl -L -o face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

YOLOv8n must be exported at **`imgsz=320`** — the server loads it via `cv2.dnn` at
a fixed 320×320 (see `_person_present`), so a default (640) or dynamic export
throws a Reshape error at load, and third-party pre-exported `yolov8n.onnx` on the
Hub (typically 640/dynamic) will *not* load. Installing `ultralytics` drags in a
full stack (its own torch/CUDA wheels + non-headless `opencv-python`) that would
overwrite this project's pinned `torch` and `opencv-python-headless` — so export
in a **throwaway env**, never the deploy env:

```bash
conda create -n yolo-export python=3.10 -y
conda activate yolo-export

# CPU-only torch is enough for export and avoids pulling multi-GB CUDA wheels.
pip install --index-url https://pypi.org/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  ultralytics onnx onnxslim

# ultralytics pulls non-headless opencv-python (needs libGL, absent on headless
# boxes -> "libGL.so.1: cannot open shared object file"). Swap to headless:
pip uninstall -y opencv-python
pip install --index-url https://pypi.org/simple/ opencv-python-headless

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)"

conda deactivate
mv yolov8n.onnx deploy/deps/checkpoints/    # move the export into place
conda env remove -n yolo-export -y          # optional: drop the throwaway env
```

| File | Purpose |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | YuNet face-presence gate |
| `yolov8n.onnx` | YOLOv8n person-presence gate |

> The detectors are **optional**: if a file is missing the server just disables
> that gate (edits run unconditionally). The DiT, VAE, and MiMo-VL weights are the
> only hard requirements.

Final tree:

```
deploy/deps/checkpoints/
├── JoyAI-Video-Edit/
│   ├── dit/joyai_video_edit_dit_0811.pth
│   └── vae/{config.json, diffusion_pytorch_model.safetensors}
├── MiMo-VL-7B-RL-2508/
├── face_detection_yunet_2023mar.onnx
└── yolov8n.onnx
```

---

## 4. Configure this machine

Copy the template matching your GPU's attention path and fill in local values
(conda location, optional prompt enhancement). `run_server.sh` sources
`deploy/.env.local` automatically if present; it is git-ignored.

```bash
cp deploy/.env.example.fa4 deploy/.env.local    # B200
cp deploy/.env.example.sage deploy/.env.local   # RTX PRO 6000 / RTX 5090

# edit deploy/.env.local
```

Key variables:

| Variable | Meaning |
| --- | --- |
| `JOYOMNI_CONDA_SH` / `JOYOMNI_CONDA_ENV` | conda `profile.d/conda.sh` + env name/prefix to activate. Leave `JOYOMNI_CONDA_ENV` empty to use the already-active shell env. |
| `JOYOMNI_DEVICE` | CUDA device for all stages (default `cuda:0`). |
| `JOYOMNI_HOST` / `JOYOMNI_PORT` | bind address (default `0.0.0.0:8080`). |
| `JOYOMNI_WIDTH` / `JOYOMNI_HEIGHT` / `JOYOMNI_FPS` | Output resolution and frame rate (default `840` / `480` / `24` = 480p @ 24 FPS). Per-GPU presets in §5. |
| `JOYOMNI_FP8_IMG` / `JOYOMNI_FP8_TXT` | FP8 image / text paths via `joyomni_ops` (default `1` / `1`). Set both `0` to run bf16 (e.g. a `JOYOMNI_OPS_NO_FP8=1` build). |
| `JOYOMNI_CUDA_GRAPH` | capture the steady-state chunk loop into a CUDA graph (default `1`; the biggest single speedup). `0` runs eager. |
| `JOYOMNI_SAGE_ATTN` | SageAttention for long-kv attention (default `1`). `0` falls back to SDPA/cuDNN. |
| `JOYOMNI_TXT_PARALLEL` | run each block's txt branch on a side CUDA stream inside the graph (default `1`). |
| `JOYOMNI_CKPT_ROOT` | override the checkpoints dir (default `deploy/deps/checkpoints`). |
| `JOYOMNI_DIT_CKPT` / `JOYOMNI_VAE_CKPT` / `JOYOMNI_TEXT_ENCODER_CKPT` / `JOYOMNI_FACE_ONNX` / `JOYOMNI_PERSON_ONNX` | override individual weight paths (default: derived from `JOYOMNI_CKPT_ROOT`). |
| `JOYOMNI_RECORD_DIR` | recording output dir. |
| `PE_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | OpenAI-compatible endpoint for prompt enhancement. If unset, the server falls back to the raw user prompt. |

---

## 5. Launch

```bash
bash deploy/run_server.sh
```

The launcher activates the env (if configured), exports the compile-cache dirs
under `deps/cache/`, wires up the vendored checkpoint paths, and starts the server
with every stage on a single device (`cuda:0` by default).

Open the UI:

```
http://<server-ip>:8080/
```

### Resolution / FPS by GPU

The server defaults to **840×480 @ 24 FPS** (480p). Override per GPU with `JOYOMNI_WIDTH` / `JOYOMNI_HEIGHT` / `JOYOMNI_FPS`:

- **NVIDIA B200** — native 720p @ 24 FPS:

  ```bash
  JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=24 bash deploy/run_server.sh
  ```

- **RTX PRO 6000** — 480p @ 24 FPS, or native 720p @ 16 FPS. The [live HuggingFace demo](https://huggingface.co/spaces/wxDai/joyai-video-edit) runs the 480p @ 24 FPS preset on this GPU:

  ```bash
  # 480p @ 24 FPS
  JOYOMNI_WIDTH=840 JOYOMNI_HEIGHT=480 JOYOMNI_FPS=24 bash deploy/run_server.sh
  # native 720p @ 16 FPS
  JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=16 bash deploy/run_server.sh
  ```

- **RTX 5090** — coming soon.

> The first launch is slow: PyTorch/Triton/CUDA kernels and the DiT attention path
> compile and warm up. Keep `deploy/deps/cache/` stable across restarts to reuse
> the compile artifacts. After moving or re-cloning the repo, the cache stores
> absolute paths and is rebuilt once.
>
> Warm restarts skip all kernel compilation. `xvideo/inductor_autotune_fix.py`
> (auto-installed from `vae_compile` / `serve main`) patches torch 2.9-2.11 defects (upstream pytorch#172819, partially fixed in 2.12; the remaining .best_config ping-pong is patched through 2.13)
> where kernels restored from the FX-graph cache re-ran coordinate-descent
> autotuning on every boot (tens of seconds of GPU re-benchmarking, and the
> `.best_config` files kept being rewritten). With the fix, the second boot
> replays everything from `deps/cache/` — warmup time is pure model load +
> cached-artifact deserialization + one eager full-pipeline pass. Set
> `JOYOMNI_NO_COORDESC_CACHE_FIX=1` to disable the patch if a future torch
> upgrade changes this code path.

---

## 6. Common overrides

Append server flags after the script (forwarded to the Python entry point):

```bash
bash deploy/run_server.sh --port 7860
```

Run without FP8 (e.g. `joyomni_ops` built with `JOYOMNI_OPS_NO_FP8=1`) — disable **both** FP8 paths:

```bash
JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 bash deploy/run_server.sh
```

Custom checkpoint locations:

```bash
JOYOMNI_DIT_CKPT=/path/to/joyai_video_edit_dit_0811.pth \
JOYOMNI_VAE_CKPT=/path/to/vae \
JOYOMNI_TEXT_ENCODER_CKPT=/path/to/MiMo-VL-7B-RL-2508 \
bash deploy/run_server.sh
```

Custom recording directory:

```bash
JOYOMNI_RECORD_DIR=/path/to/recordings bash deploy/run_server.sh
```

---

## 7. Sanity checks

Files in place:

```bash
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
test -d deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

Server health after launch:

```bash
curl http://127.0.0.1:8080/health
```

---

## HTTP / WebSocket API

| Method | Path             | Purpose                                        |
|--------|------------------|------------------------------------------------|
| GET    | `/`              | Web UI (`static/index.html`)                   |
| GET    | `/health`        | Health check                                   |
| GET    | `/debug`         | Runtime debug state                            |
| GET    | `/ref-images`    | List reference images for the UI               |
| POST   | `/load`          | Warm up / load the model                       |
| GET    | `/download_last` | Download the last produced clip                |
| WS     | `/ws`            | Stream source frames in, edited frames out     |
