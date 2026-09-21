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
├── run_server.sh          # launcher (env-driven)
├── requirements.txt       # pinned Python deps (SageAttention / flash-attn-4 / joyomni_ops built separately, §2)
├── sageattention-cudagraph-stream.patch  # stream fix for SageAttention (see §2)
├── joyomni_ops/           # in-tree CUDA op library (FP8 GEMM + fused kernels); pip install
├── xvideo/                # service code
│   ├── config.py          # runtime/model config defaults
│   ├── utils.py           # resize buckets, seeding helpers
│   ├── inductor_autotune_fix.py  # torch 2.9+ compile-cache fix
│   ├── lowvram.py         # low-VRAM mode switches (JOYOMNI_LOW_VRAM=1)
│   ├── models/            # dit/, vae/, pipeline, flow-match scheduler, loaders
│   └── serving/           # FastAPI app, streaming runtime, CUDA-graph runner, prompt-enhancement
├── static/index.html      # browser client
├── rv2v_reference/        # reference images for the UI
├── recordings/            # session recordings (created at runtime; git-ignored)
└── deps/                  # weights + compile cache — NOT in git
    ├── checkpoints/       # DiT / xVAE / MiMo-VL / onnx detectors (~51G)
    └── cache*/            # torchinductor / triton / nv_compute (one root per GPU model, §4)
```

> **`deploy/deps/` is git-ignored.** It must exist on disk for the server to
> start, but it is not tracked by this repo — you populate it in §3.

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

- **SageAttention 2.2.0** (*RTX 5090 only: GeForce runs fp32-accum SDPA at half
  rate, so int8 sage wins there; RTX PRO 6000 is net faster on plain cuDNN at
  the serving resolutions, and B200 uses FA4*) — INT8 quantized attention, used for all
  DiT denoise attention when `JOYOMNI_SAGE_ATTN=1`. Build from source with the
  bundled CUDA-graph stream fix:

  ```bash
  # from the repo root
  git clone https://github.com/thu-ml/SageAttention.git deploy/tmp/SageAttention
  cd deploy/tmp/SageAttention
  git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
  git apply ../../sageattention-cudagraph-stream.patch
  export CUDA_HOME=$CONDA_PREFIX
  export TORCH_CUDA_ARCH_LIST=12.0
  EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 python setup.py install
  cd -
  ```

  The patch routes every kernel launch through `at::cuda::getCurrentCUDAStream()`
  instead of the default stream — without it, upstream SageAttention records
  empty CUDA graphs (kernels escape capture) and the server's graph path
  produces noise. `JOYOMNI_SAGE_ATTN` is the only switch: default `0` → SDPA
  (cuDNN). Launch with `JOYOMNI_SAGE_ATTN=1` on an RTX 5090 (see §4); leave it
  unset on RTX PRO 6000 / B200.
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
  # build only this machine's arch (the default is a 5-arch fat binary —
  # sm_80..120a — which multiplies compile time ~5x); auto-detected:
  export JOYOMNI_OPS_CUDA_ARCHS=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}a' if cc[0] >= 9 else f'{cc[0]}{cc[1]}')")
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
    import sageattention; print("sageattention: OK (used when JOYOMNI_SAGE_ATTN=1 - RTX 5090)")
except Exception as e:
    print("sageattention: absent -> SDPA/cuDNN (only needed when JOYOMNI_SAGE_ATTN=1)")
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
```

**3a. DiT + xVAE** — the released JoyAI-Video-Edit weight repo:

```bash
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/JoyAI-Video-Edit \
  --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"
```

> The repo also ships the older `dit/joyai_video_edit_dit_0804.pth` (~32.5 GB);
> `--include` skips it — the server uses **0811**.

This should produce:

```
deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/config.json
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

**3b. Text/vision encoder** — MiMo-VL:

```bash
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

**3c. ONNX detectors** (*optional*):

```bash
# YuNet face detector (OpenCV Zoo, git-LFS — use the media.githubusercontent URL)
curl -L -o deploy/deps/checkpoints/face_detection_yunet_2023mar.onnx \
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

## 4. Launch

Every setting is a plain environment variable with a working default. The
launcher activates conda itself (`JOYOMNI_CONDA_SH` / `JOYOMNI_CONDA_ENV`) and
runs every stage on one device; when several GPU models share a checkout, each
gets its own `JOYOMNI_CACHE_ROOT`. UI: `http://<server-ip>:8080/`.

**NVIDIA B200 — 720p @ 30 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_b200 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=30 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 480p @ 24 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 720p @ 16 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=16 \
bash deploy/run_server.sh
```

**RTX 5090 — 480p @ 24 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_rtx5090 \
JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1 JOYOMNI_LOW_VRAM=1 \
bash deploy/run_server.sh
```

Key variables:

| Variable | Meaning |
| --- | --- |
| `JOYOMNI_CONDA_SH` / `JOYOMNI_CONDA_ENV` | conda `profile.d/conda.sh` + env name/prefix; the launcher activates it itself. Omit both to use the caller's python. |
| `JOYOMNI_DEVICE` | CUDA device for all stages (default `cuda:0`). |
| `JOYOMNI_HOST` / `JOYOMNI_PORT` | bind address (default `0.0.0.0:8080`). |
| `JOYOMNI_WIDTH` / `JOYOMNI_HEIGHT` / `JOYOMNI_FPS` | Output resolution and frame rate (default `840` / `480` / `24` = 480p @ 24 FPS). Per-GPU commands above. |
| `JOYOMNI_FP8_IMG` / `JOYOMNI_FP8_TXT` | FP8 image / text paths via `joyomni_ops` (default `1` / `1`). Set both `0` to run bf16 (e.g. a `JOYOMNI_OPS_NO_FP8=1` build). |
| `JOYOMNI_CUDA_GRAPH` | capture the steady-state chunk loop into a CUDA graph (default `1`; the biggest single speedup). `0` runs eager. |
| `JOYOMNI_SAGE_ATTN` | SageAttention for all DiT attention (default `0` → SDPA/cuDNN; set `1` on RTX 5090). |
| `JOYOMNI_FP8_FAST_ACCUM` | FP8 GEMMs accumulate in fp16 via a Triton kernel (default `0`; set `1` on RTX 5090, where fp32-accumulate tensor MMAs run at half rate — they run at full rate on RTX PRO 6000 / B200, so leave it unset there). |
| `JOYOMNI_LOW_VRAM` | low-VRAM layout — CPU-staged FP8 DiT load + text-encoder CPU offload (default `0`; set `1` on ≤48 GB cards — the full layout needs ~46.5 GB steady at 720p16). 480p24 measured ~21.5 GiB resident / ~28 GiB peak under a 30 GiB allocator cap — fits 32 GB cards. |
| `JOYOMNI_CACHE_ROOT` | compile-cache root — torchinductor / triton / nv_compute caches live under it (default `deploy/deps/cache`). Give each GPU model its own root when several share a checkout (per-card commands above). |
| `JOYOMNI_CKPT_ROOT` | override the checkpoints dir (default `deploy/deps/checkpoints`). |
| `JOYOMNI_DIT_CKPT` / `JOYOMNI_VAE_CKPT` / `JOYOMNI_TEXT_ENCODER_CKPT` / `JOYOMNI_FACE_ONNX` / `JOYOMNI_PERSON_ONNX` | override individual weight paths (default: derived from `JOYOMNI_CKPT_ROOT`). |
| `JOYOMNI_RECORD_DIR` | recording output dir. |
| `PE_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | Prompt-enhancement endpoint: OpenAI-compatible, or Anthropic-protocol when the base URL contains `/anthropic` (key sent as `Authorization: Bearer`). If unset, the server falls back to the raw user prompt. |
