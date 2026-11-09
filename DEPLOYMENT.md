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
    ├── checkpoints/       # DiT / xVAE / MiMo-VL / Qwen PE / ONNX detectors
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
  produces noise. On RTX 5090, set `JOYOMNI_FA4_ATTN=0 JOYOMNI_SAGE_ATTN=1`
  to select SageAttention (see §4). Keep SageAttention disabled on RTX PRO 6000
  and B200.
- **flash-attn-4** (`4.0.0b13`, *required on FA4 machines — B200; skip on
  RTX PRO 6000 / 5090, its JIT does not support sm_120*) — provides
  `flash_attn.cute`; kernels JIT at runtime, no build step. Deps must be
  pinned exactly:

  ```bash
  python -m pip install flash-attn-4==4.0.0b13 \
    nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 apache-tvm-ffi==0.1.12
  ```

  If it can't be imported or its kernel fails at runtime, the DiT
  falls back to SageAttention when enabled, otherwise PyTorch SDPA (cuDNN/flash).
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

Video inference and Qwen3.6-27B PE use the same `joyai-video-edit` environment.
Install the shared dependencies with `deploy/requirements.txt`, then install
the GPTQModel wheel for your GPU, as with the attention dependencies above.
Both wheels below match Linux x86_64 / Python 3.10 / Torch 2.9.1 cu128:

- **B200 / sm100** — prebuilt Marlin wheel hosted in `wxDai/joyai-video-edit-wheels`:

  ```bash
  python -m pip install --index-url https://pypi.org/simple/ 'https://huggingface.co/datasets/wxDai/joyai-video-edit-wheels/resolve/bb453ac811107503302dc1bfb96bee7881b28766/b200/cu128/torch2.9/cp310/gptqmodel-6.0.3%2B6a65d69-cp310-cp310-linux_x86_64.whl#sha256=a87bf8facef6517829998bf64cdd3d2e001b318de10935fb79e6d9d3484949cb'
  ```

- **RTX PRO 6000 / RTX 5090 / sm120** — official GPTQModel wheel:

  The RTX 5090 launch command below disables local PE; skip this wheel for that configuration.

  ```bash
  python -m pip install --index-url https://pypi.org/simple/ 'https://github.com/ModelCloud/GPTQModel/releases/download/v6.0.3/gptqmodel-6.0.3%2Bcu128torch2.9-cp310-cp310-linux_x86_64.whl'
  ```

These PE wheels do not require a source build. Use separate B200 and RTX PRO 6000
environments when they share a filesystem. For a different Python, CUDA or Torch
version, choose a matching GPTQModel wheel.

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
    print("joyomni_ops: MISSING ->", e, "(build deploy/joyomni_ops; set JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 to skip FP8)")
PY
```

---

## 3. Fetch the weights

Video-model and local PE weights live under `deploy/deps/checkpoints/`.
The local Qwen3.6-27B INT4 PE checkpoint has about 20.4 GiB of weight shards
in addition to the video weights. Disk size is not the resident GPU-memory requirement.
Create the directory and download each dependency.

```bash
mkdir -p deploy/deps/checkpoints
```

**3a. DiT + xVAE** — the released JoyAI-Video-Edit weight repo:

```bash
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/JoyAI-Video-Edit \
  --include "dit/joyai_video_edit_dit_1101_face++.pth" "vae/*"
```

> The default DiT checkpoint is `joyai_video_edit_dit_1101_face++.pth`.
> Set `JOYOMNI_DIT_CKPT` to use another checkpoint.

This should produce:

```
deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/config.json
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

**3b. Text/vision encoder** — MiMo-VL:

```bash
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

**3c. Local prompt enhancement** — Qwen3.6-27B GPTQ INT4:

```bash
hf download btbtyler09/Qwen3.6-27B-GPTQ-4bit \
  --revision bcfe8a7ec20eec6ce67ded4669819ae77e73af3a \
  --local-dir deploy/deps/checkpoints/Qwen3.6-27B-GPTQ-Int4-FOEM
```

`deploy/run_server.sh` defaults to `JOYOMNI_PE_BACKEND=qwen` and this local
directory. Override `JOYOMNI_PE_CKPT` to store the same checkpoint elsewhere.
The checkpoint is a community GPTQ INT4 quantization.

**3d. ONNX detectors** (*optional*):

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
> requirements for video inference. Local Qwen PE additionally requires the Qwen checkpoint.

Final tree:

```
deploy/deps/checkpoints/
├── JoyAI-Video-Edit/
│   ├── dit/joyai_video_edit_dit_1101_face++.pth
│   └── vae/{config.json, diffusion_pytorch_model.safetensors}
├── MiMo-VL-7B-RL-2508/
├── Qwen3.6-27B-GPTQ-Int4-FOEM/      # local PE
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
JOYOMNI_FA4_ATTN=1 JOYOMNI_SAGE_ATTN=0 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=30 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 480p @ 24 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
JOYOMNI_FA4_ATTN=0 JOYOMNI_SAGE_ATTN=0 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 720p @ 16 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
JOYOMNI_FA4_ATTN=0 JOYOMNI_SAGE_ATTN=0 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=16 \
bash deploy/run_server.sh
```

**RTX 5090 — 480p @ 24 FPS (PE disabled):**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_rtx5090 \
JOYOMNI_FA4_ATTN=0 JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1 JOYOMNI_LOW_VRAM=1 \
JOYOMNI_PE_BACKEND=none \
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
| `JOYOMNI_FA4_ATTN` | Try FA4 first when installed (default `1`). Set `1` on B200; set `0` on RTX PRO 6000 / RTX 5090. HF ZeroGPU defaults to `0`. |
| `JOYOMNI_SAGE_ATTN` | Use SageAttention when FA4 is disabled or unavailable (default `0`). Set `1` with `JOYOMNI_FA4_ATTN=0` on RTX 5090; otherwise use PyTorch SDPA. |
| `JOYOMNI_FP8_FAST_ACCUM` | FP8 GEMMs use bounded fp16 partial sums with fp32 promotion (default `0`; set `1` on RTX 5090 for faster tensor MMAs; leave unset on RTX PRO 6000 / B200). |
| `JOYOMNI_LOW_VRAM` | low-VRAM layout — CPU-staged FP8 DiT load + text-encoder CPU offload (default `0`; set `1` on ≤48 GB cards — the full layout needs ~46.5 GB steady at 720p16). Video-only 480p24 measured ~21.5 GiB resident / ~28 GiB peak under a 30 GiB allocator cap. Local Qwen PE needs additional GPU memory. |
| `JOYOMNI_CACHE_ROOT` | compile-cache root — torchinductor / triton / nv_compute caches live under it (default `deploy/deps/cache`). Give each GPU model its own root when several share a checkout (per-card commands above). |
| `JOYOMNI_CKPT_ROOT` | override the checkpoints dir (default `deploy/deps/checkpoints`). |
| `JOYOMNI_DIT_CKPT` / `JOYOMNI_VAE_CKPT` / `JOYOMNI_TEXT_ENCODER_CKPT` / `JOYOMNI_FACE_ONNX` / `JOYOMNI_PERSON_ONNX` | override individual weight paths (default: derived from `JOYOMNI_CKPT_ROOT`). |
| `JOYOMNI_RECORD_DIR` | recording output dir. |
| `JOYOMNI_PE_BACKEND` | `qwen` (default): local Qwen3.6-27B Marlin INT4 PE. `api`: external prompt-enhancement endpoint. `none`: disable PE without loading Qwen. |
| `JOYOMNI_PE_CKPT` | Local Qwen directory (default `JOYOMNI_CKPT_ROOT/Qwen3.6-27B-GPTQ-Int4-FOEM`). |
| `PE_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | Used with `JOYOMNI_PE_BACKEND=api`. Prompt-enhancement endpoint: OpenAI-compatible, or Anthropic-protocol when the base URL contains `/anthropic` (key sent as `Authorization: Bearer`). If unset, the server falls back to the raw user prompt. |
