# Deployment Guide

This guide describes how to deploy JoyAI-Video-Edit from the code repository plus the released Hugging Face weights.

## Runtime Layout

The server is launched from `deploy/`. Runtime weights are expected under `deploy/deps/`:

```text
deploy/
|-- run_server.sh
|-- static/
|-- xvideo/
`-- deps/
    |-- checkpoints/
    |   |-- JoyAI-Video-Edit/
    |   |   |-- dit/
    |   |   |   `-- joyai_video_edit_dit_0804.pth
    |   |   `-- vae/
    |   |       |-- config.json
    |   |       `-- diffusion_pytorch_model.safetensors
    |   |-- MiMo-VL-7B-RL-2508/
    |   |-- face_detection_yunet_2023mar.onnx
    |   `-- yolov8n.onnx
    `-- cache/
        |-- torchinductor/
        |-- triton/
        `-- nv_compute/
```

## 1. Prepare Environment

Create a Python 3.10 environment and install CUDA-enabled PyTorch plus the serving dependencies:

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The fused DiT CUDA kernels (RMSNorm, adaLN modulate, 3D-RoPE QK-norm, and the
FP8 quant/GEMM) are provided by the vendored `joyomni_ops` package, not by
sgl-kernel. Build and install it from the repo root:

```bash
# Full build (FP8 GEMM included): point at a cutlass checkout, tag
# 57e3cfb47a2d9e0d46eb6335c3dc411498efa198, CUDA >= 12.8 for Blackwell SASS.
JOYOMNI_OPS_CUTLASS_DIR=/path/to/cutlass python -m pip install ./joyomni_ops

# Light build (no FP8 GEMM, no cutlass needed) — set JOYOMNI_FP8_IMG=0 at runtime:
# JOYOMNI_OPS_NO_FP8=1 python -m pip install ./joyomni_ops
```

See `joyomni_ops/README.md` for op list, GPU-arch selection, and benchmarks.

Verify the key runtime imports:

```bash
python - <<'PY'
import torch
import cv2
import av
import flash_attn.cute
import joyomni_ops

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("joyomni_ops fp8:", joyomni_ops.has_fp8())
print("cv2", cv2.__version__)
PY
```

The launcher does not contain private paths or API credentials. Activate your environment before launching, or pass the conda entrypoint through environment variables:

```bash
JOYAI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYAI_CONDA_ENV=joyai-video-edit \
bash deploy/run_server.sh
```

For local private settings, copy `.env.example` to `.env`, fill in your local values, and source it before launching:

```bash
cp .env.example .env
set -a
source .env
set +a
bash deploy/run_server.sh
```

For prompt enhancement, pass an OpenAI-compatible endpoint at runtime instead of editing source files:

```bash
PE_API_KEY=<your-api-key> \
PE_BASE_URL=https://your-openai-compatible-endpoint/v1 \
PE_MODEL=<your-model-name> \
bash deploy/run_server.sh
```

If these variables are not set, prompt enhancement falls back to the raw user prompt.

### Tested Environment

The public deployment package was tested with a single NVIDIA B200 GPU:

| Item | Version / Configuration |
| --- | --- |
| GPU | 1 x NVIDIA B200 |
| CUDA runtime | `12.8` |
| Python | `3.10` |
| PyTorch | `2.9.1+cu128` |
| Transformers / Diffusers | `4.57.0` / `0.36.0` |
| FastAPI / Uvicorn | `0.117.1` / `0.37.0` |
| OpenCV / PyAV | `opencv-python-headless 4.13.0.92` / `av 13.1.0` |
| Attention / kernel packages | `flash-attn-4 4.0.0b13`, `joyomni_ops 0.1.0` (vendored, built from source), `triton 3.5.1`, `nvidia-cutlass-dsl 4.5.1` |

## 2. Clone JoyAI-Video-Edit Weights

From the code repository root:

```bash
cd deploy
cd deps/checkpoints
git lfs install
git clone https://huggingface.co/jdopensource/JoyAI-Video-Edit
```

This should create:

```text
deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0804.pth
deps/checkpoints/JoyAI-Video-Edit/vae/config.json
deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

## 3. Prepare External Dependencies

The released JoyAI-Video-Edit weight repo only contains the DiT and VAE weights. The following dependencies are still required at runtime:

| Dependency | Expected path | Notes |
| --- | --- | --- |
| MiMo-VL-7B-RL-2508 | `deploy/deps/checkpoints/MiMo-VL-7B-RL-2508` | Text and visual condition encoder. Download from the upstream MiMo-VL model repository. |
| YuNet face detector | `deploy/deps/checkpoints/face_detection_yunet_2023mar.onnx` | Used by startup and online face gates. If missing, the face gate is disabled by the server. |
| YOLOv8n person detector | `deploy/deps/checkpoints/yolov8n.onnx` | Used by online person presence checks. If missing, person checks are disabled by the server. |

Example text-encoder download:

```bash
cd deploy
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir deps/checkpoints/MiMo-VL-7B-RL-2508
```

Place the ONNX detector files at the paths shown above, or remove the corresponding flags from `run_server.sh` if you intentionally want to run without those gates.

## 4. Launch Server

From `deploy/`:

```bash
bash run_server.sh
```

The default launcher:

- binds to `0.0.0.0:8080`;
- uses `cuda:0` by default;
- places DiT, VAE encode/decode, pseudo-encode, and postprocess on the same device by default;
- enables persistent TorchInductor, Triton, and CUDA caches under `deploy/deps/cache/`;
- records generated sessions under `deploy/recordings/`.

Open the UI:

```text
http://<server-ip>:8080/
```

## 5. Common Overrides

Use a custom recording directory:

```bash
JOYOMNI_RECORD_DIR=/path/to/recordings bash run_server.sh
```

Change download re-encode quality:

```bash
JOYOMNI_DOWNLOAD_CRF=8 bash run_server.sh
```

Append additional server flags after the script:

```bash
bash run_server.sh --port 7860 --profile-timings
```

Use custom checkpoint locations:

```bash
JOYAI_DIT_CKPT=/path/to/joyai_video_edit_dit_0804.pth \
JOYAI_VAE_CKPT=/path/to/vae \
JOYAI_TEXT_ENCODER_CKPT=/path/to/MiMo-VL-7B-RL-2508 \
bash run_server.sh
```

## 6. Sanity Checks

Check files:

```bash
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0804.pth
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
test -d deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

Check server health after launch:

```bash
curl http://127.0.0.1:8080/health
```

The first launch can be slow because PyTorch, Triton, CUDA kernels, VAE paths, and DiT attention kernels need to compile and warm up. Keep `deploy/deps/cache/` stable across restarts to reuse compile artifacts.
