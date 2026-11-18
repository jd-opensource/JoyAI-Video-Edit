# JoyAI-Video-Edit on CPUs and edge devices

This directory makes the JoyAI-Video-Edit diffusion transformer run without a GPU, and
says honestly how fast that is.

**What you get**

| | |
| --- | --- |
| `joyedge` engine (`engine/`) | A C++ / [ggml](https://github.com/ggml-org/ggml) runtime for the streaming DiT: GGUF weights, 4–8 bit quantization, weights memory-mapped and paged in block by block, plain C API. No CUDA, no Python at run time. |
| Portable stock pipeline (`deploy/` patches) | The original PyTorch server code no longer needs `joyomni_ops`, FA4, SageAttention or CUDA graphs to run; it falls back to pure PyTorch on any device. |
| Tooling (`joyedge/`) | Checkpoint → GGUF exporter, quantizer, latency benchmarks for DiT and VAE, a compute-budget calculator, student-model configs and a teacher-slicing initializer. |
| Tests (`tests/`) | The engine is checked against the PyTorch reference, which is itself checked against the training code. Random weights, so no checkpoint is needed. |

**What it does not do**

The released model is a 16B-parameter *dense* transformer that runs three forward passes
per 8 frames. On a CPU that is minutes per second of video, at any quantization. Running
it *live* on a CPU needs a much smaller model, and a smaller model has to be trained.
This branch gives you the runtime and the training configs for that; it does not contain
a trained student. Details and measurements below.

Nothing here has been run on the released weights: Hugging Face was not reachable from
the environment this was built in. Every numerical check uses randomly initialised models
with the released architecture, and every timing is compute time on random inputs.

---

## 1. How much compute the model needs

Read from the code (`deploy/xvideo`, `training/src/config.py`):

| Fact | Value |
| --- | --- |
| DiT | 40 double-stream blocks, hidden 4096, 32 heads, MLP ×4 → 16.1B parameters |
| VAE compression | 24× spatial (×2/3 stem, then 16×), 8× temporal, 64 latent channels |
| One chunk | 8 frames = 1 latent frame = `(H/24)·(W/24)` tokens (700 at 840×480) |
| Per chunk | 2 denoise passes over 2×tokens (noisy + source) plus ~400 text tokens, then 1 "store" pass over the clean chunk |
| History | KV cache of chunk 0 ("sink") and the previous chunk, plus an optional reference image |

That is about **77 TFLOP per chunk** at 840×480, so 24 fps needs roughly 230 TFLOP/s of
useful arithmetic. An RTX 5090 gets there only with FP8 weights, INT8 attention and CUDA
graphs. `python -m joyedge.budget` prints this model for any size and resolution.

### Why the stock build was painful on an RTX 4090

These are inferences from `DEPLOYMENT.md` and the code, not measurements on a 4090.

- **Memory.** The guide reports ~21.5 GiB resident and ~28 GiB peak for 480p on a card
  capped at 30 GiB. A 24 GB card is under that peak, so it either fails or, on Windows,
  spills into shared system memory, which shows up as very high latency.
- **Compile time.** `joyomni_ops` builds its cutlass FP8 kernels for five GPU
  architectures unless `JOYOMNI_OPS_CUDA_ARCHS` is set (use `89` for a 4090), and the VAE
  is compiled with `torch.compile(mode="max-autotune-no-cudagraphs")` for every warm-up
  shape, including all reference-image buckets.
- **Throughput.** Real time at 480p is tuned for Blackwell cards; the 4090 has less
  compute and none of the per-card tuning.

Two knobs added here help the compile time on any GPU: `JOYOMNI_VAE_COMPILE_MODE=default`
(or `JOYOMNI_VAE_COMPILE=0`) skips the autotuning search.

## 2. Measured on a CPU

Machine: 4 vCPUs of an Intel Sapphire Rapids Xeon (AVX-512 VNNI, AMX), 15 GB RAM.
DiT only, 400 text tokens, 2 steps.

### Full 16B model, 840×480

| Path | Seconds per 8-frame chunk | fps | Weights |
| --- | --- | --- | --- |
| Stock PyTorch, fp32 | 279 ¹ | 0.03 | 64 GB |
| Stock PyTorch, bf16 (uses AMX) | 75 ¹ | 0.11 | 32 GB |
| joyedge, Q6_K | 124 ² | 0.065 | 13.2 GB file |
| joyedge, Q4_K | 117 ² | 0.069 | 9.1 GB file |

¹ One real-size block timed, multiplied by 40 blocks and 3 passes.
² All 40 blocks executed, but the benchmark file stores one block and reuses it, so the
I/O of the real 13 GB file is not included.

So the full model **works** on a CPU and is **about 350× slower than real time** on this
machine. A 32-core server would be several times faster and still far from live. On a
CPU with AMX, stock PyTorch in bf16 is the fastest path if you have 32 GB+ of RAM; the
engine wins on memory and on CPUs without AMX.

### Smaller students (random weights, same architecture, Q6_K)

Seconds per chunk, exact / with the text cache (section 4). fps = 8 ÷ seconds.

| Student | Params | 432×240 | 624×360 | 840×480 |
| --- | --- | --- | --- | --- |
| xs (768 × 12 blocks) | 0.17B | 0.92 / 0.51 | 1.95 / 1.25 | 3.43 / 2.63 |
| s (1024 × 16) | 0.40B | 1.63 / 0.87 | 3.42 / 2.40 | 6.74 / 5.17 |
| m (1536 × 20) | 1.13B | 4.09 / 2.22 | 7.32 / 5.63 | 14.1 / 10.6 |
| l (2048 × 24) | 2.42B | 7.28 / 3.95 | 14.1 / 9.83 | 25.3 / 20.3 |
| full (4096 × 40) | 16.1B | | | 124 |

Live editing needs ≤ 0.33 s per chunk for 24 fps input, ≤ 0.67 s for 12 fps.
On this small machine only the xs student at 432×240 gets there (0.51 s, DiT only).
Run `joyedge-bench` on your own device; a desktop CPU with 8–16 cores has several times
this machine's throughput.

### The VAE is the other half

The chunk also needs a VAE encode, a decode and a single-frame re-encode. The released
VAE's size is unknown to me (its `config.json` is on Hugging Face). As an indication, a
toy 4M-parameter VAE with the same structure already takes 1.0 s per chunk at 432×240 in
PyTorch on this machine. Measure the real one with:

```bash
python -m joyedge.bench_vae --vae deploy/deps/checkpoints/JoyAI-Video-Edit/vae --width 432 --height 240
```

Expect to need a small distilled decoder/encoder for live use on a CPU.

## 3. Why "like Edge0" does not make the 16B model fast

Edge0 runs very large language models on small machines because those models are
*sparse*: each token touches a few experts, and the rest stays on the SSD. JoyAI's DiT is
dense. Every token goes through all 16B weights, three times per chunk.

What carries over is the storage side, and that is implemented here:

- weights in a quantized GGUF file, opened with `mmap`;
- one block's tensors are contiguous in the file, and the engine reads ahead the next
  block and releases the one that left the window (`stream_window`).

Measured on the 1.1B student (916 MiB file, 20 blocks): peak RSS 1051 MiB without
streaming, **250 MiB** with `stream_window=2`, same speed. For the full model that means
a 13 GB file can run in a few GB of RAM. It does not reduce the arithmetic.

## 4. How the engine maps the GPU design to a CPU

| GPU deployment | CPU engine |
| --- | --- |
| FP8 weights, cutlass FP8 GEMM | K-quant GGUF weights (Q4_K/Q5_K/Q6_K), ggml integer matmul |
| `joyomni_ops` fused norm / RoPE kernels | Custom ggml ops with the same maths (`op_ln_mod`, `op_q`, `op_kv`) |
| CUDA graph of the whole chunk | One small static graph per pass, re-bound to each block's weights |
| KV pool `[sink, prev, ref]`, pre-RoPE keys | Same protocol, fp16 by default |
| SageAttention / FA4 | ggml tiled flash attention on fp16 keys and values |
| Low-VRAM staged load | `mmap` plus the sliding block window |

Two findings from the measurements that shape the defaults:

- **Use K-quants.** On this CPU, plain Q4_K/Q5_K/Q6_K matmuls ran at 830–960 GFLOP/s,
  while Q8_0, Q4_0, F16 and F32 ran at 90–160 through ggml's generic paths. Q6_K had
  about 1.8% matmul error on random data, Q4_K about 6%; the repo quotes 3.7% for its own
  FP8 path. Q6_K is the default recommendation, Q4_K when RAM or disk is tight.
- **Text tokens dominate at low resolution.** At 432×240 a denoise pass has 360 image
  tokens and ~400 text tokens. The optional text cache (`txt_cache_interval=R`) evaluates
  the text stream every R chunks and lets image tokens attend to its cached keys and
  values in between. It roughly halves the time at 240p. It is an approximation of the
  trained model and has not been evaluated on real weights; treat it as experimental.

Not done: ggml's repacked/AMX weight buffers (they need the weights copied into RAM, and
gave at most 1.3× for K-quants here; on ARM the repacked kernels are likely worth more), and
running the graph on ggml's GPU backends (the custom ops are CPU-only).

## 5. Use it

```bash
# 1. build (clones ggml at the tested commit)
edge/engine/build.sh

# 2. tests - no weights needed
python edge/tests/test_portable_block.py   # deploy block (pure torch) == training reference
python edge/tests/test_engine_parity.py    # ggml engine == PyTorch rollout, + quantized files
python edge/tests/test_cpu_streaming.py    # stock session on CPU == engine session, frames in/out
python edge/tests/test_student_tools.py    # training-code student -> GGUF -> engine; slicing

# 3. convert the released DiT (about one weight matrix of RAM; 16-33 GB of disk per file)
cd edge
python -m joyedge.export_gguf --dit-ckpt ../deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth \
    --out joyai-dit-f16.gguf --type f16
engine/build/joyedge-quantize joyai-dit-f16.gguf joyai-dit-q6_k.gguf q6_k

# 4. how fast is it on this machine?
engine/build/joyedge-bench -m joyai-dit-q6_k.gguf --lat-h 20 --lat-w 35 --stream-window 4
python -m joyedge.bench_vae --vae ../deploy/deps/checkpoints/JoyAI-Video-Edit/vae --width 840 --height 480
```

Editing a video from Python (the VAE stays in PyTorch, the DiT runs in the engine):

```python
from joyedge.engine import EdgeDiT
from joyedge.pipeline import EdgeV2VSession

dit = EdgeDiT("joyai-dit-q6_k.gguf", lat_h=10, lat_w=18, n_steps=2, stream_window=4)
session = EdgeV2VSession(vae, dit, width=432, height=240)   # vae: xvideo XVAEChunkCausal, fp32
session.set_prompt(prompt_embeds)                            # [n_tokens, 4096], see below
edited = session.push_frames(frames[:1])                     # first call: 1 frame
edited = session.push_frames(frames[1:9])                    # then 8 at a time -> uint8 [8, H, W, 3]
```

`prompt_embeds` are the MiMo-VL hidden states for (first frame, instruction). They are
computed once per session; `python -m joyedge.encode_prompt` wraps the stock code and
saves them to `.npy`, so the 7B encoder can run on another machine. That helper is
untested here for lack of the encoder weights.

To run the **stock server code** without CUDA kernels (reference / debugging; needs the
RAM for the unquantized model):

```bash
JOYOMNI_PRECISION=fp32 JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 JOYOMNI_CUDA_GRAPH=0 \
JOYOMNI_VAE_COMPILE=0 JOYOMNI_PE_BACKEND=none JOYOMNI_DEVICE=cpu bash deploy/run_server.sh
```

The session class is verified on CPU with random weights (`test_cpu_streaming.py`); the
full server start-up path with real checkpoints is not.

## 6. The route to live editing on a CPU

1. **Pick a size from measurements.** Build a random model of the candidate size, time it
   on the target device, add the VAE:
   ```bash
   python -m joyedge.make_bench_model --preset student-s --out s-f16.gguf
   engine/build/joyedge-quantize s-f16.gguf s-q6_k.gguf q6_k
   engine/build/joyedge-bench -m s-q6_k.gguf --lat-h 10 --lat-w 18
   ```
2. **Train the student.** `training/configs/edge_student_stage1_ar.py` and
   `edge_student_stage2_distillation.py` reuse the repo's trainer with a smaller
   architecture (same latent space, same encoder, same KV protocol). Optional
   initialisation from the teacher: `python -m joyedge.slice_teacher`. This needs GPUs and
   editing data; the toy dataset in the repo is for smoke tests only.
3. **Distil with the 16B model as teacher.** As released, the DMD trainer builds
   generator, critic and teacher from one architecture config, so the provided Stage 2
   config self-distils the student. Using the 16B model as the real-score teacher needs a
   second architecture config for that role in `src/train_t2v_dmd.py` (`load_stage1_dit`).
   Not implemented here.
4. **Replace the VAE path** with a small causal encoder/decoder trained on the same
   latents, once `bench_vae` shows the real cost.
5. **Export and run**: a student checkpoint exports with the same commands as section 5
   (`test_student_tools.py` checks this for a model built by the training code).

Expect lower quality than the 16B model; how much is an empirical question nobody can
answer without training it.

## 7. Limits to know about

- No test used the released weights, and no image quality was evaluated.
- Quantization error and the text cache are characterised on random weights only.
- Benchmarks reuse one stored block for the 16B model; disk read speed of the real file
  on a machine with less RAM than the file is not measured.
- The static-scene KV freeze of the server maps to `update_history=False`; for
  `max_temporal_ids > 2` its positional handling during the first few chunks differs
  slightly from the server.
- Engine patch size is fixed at `[1, 1, 1]`, chunk size 1 (the released configuration).
- Windows builds fall back to reading the file into memory (no `mmap` streaming).
- ARM, Apple silicon and Android are expected to work through ggml but were not built or
  timed here.
