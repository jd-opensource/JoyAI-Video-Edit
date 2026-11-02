# Training & Evaluation

This directory provides autoregressive adaptation, Resilient-DMD distillation, long-horizon fine-tuning, and evaluation on LongV2VBench and OpenVE-Bench for JoyAI-Video-Edit.

We provide a [bidirectional version](https://huggingface.co/jdopensource/JoyAI-Video-Edit/blob/main/dit/pretrain_editing_bid_version.pth) of the model that supports T2I, T2V, I2I, and V2V. You can use this checkpoint as a starting point for autoregressive adaptation followed by Resilient-DMD distillation, or further adapt and optimize it for your own tasks.

Follow the [main README](../README.md#quick-start) to set up the environment and prepare model weights. Activate that environment, then enter `training/` from the repository root. Run all commands below from this directory:

```bash
cd training
python -m pip install -r requirements.txt
```

## Method Overview

![Resilient-DMD: autoregressive generation, distribution matching, and dynamic source guidance](assets/resilient_dmd.png)

**Resilient-DMD** monitors relative source attention to detect overreliance on generated history. When source grounding weakens, it strengthens the real-score teacher's source guidance and trains a two-step generator through distribution matching. The generator and fake-score model share a backbone with separate LoRA adapters; monitoring and control are active only during training.

**Long-Horizon Autoregressive Distillation (LHAD)** extends training to longer rollouts using forward–reverse traversal of source videos, segmented backpropagation, and detached history reuse, keeping activation memory bounded.

## 1. Training Data

Download [JoyAI-Video-Edit ToyData](https://huggingface.co/datasets/EasonXiao-888/JoyAI-Video-Edit_ToyData):

```bash
hf download EasonXiao-888/JoyAI-Video-Edit_ToyData \
    --repo-type dataset --local-dir data
```

Keep the TAR files intact; no extraction is needed. Convert `resolved_files` in the four manifests to local absolute paths:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data").resolve()
for task in ("t2i", "i2i", "t2v", "v2v"):
    manifest_path = root / f"{task}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [root / Path(shard).name for shard in manifest["resolved_files"]]
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TAR files: {missing}")
    manifest["resolved_files"] = [str(shard) for shard in shards]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
```

The default configs use `data/t2i.json`, `data/i2i.json`, `data/t2v.json`, and `data/v2v.json`, with `caption` as the text field. Override these paths with `JOYAI_T2I_DATA`, `JOYAI_I2I_DATA`, `JOYAI_T2V_DATA`, and `JOYAI_V2V_DATA` if needed.

> ToyData is intended for data-loading checks and training smoke tests, not full-scale training.

## 2. Training

Set the model paths for your machine. Compatible weights prepared for the main repository can be reused:

```bash
export JOYAI_BASE_MODEL="/absolute/path/to/base_model.pth"
export JOYAI_VAE="/absolute/path/to/vae"
export JOYAI_TEXT_ENCODER="/absolute/path/to/text_encoder"
```

### Configure Your Experiment

| Config | Purpose |
| --- | --- |
| `configs/stage1_ar.py` | Stage1 autoregressive adaptation |
| `configs/stage2_distillation.py` | Stage2 Resilient-DMD distillation |
| `configs/stage2_distillation_long-tuning.py` | LHAD fine-tuning from a Stage2 checkpoint |

For an initial smoke test, adjust the relevant config as follows:

| Setting | Example |
| --- | --- |
| Experiment and steps | `exp_name="toy"`, `max_train_steps=20`, `checkpoint_interval=10`, `lr_warmup_steps=5` |
| Resolution and length | `TRAIN_RESOLUTION=(480, 840)`; set `frame_buckets=[49]` |
| Batch size | Set bucket `bs_*` values and `gradient_accumulation_steps`; each task keeps its configured batch size across clip lengths |
| Learning rate | Set `learning_rate` for Stage1; set `dmd_generator_learning_rate` and `dmd_fake_score_learning_rate` separately for Stage2 |
| Task mixture | Adjust `image_sampling_prob`, `multiple_images_sampling_prob`, `video_sampling_prob`, and `multiple_videos_sampling_prob`; use `0, 0, 1, 0` for T2V only |
| Long-horizon training | `dmd_rollout_num_chunks` controls generator rollout length, `dmd_grad_window_chunks` controls segment length, and `dmd_critic_rollout_num_chunks` controls critic rollout length independently |

Keep V2V sampling enabled to test dynamic source guidance. Single-GPU training requires sufficient VRAM. Multi-GPU training uses data parallelism and FSDP2; set `hsdp_shard_dim` to a positive divisor of the total process count (for example, 8 to shard across 8 GPUs). Each rank loads a different batch. Their frame counts are trimmed to the shortest batch in that micro-batch so chunk-by-chunk rollouts execute the same number of forwards.

Bucket tuples are `(batch_size, num_items, num_frames, height, width)`. Images and videos share spatial buckets scaled from `TRAIN_RESOLUTION`, with both dimensions aligned to 24 pixels. The target media selects the largest usable frame count, then the closest aspect ratio. Reference images use separate buckets with an area budget of `768²`, a 24-pixel grid, and a maximum aspect ratio of 4. The `bs_*` values set a fixed per-rank batch size for each task. Effective batch size is the sum of actual per-rank batch sizes over the accumulation steps.

The long-horizon config inherits shared settings from `configs/stage2_distillation.py` and overrides only its resume path, experiment name, generator learning rate, rollout length, and temporal positions.

`max_temporal_ids` sets a temporal position ceiling independently of `dmd_rollout_use_relative_temporal_ids`. When a window exceeds the ceiling, its absolute positions shift down and values below zero are floored at zero (for example, `0,49,50` becomes `0,7,8` with a ceiling of 8). Without a ceiling, `dmd_rollout_use_relative_temporal_ids=True` uses dense window positions and `False` retains absolute positions. The default ceilings are 2 for short rollouts and 8 for long rollouts.

### Stage1 → Stage2

The following example runs Stage1 with `exp_name="toy"`, one GPU, and 20 training steps:

```bash
bash scripts/train/stage1_ar.sh configs/stage1_ar.py 1
```

Initialize distillation from the Stage1 checkpoint:

```bash
export JOYAI_STAGE1_CKPT="$PWD/outputs/stage1_ar/toy_world1/checkpoints/global_step20/step_20.pth"
bash scripts/train/stage2_distillation.sh configs/stage2_distillation.py 1
```

The final `1` specifies the number of GPU processes per node. Logs and checkpoints are saved under `outputs/<stage>/<exp_name>_world<world_size>/`. Set `JOYAI_OUTPUT_DIR` to override the output root. Set `resume_from_checkpoint` in the config to a checkpoint path to resume training.

### Long-Horizon Fine-Tuning

```bash
export JOYAI_STAGE2_CKPT="$PWD/outputs/stage2_distillation/toy_world1/checkpoints/global_step20/step_20.pth"
JOYAI_OUTPUT_DIR=outputs/stage2_longtuning \
    bash scripts/train/stage2_distillation.sh configs/stage2_distillation_long-tuning.py 1
```

Set `max_train_steps` in the long-horizon config above the resumed checkpoint's step. **Stage2 / LHAD LoRA checkpoints do not include the full backbone. Keep the same `JOYAI_STAGE1_CKPT` for continued training and evaluation.**

## 3. Evaluation

### Download the Benchmarks

**LongV2VBench**: [EasonXiao-888/LongV2VBench](https://huggingface.co/datasets/EasonXiao-888/LongV2VBench).

```bash
hf download EasonXiao-888/LongV2VBench \
    --repo-type dataset --local-dir data/LongV2VBench
```

**OpenVE-Bench**: use the [OpenVE-Bench dataset](https://huggingface.co/datasets/Lewandofski/OpenVE-Bench) provided by the [official OpenVE-3M project](https://github.com/OpenVE-Team/OpenVE-3M).

```bash
hf download Lewandofski/OpenVE-Bench \
    --repo-type dataset --local-dir data/OpenVE-Bench
```

If the data is distributed as an archive, extract it following the dataset instructions. Each benchmark uses `benchmark_videos.csv` in its data directory. Preserve the video directory structure so that `original_video` paths resolve correctly. For custom locations, set `DATASET_ROOT` and either `METADATA_PATH` for LongV2VBench or `CSV_PATH` for OpenVE-Bench.

### Generate Videos

Specify the `.pth` checkpoint to evaluate, using the matching config, Stage1 backbone, VAE, and text encoder:

```bash
export JOYAI_CHECKPOINT="/absolute/path/to/stage2/step_XXXX.pth"

GPUS_PER_NODE=1 SAVE=outputs_eval/longv2vbench \
    bash scripts/evaluation/longv2vbench.sh \
    configs/stage2_distillation.py "$JOYAI_CHECKPOINT"

GPUS_PER_NODE=1 SAVE=outputs_eval/openvebench \
    bash scripts/evaluation/openvebench.sh \
    configs/stage2_distillation.py "$JOYAI_CHECKPOINT"
```

Override inference settings with `INFERENCE_STEP`, `CFG`, `SRC_CFG`, and `RESOLUTION` (`480` or `720`), or append `--max-items 1` for a single-sample check. Results are written to `SAVE/fullset/<edited_type>/`. Existing videos are skipped, so use a new `SAVE` directory when changing models or settings.

Benchmark inference runs an independent, unsharded model on each GPU and assigns different videos to each rank.

`MAX_TEMPORAL_IDS` sets the inference ceiling independently of `USE_RELATIVE_ROPE`. Set `MAX_TEMPORAL_IDS=none` (or pass `--max-temporal-ids none`) to disable the ceiling. The benchmark scripts default to 8; set `MAX_TEMPORAL_IDS=2` when evaluating the default short-rollout model.

### Evaluation

The commands above generate videos without scoring by default. To enable scoring, configure a Gemini-compatible gateway that supports video requests:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
export GEMINI_BASE_URL="https://YOUR_VIDEO_CAPABLE_GATEWAY/v1"
export MODEL_ID="Gemini-2.5-pro"
```

Prefix either evaluation command with `RUN_EVAL=True MAX_WORKERS=4` to score videos after inference. LongV2VBench saves JSONL results and a TXT summary; OpenVE-Bench saves CSV results and JSON statistics, all under the corresponding `SAVE` directory.

Scoring uploads videos and may incur API charges. Video generation alone does not require an API key.

## Acknowledgements

We thank the authors and contributors of the following projects for sharing their open-source work:

- [LongLive](https://github.com/NVlabs/LongLive)
- [Self Forcing](https://github.com/guandeh17/Self-Forcing)
- [Causal Forcing](https://github.com/thu-ml/Causal-Forcing)

## Citation

If you find this project helpful, please cite:

```bibtex
@article{xiao2026joyai,
  title={JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion},
  author={Xiao, Yicheng and Dai, Wenxun and Qin, Xinran and Song, Lin and Zhang, Maoquan and Xu, Hang and Chen, Yukang and Li, Yitong and Zhang, Guohui and Zhang, Yuan and Zhang, Xuying and Zhang, Tommy and Yuan, Jianlong and Li, Peihao and Lu, Shuai and Fu, Siming and Zhao, Chuyang and Han, Xin and Huang, Jie and Li, Wenbo and Ma, Guoqing and Huang, Wei and Qi, Xiaojuan and Huang, Haoyang and Duan, Nan},
  journal={arXiv preprint arXiv:2608.03974},
  year={2026}
}
```
