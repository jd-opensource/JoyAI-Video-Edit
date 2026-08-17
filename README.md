<h1 align="center">JoyAI-Video-Edit</h1>
<h3 align="center">Real-Time Open-Ended Video Editing with Autoregressive Diffusion</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2608.03974"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Project-JoyAI--Video--Edit-333399" alt="Project"></a>
  <a href="https://huggingface.co/jdopensource/JoyAI-Video-Edit"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-JoyAI--Video--Edit-yellow" alt="Hugging Face"></a>
  <a href="https://huggingface.co/spaces/wxDai/joyai-video-edit"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Demo-Streaming--V2V-orange" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="96%" alt="JoyAI-Video-Edit teaser">
</p>

JoyAI-Video-Edit is a real-time, instruction-guided video editing system for open-ended video streams. Given a live camera stream or uploaded video and a natural-language edit instruction, it edits frames causally as they arrive, without waiting for the full video, requiring a predefined video length, or revisiting future frames. In our deployment benchmark, the full end-to-end pipeline reaches 30 FPS at 720 × 1248, pushing video editing from offline batch processing toward interactive streaming generation.

The system combines an MLLM-based condition encoder, a causal video VAE, and a 16B-parameter multimodal diffusion transformer. It is trained and deployed as an autoregressive diffusion editor, then accelerated with aligned autoregressive distribution matching distillation, long-horizon optimization, bounded KV-state inference, and deployment-oriented scheduling to sustain high-throughput 720p editing while reducing train-inference mismatch and accumulated temporal drift.

## 🔥🔥🔥 News!!

- 2026.08.15: 🎉 Live demo released — real-time streaming video editing on a single RTX PRO 6000 (Blackwell) GPU: 840 × 480 @ 24 FPS or 720p @ 16 FPS. **[Try HuggingFace Demo](https://huggingface.co/spaces/wxDai/joyai-video-edit)**
- 2026.08.14: 🎉 Released an upgraded checkpoint with significantly stronger reference-image-guided video editing (RV2V), delivering better subject and identity preservation, more faithful reference conditioning, and improved temporal consistency across long streams. Grab the new **[DiT weights](https://huggingface.co/jdopensource/JoyAI-Video-Edit/blob/main/dit/joyai_video_edit_dit_0811.pth)**.
- 2026.08.05: 🎉 We release the deployment code, [technical report](https://arxiv.org/pdf/2608.03974), and JoyAI-Video-Edit checkpoints. Please check the links above for details.

## 💎 Highlights

- **Real-time open-ended editing.** Edits live or uploaded videos as frames arrive, without requiring the full sequence upfront.
- **Diverse instruction control.** Supports subject edits, local edits, background changes, style transfer, motion changes, and reference-guided editing.
- **Autoregressive diffusion design.** Combines an MLLM condition encoder, causal video VAE, and MMDiT backbone for streaming video editing.
- **High-throughput 720p deployment.** Reaches 30 FPS end-to-end throughput at 720 × 1248 with bounded KV-state inference and stable per-chunk compute.

## 🚧 TODO

- [x] **Stronger model version in progress.** A more powerful version is under active development, with a particular focus on advancing reference-image-guided video editing (RV2V) capabilities.
- [ ] **Consumer GPU support.** Optimize deployment for consumer-grade GPUs such as GeForce RTX 5090.
- [ ] **Diffusers support.** Provide a 🤗 Diffusers pipeline for JoyAI-Video-Edit to streamline loading and inference.
- [ ] **LongV2VBench release.** Release LongV2VBench for long-form video-to-video editing evaluation.
- [ ] **Release full training and data pipelines.** Open-source the complete training framework and data generation pipeline.

## 🎬 Showcase

JoyAI-Video-Edit is designed for broad video editing tasks, including global appearance changes, local object edits, subject add/remove/replace, background replacement, style transfer, and reference-guided edits.

https://github.com/user-attachments/assets/bca232c9-75df-46f9-b366-14cfa2651994

<table>
  <tr>
    <th>Source</th>
    <th>Prompt</th>
    <th>Edited</th>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/dc7c3cf6-de0b-4b7b-9afe-d4e38c4ea6e1"><img src="assets/cases/case01_source.gif" width="220" alt="Case 01 source"></a></td>
    <td>Transform the people, hairstyles, and interior into a British castle aristocratic style.</td>
    <td><a href="https://github.com/user-attachments/assets/dbb9b6de-9d63-4879-8418-96610eac79b3"><img src="assets/cases/case01_edited.gif" width="220" alt="Case 01 edited"></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/79792d23-7037-43c2-b185-dfd529aac3b7"><img src="assets/cases/case02_source.gif" width="220" alt="Case 02 source"></a></td>
    <td>Turn the video into a watercolor wash style.</td>
    <td><a href="https://github.com/user-attachments/assets/a20dea9d-715f-45b6-ae1d-b8c694983910"><img src="assets/cases/case02_edited.gif" width="220" alt="Case 02 edited"></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/92c1723b-78a6-4b04-837d-21545aeff823"><img src="assets/cases/case03_source.gif" width="220" alt="Case 03 source"></a></td>
    <td>Make all dogs white, add colorful hats, and turn the sunglasses hot pink.</td>
    <td><a href="https://github.com/user-attachments/assets/8412d011-9b08-46c8-820e-ab59e97773d4"><img src="assets/cases/case03_edited.gif" width="220" alt="Case 03 edited"></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/e9d70802-96e8-4aa4-a577-9e9cf0f8705c"><img src="assets/cases/case04_source.gif" width="220" alt="Case 04 source"></a></td>
    <td>Dress the girl in a brown down jacket and blue baseball cap.</td>
    <td><a href="https://github.com/user-attachments/assets/5ae898bd-ee26-4a63-9342-72430c28b82f"><img src="assets/cases/case04_edited.gif" width="220" alt="Case 04 edited"></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/b7b5e199-080c-4344-924e-01951fd05912"><img src="assets/cases/case05_source.gif" width="220" alt="Case 05 source"></a></td>
    <td>Remove the two white cats in pink clothes on both sides.</td>
    <td><a href="https://github.com/user-attachments/assets/8e1874b8-755c-4bb0-914d-155110501076"><img src="assets/cases/case05_edited.gif" width="220" alt="Case 05 edited"></a></td>
  </tr>
</table>

## 📦 Model Download

Download the released JoyAI-Video-Edit weights from [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Video-Edit), then place them under:

```text
deploy/deps/checkpoints/JoyAI-Video-Edit/
|-- dit/
|   `-- joyai_video_edit_dit_0811.pth
`-- vae/
    |-- config.json
    `-- diffusion_pytorch_model.safetensors
```

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit
python -m pip install -r deploy/requirements.txt
```

### 2. Prepare Checkpoints

Download the released weights from the Hugging Face link above. MiMo-VL and the ONNX detector files are external runtime dependencies; see [`DEPLOYMENT.md`](DEPLOYMENT.md) for deployment details.

### 3. Launch

```bash
cd deploy
bash run_server.sh
```

Then open:

```text
http://localhost:8080
```

For remote machines, bind the server to `0.0.0.0` and open the selected port, or use SSH port forwarding.

## 📚 Citation

If JoyAI-Video-Edit is useful for your research or product prototype, please cite:

```bibtex
@article{xiao2026joyai,
  title={JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion},
  author={Xiao, Yicheng and Dai, Wenxun and Qin, Xinran and Song, Lin and Zhang, Maoquan and Xu, Hang and Chen, Yukang and Li, Yitong and Zhang, Guohui and Zhang, Yuan and Zhang, Xuying and Zhang, Tommy and Yuan, Jianlong and Li, Peihao and Lu, Shuai and Fu, Siming and Zhao, Chuyang and Han, Xin and Huang, Jie and Li, Wenbo and Ma, Guoqing and Huang, Wei and Qi, Xiaojuan and Huang, Haoyang and Duan, Nan},
  journal={arXiv preprint arXiv:2608.03974},
  year={2026}
}
```

<a id="license"></a>

## ⚖️ License Agreement

JoyAI-Video-Edit is licensed under Apache 2.0.
