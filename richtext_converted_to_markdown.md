### B. Run it on your CPU device

1.  **Install dependencies and build the engine.** You need Python 3.10, cmake and a C++ compiler; on a GPU-less machine use the CPU build of torch.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install torch==2.9.1 torchvision==0.24.1 diffusers==0.37.1 transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 einops numpy pillow loguru gguf   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   edge/engine/build.sh   `

1.  **Run the four tests.** They need no weights and each should end in OK.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   for t in test_portable_block test_engine_parity test_cpu_streaming test_student_tools; do python edge/tests/$t.py | tail -1; done   `

1.  **Download the weights.** The transformer is about 32 GB; the text encoder is separate.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   hf download jdopensource/JoyAI-Video-Edit --local-dir deploy/deps/checkpoints/JoyAI-Video-Edit --include "dit/joyai_video_edit_dit_1101_face++.pth" "vae/*"   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 --local-dir deploy/deps/checkpoints/MiMo-VL-7B-RL-2508   `

1.  **Convert and quantize the transformer.** This needs about 45 GB of free disk; the result is a 13 GB file.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   cd edge && python -m joyedge.export_gguf --dit-ckpt "../deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth" --out joyai-dit-f16.gguf --type f16   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   engine/build/joyedge-quantize joyai-dit-f16.gguf joyai-dit-q6_k.gguf q6_k   `

1.  **Measure your device.** This is the decision point: it tells you how many seconds each 8 frames takes.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   engine/build/joyedge-bench -m joyai-dit-q6_k.gguf --lat-h 10 --lat-w 18 --stream-window 4   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m joyedge.bench_vae --vae ../deploy/deps/checkpoints/JoyAI-Video-Edit/vae --width 432 --height 240   `

1.  **Compute the prompt embeddings once per edit.** This loads the 7B text encoder, so it needs roughly 28 GB of RAM at fp32 and can run on a different machine. I could not test this helper.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m joyedge.encode_prompt --text-encoder ../deploy/deps/checkpoints/MiMo-VL-7B-RL-2508 --image first_frame.png --width 432 --height 240 --prompt "Turn the video into a watercolor wash style." --out prompt.npy   `

1.  **Edit a video.** Save this as edge/run\_edit.py and run it from edge/ with python run\_edit.py. It expects source frames as frames/00000.png onward, numbering 1 + 8·k frames. Running it standalone like this is untested, because it needs the real VAE.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import glob, numpy as np, torch  from PIL import Image  from joyedge.engine import EdgeDiT  from joyedge.pipeline import EdgeV2VSession  from xvideo.models.vae import XVAEChunkCausal  W, H = 432, 240  vae = XVAEChunkCausal.from_pretrained("../deploy/deps/checkpoints/JoyAI-Video-Edit/vae", torch_dtype=torch.float32)  dit = EdgeDiT("joyai-dit-q6_k.gguf", lat_h=H // 24, lat_w=W // 24, n_steps=2, stream_window=4)  session = EdgeV2VSession(vae, dit, width=W, height=H)  session.set_prompt(np.load("prompt.npy"))  frames = [Image.open(p) for p in sorted(glob.glob("frames/*.png"))]  i, n_out = 0, 0  while i + session.frames_per_next_chunk <= len(frames):      n = session.frames_per_next_chunk      for frame in session.push_frames(frames[i:i + n]):          Image.fromarray(frame).save(f"out_{n_out:05d}.png"); n_out += 1      i += n   `

### C. If step 5 is too slow for you

With the full model it will be: expect minutes per second of video. To get live speed you have to train a smaller model, on GPUs with real editing data:

1.  Pick a size by timing random models of each size on your device:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m joyedge.make_bench_model --preset student-s --out s-f16.gguf && engine/build/joyedge-quantize s-f16.gguf s-q6_k.gguf q6_k && engine/build/joyedge-bench -m s-q6_k.gguf --lat-h 10 --lat-w 18   `

1.  Optionally initialise the student from the big model:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m joyedge.slice_teacher --teacher "../deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth" --hidden 1024 --heads 8 --depth 16 --out student_s_init.pth   `

1.  Train stage 1, then stage 2, from the training/ directory:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   JOYAI_STUDENT=s JOYAI_STUDENT_RES=240,432 JOYAI_STUDENT_INIT=../edge/student_s_init.pth bash scripts/train/stage1_ar.sh configs/edge_student_stage1_ar.py 8   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   JOYAI_STUDENT=s JOYAI_STUDENT_RES=240,432 JOYAI_STAGE1_CKPT= bash scripts/train/stage2_distillation.sh configs/edge_student_stage2_distillation.py 8   `

1.  Convert the trained student with the same commands as step B4 and run it with B5–B7.
    

Stage 2 saves a LoRA-only checkpoint, which the exporter rejects until it is merged into the stage-1 weights, and I did not write that merge step. Two other things are also not built: a small replacement for the VAE, and using the 16B model as the teacher in stage 2.

i will use 4090 for above all parts will it make it work or exceeds the things like gpu ?

**Mostly it doesn’t matter, because the engine I built runs on the CPU and never touches the 4090.** The GPU only comes into play in three places, and two of them exceed 24 GB.

PartUses the 4090?Fits in 24 GB?A. Push to forkNon/aB1–B5. Build, tests, convert, quantize, benchmarkNo, CPU and disk onlyn/aB6. Prompt embeddings (7B text encoder)Yes, if you add --device cuda --dtype bf16Yes, about 14–16 GB. At fp32 (about 28 GB) it does not fit.B7. Editing a video with my engineNo. The transformer and the VAE both run on the CPU as written.n/aC1–C2. Sizing and slicing a studentNon/aC3. Training a studentYesVery unlikely beyond the smallest student, see below

So for part B the 4090 will not make anything faster except B6. Your speed is set by the CPU in that machine, and the full model will still take minutes per second of video.

### Where the 4090 does exceed

*   **The original GPU server at 480p.** The repo’s own guide reports about 21.5 GiB resident and 28 GiB peak, which is over 24 GB. This is the problem you already hit.
    
*   **The unquantized transformer in PyTorch.** It is 32 GB in bf16, so it does not fit at all without the FP8 path.
    
*   **Training (C3).** The trainer keeps the 7B text encoder (about 14–16 GB) and the VAE on the GPU next to the student, its gradients and optimizer state, and the video activations. I have not measured it, but I expect only the smallest student at 240p to have a chance, and stage 2 is heavier than stage 1. Using the 16B model as teacher is impossible on this card.
    

Apart from memory, one 4090 is not enough compute to train a video editing model to useful quality in reasonable time, and the repo only ships toy data.

### What I would actually do with a 4090

Use the original GPU server at a lower resolution, since that is where the card’s speed is usable. The command below is untested — I have no 4090 — but it is the configuration most likely to fit: 624×360, low-VRAM mode, no local prompt enhancer, and the new knob that skips the long VAE autotuning. Build joyomni\_ops first with JOYOMNI\_OPS\_CUDA\_ARCHS=89, which makes it compile for the 4090 only.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   JOYOMNI_FA4_ATTN=0 JOYOMNI_SAGE_ATTN=0 JOYOMNI_LOW_VRAM=1 JOYOMNI_PE_BACKEND=none JOYOMNI_VAE_COMPILE_MODE=default JOYOMNI_WIDTH=624 JOYOMNI_HEIGHT=360 bash deploy/run_server.sh   `

If that still runs out of memory, drop to JOYOMNI\_WIDTH=432 JOYOMNI\_HEIGHT=240. The FP8 weights alone take about 16 GB, so there is not much headroom either way.