import concurrent.futures
import json
import re
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    Qwen3_5MoeForConditionalGeneration,
)

from .pe_qwen_kernels import Int4Experts, prepare_kernels


def load_model(path):
    path = Path(path)
    start = time.perf_counter()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    q = config.quantization_config
    assert (
        q["bits"] == 4 and q["group_size"] == 128 and q["sym"] and (not q["desc_act"])
    )
    del config.quantization_config
    with torch.device("meta"):
        model = Qwen3_5MoeForConditionalGeneration(config)
        for layer in model.model.language_model.layers:
            layer.mlp.experts = Int4Experts(config.text_config)
    model.set_attn_implementation("sdpa")
    model.to(dtype=torch.bfloat16).to_empty(device="cpu")
    state = model.state_dict()
    loaded = set()
    seen = set()
    pattern = re.compile(
        "model.language_model.layers\\.(\\d+)\\.mlp.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.(qweight|scales|qzeros|g_idx)$"
    )

    def read(shard):
        tensors = load_file(str(shard), device="cpu")
        for key, tensor in tensors.items():
            match = pattern.fullmatch(key)
            if match:
                (layer, expert, proj, kind) = match.groups()
                if kind in ("qzeros", "g_idx"):
                    if kind == "g_idx":
                        assert torch.equal(
                            tensor,
                            torch.arange(tensor.numel(), dtype=tensor.dtype) // 128,
                        )
                    else:
                        assert bool((tensor == -2004318072).all()), (
                            key,
                            tensor.flatten()[:8],
                        )
                    continue
                module = model.model.language_model.layers[int(layer)].mlp.experts
                target = getattr(
                    module,
                    ("down" if proj == "down_proj" else "gate_up")
                    + ("_q" if kind == "qweight" else "_s"),
                )[int(expert)]
                if proj != "down_proj":
                    target = target[0:512] if proj == "gate_proj" else target[512:1024]
                target.copy_(tensor.T)
                seen.add((layer, expert, proj, kind))
            else:
                if key.startswith("mtp.") or ".mtp." in key:
                    continue
                if key not in state:
                    raise KeyError(key)
                state[key].copy_(tensor)
                loaded.add(key)
        print("SHARD", shard.name, flush=True)

    files = sorted(
        set(
            json.loads((path / "model.safetensors.index.json").read_text())[
                "weight_map"
            ].values()
        )
    )
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(read, [path / f for f in files]))
    missing = [k for k in state if k not in loaded and ".mlp.experts." not in k]
    assert not missing, (len(missing), missing[:8])
    assert len(seen) == 40 * 256 * 3 * 2, len(seen)
    model = model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    for module in model.modules():
        init = getattr(
            module,
            "compute_axial_rope_parameters",
            getattr(module, "compute_default_rope_parameters", None),
        )
        if init is not None:
            (inv, scaling) = init(module.config, device=torch.device("cpu"))
            module.inv_freq = inv
            module.original_inv_freq = inv.clone()
            module.attention_scaling = scaling
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model.generation_config = GenerationConfig.from_pretrained(
        path, local_files_only=True
    )
    prepare_kernels()
    print(
        f"[pe] Qwen INT4 CPU preparation: {time.perf_counter() - start:.2f}s",
        flush=True,
    )
    return (model, processor)


from transformers import StaticCache


class _GraphDecoder:
    def __init__(self, model):
        self.model = model
        self.capacity = 4096
        self.cache = StaticCache(config=model.config, max_cache_len=self.capacity)
        self.graph = None

    def copy_cache(self, source, length):
        for dst, src in zip(self.cache.layers, source.layers):
            if hasattr(src, "keys"):
                if not dst.is_initialized:
                    dst.lazy_initialization(src.keys, src.values)
                dst.keys[:, :, :length].copy_(src.keys)
                dst.values[:, :, :length].copy_(src.values)
                dst.cumulative_length.fill_(length)
            else:
                for i, state in src.conv_states.items():
                    if state is not None:
                        if not dst.is_conv_states_initialized[i]:
                            dst.lazy_initialization(conv_states=state, state_idx=i)
                        dst.conv_states[i].copy_(state)
                        dst.has_previous_state[i] = src.has_previous_state[i]
                for i, state in src.recurrent_states.items():
                    if state is not None:
                        if not dst.is_recurrent_states_initialized[i]:
                            dst.lazy_initialization(recurrent_states=state, state_idx=i)
                        dst.recurrent_states[i].copy_(state)

    def step(self):
        model = self.model
        self.mask.copy_(
            torch.where(self.slots <= self.pos, 0.0, torch.finfo(torch.bfloat16).min)
        )
        out = model.model.language_model(
            input_ids=self.token,
            position_ids=self.rope,
            attention_mask={"full_attention": self.mask, "linear_attention": None},
            past_key_values=self.cache,
            use_cache=True,
        )
        self.token.copy_(
            model.lm_head(out.last_hidden_state[:, -1:])[:, -1].argmax(-1).view(1, 1)
        )
        self.pos.add_(1)
        self.rope.add_(1)

    def generate(self, inputs, limit, should_stop):
        model = self.model
        if should_stop():
            raise TimeoutError("PE cancelled before prefill")
        if inputs.input_ids.shape[1] + limit > self.capacity:
            raise ValueError("PE input exceeds the decode cache capacity")
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.model.rope_deltas = None
        out = model(**inputs, use_cache=True, logits_to_keep=1)
        first = out.logits[:, -1].argmax(-1).view(1, 1)
        length = inputs.input_ids.shape[1]
        delta = model.model.rope_deltas.clone()
        self.copy_cache(out.past_key_values, length)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - started
        capture = 0

        def positions():
            return torch.cat(
                [
                    self.pos.view(1, 1, 1),
                    (self.pos + delta).view(1, 1, 1).expand(3, 1, 1),
                ],
                dim=0,
            )

        if self.graph is None:
            started_capture = time.perf_counter()
            self.token = first.clone()
            self.pos = torch.tensor([length], device="cuda")
            self.rope = positions().clone()
            self.slots = torch.arange(self.capacity, device="cuda").view(1, 1, 1, -1)
            self.mask = torch.empty(
                (1, 1, 1, self.capacity), device="cuda", dtype=torch.bfloat16
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.step()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.step()
            torch.cuda.synchronize()
            capture = time.perf_counter() - started_capture
        self.copy_cache(out.past_key_values, length)
        self.token.copy_(first)
        self.pos.fill_(length)
        self.rope.copy_(positions())
        del out
        tokens = [first.item()]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        while tokens[-1] not in eos and len(tokens) < limit:
            if should_stop():
                raise TimeoutError("PE generation cancelled")
            self.graph.replay()
            tokens.append(self.token.item())
        return (tokens, dict(prefill_seconds=prefill, capture_seconds=capture))


MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
MODEL_REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"


def _linear_patch(self, x):
    return torch.nn.functional.linear(
        x.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype),
        self.proj.weight.flatten(1),
        self.proj.bias,
    )


class QwenRuntime:
    def __init__(self, path, device="cuda"):
        import types

        (self.model, self.processor) = load_model(path)
        self.model.model.visual.patch_embed.forward = types.MethodType(
            _linear_patch, self.model.model.visual.patch_embed
        )
        self.model.to(device)
        self.decoder = None

    def generate(self, system, text, images, should_stop):
        content = [{"type": "text", "text": text}]
        for i in range(len(images)):
            content.extend(
                [{"type": "text", "text": f"\n[Image {i}]:"}, {"type": "image"}]
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=[rendered], images=images or None, padding=True, return_tensors="pt"
        ).to(self.model.device)
        if self.decoder is None:
            self.decoder = _GraphDecoder(self.model)
        # CUDA RNG graph state is shared with video; keep it outside inference mode.
        with torch.no_grad():
            (tokens, timings) = self.decoder.generate(inputs, 384, should_stop)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if tokens[-1] not in eos:
            raise ValueError("Qwen PE output was truncated")
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not text or "<think" in text.lower() or "</think" in text.lower():
            raise ValueError("Qwen PE did not return an editing prompt")
        return (text, timings)
