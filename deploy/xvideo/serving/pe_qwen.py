"""Local Qwen3.6-27B GPTQ PE with Marlin and CUDA Graph decoding."""

import time
import types
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, StaticCache


def _gptq_backend():
    # GPTQModel 6.0.3 needs these aliases with Transformers 5.20.2.
    import huggingface_hub as hub
    from transformers.utils import hub as transformers_hub

    transformers_hub.create_repo = hub.create_repo
    transformers_hub.list_repo_tree = hub.list_repo_tree
    from gptqmodel import BACKEND, GPTQModel

    return GPTQModel, BACKEND


def load_model(path, device="cuda"):
    start = time.perf_counter()
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen3_5" or config.text_config.num_hidden_layers != 64:
        raise ValueError("Local PE requires the dense Qwen3.6-27B GPTQ checkpoint")
    GPTQModel, BACKEND = _gptq_backend()
    wrapped = GPTQModel.load(
        str(path), device=device, backend=BACKEND.MARLIN,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
    )
    model = wrapped.model.eval().requires_grad_(False)
    quantized = sum(
        "Marlin" in type(module).__name__ and hasattr(module, "qweight")
        for module in model.modules()
    )
    if quantized != 400:
        raise ValueError(f"Expected 400 Marlin INT4 modules, got {quantized}")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    print(f"[pe] Qwen3.6-27B Marlin INT4: {quantized} modules, "
          f"loaded in {time.perf_counter() - start:.2f}s", flush=True)
    return model, processor


class _GraphDecoder:
    def __init__(self, model):
        self.model = model
        self.capacity = 4096
        self.cache = StaticCache(config=model.config, max_cache_len=self.capacity)
        self.graph = None
        eos = model.generation_config.eos_token_id
        self.eos_token_ids = frozenset([eos] if isinstance(eos, int) else eos)

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
            self.copy_cache(out.past_key_values, length)
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
        while tokens[-1] not in self.eos_token_ids and len(tokens) < limit:
            if should_stop():
                raise TimeoutError("PE generation cancelled")
            self.graph.replay()
            tokens.append(self.token.item())
        return (tokens, dict(prefill_seconds=prefill, capture_seconds=capture))


MODEL_ID = "btbtyler09/Qwen3.6-27B-GPTQ-4bit"
MODEL_REVISION = "bcfe8a7ec20eec6ce67ded4669819ae77e73af3a"


def _linear_patch(self, x):
    return torch.nn.functional.linear(
        x.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype),
        self.proj.weight.flatten(1),
        self.proj.bias,
    )


class QwenRuntime:
    def __init__(self, path, device="cuda"):
        (self.model, self.processor) = load_model(path, device=device)
        self.model.model.visual.patch_embed.forward = types.MethodType(
            _linear_patch, self.model.model.visual.patch_embed
        )
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
        with torch.cuda.device(self.model.device), torch.no_grad():
            (tokens, timings) = self.decoder.generate(inputs, 384, should_stop)
        if tokens[-1] not in self.decoder.eos_token_ids:
            raise ValueError("Qwen PE output was truncated")
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not text or "<think" in text.lower() or "</think" in text.lower():
            raise ValueError("Qwen PE did not return an editing prompt")
        return (text, timings)
