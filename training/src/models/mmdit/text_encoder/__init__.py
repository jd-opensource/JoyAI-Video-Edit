import torch
from transformers import AutoModel, Qwen2Tokenizer


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load a frozen multimodal backbone and tokenizer for prompt conditioning."""
    model, loading = AutoModel.from_pretrained(
        text_encoder_ckpt,
        dtype=torch_dtype,
        local_files_only=True,
        attn_implementation=attn_implementation,
        output_loading_info=True,
    )
    unexpected = set(loading["unexpected_keys"]) - {"lm_head.weight"}
    if loading["missing_keys"] or loading["mismatched_keys"] or unexpected:
        raise ValueError(f"Text encoder backbone checkpoint mismatch: {loading}")
    model.to(device).eval().requires_grad_(False)
    tokenizer = Qwen2Tokenizer.from_pretrained(text_encoder_ckpt, local_files_only=True)
    return tokenizer, model
