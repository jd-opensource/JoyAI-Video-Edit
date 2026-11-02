import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLModel, Qwen3VLModel


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load a frozen multimodal backbone and tokenizer for prompt conditioning."""
    ckpt_name = text_encoder_ckpt.lower()
    is_qwen3_vl_family = (
        'qwen3-vl-8b-instruct' in ckpt_name
        or 'joyomni' in ckpt_name
    )

    model_class = Qwen3VLModel if is_qwen3_vl_family else Qwen2_5_VLModel
    model, loading = model_class.from_pretrained(
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
