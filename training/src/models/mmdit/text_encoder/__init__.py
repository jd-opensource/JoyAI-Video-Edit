import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
):
    ckpt_name = text_encoder_ckpt.lower()
    is_qwen3_vl_family = (
        'qwen3-vl-8b-instruct' in ckpt_name
        or 'joyomni' in ckpt_name
    )

    if is_qwen3_vl_family:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(device).eval().requires_grad_(False)
        tokenizer = Qwen2Tokenizer.from_pretrained(
            text_encoder_ckpt,
            local_files_only=True,
        )
        return tokenizer, model
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            torch_dtype=torch_dtype,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        ).to(device).eval().requires_grad_(False)
        tokenizer = Qwen2Tokenizer.from_pretrained(
            text_encoder_ckpt,
            local_files_only=True,
        )
        return tokenizer, model
