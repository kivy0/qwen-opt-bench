import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_mapping.get(dtype_str, torch.bfloat16)


def build_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(
    name_or_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str | None = "flash_attention_2",
    gradient_checkpointing: bool = False,
) -> PreTrainedModel:
    logger.info("Loading model...")

    dtype = str_to_torch_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    logger.info("Done.")
    return model
