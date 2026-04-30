import logging
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def get_dataloader(
    dataset,
    tokenizer=None,
    seed: int = 42,
    batch_size: int = 1,
    mlm: bool = False,
    **kwargs,
) -> DataLoader:
    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    collate_fn = kwargs.pop("collate_fn", None)
    if collate_fn is None and tokenizer is not None:
        collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
        **kwargs,
    )


def prepare_dataset(hf_dataset_name: str, tokenizer, max_length: int = 512):
    logger.info(f"Loading dataset {hf_dataset_name}...")
    dataset = load_dataset(hf_dataset_name, split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    logger.info(f"Tokenizing dataset (max_length={max_length})...")
    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )
    logger.info("Formatting dataset to PyTorch tensors...")
    tokenized.set_format("torch")
    logger.info("Done.")
    return tokenized
