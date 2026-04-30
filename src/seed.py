import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set all random seeds for reproducibility"""
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False

        # Full determinism of all operations PyTorch
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    torch.set_float32_matmul_precision("highest")

    logger.info(f"Set all seeds to {seed}")
