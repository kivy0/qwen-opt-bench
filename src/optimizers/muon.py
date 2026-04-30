"""
The original Muon optimizer logic from Moonlight was used,
but with our own implementation of the hybrid optimizer and
optimizers from torch.optim: AdamW, Muon.
"""

import torch
from torch.optim import AdamW, Muon

from src.optimizers.hybrid import HybridOptimizer


class MoonlightMuonOptimizer(HybridOptimizer):
    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter],
        muon_lr: float = 1e-3,
        adamw_lr: float = 1e-3,
        muon_kwargs: dict = {},
        adamw_kwargs: dict = {},
    ):

        optimizers = []

        optimizers.append(Muon(muon_params, lr=muon_lr, **muon_kwargs))

        optimizers.append(AdamW(adamw_params, lr=adamw_lr, **adamw_kwargs))

        super().__init__(optimizers)
