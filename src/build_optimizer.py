"""
Factory and utilities for building optimizers
with parameter grouping strategies
for transformer-style models.
"""

import torch
from torch.optim import AdamW

from src.config import AdamWConfig, Config, HybridOptimizerConfig, MoonlightMuonConfig
from src.optimizers.hybrid import HybridOptimizer
from src.optimizers.muon import MoonlightMuonOptimizer

# TODO Add creation without configs for notebook etc


def get_muon_adamw_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Divides the model parameters into two groups:
    1. Muon: 2D matrices (excluding the embed and head layers)
    2. AdamW: all other parameters (bias, LayerNorm, 1D tensors, embed, head)
    """
    muon_params, adamw_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim >= 2 and "embed" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params


def get_three_way_params(
    model: torch.nn.Module,
    muon_layers_limit: int = 12,
) -> tuple[
    list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]
]:
    """
    Splits model parameters into three groups:
    1. muon_params: 2D matrices for the first muon_layers_limit layers.
    2. adamw_2d_params: 2D matrices for the remaining layers (starting from muon_layers_limit).
    3. adamw_other_params: 1D tensors, LayerNorm, bias, embed_tokens, lm_head.
    """
    muon_params = []
    adamw_2d_params = []
    adamw_other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_2d_matrix = p.ndim >= 2 and "embed" not in name and "head" not in name

        if is_2d_matrix:
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            if layer_idx is not None and layer_idx < muon_layers_limit:
                muon_params.append(p)
            else:
                adamw_2d_params.append(p)
        else:
            adamw_other_params.append(p)

    return muon_params, adamw_2d_params, adamw_other_params


def _build_adamw(model: torch.nn.Module, config: AdamWConfig):
    return AdamW(model.parameters(), **config.model_dump(exclude={"type"}))


def _build_moonlight_muon(model: torch.nn.Module, config: MoonlightMuonConfig):
    muon_params, adamw_params = get_muon_adamw_params(model)
    return MoonlightMuonOptimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_lr=config.muon.lr,
        adamw_lr=config.adamw.lr,
        muon_kwargs=config.muon.model_dump(exclude={"lr"}),
        adamw_kwargs=config.adamw.model_dump(exclude={"type", "lr"}),
    )


def _build_hybrid(
    model: torch.nn.Module,
    config: HybridOptimizerConfig,
) -> HybridOptimizer:
    muon_params, adamw_2d_params, adamw_other_params = get_three_way_params(
        model, muon_layers_limit=config.muon_layers_limit
    )

    muon_adamw_opt = MoonlightMuonOptimizer(
        muon_params=muon_params,
        adamw_params=adamw_2d_params,
        muon_lr=config.muon.lr,
        adamw_lr=config.adamw_2d.lr,
        muon_kwargs=config.muon.model_dump(exclude={"lr"}),
        adamw_kwargs=config.adamw_2d.model_dump(exclude={"type", "lr"}),
    )

    other_opt = AdamW(
        adamw_other_params,
        lr=config.adamw_other.lr,
        **config.adamw_other.model_dump(exclude={"type", "lr"}),
    )

    return HybridOptimizer([muon_adamw_opt, other_opt])


def _build_mezo(
    model: torch.nn.Module,
    config,
):
    raise NotImplementedError()


OPTIMIZER_REGISTRY = {
    "adamw": _build_adamw,
    "moonlight_muon": _build_moonlight_muon,
    "hybrid": _build_hybrid,
    "mezo": _build_mezo,
}


def build_optimizer(model: torch.nn.Module, config: Config):
    opt_config = config.optimizer
    opt_type = opt_config.type.lower()

    if opt_type not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer {opt_type}. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    return OPTIMIZER_REGISTRY[opt_type](model, opt_config)
