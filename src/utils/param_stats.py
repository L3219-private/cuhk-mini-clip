# src/utils/param_stats.py

"""
parameter statistics for lightweight scoring.

How to use:
    from src.utils.param_stats import param_stats
    stats = param_stats(model)
    print(stats)
"""

import torch.nn as nn


def param_stats(model: nn.Module) -> dict:
    """Return a dict summarising parameter counts and freeze status.

    Returns：
    dict with keys:
        total_params：int, every parameter in the model
        trainable_params：int, parameters with requires_grad=True
        frozen_params：int, parameters with requires_grad=False
        trainable_ratio：float, trainable / total (0.0–1.0)
        total_M：str, e.g. "11.50M"
        trainable_M：str, e.g. "11.50M"
        backbone_frozen：bool, True if the model has a .backbone
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    frozen = total - trainable

    # Check whether .backbone exists and is fully frozen
    backbone_frozen = False
    if hasattr(model, "backbone"):
        bb_params = list(model.backbone.parameters())
        if bb_params:  # non-empty
            backbone_frozen = all(not p.requires_grad for p in bb_params)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
        "total_M": f"{total / 1e6:.2f}M",
        "trainable_M": f"{trainable / 1e6:.2f}M",
        "backbone_frozen": backbone_frozen,
    }


def print_param_stats(model: nn.Module, name: str = "Model") -> dict:
    """Pretty-print parameter stats and return the dict."""
    s = param_stats(model)
    print(f"{'─' * 40}")
    print(f" {name}")
    print(f"{'─' * 40}")
    print(f"  Total params:      {s['total_params']:>12,}  ({s['total_M']})")
    print(f"  Trainable params:  {s['trainable_params']:>12,}  ({s['trainable_M']})")
    print(f"  Frozen params:     {s['frozen_params']:>12,}")
    print(f"  Trainable ratio:   {s['trainable_ratio']:>11.1%}")
    print(f"  Backbone frozen:   {str(s['backbone_frozen']):>11}")
    print(f"{'─' * 40}")
    return s
