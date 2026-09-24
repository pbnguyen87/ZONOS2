"""Minimal LoRA for Zonos2Trainable.

Wraps nn.Linear / Linear3D leaves in-place; ``export_merged_state_dict`` folds the
adapters back into checkpoint-format weights so the saved model needs no LoRA code
at inference time.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Linear3D

# Attribute names eligible for adaptation (last path component of the module).
#   wq/wkv/wo/gater : attention projections
#   w_in/w_out      : dense FFN
#   multi_output    : output head
#   speaker_projection : speaker conditioning projection
DEFAULT_TARGETS = ("wq", "wkv", "wo", "w_in", "w_out")


class LoRAAdapter(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        weight = base.weight
        if weight.dim() == 3:
            out_features = weight.shape[0] * weight.shape[1]
            in_features = weight.shape[2]
        else:
            out_features, in_features = weight.shape
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dtype = weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return y + delta * self.scaling

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        w = self.base.weight
        return (w.float() + delta.view(w.shape).float()).to(w.dtype)


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> List[str]:
    """Freeze the model and wrap target linears with LoRA adapters.

    Returns the module paths that were adapted.
    """
    targets = set(targets)
    model.requires_grad_(False)

    adapted: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for attr, child in list(parent.named_children()):
            if attr not in targets:
                continue
            if not isinstance(child, (nn.Linear, Linear3D)):
                continue
            adapter = LoRAAdapter(child, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, adapter)
            adapted.append(f"{parent_name}.{attr}" if parent_name else attr)

    for name, p in model.named_parameters():
        p.requires_grad = "lora_A" in name or "lora_B" in name
    if not adapted:
        raise ValueError(f"No modules matched LoRA targets {sorted(targets)}")
    return adapted


def export_merged_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict in checkpoint format with LoRA deltas folded into the weights.

    Adapter wrapping renames keys (``...wq.weight`` -> ``...wq.base.weight`` plus
    ``lora_A``/``lora_B``); this reverses that and merges, without mutating the
    live model.
    """
    merged_by_path: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            merged_by_path[name] = module.merged_weight()

    out: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if ".lora_A" in key or ".lora_B" in key:
            continue
        if key.endswith(".base.weight"):
            path = key[: -len(".base.weight")]
            out[path + ".weight"] = merged_by_path.get(path, value).detach().cpu()
        elif ".base." in key:  # e.g. a bias on a wrapped linear
            out[key.replace(".base.", ".")] = value.detach().cpu()
        else:
            out[key] = value.detach().cpu()
    return out


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Just the adapter weights, for lightweight intermediate saves."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k
    }


__all__ = [
    "LoRAAdapter",
    "apply_lora",
    "export_merged_state_dict",
    "lora_state_dict",
    "DEFAULT_TARGETS",
]
