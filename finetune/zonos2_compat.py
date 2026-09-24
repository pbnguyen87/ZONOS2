"""Bridging helpers between the fine-tuning code and the zonos2 inference package.

The inference package's sub-package ``__init__``s pull in flashinfer / sgl_kernel /
zmq at import time (e.g. ``zonos2.tts`` imports the whole scheduler stack). Training
must not depend on those, so the pure modules we need (prompt building, the speaker
encoder, text normalization) are loaded directly by file path instead of through the
package machinery.

Also holds checkpoint I/O: loading a Zonos2 ``model.pth`` (with training-key
normalization and expert-weight fusion) and saving a servable checkpoint back.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ZONOS2_SRC = REPO_ROOT / "python" / "zonos2"

_MODULE_CACHE: Dict[str, ModuleType] = {}


def load_zonos2_module(relpath: str) -> ModuleType:
    """Load a module from python/zonos2/<relpath> without importing its package.

    This skips the sub-package ``__init__`` (which may import flashinfer/zmq) while
    still letting the module's own absolute imports resolve normally.
    """
    if relpath in _MODULE_CACHE:
        return _MODULE_CACHE[relpath]
    path = ZONOS2_SRC / relpath
    if not path.is_file():
        raise FileNotFoundError(f"zonos2 source module not found: {path}")
    mod_name = "zonos2_finetune_compat." + relpath.replace("/", ".").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[relpath] = module
    return module


def prompt_module() -> ModuleType:
    """zonos2/tts/prompt.py — byte tokenization, conditioning ids, shear, silence."""
    return load_zonos2_module("tts/prompt.py")


def speaker_cloning_module() -> ModuleType:
    """zonos2/models/speaker_cloning.py — Qwen3SpeakerEmbedding."""
    return load_zonos2_module("models/speaker_cloning.py")


def textnorm_module() -> ModuleType:
    """zonos2/tokenizer/textnorm.py — NeMo forward text normalization."""
    return load_zonos2_module("tokenizer/textnorm.py")


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def load_config(model_path: str):
    """Load the Zonos2Config sidecar (params.json / config.yaml) for a checkpoint.

    Resolves HF repo ids the same way the server does.
    """
    # zonos2.utils has no heavy GPU imports (logger/zmq/hf only), safe to import.
    if str(REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "python"))
    from zonos2.utils.hf import cached_load_checkpoint_config

    return cached_load_checkpoint_config(model_path)


def resolve_model_dir(model_path: str) -> Path:
    if str(REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "python"))
    from zonos2.utils.hf import resolve_model_path

    return Path(resolve_model_path(model_path))


# ---------------------------------------------------------------------------
# Checkpoint state-dict I/O
# ---------------------------------------------------------------------------

_TRAINING_ONLY_KEY_PARTS = (".router.ent_denom", ".router.normalized_entropy")


def _normalize_training_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mirror zonos2.models.weight._normalize_zonos2_state_dict."""
    for key in list(sd.keys()):
        if ".parametrizations." in key and ".original" in key:
            new_key = key.replace(".parametrizations.", ".").replace(".original", "")
            sd[new_key] = sd.pop(key)
        elif any(part in key for part in _TRAINING_ONLY_KEY_PARTS):
            sd.pop(key)
    return sd


def _fuse_expert_weights(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert per-checkpoint expert layouts to fused gate_up_proj / down_proj.

    Mirrors FusedGroupedExperts.load_state_dict: supports the unfused
    ``w1.weight``/``w3.weight``/``w2.weight`` layout, the SonicMoE interleaved
    ``w13``/``w2`` layout, and the already-fused layout.
    """
    prefixes = set()
    for key in sd:
        marker = ".experts."
        idx = key.find(marker)
        if idx >= 0:
            prefixes.add(key[: idx + len(marker)])

    for prefix in prefixes:
        gate_up = prefix + "gate_up_proj"
        down = prefix + "down_proj"
        if gate_up not in sd:
            if prefix + "w13" in sd:
                w13 = sd.pop(prefix + "w13")
                assert w13.dim() == 3 and w13.shape[1] % 2 == 0, w13.shape
                sd[gate_up] = torch.cat([w13[:, 0::2, :], w13[:, 1::2, :]], dim=1)
            elif prefix + "w1.weight" in sd and prefix + "w3.weight" in sd:
                w1 = sd.pop(prefix + "w1.weight")
                w3 = sd.pop(prefix + "w3.weight")
                sd[gate_up] = torch.cat([w1, w3], dim=1)
        if down not in sd:
            if prefix + "w2" in sd:
                sd[down] = sd.pop(prefix + "w2")
            elif prefix + "w2.weight" in sd:
                sd[down] = sd.pop(prefix + "w2.weight")
    return sd


def load_checkpoint_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Load model.pth from a checkpoint dir (or a direct .pt/.pth file)."""
    model_dir = resolve_model_dir(model_path)
    candidate = None
    if model_dir.is_dir():
        for name in ("model.pth", "model.pt", "consolidated/consolidated.pth"):
            if (model_dir / name).is_file():
                candidate = model_dir / name
                break
    elif model_dir.is_file() and model_dir.suffix in (".pt", ".pth"):
        candidate = model_dir
    if candidate is None:
        raise FileNotFoundError(f"No model.pth/model.pt found under {model_dir}")

    state = torch.load(candidate, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = _normalize_training_keys(dict(state))
    state = _fuse_expert_weights(state)
    return state


def save_servable_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    source_model_path: str,
    output_dir: str | Path,
) -> Path:
    """Write model.pth plus the config sidecars so the dir is directly servable.

    The result can be passed straight to ``python -m zonos2 --model-path <dir>``
    or ``TTSLLM(model_path=<dir>)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "model.pth")

    src = resolve_model_dir(source_model_path)
    copied_config = False
    for name in ("params.json", "config.yaml"):
        if (src / name).is_file():
            shutil.copy2(src / name, output_dir / name)
            copied_config = True
    if not copied_config:
        raise FileNotFoundError(
            f"No params.json/config.yaml found next to {src}; the saved checkpoint "
            "would not be loadable by the server."
        )
    return output_dir


def save_json(path: str | Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
