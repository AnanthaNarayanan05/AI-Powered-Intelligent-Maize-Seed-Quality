"""Central config loader. Reads configs/config.yaml so every script/service shares
one source of truth for paths, hyperparameters, and dataset metadata."""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path (e.g. cfg['paths']['checkpoints']) against the
    project root, creating parent directories for output-style paths on demand."""
    p = PROJECT_ROOT / relative
    return p


def get_device(prefer: str = "cuda"):
    """Lazy-import torch so non-ML code (backend, data scripts) never needs it
    installed. Returns a torch.device, auto-falling back to CPU."""
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
