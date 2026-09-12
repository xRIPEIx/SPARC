"""Shared superpixel mask directory metadata (precomputed or synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

META_FILENAME = "superpixel_mask_meta.json"
FORMAT_ID = "sparc_region_mask_v1"
MODE_PRECOMPUTED = "precomputed_npy"
MODE_SYNTHETIC = "synthetic_constant"


def meta_path(mask_root: Path) -> Path:
    return Path(mask_root) / META_FILENAME


def load_meta(mask_root: Path) -> dict[str, Any] | None:
    path = meta_path(mask_root)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid superpixel mask meta (expected object): {path}")
    return data


def write_meta(mask_root: Path, meta: dict[str, Any]) -> Path:
    mask_root = Path(mask_root)
    mask_root.mkdir(parents=True, exist_ok=True)
    path = meta_path(mask_root)
    payload = {"format": FORMAT_ID, **meta}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def is_synthetic_constant(meta: dict[str, Any] | None) -> bool:
    return bool(meta) and meta.get("mode") == MODE_SYNTHETIC
