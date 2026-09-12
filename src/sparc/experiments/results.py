"""Downstream result rows and their CSV files.

One CSV per run, named by run_id, so that concurrent array tasks never contend
for a single file. Aggregation happens later, from whatever files exist.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

#: Written for every run, before any task-specific metric columns.
BASE_COLUMNS = (
    "run_id",
    "config_id",
    "task",
    "init",
    "backbone",
    "ssl_method",
    "pretrain_dataset",
    "ssl_epochs",
    "ssl_ckpt",
    "seed",
    "downstream_epochs",
    "batch_size",
    "lr",
    "weight_decay",
)


def read_ssl_metadata(ssl_ckpt: str | Path | None) -> dict[str, Any]:
    """Read method/dataset/epoch out of a checkpoint so results are
    self-describing rather than relying on what the caller claimed."""
    meta: dict[str, Any] = {"ssl_method": "", "pretrain_dataset": "", "ssl_epochs": -1}
    if not ssl_ckpt or not Path(ssl_ckpt).exists():
        return meta
    try:
        payload = torch.load(ssl_ckpt, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - metadata is best-effort, never fatal
        return meta
    meta["ssl_method"] = payload.get("method", "")
    config = payload.get("config", {}) or {}
    meta["pretrain_dataset"] = (config.get("data", {}) or {}).get("dataset_name", "")
    meta["ssl_epochs"] = payload.get("epoch", -1)
    return meta


def build_result_row(
    *,
    run_id: str,
    config_id: str,
    task: str,
    init: str,
    backbone: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    ssl_ckpt: str | None,
    metrics: dict[str, float],
) -> dict[str, Any]:
    meta = read_ssl_metadata(ssl_ckpt)
    row = {
        "run_id": run_id,
        "config_id": config_id,
        "task": task,
        "init": init,
        "backbone": backbone,
        "ssl_method": meta["ssl_method"],
        "pretrain_dataset": meta["pretrain_dataset"],
        "ssl_epochs": meta["ssl_epochs"],
        "ssl_ckpt": ssl_ckpt or "",
        "seed": seed,
        "downstream_epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    row.update(metrics)
    return row


def write_result_csv(row: dict[str, Any], path: str | Path) -> Path:
    """Write a single-row CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(BASE_COLUMNS) + [k for k in row if k not in BASE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    return path


def read_result_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
