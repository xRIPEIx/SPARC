"""Aggregate per-run results into per-configuration statistics.

One CSV per run comes in; one row per configuration goes out, carrying the mean
and the *sample* standard deviation (n-1) over the seeds that completed, the
seed count, and the raw values. Everything the paper's tables and figures show
is computed here, once, so a table and a figure cannot disagree.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from sparc.experiments.manifest import read_manifest
from sparc.experiments.results import BASE_COLUMNS, read_result_csv
from sparc.experiments.run_id import parse_run_id

#: Metric columns per task, in report order.
TASK_METRICS = {
    "segmentation": ("mIoU", "pixel_acc"),
    "detection": ("AP", "AP50", "AP75"),
}


def _metric_columns(rows: list[dict[str, str]], task: str) -> list[str]:
    present = set().union(*(r.keys() for r in rows)) if rows else set()
    return [m for m in TASK_METRICS[task] if m in present] or sorted(
        c for c in present if c not in BASE_COLUMNS
    )


def load_runs(results_root: str | Path, task: str, run_ids: list[str] | None = None):
    """Read every per-run CSV for a task, optionally restricted to a run list."""
    task_dir = Path(results_root) / task
    rows: list[dict[str, str]] = []
    wanted = set(run_ids) if run_ids is not None else None
    for path in sorted(task_dir.glob("*.csv")):
        run_id = path.stem
        if wanted is not None and run_id not in wanted:
            continue
        for row in read_result_csv(path):
            row = dict(row)
            row.setdefault("run_id", run_id)
            if not row.get("config_id"):
                row["config_id"] = parse_run_id(row["run_id"])[0]
            if not row.get("seed"):
                row["seed"] = str(parse_run_id(row["run_id"])[1])
            rows.append(row)
    return rows


def aggregate(
    rows: list[dict[str, str]],
    task: str,
    *,
    manifest: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Group runs by config_id and compute per-metric statistics."""
    metrics = _metric_columns(rows, task)
    planned: dict[str, set[int]] = {}
    axis: dict[str, tuple[str, str]] = {}
    if manifest:
        for m in manifest:
            planned.setdefault(m["config_id"], set()).add(int(m["seed"]))
            axis[m["config_id"]] = (m.get("axis_key", ""), m.get("axis_value", ""))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)

    records: list[dict[str, Any]] = []
    config_ids = sorted(set(grouped) | set(planned))
    for config_id in config_ids:
        runs = sorted(grouped.get(config_id, []), key=lambda r: int(r["seed"]))
        seeds = [int(r["seed"]) for r in runs]
        record: dict[str, Any] = {
            "config_id": config_id,
            "task": task,
            "axis_key": axis.get(config_id, ("", ""))[0],
            "axis_value": axis.get(config_id, ("", ""))[1],
            "seeds_planned": len(planned.get(config_id, set(seeds))),
            "seeds_complete": len(seeds),
            "seeds": ";".join(map(str, seeds)),
        }
        for name in metrics:
            values = [float(r[name]) for r in runs if r.get(name) not in (None, "")]
            record[f"{name}_mean"] = statistics.fmean(values) if values else math.nan
            record[f"{name}_std"] = statistics.stdev(values) if len(values) >= 2 else math.nan
            record[f"{name}_n"] = len(values)
            record[f"{name}_values"] = ";".join(f"{v:.6f}" for v in values)
        records.append(record)
    return records


def write_aggregate(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0]) if records else ["config_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    k: ("" if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in record.items()
                }
            )
    return path


def read_aggregate(path: str | Path) -> list[dict[str, Any]]:
    """Read a by_config CSV back with numeric fields parsed."""
    out = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rec: dict[str, Any] = dict(row)
            for k, v in row.items():
                if k.endswith(("_mean", "_std")):
                    rec[k] = float(v) if v not in ("", None) else math.nan
                elif k.endswith("_n") or k in ("seeds_planned", "seeds_complete"):
                    rec[k] = int(v) if v not in ("", None) else 0
            out.append(rec)
    return out


def merge_aggregates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine aggregates from several sweeps into one list, deduplicating by config_id.

    A config_id that appears in two sweeps must be the SAME run reported twice
    (densecl_lambda_0p5 is both the DenseCL baseline and a point on the DenseCL
    lambda curve). Identical statistics are merged; differing ones are an error,
    because that would mean two different runs are wearing one name.
    """
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for record in group:
            key = record["config_id"]
            if key in merged:
                for k, v in record.items():
                    prev = merged[key].get(k)
                    if k.endswith(("_mean", "_std")) and not _close(prev, v):
                        raise ValueError(
                            f"config_id {key!r} appears in two aggregates with different "
                            f"{k}: {prev} vs {v}. Two different runs share one name."
                        )
                continue
            merged[key] = dict(record)
    return list(merged.values())


def _close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= tol


def load_manifest_run_ids(manifest_path: str | Path) -> list[str]:
    return [m["run_id"] for m in read_manifest(manifest_path)]
