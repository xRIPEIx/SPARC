"""Sweep expansion.

A sweep YAML expands to a manifest: one row per (configuration, seed). Both the
pretraining job and the evaluation job read the same row, so the swept value a
checkpoint was trained with and the one recorded against its result cannot
disagree.

Array layout mirrors the scheduler's: `array_index` identifies the
configuration and repeats across seeds, so one submission covers one seed
(`--array 0-6` with SEED=3). That keeps job arrays small and lets seeds be
added later without renumbering anything.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from sparc.config.schema import SweepConfig
from sparc.experiments.run_id import resolve_run_id

MANIFEST_COLUMNS = (
    "array_index",
    "run_id",
    "config_id",
    "sweep",
    "seed",
    "axis_key",
    "axis_value",
    "pretrain_config",
    "pretrain_overrides",
    "downstream_overrides",
)


def format_value(value: Any) -> str:
    """Render a swept value for use inside an identifier.

    Decimal points become "p" and trailing zeros are dropped, so 0.5 -> "0p5"
    and 1.0 -> "1". This reproduces the naming already embedded in existing
    checkpoint directories and result filenames.
    """
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "neg")


def load_sweep(path: str | Path) -> DictConfig:
    """Load and validate a sweep definition."""
    path = Path(path)
    node = OmegaConf.load(path)
    sweep = OmegaConf.merge(OmegaConf.structured(SweepConfig), node)

    # NOTE: item access, not attribute access. OmegaConf resolves
    # `axis.values` to dict.values -- the bound method -- rather than to the
    # key named "values". Same trap awaits `keys` and `items`.
    has_axis = sweep.axis is not None and len(sweep.axis["values"]) > 0
    has_entries = len(sweep.entries) > 0
    if has_axis == has_entries:
        raise ValueError(
            f"{path}: give exactly one of `axis` (a one-factor sweep) or "
            f"`entries` (an explicit list), not both and not neither."
        )
    return sweep  # type: ignore[return-value]


def _resolve_relative(base: Path, value: str | None) -> str:
    """Resolve a config reference to a repo-relative path.

    References inside a sweep file are written relative to that file
    (../pretrain/x.yaml), but the manifest is committed and read on other
    machines, so an absolute path would bake this checkout's location into it.
    Anything outside the repository is left absolute, since nothing better
    exists.
    """
    if not value:
        return ""
    candidate = (base.parent / value).resolve()
    try:
        from sparc.config.paths import configs_dir

        return str(candidate.relative_to(configs_dir().parent))
    except (ValueError, FileNotFoundError):
        return str(candidate)


def expand(sweep_path: str | Path) -> list[dict[str, Any]]:
    """Expand a sweep definition into manifest rows."""
    sweep_path = Path(sweep_path)
    sweep = load_sweep(sweep_path)

    if sweep.axis is not None and len(sweep.axis["values"]) > 0:
        entries = [
            {
                "config_id": sweep.axis.config_id.format(value=format_value(value)),
                "pretrain_config": sweep.pretrain_config,
                "pretrain_overrides": [f"{sweep.axis.key}={value}"],
                "downstream_overrides": [],
                "axis_key": sweep.axis.key,
                "axis_value": value,
            }
            for value in sweep.axis["values"]
        ]
    else:
        entries = [
            {
                "config_id": entry.config_id,
                "pretrain_config": entry.pretrain_config or sweep.pretrain_config,
                "pretrain_overrides": list(entry.pretrain_overrides),
                "downstream_overrides": list(entry.downstream_overrides),
                "axis_key": "",
                "axis_value": "",
            }
            for entry in sweep.entries
        ]

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        for seed in sweep.seeds:
            run_id = resolve_run_id(entry["config_id"], seed)
            pretrain_overrides = [
                *entry["pretrain_overrides"],
                f"train.seed={seed}",
                f"experiment.config_id={entry['config_id']}",
                f"experiment.run_id={run_id}",
            ]
            downstream_overrides = [
                *entry["downstream_overrides"],
                f"train.seed={seed}",
                f"experiment.config_id={entry['config_id']}",
                f"experiment.run_id={run_id}",
            ]
            rows.append(
                {
                    "array_index": index,
                    "run_id": run_id,
                    "config_id": entry["config_id"],
                    "sweep": sweep.name,
                    "seed": seed,
                    "axis_key": entry["axis_key"],
                    "axis_value": entry["axis_value"],
                    "pretrain_config": _resolve_relative(sweep_path, entry["pretrain_config"]),
                    "pretrain_overrides": " ".join(pretrain_overrides),
                    "downstream_overrides": " ".join(downstream_overrides),
                }
            )
    return rows


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_row(rows: list[dict[str, str]], array_index: int, seed: int) -> dict[str, str]:
    """Find the row a scheduler task should run.

    Raises rather than returning a default: a task that cannot find its row must
    fail immediately, not silently run some other configuration.
    """
    for row in rows:
        if int(row["array_index"]) == array_index and int(row["seed"]) == seed:
            return row
    raise KeyError(
        f"No manifest row for array_index={array_index} seed={seed}. "
        f"Available indices: {sorted({int(r['array_index']) for r in rows})}; "
        f"seeds: {sorted({int(r['seed']) for r in rows})}"
    )
