"""`sparc-sweep` -- expand sweep definitions and report progress.

sparc-sweep expand configs/sweeps/sparc_lambda.yaml -o experiments/sparc_lambda/manifest.csv
sparc-sweep status experiments/sparc_lambda/manifest.csv --results-root runs/results
sparc-sweep run-id --config-id sparc_lambda_0p5 --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparc.experiments.manifest import expand, read_manifest, select_row, write_manifest
from sparc.experiments.run_id import resolve_run_id

TASKS = ("segmentation", "detection")


def _cmd_expand(args) -> int:
    rows = expand(args.sweep)
    if args.output:
        path = write_manifest(rows, args.output)
        configs = len({r["config_id"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        print(f"{path}: {len(rows)} rows ({configs} configs x seeds {seeds})")
    else:
        import csv

        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def _cmd_row(args) -> int:
    """Print one row's field, for a scheduler task to consume.

    Shell scripts read single fields rather than parsing CSV, which keeps the
    job scripts trivial and keeps the parsing in tested Python.
    """
    row = select_row(read_manifest(args.manifest), args.array_index, args.seed)
    if args.field:
        if args.field not in row:
            raise SystemExit(f"Unknown field {args.field!r}. Available: {sorted(row)}")
        print(row[args.field])
    else:
        for key, value in row.items():
            print(f"{key}={value}")
    return 0


def _cmd_status(args) -> int:
    """Report completion, derived from files on disk.

    Never from a `status` column: a manifest records what was planned, and the
    filesystem records what actually finished. Keeping those separate is what
    stops a status table from drifting out of date in either direction.
    """
    rows = read_manifest(args.manifest)
    results_root = Path(args.results_root)
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else None

    by_config: dict[str, dict[str, int]] = {}
    for row in rows:
        stats = by_config.setdefault(
            row["config_id"], {"planned": 0, "pretrained": 0, **{t: 0 for t in TASKS}}
        )
        stats["planned"] += 1
        if checkpoint_root is not None and any((checkpoint_root / row["run_id"]).glob("*.pth")):
            stats["pretrained"] += 1
        for task in TASKS:
            if (results_root / task / f"{row['run_id']}.csv").is_file():
                stats[task] += 1

    width = max(len(c) for c in by_config)
    header = f"{'config_id':<{width}}  pretrain  " + "  ".join(f"{t[:3]}" for t in TASKS)
    print(header)
    print("-" * len(header))
    for config_id, stats in sorted(by_config.items()):
        planned = stats["planned"]
        cells = "  ".join(f"{stats[t]}/{planned}" for t in TASKS)
        pre = f"{stats['pretrained']}/{planned}" if checkpoint_root else "   -"
        print(f"{config_id:<{width}}  {pre:>8}  {cells}")

    total = sum(s["planned"] for s in by_config.values())
    done = {t: sum(s[t] for s in by_config.values()) for t in TASKS}
    print()
    print(f"total runs: {total};  " + ";  ".join(f"{t}: {done[t]}/{total}" for t in TASKS))
    return 0


def _cmd_run_id(args) -> int:
    print(resolve_run_id(args.config_id, args.seed))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="sparc-sweep", description="Sweep manifests and status.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_expand = sub.add_parser("expand", help="Expand a sweep YAML into a manifest CSV.")
    p_expand.add_argument("sweep")
    p_expand.add_argument("-o", "--output", default=None)
    p_expand.set_defaults(func=_cmd_expand)

    p_row = sub.add_parser("row", help="Print one manifest row (for job scripts).")
    p_row.add_argument("manifest")
    p_row.add_argument("--array-index", type=int, required=True)
    p_row.add_argument("--seed", type=int, required=True)
    p_row.add_argument("--field", default=None)
    p_row.set_defaults(func=_cmd_row)

    p_status = sub.add_parser("status", help="Report completion from files on disk.")
    p_status.add_argument("manifest")
    p_status.add_argument("--results-root", required=True)
    p_status.add_argument("--checkpoint-root", default=None)
    p_status.set_defaults(func=_cmd_status)

    p_run = sub.add_parser("run-id", help="Resolve config_id + seed to a run_id.")
    p_run.add_argument("--config-id", required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.set_defaults(func=_cmd_run_id)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
