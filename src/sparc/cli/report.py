"""`sparc-report` -- aggregate results and render tables, figures and checks.

    sparc-report aggregate --manifest experiments/sparc_lambda/manifest.csv \
                           --results-root runs/results --task segmentation \
                           -o experiments/results/aggregate/sparc_lambda_segmentation_by_config.csv
    sparc-report table  --aggregates experiments/results/aggregate --format markdown
    sparc-report readme --aggregates experiments/results/aggregate --readme README.md [--check]
    sparc-report figure --aggregates experiments/results/aggregate --metric mIoU -o fig.png
    sparc-report check  --aggregates experiments/results/aggregate
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

from sparc.experiments.aggregate import (
    aggregate,
    load_runs,
    merge_aggregates,
    read_aggregate,
    write_aggregate,
)
from sparc.experiments.manifest import read_manifest

SWEEPS = ("sparc_lambda", "densecl_lambda", "baselines")
TASKS = ("segmentation", "detection")
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"

BASELINE_LABELS = {
    "random": "Random init",
    "imagenet": "Supervised ImageNet",
    "moco": "MoCo v2",
    "densecl_lambda_0p5": "DenseCL",
    "sparc_lambda_0p5": "**SPARC** (λ = 0.5)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _agg_path(aggregates: Path, sweep: str, task: str) -> Path:
    return aggregates / f"{sweep}_{task}_by_config.csv"


def load_all(aggregates: Path) -> dict[str, list[dict[str, Any]]]:
    """{task: merged records across all sweeps}. Missing files are skipped."""
    out: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        groups = [
            read_aggregate(p) for s in SWEEPS if (p := _agg_path(aggregates, s, task)).is_file()
        ]
        if groups:
            out[task] = merge_aggregates(*groups)
    if not out:
        raise SystemExit(f"No *_by_config.csv files under {aggregates}")
    return out


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["config_id"]: r for r in records}


def _cell(rec: dict[str, Any] | None, metric: str, *, percent: bool = True, bold=False) -> str:
    if rec is None or f"{metric}_mean" not in rec:
        return "—"
    mean, std, n = rec[f"{metric}_mean"], rec[f"{metric}_std"], rec.get(f"{metric}_n", 0)
    if math.isnan(mean):
        return "—"
    scale = 100.0 if percent else 1.0
    text = f"{mean * scale:.1f}" if math.isnan(std) else f"{mean * scale:.1f} ± {std * scale:.1f}"
    if n and n < rec.get("seeds_planned", n):
        text += f" (n={n})"
    return f"**{text}**" if bold else text


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def render_markdown(data: dict[str, list[dict[str, Any]]]) -> str:
    seg = _by_id(data.get("segmentation", []))
    det = _by_id(data.get("detection", []))
    lines: list[str] = []

    everything = list(seg.values()) + list(det.values())
    n_seeds = max((r.get("seeds_complete", 0) for r in everything), default=0)
    lines.append(
        "VOC2012 transfer from COCO pretraining, ResNet-18, "
        f"mean ± s.d. over {n_seeds} seeds, in percent."
    )
    lines.append("")
    lines.append("| Initialisation | mIoU | pixel acc. | AP | AP50 | AP75 |")
    lines.append("|---|---|---|---|---|---|")
    for cid, label in BASELINE_LABELS.items():
        if cid not in seg and cid not in det:
            continue
        bold = cid.startswith("sparc")
        cells = [
            _cell(seg.get(cid), "mIoU", bold=bold),
            _cell(seg.get(cid), "pixel_acc", bold=bold),
            _cell(det.get(cid), "AP", bold=bold),
            _cell(det.get(cid), "AP50", bold=bold),
            _cell(det.get(cid), "AP75", bold=bold),
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    seg_recs = data.get("segmentation", [])
    lam = sorted({_axis_value(r) for r in seg_recs if _axis_value(r) is not None})
    if lam:
        lines += [
            "",
            "**λ ablation.** Both objectives are `(1 − λ)·L_global + λ·L_local`; "
            "at λ = 0 each reduces to MoCo v2.",
            "",
            "| λ | SPARC mIoU | DenseCL mIoU | SPARC AP | DenseCL AP |",
            "|---|---|---|---|---|",
        ]
        for v in lam:
            s_id, d_id = f"sparc_lambda_{_fmt(v)}", f"densecl_lambda_{_fmt(v)}"
            cells = [
                _cell(seg.get(s_id), "mIoU"),
                _cell(seg.get(d_id), "mIoU"),
                _cell(det.get(s_id), "AP"),
                _cell(det.get(d_id), "AP"),
            ]
            lines.append(f"| {v:g} | " + " | ".join(cells) + " |")

    sd = _seed_sd(seg, "mIoU")
    if sd is not None:
        lines += [
            "",
            f"Seed-to-seed s.d. on mIoU is ≈{100 * sd:.1f} points, so differences below "
            f"≈{200 * sd:.1f} points are not resolvable at this sample size.",
        ]
    return "\n".join(lines) + "\n"


def _axis_value(rec: dict[str, Any]):
    v = rec.get("axis_value", "")
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _fmt(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def _seed_sd(recs: dict[str, dict[str, Any]], metric: str):
    sds = [
        r[f"{metric}_std"]
        for r in recs.values()
        if not math.isnan(r.get(f"{metric}_std", math.nan))
    ]
    return (sum(sds) / len(sds)) if sds else None


def render_latex(data: dict[str, list[dict[str, Any]]]) -> str:
    seg, det = _by_id(data.get("segmentation", [])), _by_id(data.get("detection", []))
    rows = []
    for cid, label in BASELINE_LABELS.items():
        if cid not in seg and cid not in det:
            continue
        label = label.replace("**", "").replace("λ", r"$\lambda$")
        cells = [
            _cell(seg.get(cid), "mIoU"),
            _cell(seg.get(cid), "pixel_acc"),
            _cell(det.get(cid), "AP"),
            _cell(det.get(cid), "AP50"),
            _cell(det.get(cid), "AP75"),
        ]
        rows.append(
            f"  {label} & "
            + " & ".join(c.replace("±", r"$\pm$").replace("**", "") for c in cells)
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"  Initialisation & mIoU & pixel acc. & AP & AP$_{50}$ & AP$_{75}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def lambda_zero_identity(data: dict[str, list[dict[str, Any]]], *, k: float = 2.0) -> list[str]:
    """sparc_lambda_0, densecl_lambda_0 and moco are three routes to the same
    objective and must agree within seed noise.

    Two configurations agree when |Δmean| <= k * sqrt(s1²/n1 + s2²/n2). Returns
    a human-readable line per comparison; raises on any failure.
    """
    lines, failures = [], []
    for task, metric in (("segmentation", "mIoU"), ("detection", "AP")):
        recs = _by_id(data.get(task, []))
        trio = [c for c in ("sparc_lambda_0", "densecl_lambda_0", "moco") if c in recs]
        for i, a in enumerate(trio):
            for b in trio[i + 1 :]:
                ra, rb = recs[a], recs[b]
                ma, mb = ra[f"{metric}_mean"], rb[f"{metric}_mean"]
                sa, sb = ra[f"{metric}_std"], rb[f"{metric}_std"]
                na, nb = ra[f"{metric}_n"], rb[f"{metric}_n"]
                if any(math.isnan(x) for x in (ma, mb, sa, sb)) or not na or not nb:
                    lines.append(f"  {task}/{metric}: {a} vs {b}: insufficient seeds, skipped")
                    continue
                se = math.sqrt(sa**2 / na + sb**2 / nb)
                delta = abs(ma - mb)
                ok = delta <= k * se
                verdict = "OK" if ok else "FAIL"
                lines.append(
                    f"  {task}/{metric}: {a} vs {b}: |Δ|={100 * delta:.2f} pts, "
                    f"{k:g}·SE={100 * k * se:.2f} pts -> {verdict}"
                )
                if not ok:
                    failures.append(lines[-1])
    if failures:
        raise SystemExit("λ=0 identity check FAILED:\n" + "\n".join(failures))
    return lines


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _curve(data, task, metric, prefix):
    recs = [
        r
        for r in data.get(task, [])
        if r["config_id"].startswith(prefix + "_") and _axis_value(r) is not None
    ]
    recs.sort(key=_axis_value)
    xs = [_axis_value(r) for r in recs]
    ys = [100 * r[f"{metric}_mean"] for r in recs]
    es = [100 * (0.0 if math.isnan(r[f"{metric}_std"]) else r[f"{metric}_std"]) for r in recs]
    return xs, ys, es


def render_pgfplots(data, task, metric) -> str:
    """Emit addplot coordinate blocks in percent, one per objective, plus the
    MoCo baseline as a horizontal reference. Styling is left to the document."""
    out = []
    for prefix, name in (("sparc_lambda", "SPARC"), ("densecl_lambda", "DenseCL")):
        xs, ys, es = _curve(data, task, metric, prefix)
        if not xs:
            continue
        coords = "\n".join(
            f"    ({x:g}, {y:.4f}) +- (0, {e:.4f})" for x, y, e in zip(xs, ys, es, strict=True)
        )
        out.append(
            f"% {name}, {metric} (%)\n"
            "\\addplot+[error bars/.cd, y dir=both, y explicit]\n"
            f"  coordinates {{\n{coords}\n  }};"
        )
    moco = _by_id(data.get(task, [])).get("moco")
    if moco is not None and not math.isnan(moco[f"{metric}_mean"]):
        level = 100 * moco[f"{metric}_mean"]
        out.append(f"% MoCo v2 baseline\n\\addplot[dashed, domain=0:1] {{{level:.4f}}};")
    return "\n".join(out) + "\n"


def render_png(data, task, metric, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=160)
    for prefix, name, color in (
        ("sparc_lambda", "SPARC (λ = weight on region term)", "#1f77b4"),
        ("densecl_lambda", "DenseCL (λ = weight on dense term)", "#ff7f0e"),
    ):
        xs, ys, es = _curve(data, task, metric, prefix)
        if xs:
            ax.errorbar(
                xs, ys, yerr=es, marker="o", ms=4, capsize=3, lw=1.5, color=color, label=name
            )
    moco = _by_id(data.get(task, [])).get("moco")
    if moco is not None and not math.isnan(moco[f"{metric}_mean"]):
        ax.axhline(100 * moco[f"{metric}_mean"], ls="--", color="gray", lw=1, label="MoCo v2")
    ax.set_xlabel("λ")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xticks([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"VOC2012 {task}: both curves meet MoCo v2 at λ = 0", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# README block
# ---------------------------------------------------------------------------


def splice_readme(readme: Path, block: str) -> str:
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{readme} has no {README_START} / {README_END} markers")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    return pattern.sub(f"{README_START}\n{block}{README_END}", text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_aggregate(a):
    manifest = read_manifest(a.manifest) if a.manifest else None
    run_ids = [m["run_id"] for m in manifest] if manifest else None
    rows = load_runs(a.results_root, a.task, run_ids)
    records = aggregate(rows, a.task, manifest=manifest)
    path = write_aggregate(records, a.output)
    done = sum(r["seeds_complete"] for r in records)
    planned = sum(r["seeds_planned"] for r in records)
    print(f"{path}: {len(records)} configurations, {done}/{planned} runs")
    return 0


def _cmd_table(a):
    data = load_all(Path(a.aggregates))
    sys.stdout.write(render_markdown(data) if a.format == "markdown" else render_latex(data))
    return 0


def _cmd_readme(a):
    data = load_all(Path(a.aggregates))
    block = render_markdown(data)
    readme = Path(a.readme)
    new_text = splice_readme(readme, block)
    if a.check:
        if new_text != readme.read_text():
            raise SystemExit(
                f"{readme} results block is stale. Regenerate with: sparc-report readme ..."
            )
        print("README results block is current.")
        return 0
    readme.write_text(new_text)
    print(f"{readme}: results block updated")
    return 0


def _cmd_figure(a):
    data = load_all(Path(a.aggregates))
    out = Path(a.output)
    if a.format == "png":
        render_png(data, a.task, a.metric, out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_pgfplots(data, a.task, a.metric))
    print(out)
    return 0


def _cmd_check(a):
    data = load_all(Path(a.aggregates))  # merge_aggregates raises on a double-counted config_id
    print("double-count check: OK (every config_id names one run)")
    print("λ=0 identity check:")
    for line in lambda_zero_identity(data, k=a.k):
        print(line)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="sparc-report")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("aggregate", help="Per-run CSVs -> one row per configuration.")
    s.add_argument("--manifest")
    s.add_argument("--results-root", required=True)
    s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=_cmd_aggregate)

    s = sub.add_parser("table", help="Render the results table.")
    s.add_argument("--aggregates", required=True)
    s.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    s.set_defaults(func=_cmd_table)

    s = sub.add_parser("readme", help="Splice the results table into README.md.")
    s.add_argument("--aggregates", required=True)
    s.add_argument("--readme", default="README.md")
    s.add_argument("--check", action="store_true", help="Fail if the README is stale.")
    s.set_defaults(func=_cmd_readme)

    s = sub.add_parser("figure", help="The paired lambda curves.")
    s.add_argument("--aggregates", required=True)
    s.add_argument("--task", choices=TASKS, default="segmentation")
    s.add_argument("--metric", default="mIoU")
    s.add_argument("--format", choices=("png", "pgfplots"), default="png")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=_cmd_figure)

    s = sub.add_parser("check", help="Double-count and lambda=0 identity checks.")
    s.add_argument("--aggregates", required=True)
    s.add_argument("--k", type=float, default=2.0)
    s.set_defaults(func=_cmd_check)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
