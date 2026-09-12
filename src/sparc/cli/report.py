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
# Figure: the paired lambda curves, as the paper draws them
# ---------------------------------------------------------------------------

METRIC_TASK = {
    "mIoU": "segmentation",
    "pixel_acc": "segmentation",
    "AP": "detection",
    "AP50": "detection",
    "AP75": "detection",
}
METRIC_LABEL = {
    "mIoU": r"VOC2012 mIoU (\%)",
    "pixel_acc": r"VOC2012 pixel acc.\ (\%)",
    "AP": r"VOC2012 AP (\%)",
    "AP50": r"VOC2012 AP50 (\%)",
    "AP75": r"VOC2012 AP75 (\%)",
}
DEFAULT_METRICS = ("mIoU", "pixel_acc", "AP", "AP50")
#: The paper's palette: SPARC blue circles, DenseCL orange squares, MoCo v2 a
#: single green triangle at lambda = 0 (it is the shared endpoint, not a curve).
PALETTE = {
    "series1": "2A78D6",
    "series2": "EB6834",
    "series3": "2CA02C",
    "ink2": "52514E",
    "grid": "E1E0D9",
    "rule": "C3C2B7",
}
SERIES = (
    ("sparc_lambda", "SPARC", "series1, mark=*, mark options={fill=series1}"),
    ("densecl_lambda", "DenseCL", "series2, mark=square*, mark options={fill=series2}"),
)
MOCO_STYLE = "series3, only marks, mark=triangle*, mark size=3.5pt, mark options={fill=series3}"
DEFAULT_CAPTION = (
    r"\textbf{Semantic Segmentation and Object Detection Performance} between SPARC, "
    r"DenseCL, and MoCo-v2 across different $\lambda$ values. It is observed that the "
    r"proposed SPARC's pure region-level significantly outperforms pure pixel-level and "
    r"global-level contrastive learning framework."
)


def _curve(data, metric, prefix):
    task = METRIC_TASK[metric]
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


def _moco_point(data, metric):
    rec = _by_id(data.get(METRIC_TASK[metric], [])).get("moco")
    if rec is None or math.isnan(rec[f"{metric}_mean"]):
        return None
    std = rec[f"{metric}_std"]
    return 100 * rec[f"{metric}_mean"], 100 * (0.0 if math.isnan(std) else std)


def _panel_extent(data, metric):
    lo, hi = math.inf, -math.inf
    for prefix, _name, _style in SERIES:
        _xs, ys, es = _curve(data, metric, prefix)
        for y, e in zip(ys, es, strict=True):
            lo, hi = min(lo, y - e), max(hi, y + e)
    moco = _moco_point(data, metric)
    if moco:
        lo, hi = min(lo, moco[0] - moco[1]), max(hi, moco[0] + moco[1])
    return lo, hi


def _nice_axis(lo, hi):
    """ymin, ymax and a tick step giving roughly 5-8 gridlines, with headroom
    for the legend. Hand-tuning in the .tex afterwards is expected."""
    span = max(hi - lo, 1e-6)
    step = next(s for s in (1, 2, 3, 5, 10, 20) if span / s <= 8)
    ymin = math.floor(lo / step) * step
    ymax = math.ceil((hi + 0.12 * span) / step) * step
    return ymin, ymax, step


def _coords(xs, ys, es, per_line=3):
    items = [f"({x:g},{y:.2f})+-(0,{e:.2f})" for x, y, e in zip(xs, ys, es, strict=True)]
    lines = [" ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)]
    return "\n".join("    " + line for line in lines)


def render_pgfplots(
    data, metrics=DEFAULT_METRICS, cols=2, caption=DEFAULT_CAPTION, label="fig:lambda-all"
) -> str:
    """A complete figure environment: a groupplot with one panel per metric."""
    rows = math.ceil(len(metrics) / cols)
    out = [
        "% Generated by `sparc-report figure --format pgfplots`. Do not edit; regenerate.",
        "% Preamble needs: \\usepackage{xcolor,pgfplots} \\usepgfplotslibrary{groupplots}",
        "% and \\pgfplotsset{compat=1.17} (or newer where available).",
        r"\begin{figure}[t]",
        r"\centering",
    ]
    out += [rf"\providecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in PALETTE.items()]
    out += [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        rf"  group style={{group size={cols} by {rows}, horizontal sep=1.3cm, "
        r"vertical sep=0.75cm, xticklabels at=edge bottom, xlabels at=edge bottom},",
        r"  width=4.5cm, height=4.5cm,",
        r"  xlabel={loss weight $\lambda$}, xlabel style={yshift=2pt},",
        r"  xmin=-0.04, xmax=1.04,",
        r"  xtick={0,0.3,0.5,0.7,1.0}, xticklabels={0,0.3,0.5,0.7,1},",
        r"  axis line style={rule, thin},",
        r"  axis x line*=bottom, axis y line*=left,",
        r"  tick style={rule, thin}, tick label style={font=\scriptsize, color=ink2},",
        r"  label style={font=\footnotesize, color=ink2},",
        r"  ymajorgrids, grid style={grid, thin},",
        r"  clip=false,",
        r"  legend cell align=left,",
        r"  error bars/y dir=both, error bars/y explicit,",
        r"  error bars/error bar style={thin},",
        r"  every axis plot/.append style={line width=1.4pt, mark size=2.2pt},",
        r"]",
    ]
    for i, metric in enumerate(metrics):
        ymin, ymax, step = _nice_axis(*_panel_extent(data, metric))
        legend = (
            ", legend columns=-1, legend to name=lambdalegend, "
            "legend style={draw=none, fill=none, "
            "/tikz/every even column/.append style={column sep=10pt}}"
            if i == 0
            else ""
        )
        out.append(f"% {METRIC_LABEL[metric]}")
        out.append(
            rf"\nextgroupplot[ylabel={{{METRIC_LABEL[metric]}}}, ymin={ymin:g}, ymax={ymax:g}, "
            rf"ytick={{{ymin:g},{ymin + step:g},...,{ymax:g}}}{legend}]"
        )
        for prefix, name, style in SERIES:
            xs, ys, es = _curve(data, metric, prefix)
            if not xs:
                continue
            out += [
                f"% {name}",
                rf"\addplot[{style}]",
                "  coordinates {",
                _coords(xs, ys, es),
                "  };",
            ]
            if i == 0:
                out.append(rf"\addlegendentry{{{name}}}")
        moco = _moco_point(data, metric)
        if moco:
            out += [
                "% MoCo-v2",
                rf"\addplot[{MOCO_STYLE}]",
                f"  coordinates {{(0,{moco[0]:.2f})+-(0,{moco[1]:.2f})}};",
            ]
            if i == 0:
                out.append(r"\addlegendentry{MoCo-v2}")
    out += [
        r"\end{groupplot}",
        r"\end{tikzpicture}\\[2pt]",
        # \ref* rather than \ref: with hyperref loaded, a plain \ref paints the
        # legend in the link colour.
        r"\ref*{lambdalegend}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(out)


def render_png(data, path: Path, metrics=DEFAULT_METRICS, cols=2) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), dpi=160, squeeze=False)
    colour = {k: "#" + v for k, v in PALETTE.items()}
    for ax, metric in zip(axes.flat, metrics, strict=False):
        for (prefix, name, _style), key, marker in zip(
            SERIES, ("series1", "series2"), ("o", "s"), strict=True
        ):
            xs, ys, es = _curve(data, metric, prefix)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker=marker,
                    ms=4,
                    capsize=2,
                    lw=1.4,
                    color=colour[key],
                    label=name,
                )
        moco = _moco_point(data, metric)
        if moco:
            ax.errorbar(
                [0],
                [moco[0]],
                yerr=[moco[1]],
                fmt="^",
                ms=7,
                capsize=2,
                color=colour["series3"],
                label="MoCo-v2",
                zorder=5,
            )
        ax.set_ylabel(
            METRIC_LABEL[metric].replace(r"\%", "%").replace(r"\ ", " "), color=colour["ink2"]
        )
        ax.set_xlabel("loss weight λ", color=colour["ink2"])
        ax.set_xticks([0, 0.3, 0.5, 0.7, 1.0])
        ax.set_xlim(-0.04, 1.04)
        ax.grid(axis="y", color=colour["grid"], lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
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
    metrics = tuple(a.metrics)
    unknown = [m for m in metrics if m not in METRIC_TASK]
    if unknown:
        raise SystemExit(f"Unknown metric(s) {unknown}; choose from {sorted(METRIC_TASK)}")
    if a.format == "png":
        render_png(data, out, metrics=metrics, cols=a.cols)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_pgfplots(data, metrics=metrics, cols=a.cols))
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

    s = sub.add_parser("figure", help="The paired lambda curves, one panel per metric.")
    s.add_argument("--aggregates", required=True)
    s.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    s.add_argument("--cols", type=int, default=2)
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
