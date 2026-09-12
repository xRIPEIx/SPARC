"""The paper's lambda figure is reproduced coordinate-for-coordinate from the
shipped aggregates.

The numbers below are the ones typeset in the paper. If a change to the
aggregation or the figure emitter moves any of them, this is where it shows up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparc.cli.report import DEFAULT_METRICS, load_all, render_pgfplots

AGG = Path("experiments/results/aggregate")

#: metric -> series -> "(x,y)+-(0,e)" strings exactly as they appear in the paper.
PAPER = {
    "mIoU": {
        "SPARC": "(0,34.41)+-(0,0.31) (0.1,36.75)+-(0,0.22) (0.3,38.41)+-(0,0.06) "
        "(0.5,38.80)+-(0,0.39) (0.7,38.88)+-(0,0.45) (0.9,38.78)+-(0,0.35) (1,38.32)+-(0,0.29)",
        "DenseCL": "(0,34.19)+-(0,0.32) (0.1,36.49)+-(0,0.43) (0.3,37.50)+-(0,0.50) "
        "(0.5,37.38)+-(0,0.19) (0.7,36.61)+-(0,0.39) (0.9,33.66)+-(0,0.68) (1,27.56)+-(0,1.47)",
        "MoCo-v2": "(0,34.24)+-(0,0.52)",
    },
    "pixel_acc": {
        "SPARC": "(0,82.96)+-(0,0.18) (0.1,83.66)+-(0,0.24) (0.3,84.26)+-(0,0.14) "
        "(0.5,84.34)+-(0,0.16) (0.7,84.23)+-(0,0.25) (0.9,84.21)+-(0,0.13) (1,84.05)+-(0,0.12)",
        "DenseCL": "(0,82.80)+-(0,0.25) (0.1,83.55)+-(0,0.34) (0.3,83.92)+-(0,0.08) "
        "(0.5,83.86)+-(0,0.10) (0.7,83.63)+-(0,0.14) (0.9,82.76)+-(0,0.31) (1,81.08)+-(0,0.63)",
        "MoCo-v2": "(0,82.85)+-(0,0.11)",
    },
    "AP": {
        "SPARC": "(0,23.66)+-(0,0.26) (0.1,24.47)+-(0,0.29) (0.3,25.10)+-(0,0.21) "
        "(0.5,25.41)+-(0,0.16) (0.7,25.53)+-(0,0.11) (0.9,25.30)+-(0,0.14) (1,25.55)+-(0,0.15)",
        "DenseCL": "(0,23.66)+-(0,0.32) (0.1,24.04)+-(0,0.16) (0.3,24.21)+-(0,0.24) "
        "(0.5,24.03)+-(0,0.18) (0.7,23.52)+-(0,0.22) (0.9,21.74)+-(0,0.19) (1,18.36)+-(0,0.89)",
        "MoCo-v2": "(0,23.68)+-(0,0.14)",
    },
    "AP50": {
        "SPARC": "(0,47.01)+-(0,0.28) (0.1,48.01)+-(0,0.35) (0.3,48.81)+-(0,0.18) "
        "(0.5,49.28)+-(0,0.24) (0.7,49.34)+-(0,0.28) (0.9,48.97)+-(0,0.25) (1,49.39)+-(0,0.25)",
        "DenseCL": "(0,47.11)+-(0,0.48) (0.1,47.50)+-(0,0.41) (0.3,47.68)+-(0,0.24) "
        "(0.5,47.15)+-(0,0.29) (0.7,46.18)+-(0,0.41) (0.9,43.52)+-(0,0.30) (1,37.84)+-(0,1.38)",
        "MoCo-v2": "(0,47.06)+-(0,0.24)",
    },
}


@pytest.fixture(scope="module")
def tex():
    return render_pgfplots(load_all(AGG))


def _panel(tex: str, metric: str) -> str:
    start = tex.index(f"% VOC2012 {metric}" if metric != "pixel_acc" else "% VOC2012 pixel acc")
    rest = tex[start + 1 :]
    end = rest.find("\n% VOC2012")
    return rest if end == -1 else rest[:end]


@pytest.mark.parametrize("metric", DEFAULT_METRICS)
@pytest.mark.parametrize("series", ["SPARC", "DenseCL", "MoCo-v2"])
def test_every_paper_coordinate_is_reproduced(tex, metric, series):
    panel = " ".join(_panel(tex, metric).split())
    for coord in PAPER[metric][series].split(" "):
        assert coord in panel, f"{metric}/{series}: {coord} not produced by the aggregates"


def test_moco_is_a_single_triangle_at_lambda_zero(tex):
    assert tex.count("mark=triangle*") == len(DEFAULT_METRICS)
    assert r"\addlegendentry{MoCo-v2}" in tex


def test_figure_is_a_complete_environment(tex):
    for token in (
        r"\begin{figure}",
        r"\begin{groupplot}",
        "group size=2 by 2",
        r"legend to name=lambdalegend",
        r"\ref*{lambdalegend}",
        r"\end{figure}",
    ):
        assert token in tex
