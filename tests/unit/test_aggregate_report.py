"""Tests for aggregation, the merge safeguard, the λ=0 identity check, and the
README splice."""

from __future__ import annotations

import math

import pytest

from sparc.cli.report import (
    README_END,
    README_START,
    lambda_zero_identity,
    render_markdown,
    render_pgfplots,
    splice_readme,
)
from sparc.experiments.aggregate import (
    aggregate,
    load_runs,
    merge_aggregates,
    read_aggregate,
    write_aggregate,
)
from sparc.experiments.results import build_result_row, write_result_csv


def _write_runs(root, task, config_id, values, metric, axis=None):
    for seed, v in enumerate(values, start=1):
        rid = config_id if seed == 1 else f"{config_id}_seed{seed}"
        row = build_result_row(
            run_id=rid,
            config_id=config_id,
            task=task,
            init="ssl",
            backbone="resnet18",
            seed=seed,
            epochs=1,
            batch_size=1,
            lr=1e-4,
            weight_decay=1e-4,
            ssl_ckpt=None,
            metrics={metric: v},
        )
        write_result_csv(row, root / task / f"{rid}.csv")


class TestAggregate:
    def test_mean_sample_std_and_values(self, tmp_path):
        _write_runs(tmp_path, "segmentation", "c", [0.30, 0.32, 0.34], "mIoU")
        rec = aggregate(load_runs(tmp_path, "segmentation"), "segmentation")[0]
        assert rec["mIoU_mean"] == pytest.approx(0.32)
        assert rec["mIoU_std"] == pytest.approx(0.02)  # n-1, not n
        assert rec["mIoU_n"] == 3 and rec["seeds"] == "1;2;3"

    def test_single_seed_has_undefined_std(self, tmp_path):
        _write_runs(tmp_path, "segmentation", "c", [0.30], "mIoU")
        rec = aggregate(load_runs(tmp_path, "segmentation"), "segmentation")[0]
        assert math.isnan(rec["mIoU_std"]) and rec["mIoU_n"] == 1

    def test_manifest_supplies_planned_counts_and_axis(self, tmp_path):
        _write_runs(tmp_path, "segmentation", "sparc_lambda_0p5", [0.3, 0.4], "mIoU")
        manifest = [
            {
                "config_id": "sparc_lambda_0p5",
                "seed": str(s),
                "run_id": "x",
                "axis_key": "method.lambda_region",
                "axis_value": "0.5",
            }
            for s in range(1, 6)
        ]
        rec = aggregate(load_runs(tmp_path, "segmentation"), "segmentation", manifest=manifest)[0]
        assert rec["seeds_planned"] == 5 and rec["seeds_complete"] == 2
        assert rec["axis_value"] == "0.5"

    def test_write_then_read_round_trips(self, tmp_path):
        _write_runs(tmp_path, "detection", "c", [0.2, 0.25], "AP")
        recs = aggregate(load_runs(tmp_path, "detection"), "detection")
        back = read_aggregate(write_aggregate(recs, tmp_path / "agg.csv"))
        assert back[0]["AP_mean"] == pytest.approx(0.225) and back[0]["AP_n"] == 2


class TestMergeSafeguard:
    def test_identical_duplicate_is_merged_once(self):
        a = [{"config_id": "densecl_lambda_0p5", "mIoU_mean": 0.37, "mIoU_std": 0.01, "mIoU_n": 5}]
        assert len(merge_aggregates(a, list(a))) == 1

    def test_conflicting_duplicate_raises(self):
        """Two different runs must not share one name in a results table."""
        a = [
            {"config_id": "densecl_lambda_0p5", "mIoU_mean": 0.3749, "mIoU_std": 0.01, "mIoU_n": 1}
        ]
        b = [
            {"config_id": "densecl_lambda_0p5", "mIoU_mean": 0.3787, "mIoU_std": 0.01, "mIoU_n": 1}
        ]
        with pytest.raises(ValueError, match="Two different runs share one name"):
            merge_aggregates(a, b)


def _rec(cid, mean, std=0.004, n=5, metric="mIoU", axis=""):
    return {
        "config_id": cid,
        f"{metric}_mean": mean,
        f"{metric}_std": std,
        f"{metric}_n": n,
        "seeds_planned": n,
        "seeds_complete": n,
        "axis_value": axis,
    }


class TestLambdaZeroIdentity:
    def test_agreeing_endpoints_pass(self):
        data = {
            "segmentation": [
                _rec("sparc_lambda_0", 0.340),
                _rec("densecl_lambda_0", 0.342),
                _rec("moco", 0.339),
            ]
        }
        lines = lambda_zero_identity(data)
        assert len(lines) == 3 and all("OK" in line for line in lines)

    def test_disagreeing_endpoint_fails_loudly(self):
        data = {
            "segmentation": [
                _rec("sparc_lambda_0", 0.340),
                _rec("densecl_lambda_0", 0.380),
                _rec("moco", 0.339),
            ]
        }
        with pytest.raises(SystemExit, match="FAIL"):
            lambda_zero_identity(data)


class TestRendering:
    def test_markdown_table_has_baselines_and_lambda_rows(self):
        data = {
            "segmentation": [
                _rec("moco", 0.34),
                _rec("sparc_lambda_0p5", 0.39),
                _rec("sparc_lambda_0", 0.34, axis="0"),
                _rec("densecl_lambda_0", 0.34, axis="0"),
            ],
            "detection": [
                _rec("moco", 0.23, metric="AP"),
                _rec("sparc_lambda_0p5", 0.255, metric="AP"),
            ],
        }
        text = render_markdown(data)
        assert "| MoCo v2 |" in text and "**SPARC**" in text
        assert "| 0 |" in text and "39.0" in text

    def test_pgfplots_coordinates_are_in_percent(self):
        data = {
            "segmentation": [
                _rec("sparc_lambda_0", 0.34, axis="0"),
                _rec("sparc_lambda_1", 0.36, axis="1"),
            ]
        }
        text = render_pgfplots(data, "segmentation", "mIoU")
        assert "(0, 34.0000)" in text and "(1, 36.0000)" in text

    def test_readme_splice_replaces_only_the_marked_block(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(f"# T\n\nintro\n\n{README_START}\nold\n{README_END}\n\nfooter\n")
        new = splice_readme(readme, "NEW\n")
        assert "old" not in new and "NEW" in new and "intro" in new and "footer" in new

    def test_readme_without_markers_is_an_error(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# no markers\n")
        with pytest.raises(SystemExit, match="markers"):
            splice_readme(readme, "x")
