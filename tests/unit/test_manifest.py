"""Tests for sweep expansion.

The manifest is the single source of truth that replaces bash arrays duplicated
between each sweep's pretrain and evaluate script. A one-sided edit to those
arrays would evaluate the wrong checkpoint under the right run_id and report no
error, so the properties here are the ones that made that class of bug possible.
"""

from __future__ import annotations

import pytest

from sparc.experiments.manifest import (
    expand,
    format_value,
    read_manifest,
    select_row,
    write_manifest,
)

SWEEPS = {
    "sparc_lambda": "configs/sweeps/sparc_lambda.yaml",
    "densecl_lambda": "configs/sweeps/densecl_lambda.yaml",
    "baselines": "configs/sweeps/baselines.yaml",
}
#: The seven weights swept by both lambda ablations.
LAMBDA_VALUES = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]


class TestValueFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0"), (0.1, "0p1"), (0.3, "0p3"), (0.5, "0p5"), (0.9, "0p9"), (1, "1"), (1.0, "1")],
    )
    def test_matches_the_established_naming(self, value, expected):
        """Reproduces the naming already embedded in existing checkpoint
        directories and result filenames, so 1.0 renders "1", not "1p0"."""
        assert format_value(value) == expected


class TestLambdaSweeps:
    @pytest.mark.parametrize("sweep", ["sparc_lambda", "densecl_lambda"])
    def test_seven_values_by_five_seeds(self, sweep):
        rows = expand(SWEEPS[sweep])
        assert len(rows) == 35
        assert len({r["config_id"] for r in rows}) == 7
        assert sorted({r["seed"] for r in rows}) == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("sweep", ["sparc_lambda", "densecl_lambda"])
    def test_swept_values_match_the_published_study(self, sweep):
        """Cross-check against the values the archived results were produced
        with. The identifiers were renamed; the values were not."""
        rows = expand(SWEEPS[sweep])
        assert sorted({float(r["axis_value"]) for r in rows}) == sorted(
            float(v) for v in LAMBDA_VALUES
        )

    def test_sparc_varies_lambda_region(self):
        assert {r["axis_key"] for r in expand(SWEEPS["sparc_lambda"])} == {"method.lambda_region"}

    def test_densecl_varies_its_own_dense_lambda(self):
        """DenseCL's mix is its own hyper-parameter, not one SPARC introduces."""
        assert {r["axis_key"] for r in expand(SWEEPS["densecl_lambda"])} == {"method.dense_lambda"}

    def test_config_ids_match_the_archived_naming(self):
        ids = {r["config_id"] for r in expand(SWEEPS["densecl_lambda"])}
        assert ids == {f"densecl_lambda_{v}" for v in ["0", "0p1", "0p3", "0p5", "0p7", "0p9", "1"]}

    @pytest.mark.parametrize("sweep", ["sparc_lambda", "densecl_lambda"])
    def test_lambda_zero_endpoint_exists_in_both(self, sweep):
        """Both sweeps must reach lambda = 0, where each reduces to the
        MoCo-global objective. That shared endpoint is the study's cheapest
        validity check."""
        assert any(float(r["axis_value"]) == 0.0 for r in expand(SWEEPS[sweep]))


class TestRunIds:
    def test_seed_one_keeps_the_bare_config_id(self):
        rows = expand(SWEEPS["sparc_lambda"])
        seed1 = [r for r in rows if r["seed"] == 1]
        assert all(r["run_id"] == r["config_id"] for r in seed1)

    def test_later_seeds_are_suffixed(self):
        rows = expand(SWEEPS["sparc_lambda"])
        for row in rows:
            if row["seed"] != 1:
                assert row["run_id"] == f"{row['config_id']}_seed{row['seed']}"

    def test_run_ids_are_unique(self):
        rows = expand(SWEEPS["sparc_lambda"])
        assert len({r["run_id"] for r in rows}) == len(rows)


class TestOverrideConsistency:
    """The property that makes the duplicated-array bug impossible."""

    def test_pretrain_and_downstream_agree_on_identity(self):
        for row in expand(SWEEPS["sparc_lambda"]):
            for field in ("experiment.run_id", "experiment.config_id", "train.seed"):
                pre = [o for o in row["pretrain_overrides"].split() if o.startswith(field + "=")]
                post = [o for o in row["downstream_overrides"].split() if o.startswith(field + "=")]
                assert pre == post, f"{field} disagrees between stages for {row['run_id']}"

    def test_swept_value_appears_in_the_pretrain_overrides(self):
        for row in expand(SWEEPS["sparc_lambda"]):
            assert f"{row['axis_key']}={row['axis_value']}" in row["pretrain_overrides"]


class TestBaselines:
    def test_four_arms_five_seeds(self):
        rows = expand(SWEEPS["baselines"])
        assert len(rows) == 20
        assert {r["config_id"] for r in rows} == {
            "moco",
            "densecl_lambda_0p5",
            "random",
            "imagenet",
        }

    def test_densecl_baseline_reuses_the_lambda_sweep_identity(self):
        """dense_lambda defaults to 0.5, so the DenseCL baseline and
        densecl_lambda_0p5 are literally the same run. Sharing the config_id is
        what stops them being reported as two results."""
        baseline_ids = {r["config_id"] for r in expand(SWEEPS["baselines"])}
        sweep_ids = {r["config_id"] for r in expand(SWEEPS["densecl_lambda"])}
        assert baseline_ids & sweep_ids == {"densecl_lambda_0p5"}

    @pytest.mark.parametrize("arm", ["random", "imagenet"])
    def test_checkpoint_free_arms_have_no_pretrain_config(self, arm):
        rows = [r for r in expand(SWEEPS["baselines"]) if r["config_id"] == arm]
        assert rows
        assert all(r["pretrain_config"] == "" for r in rows)
        assert all("model.init=" in r["downstream_overrides"] for r in rows)


class TestPortability:
    def test_config_paths_are_repo_relative(self):
        """Manifests are committed and read on other machines, so an absolute
        path would bake one checkout's location into the repository."""
        for sweep in SWEEPS.values():
            for row in expand(sweep):
                assert not row["pretrain_config"].startswith("/"), row["pretrain_config"]


class TestRoundTrip:
    def test_write_then_read_preserves_rows(self, tmp_path):
        rows = expand(SWEEPS["sparc_lambda"])
        path = write_manifest(rows, tmp_path / "m.csv")
        back = read_manifest(path)
        assert len(back) == len(rows)
        assert back[0]["run_id"] == rows[0]["run_id"]

    def test_select_row_finds_the_scheduler_task(self, tmp_path):
        path = write_manifest(expand(SWEEPS["sparc_lambda"]), tmp_path / "m.csv")
        row = select_row(read_manifest(path), array_index=4, seed=3)
        assert row["config_id"] == "sparc_lambda_0p7"
        assert row["seed"] == "3"

    def test_missing_row_raises_rather_than_defaulting(self, tmp_path):
        """A scheduler task that cannot find its row must fail, not silently run
        some other configuration."""
        path = write_manifest(expand(SWEEPS["sparc_lambda"]), tmp_path / "m.csv")
        with pytest.raises(KeyError, match="No manifest row"):
            select_row(read_manifest(path), array_index=99, seed=1)


class TestValidation:
    def test_rejects_a_sweep_with_neither_axis_nor_entries(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: bad\nseeds: [1]\n")
        with pytest.raises(ValueError, match="exactly one"):
            expand(path)

    def test_rejects_a_sweep_with_both(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\nseeds: [1]\naxis: {key: a.b, values: [1]}\nentries: [{config_id: x}]\n"
        )
        with pytest.raises(ValueError, match="exactly one"):
            expand(path)


class TestCommittedManifests:
    """The committed manifests must match what the definitions expand to."""

    @pytest.mark.parametrize("sweep", sorted(SWEEPS))
    def test_committed_manifest_is_current(self, sweep):
        committed = read_manifest(f"experiments/{sweep}/manifest.csv")
        expected = expand(SWEEPS[sweep])
        assert len(committed) == len(expected)
        for got, want in zip(committed, expected, strict=True):
            assert got["run_id"] == want["run_id"]
            assert got["pretrain_overrides"] == want["pretrain_overrides"]
