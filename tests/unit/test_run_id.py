"""Tests for run identity.

Seed 1 keeps the bare config_id; seeds 2+ get a suffix. This must agree exactly
with the shell implementation it replaces, because manifests, checkpoint
directories and result filenames all depend on it.
"""

from __future__ import annotations

import pytest

from sparc.experiments.run_id import parse_run_id, resolve_run_id


class TestResolve:
    def test_seed_one_keeps_the_bare_config_id(self):
        """The asymmetry that let an existing single-seed study be reused as
        replicate #1 instead of being renamed and re-run."""
        assert resolve_run_id("sparc_lambda_0p5", 1) == "sparc_lambda_0p5"

    @pytest.mark.parametrize("seed", [2, 3, 4, 5])
    def test_later_seeds_are_suffixed(self, seed):
        assert resolve_run_id("sparc_lambda_0p5", seed) == f"sparc_lambda_0p5_seed{seed}"

    def test_rejects_seed_zero(self):
        with pytest.raises(ValueError, match="seed"):
            resolve_run_id("x", 0)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "config_id", ["sparc_lambda_0p5", "densecl_lambda_0", "moco", "sparc_lambda_1"]
    )
    @pytest.mark.parametrize("seed", [1, 2, 5])
    def test_parse_inverts_resolve(self, config_id, seed):
        assert parse_run_id(resolve_run_id(config_id, seed)) == (config_id, seed)

    def test_config_id_containing_seed_like_text_is_not_misparsed(self):
        assert parse_run_id("weird_seedling") == ("weird_seedling", 1)
