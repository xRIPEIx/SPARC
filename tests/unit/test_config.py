"""Tests for config composition, overrides and path resolution.

The previous workflow kept experiment configuration in 46 shell scripts, where a
mistyped flag silently did nothing and the swept values were duplicated between
each sweep's pretrain and evaluate script. These tests pin the behaviour that
replaces it.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from sparc.config.loader import _parse_value, apply_overrides, load_config
from sparc.config.schema import PretrainConfig

REFERENCE = "configs/pretrain/sparc_coco_r18.yaml"


class TestValueParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("null", None),
            ("0.7", 0.7),
            ("16", 16),
            ("true", True),
            ("false", False),
            ("sparc", "sparc"),
            ("/tmp/a/b", "/tmp/a/b"),
            ("${paths.x}/y", "${paths.x}/y"),
        ],
    )
    def test_scalars(self, raw, expected):
        assert _parse_value(raw) == expected

    def test_null_is_none_not_the_string(self):
        """`data.masks=null` must clear the value. As a string it is a perfectly
        valid `str | None`, so the mistake would surface much later as a
        confusing missing-directory error."""
        assert _parse_value("null") is None


class TestOverrides:
    def test_unknown_key_is_rejected(self):
        config = OmegaConf.structured(PretrainConfig)
        with pytest.raises(KeyError, match="Unknown config key"):
            apply_overrides(config, ["method.lambda_regionn=0.7"])

    def test_malformed_override_is_rejected(self):
        config = OmegaConf.structured(PretrainConfig)
        with pytest.raises(ValueError, match="key.path=value"):
            apply_overrides(config, ["method.lambda_region"])

    def test_values_are_coerced_to_the_schema_type(self):
        config = load_config(REFERENCE, ["method.lambda_region=0.7", "train.epochs=3"])
        assert isinstance(config.method.lambda_region, float)
        assert isinstance(config.train.epochs, int)

    def test_override_may_reference_a_path(self):
        """Overrides are applied after paths are merged, so this resolves."""
        config = load_config(REFERENCE, ["data.masks=${paths.superpixel_root}/slic_n250"])
        assert config.data.masks.endswith("/superpixel_masks/slic_n250")


class TestComposition:
    def test_reference_config_resolves(self, monkeypatch):
        monkeypatch.setenv("SPARC_DATA_ROOT", "/data/here")
        config = load_config(REFERENCE)
        assert config.method.name == "sparc"
        assert config.backbone.name == "resnet18"
        # From _base.yaml, inherited via `defaults:`.
        assert config.train.optimizer == "adam"
        # From ../backbone/resnet18.yaml.
        assert config.backbone.imagenet_init is False

    def test_paths_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("SPARC_DATA_ROOT", "/somewhere/else")
        config = load_config(REFERENCE)
        assert config.data.images == "/somewhere/else/coco/train2017"

    def test_explicit_env_var_beats_data_root(self, monkeypatch):
        monkeypatch.setenv("SPARC_DATA_ROOT", "/generic")
        monkeypatch.setenv("SPARC_COCO_IMAGES", "/specific/images")
        assert load_config(REFERENCE).data.images == "/specific/images"

    def test_paths_node_is_not_baked_into_the_config(self):
        """Paths describe the machine, not the experiment, so they must not end
        up in the checkpoint as though they were part of the method."""
        assert "paths" not in load_config(REFERENCE)

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/pretrain/does_not_exist.yaml")


class TestBaselineConfigs:
    def test_moco_needs_no_masks(self):
        assert load_config("configs/pretrain/moco_coco_r18.yaml").data.masks is None

    def test_densecl_baseline_is_the_lambda_sweep_midpoint(self):
        """dense_lambda defaults to 0.5, so this config IS densecl_lambda_0p5.
        They are one run and must never be reported as two results."""
        config = load_config("configs/pretrain/densecl_coco_r18.yaml")
        assert config.method.dense_lambda == 0.5
        assert config.experiment.config_id == "densecl_lambda_0p5"
