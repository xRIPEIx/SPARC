"""Tests for checkpoint transfer.

This is the hinge between pretraining and downstream evaluation, and the one
place where a mistake produces a randomly-initialised backbone together with a
completely plausible-looking metric. The previous implementation called
load_state_dict(strict=False) and merely *printed* the missing-key count, so a
mistyped architecture trained happily and reported a publishable number with no
error anywhere.
"""

from __future__ import annotations

import pytest
import torch
import torchvision
from torch import nn

from sparc.engine.checkpoint import (
    LoadReport,
    find_resume_checkpoint,
    load_pretrained_encoder,
    strip_encoder_state,
)


@pytest.fixture(scope="module")
def resnet18_state():
    """One ResNet-18 state dict, shared. Tests copy it rather than rebuilding."""
    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return {k: v.clone() for k, v in net.state_dict().items()}


def _resnet18():
    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net


def _write_encoder_checkpoint(path, state, key="encoder"):
    torch.save({key: state, "epoch": 1, "method": "sparc"}, path)
    return path


class TestStripEncoderState:
    def test_strips_nested_wrapper_prefixes(self):
        raw = {
            "module.encoder.conv1.weight": torch.zeros(1),
            "backbone.layer1.0.conv1.weight": torch.zeros(1),
            "layer4.1.bn2.bias": torch.zeros(1),
        }
        out = strip_encoder_state(raw)
        assert set(out) == {"conv1.weight", "layer1.0.conv1.weight", "layer4.1.bn2.bias"}

    def test_drops_non_backbone_keys(self):
        """Projection heads, the region projector, the classifier and loss-module
        state are all method-specific and must not travel downstream."""
        raw = {
            "conv1.weight": torch.zeros(1),
            "fc.weight": torch.zeros(1),
            "projection_head.layers.0.weight": torch.zeros(1),
            "region_projector.net.0.weight": torch.zeros(1),
            "criterion_global.bank": torch.zeros(1),
        }
        assert set(strip_encoder_state(raw)) == {"conv1.weight"}


class TestLoadPretrainedEncoder:
    def test_loads_matching_weights_and_reports_full_coverage(self, tmp_path, resnet18_state):
        state = dict(resnet18_state)
        state["conv1.weight"] = torch.full_like(state["conv1.weight"], 0.5)
        path = _write_encoder_checkpoint(tmp_path / "ckpt.pth", state)

        target = _resnet18()
        report = load_pretrained_encoder(target, path)

        assert report.fraction == 1.0
        assert report.missing == [] and report.unexpected == []
        assert torch.allclose(target.conv1.weight, torch.full_like(target.conv1.weight, 0.5))

    def test_architecture_mismatch_raises_instead_of_half_loading(self, tmp_path, resnet18_state):
        """The defect this gate exists for: a checkpoint from a different
        architecture used to leave most of the network randomly initialised and
        still report a plausible metric.

        Simulated by keeping only a fifth of the parameters rather than by
        building a second large network, which keeps this test cheap.
        """
        partial = dict(list(resnet18_state.items())[: len(resnet18_state) // 5])
        path = _write_encoder_checkpoint(tmp_path / "partial.pth", partial)

        with pytest.raises(RuntimeError) as exc:
            load_pretrained_encoder(_resnet18(), path)
        message = str(exc.value)
        assert "Refusing to load" in message
        assert "architecture mismatch" in message
        assert "allow_partial" in message, "error should say how to override deliberately"

    def test_allow_partial_overrides_the_gate(self, tmp_path, resnet18_state):
        partial = dict(list(resnet18_state.items())[: len(resnet18_state) // 5])
        path = _write_encoder_checkpoint(tmp_path / "partial.pth", partial)
        report = load_pretrained_encoder(_resnet18(), path, allow_partial=True)
        assert report.fraction < 0.95

    def test_reads_encoder_then_state_dict_then_model(self, tmp_path, resnet18_state):
        """Checkpoints in the wild use different top-level keys."""
        for key in ("encoder", "state_dict", "model"):
            path = _write_encoder_checkpoint(tmp_path / f"{key}.pth", resnet18_state, key=key)
            assert load_pretrained_encoder(_resnet18(), path).fraction == 1.0

    def test_accepts_a_bare_state_dict(self, tmp_path, resnet18_state):
        path = tmp_path / "bare.pth"
        torch.save(resnet18_state, path)
        assert load_pretrained_encoder(_resnet18(), path).fraction == 1.0

    def test_a_trivially_empty_checkpoint_is_refused(self, tmp_path):
        path = _write_encoder_checkpoint(tmp_path / "empty.pth", {})
        with pytest.raises(RuntimeError, match="Refusing to load"):
            load_pretrained_encoder(_resnet18(), path)


class TestLoadReport:
    def test_fraction_handles_an_empty_target(self):
        assert LoadReport(matched=0, total=0, missing=[], unexpected=[]).fraction == 0.0


class TestFindResumeCheckpoint:
    def test_prefers_last(self, tmp_path):
        (tmp_path / "last.pth").touch()
        (tmp_path / "sparc_coco_resnet18_ep0010.pth").touch()
        assert find_resume_checkpoint(tmp_path).name == "last.pth"

    def test_falls_back_to_the_highest_epoch(self, tmp_path):
        for epoch in (10, 90, 100):
            (tmp_path / f"sparc_coco_resnet18_ep{epoch:04d}.pth").touch()
        assert find_resume_checkpoint(tmp_path).name.endswith("ep0100.pth")

    def test_sorts_numerically_not_lexicographically(self, tmp_path):
        """Zero padding makes string order agree today, but the parse is what
        makes that a guarantee rather than a coincidence."""
        for epoch in (9, 10, 100):
            (tmp_path / f"m_d_a_ep{epoch}.pth").touch()  # deliberately unpadded
        assert find_resume_checkpoint(tmp_path).name == "m_d_a_ep100.pth"

    def test_returns_none_when_empty(self, tmp_path):
        assert find_resume_checkpoint(tmp_path) is None

    def test_explicit_path_that_is_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="resume"):
            find_resume_checkpoint(tmp_path, str(tmp_path / "nope.pth"))
