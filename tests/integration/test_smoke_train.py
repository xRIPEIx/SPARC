"""End-to-end pretraining on synthetic data.

Replaces the previous `tools/debug_regioncl_batch.py`, which had to be run by
hand and only checked shapes. Everything here runs in CI.
"""

from __future__ import annotations

import pytest
import torch

from sparc.cli.pretrain import main

METHOD_CONFIGS = {
    "sparc": "configs/pretrain/sparc_coco_r18.yaml",
    "moco": "configs/pretrain/moco_coco_r18.yaml",
    "densecl": "configs/pretrain/densecl_coco_r18.yaml",
}


def _argv(config, images, masks, out, **extra):
    settings = {
        "data.images": str(images),
        "data.input_size": "32",
        "data.num_workers": "0",
        "train.epochs": "2",
        "train.batch_size": "4",
        "train.amp": "false",
        "train.save_every": "1",
        "method.proj_dim": "16",
        "method.memory_bank_size": "32",
        "output.dir": str(out),
    }
    if masks is not None:
        settings["data.masks"] = str(masks)
    settings.update(extra)
    return ["--config", config, "--set", *[f"{k}={v}" for k, v in settings.items()]]


@pytest.mark.parametrize("method", sorted(METHOD_CONFIGS))
def test_each_method_trains_and_checkpoints(method, synthetic_corpus, tmp_path, capsys):
    images, masks = synthetic_corpus
    out = tmp_path / f"out_{method}"
    needs_masks = method == "sparc"

    assert main(_argv(METHOD_CONFIGS[method], images, masks if needs_masks else None, out)) == 0

    assert (out / "config.yaml").is_file(), "resolved config not saved beside the checkpoint"
    assert (out / "last.pth").is_file()
    checkpoints = sorted(out.glob(f"{method}_*_ep*.pth"))
    assert checkpoints, f"no periodic checkpoint written for {method}"

    payload = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
    assert payload["method"] == method
    assert payload["epoch"] == 2
    assert "encoder" in payload and payload["encoder"]
    # The saved config is what makes a checkpoint self-describing.
    assert payload["config"]["method"]["name"] == method

    out_text = capsys.readouterr().out
    assert "epoch=0002/2" in out_text


def test_encoder_state_is_plain_backbone_keys(synthetic_corpus, tmp_path):
    """Downstream transfer loads this straight into a fresh backbone, so no
    wrapper prefixes and no projection-head weights may leak in."""
    images, masks = synthetic_corpus
    out = tmp_path / "out"
    main(_argv(METHOD_CONFIGS["sparc"], images, masks, out, **{"train.epochs": "1"}))

    encoder = torch.load(out / "last.pth", map_location="cpu", weights_only=False)["encoder"]
    assert "conv1.weight" in encoder
    assert not any(k.startswith(("module.", "encoder.", "backbone.")) for k in encoder)
    assert not any("projection_head" in k or "region_projector" in k for k in encoder)


def test_training_actually_updates_weights(synthetic_corpus, tmp_path):
    """A loss that decreases proves nothing if nothing is being optimised."""
    images, masks = synthetic_corpus
    out = tmp_path / "out"
    main(_argv(METHOD_CONFIGS["sparc"], images, masks, out, **{"train.save_every": "1"}))

    first = torch.load(
        out / "sparc_coco_resnet18_ep0001.pth", map_location="cpu", weights_only=False
    )
    second = torch.load(
        out / "sparc_coco_resnet18_ep0002.pth", map_location="cpu", weights_only=False
    )
    changed = [
        k for k in first["encoder"] if not torch.equal(first["encoder"][k], second["encoder"][k])
    ]
    assert changed, "encoder weights identical across epochs; nothing trained"


def test_resume_continues_rather_than_restarting(synthetic_corpus, tmp_path, capsys):
    images, masks = synthetic_corpus
    out = tmp_path / "out"
    main(_argv(METHOD_CONFIGS["sparc"], images, masks, out, **{"train.epochs": "1"}))
    capsys.readouterr()

    main(_argv(METHOD_CONFIGS["sparc"], images, masks, out, **{"train.epochs": "2"}))
    text = capsys.readouterr().out
    assert "Resumed from" in text
    assert "epoch=0002/2" in text
    assert "epoch=0001/2" not in text, "resume restarted from the beginning"


class TestMaskPreconditions:
    """requires_region_masks is checked at config time, not mid-epoch. Failing
    an hour into a run because a mask directory was mistyped is exactly the kind
    of wasted GPU time this repository is meant to avoid."""

    def test_unset_masks_is_rejected(self, synthetic_corpus, tmp_path):
        images, _ = synthetic_corpus
        argv = _argv(METHOD_CONFIGS["sparc"], images, None, tmp_path / "out")
        argv += ["data.masks=null"]
        with pytest.raises(SystemExit, match="data.masks is unset"):
            main(argv)

    def test_missing_mask_directory_is_rejected_with_a_fix(self, synthetic_corpus, tmp_path):
        images, _ = synthetic_corpus
        argv = _argv(METHOD_CONFIGS["sparc"], images, tmp_path / "nope", tmp_path / "out")
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert "sparc-masks" in str(exc.value), "error should say how to generate masks"


class TestBackboneSwap:
    """The property the registry exists for: a different architecture is a
    config change, not a code change."""

    def test_resnet50_needs_no_code_change(self, synthetic_corpus, tmp_path):
        images, masks = synthetic_corpus
        out = tmp_path / "out50"
        argv = _argv(
            METHOD_CONFIGS["sparc"],
            images,
            masks,
            out,
            **{"backbone.name": "resnet50", "train.epochs": "1"},
        )
        assert main(argv) == 0
        encoder = torch.load(out / "last.pth", map_location="cpu", weights_only=False)["encoder"]
        # Bottleneck expansion: layer1 output is 256 channels, not 64.
        assert encoder["layer1.0.conv3.weight"].shape[0] == 256

    def test_timm_backbone_needs_no_code_change(self, synthetic_corpus, tmp_path):
        pytest.importorskip("timm")
        images, masks = synthetic_corpus
        out = tmp_path / "outtimm"
        argv = _argv(
            METHOD_CONFIGS["sparc"],
            images,
            masks,
            out,
            **{"backbone.name": "timm:convnext_tiny", "train.epochs": "1"},
        )
        assert main(argv) == 0
        assert (out / "last.pth").is_file()
