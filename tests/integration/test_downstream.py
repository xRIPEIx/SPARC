"""End-to-end downstream evaluation on a synthetic VOC tree.

Both tasks, over multiple backbones, without the real dataset.
"""

from __future__ import annotations

import pytest
import torch

from sparc.cli.evaluate import main
from sparc.experiments.results import read_result_csv

CONFIGS = {
    "segmentation": "configs/downstream/voc_seg_fcn_r18.yaml",
    "detection": "configs/downstream/voc_det_frcnn_r18.yaml",
}


def _argv(task, voc_root, out, results_csv, **extra):
    settings = {
        "data.voc_root": str(voc_root),
        "data.num_workers": "0",
        "train.epochs": "1",
        "train.batch_size": "2",
        "train.eval_batch_size": "2",
        "train.amp": "false",
        "train.print_freq": "0",
        "data.crop_size": "64",
        "data.eval_size": "64",
        "model.init": "random",
        "model.ssl_ckpt": "null",
        "output.dir": str(out),
        "results_csv": str(results_csv),
        "experiment.config_id": "smoke",
        "experiment.run_id": "smoke",
    }
    settings.update(extra)
    return ["--config", CONFIGS[task], "--set", *[f"{k}={v}" for k, v in settings.items()]]


@pytest.mark.parametrize("task", sorted(CONFIGS))
def test_task_runs_and_writes_a_result_row(task, synthetic_voc, tmp_path):
    csv_path = tmp_path / "results" / f"{task}.csv"
    assert main(_argv(task, synthetic_voc, tmp_path / "out", csv_path)) == 0

    rows = read_result_csv(csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == task
    assert row["run_id"] == "smoke"
    assert row["backbone"] == "resnet18"
    metric = "mIoU" if task == "segmentation" else "AP"
    assert metric in row and float(row[metric]) == float(row[metric])  # not NaN


def test_segmentation_reports_both_metrics(synthetic_voc, tmp_path):
    csv_path = tmp_path / "seg.csv"
    main(_argv("segmentation", synthetic_voc, tmp_path / "out", csv_path))
    row = read_result_csv(csv_path)[0]
    assert {"mIoU", "pixel_acc"} <= set(row)


def test_detection_reports_ap_at_both_thresholds(synthetic_voc, tmp_path):
    csv_path = tmp_path / "det.csv"
    main(_argv("detection", synthetic_voc, tmp_path / "out", csv_path))
    row = read_result_csv(csv_path)[0]
    assert {"AP", "AP50", "AP75"} <= set(row)


def test_ssl_checkpoint_is_transferred(synthetic_voc, tmp_path, capsys):
    """A pretrained encoder must actually reach the downstream backbone."""
    import torchvision
    from torch import nn

    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    ckpt = tmp_path / "ssl.pth"
    torch.save({"encoder": net.state_dict(), "epoch": 100, "method": "sparc"}, ckpt)

    csv_path = tmp_path / "seg.csv"
    argv = _argv(
        "segmentation",
        synthetic_voc,
        tmp_path / "out",
        csv_path,
        **{"model.init": "ssl", "model.ssl_ckpt": str(ckpt)},
    )
    assert main(argv) == 0
    assert "100.0%" in capsys.readouterr().out, "full encoder transfer not reported"

    row = read_result_csv(csv_path)[0]
    assert row["ssl_method"] == "sparc"
    assert row["ssl_epochs"] == "100"


def test_mismatched_checkpoint_is_refused(synthetic_voc, tmp_path):
    """The gate that replaces strict=False: a checkpoint that does not fit the
    requested architecture must stop the run, not train a random backbone."""
    torch.save({"encoder": {"nothing.useful": torch.zeros(1)}}, tmp_path / "bad.pth")
    argv = _argv(
        "segmentation",
        synthetic_voc,
        tmp_path / "out",
        tmp_path / "r.csv",
        **{"model.init": "ssl", "model.ssl_ckpt": str(tmp_path / "bad.pth")},
    )
    with pytest.raises(RuntimeError, match="Refusing to load"):
        main(argv)


def test_save_model_writes_the_finetuned_head(synthetic_voc, tmp_path):
    out = tmp_path / "out"
    argv = _argv(
        "segmentation",
        synthetic_voc,
        out,
        tmp_path / "r.csv",
        **{"train.save_model": "true", "train.seed": "1"},
    )
    main(argv)
    assert (out / "best_segmentation_seed1.pth").is_file()


class TestBackboneAgnostic:
    """Both heads build themselves from the backbone's reported channels, so a
    different architecture needs no code change here either."""

    @pytest.mark.parametrize("task", sorted(CONFIGS))
    def test_resnet50(self, task, synthetic_voc, tmp_path):
        argv = _argv(
            task,
            synthetic_voc,
            tmp_path / "out",
            tmp_path / "r.csv",
            **{"backbone.name": "resnet50"},
        )
        assert main(argv) == 0

    @pytest.mark.parametrize("task", sorted(CONFIGS))
    def test_timm_backbone(self, task, synthetic_voc, tmp_path):
        pytest.importorskip("timm")
        argv = _argv(
            task,
            synthetic_voc,
            tmp_path / "out",
            tmp_path / "r.csv",
            **{"backbone.name": "timm:convnext_tiny"},
        )
        assert main(argv) == 0
