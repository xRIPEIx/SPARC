"""Typed configuration schema.

Frozen dataclasses validated by OmegaConf. A YAML file is checked against these
before anything is constructed, so a typo is an error at second zero rather than
a surprise after an hour of GPU time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackboneConfig:
    #: Any registry key, or "timm:<model>" for any timm architecture.
    name: str = "resnet18"
    #: Start from ImageNet-supervised weights instead of random init.
    imagenet_init: bool = False


@dataclass
class MethodConfig:
    """Objective hyper-parameters.

    Every method reads the fields it needs and ignores the rest, so one schema
    covers all of them. `params` is the escape hatch for out-of-tree methods
    registered via a plugin.
    """

    name: str = "sparc"
    proj_dim: int = 128
    memory_bank_size: int = 4096
    temperature: float = 0.1

    # --- SPARC ---
    #: Weight on the region term: L = (1 - l) * L_global + l * L_region.
    #: 0 reduces SPARC to the MoCo-global objective.
    lambda_region: float = 0.5
    region_temperature: float = 0.2
    max_regions: int = 16
    #: Side length of the RxR grid regions are pooled onto.
    region_pool_size: int = 7
    #: Minimum region area in full-resolution transformed mask pixels.
    region_min_area: float = 1.0

    # --- DenseCL ---
    #: DenseCL's own global-vs-dense mix. NOTE: 0.5 is the DenseCL baseline.
    dense_lambda: float = 0.5

    #: Extra keyword arguments for methods registered outside this package.
    params: dict[str, Any] = field(default_factory=dict)
    #: Modules to import before resolving `name`, so a plugin can register.
    plugins: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    #: Image root. Any directory tree of images.
    images: str = "???"
    #: Superpixel mask root, mirroring the image tree. Required by SPARC only.
    masks: str | None = None
    #: Label recorded in checkpoints and run ids, e.g. "coco".
    dataset_name: str = "coco"
    input_size: int = 224
    num_workers: int = 8
    #: Cap the dataset size. For smoke tests only -- never for a real run.
    limit: int | None = None


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    momentum_start: float = 0.996
    momentum_end: float = 1.0
    amp: bool = True
    seed: int = 1
    save_every: int = 10
    save_last_every_epoch: bool = True
    #: None, "auto", or an explicit checkpoint path.
    resume: str | None = None
    device: str = "auto"
    gpu: int | None = None
    log_every: int = 0


@dataclass
class OutputConfig:
    dir: str = "./runs/${experiment.run_id}"


@dataclass
class ExperimentConfig:
    #: Stable identity for this configuration across seeds. Manifests group on it.
    config_id: str = "default"
    #: Resolved from config_id and seed; seed 1 keeps the bare config_id.
    run_id: str = "default"
    tags: list[str] = field(default_factory=list)


@dataclass
class PretrainConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Downstream evaluation
# ---------------------------------------------------------------------------


@dataclass
class DownstreamDataConfig:
    voc_root: str = "???"
    sbd_root: str | None = None
    #: "voc2012" or "sbd" (SBD train_noval). Validation is always VOC2012 val.
    seg_train_source: str = "voc2012"
    crop_size: int = 512
    eval_size: int = 520
    num_workers: int = 4
    download: bool = False
    #: VOC marks hard examples "difficult"; the standard protocol excludes them.
    ignore_difficult: bool = True


@dataclass
class DownstreamModelConfig:
    #: "random", "supervised_imagenet", or "ssl".
    init: str = "ssl"
    ssl_ckpt: str | None = None
    #: Proceed even when few checkpoint parameters matched. Off by design: the
    #: default refuses, because a silent partial load yields a plausible metric.
    allow_partial_load: bool = False


@dataclass
class DownstreamTrainConfig:
    epochs: int = 40
    batch_size: int = 16
    eval_batch_size: int = 4
    #: Tuned once by an Optuna search over the segmentation task and then frozen
    #: for every run, including detection, so that all arms share one protocol.
    lr: float = 7.47977835764546e-05
    weight_decay: float = 0.000119573094297164
    eval_every: int = 1
    amp: bool = True
    seed: int = 1
    device: str = "auto"
    gpu: int | None = None
    print_freq: int = 50
    #: Keep the fine-tuned model, not just its metric. Needed for prediction
    #: visualisation; a previous study discarded every head and had to refit.
    save_model: bool = False


@dataclass
class DownstreamConfig:
    #: "segmentation" or "detection".
    task: str = "segmentation"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: DownstreamModelConfig = field(default_factory=DownstreamModelConfig)
    data: DownstreamDataConfig = field(default_factory=DownstreamDataConfig)
    train: DownstreamTrainConfig = field(default_factory=DownstreamTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    #: Where the single-row result CSV is written.
    results_csv: str = "${paths.results_root}/${task}/${experiment.run_id}.csv"
