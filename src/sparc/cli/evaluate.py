"""`sparc-eval` -- downstream transfer evaluation on PASCAL VOC.

    sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
               --set model.ssl_ckpt=/path/to/last.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sparc.config.loader import load_config, save_config
from sparc.config.schema import DownstreamConfig
from sparc.eval import detection as det_task
from sparc.eval import segmentation as seg_task
from sparc.eval.builders import build_downstream_model
from sparc.experiments.results import build_result_row, write_result_csv
from sparc.experiments.run_id import resolve_run_id
from sparc.utils.amp import build_grad_scaler
from sparc.utils.device import resolve_device
from sparc.utils.seed import dataloader_generator, seed_everything, worker_init_fn

#: Which metric selects the best epoch, per task.
PRIMARY_METRIC = {"segmentation": "mIoU", "detection": "AP"}


def build_loaders(config):
    common = dict(
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=dataloader_generator(config.train.seed),
    )
    if config.task == "segmentation":
        from sparc.data.voc.segmentation import (
            build_segmentation_datasets,
            collate_segmentation,
        )

        train_set, val_set = build_segmentation_datasets(
            config.data.voc_root,
            sbd_root=config.data.sbd_root,
            train_source=config.data.seg_train_source,
            crop_size=config.data.crop_size,
            eval_size=config.data.eval_size,
            download=config.data.download,
        )
        collate = collate_segmentation
    else:
        from sparc.data.voc.detection import build_detection_datasets, collate_detection

        train_set, val_set = build_detection_datasets(
            config.data.voc_root,
            download=config.data.download,
            ignore_difficult=config.data.ignore_difficult,
        )
        collate = collate_detection

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        **common,
    )
    return train_loader, val_loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sparc-eval",
        description="Fine-tune a pretrained backbone on VOC segmentation or detection.",
    )
    parser.add_argument("--config", help="Path to a downstream YAML config.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--paths", default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.config:
        raise SystemExit("--config is required")

    config = load_config(args.config, args.set, paths_file=args.paths, schema=DownstreamConfig)
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0

    if config.task not in PRIMARY_METRIC:
        raise SystemExit(f"Unknown task {config.task!r}; expected one of {sorted(PRIMARY_METRIC)}")

    seed_everything(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu)
    task_module = seg_task if config.task == "segmentation" else det_task

    model, _ = build_downstream_model(
        config.task,
        backbone_name=config.backbone.name,
        init=config.model.init,
        ssl_ckpt=config.model.ssl_ckpt,
        allow_partial_load=config.model.allow_partial_load,
    )
    model = model.to(device)

    train_loader, val_loader = build_loaders(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.train.epochs, 1)
    )
    amp_enabled = bool(config.train.amp) and device.type == "cuda"
    scaler = build_grad_scaler(amp_enabled)

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / f"config_{config.task}.yaml")

    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"device={device} amp={amp_enabled} task={config.task}", flush=True)

    primary = PRIMARY_METRIC[config.task]
    best: dict[str, float] | None = None

    for epoch in range(1, config.train.epochs + 1):
        task_module.train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            amp=amp_enabled,
            print_freq=config.train.print_freq,
        )
        scheduler.step()

        if epoch % config.train.eval_every == 0 or epoch == config.train.epochs:
            metrics = task_module.evaluate(model, val_loader, device)
            if best is None or metrics[primary] > best[primary]:
                best = metrics
                if config.train.save_model:
                    path = output_dir / f"best_{config.task}_seed{config.train.seed}.pth"
                    torch.save(model.state_dict(), path)
                    print(f"Saved best {config.task} model: {path}", flush=True)

    if best is None:  # epochs == 0
        best = task_module.evaluate(model, val_loader, device)

    run_id = config.experiment.run_id or resolve_run_id(
        config.experiment.config_id, config.train.seed
    )
    row = build_result_row(
        run_id=run_id,
        config_id=config.experiment.config_id,
        task=config.task,
        init=config.model.init,
        backbone=config.backbone.name,
        seed=config.train.seed,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        ssl_ckpt=config.model.ssl_ckpt,
        metrics=best,
    )
    path = write_result_csv(row, config.results_csv)
    print(f"best {primary}={best[primary]:.5f} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
