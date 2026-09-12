"""`sparc-pretrain` -- self-supervised pretraining.

    sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml
    sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml \
                   --set method.lambda_region=0.7 train.seed=3
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from omegaconf import OmegaConf

from sparc.config.loader import load_config, save_config, to_dict
from sparc.engine.checkpoint import (
    LAST_NAME,
    checkpoint_name,
    find_resume_checkpoint,
    load_resume_checkpoint,
    save_checkpoint,
)
from sparc.engine.optim import build_optimizer, build_scheduler
from sparc.engine.trainer import EpochResult, train
from sparc.methods import get_method, list_methods
from sparc.models.backbones import build_backbone
from sparc.utils.amp import build_grad_scaler
from sparc.utils.device import resolve_device
from sparc.utils.seed import seed_everything


def build_method(config):
    """Instantiate the method, passing only the parameters it accepts.

    One flat schema covers every objective; each constructor takes what it
    needs. That is what lets a new method be added without touching the config
    schema, the CLI, or the trainer.
    """
    for module in config.method.plugins:
        __import__(module)

    spec = build_backbone(config.backbone.name, imagenet_init=config.backbone.imagenet_init)
    cls = get_method(config.method.name)

    accepted = set(inspect.signature(cls.__init__).parameters)
    known = {
        k: v
        for k, v in to_dict(config.method).items()
        if k not in ("name", "params", "plugins") and k in accepted
    }
    known.update(dict(config.method.params))
    return cls(spec, **known), spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sparc-pretrain",
        description="Self-supervised pretraining (SPARC, MoCo v2, DenseCL).",
    )
    # Not required=True: the informational flags below must work without one.
    parser.add_argument("--config", help="Path to a pretraining YAML config.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. method.lambda_region=0.7 train.seed=3",
    )
    parser.add_argument("--paths", default=None, help="Override configs/paths.yaml.")
    parser.add_argument(
        "--print-config", action="store_true", help="Print the resolved config and exit."
    )
    parser.add_argument(
        "--list-methods", action="store_true", help="List registered methods and exit."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_methods:
        print("\n".join(list_methods()))
        return 0
    if not args.config:
        raise SystemExit("--config is required (or pass --list-methods)")

    config = load_config(args.config, args.set, paths_file=args.paths)
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0

    seed_everything(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu)

    method, spec = build_method(config)

    # Checked before the dataset is built and before any compute is spent.
    if method.requires_region_masks:
        if not config.data.masks:
            raise SystemExit(
                f"Method {config.method.name!r} requires region masks, but data.masks is unset.\n"
                f"Generate them with `sparc-masks`, then set data.masks."
            )
        if not Path(config.data.masks).exists():
            raise SystemExit(
                f"Method {config.method.name!r} requires region masks, but the mask directory "
                f"does not exist:\n  {config.data.masks}\n"
                f"Generate it with:\n"
                f"  sparc-masks --config configs/superpixel/slic_n100.yaml\n"
                f"or point data.masks somewhere else. Paths come from configs/paths.yaml."
            )

    from sparc.data.pretrain_dataset import build_pretrain_dataloader

    loader = build_pretrain_dataloader(config, requires_masks=method.requires_region_masks)
    method = method.to(device)

    optimizer = build_optimizer(
        method,
        name=config.train.optimizer,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, name=config.train.scheduler, epochs=config.train.epochs)
    amp_enabled = bool(config.train.amp) and device.type == "cuda"
    scaler = build_grad_scaler(amp_enabled)

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    start_epoch = 1
    if config.train.resume:
        found = find_resume_checkpoint(output_dir, config.train.resume)
        if found is not None:
            start_epoch = load_resume_checkpoint(
                found,
                method=method,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if amp_enabled else None,
            )
            print(f"Resumed from {found} at epoch {start_epoch}", flush=True)
            if start_epoch > config.train.epochs:
                print("Already at the requested number of epochs; nothing to do.")
                return 0

    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"device={device} amp={amp_enabled} batches/epoch={len(loader)}", flush=True)

    resolved = to_dict(config)

    def on_epoch_end(result: EpochResult) -> None:
        periodic = (
            result.epoch % config.train.save_every == 0 or result.epoch == config.train.epochs
        )
        if not (periodic or config.train.save_last_every_epoch):
            return
        payload = dict(
            method=method,
            method_name=config.method.name,
            config=resolved,
            epoch=result.epoch,
            avg_loss=result.avg_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if amp_enabled else None,
        )
        if periodic:
            save_checkpoint(
                output_dir
                / checkpoint_name(
                    config.method.name,
                    config.data.dataset_name,
                    spec.name.replace(":", "_"),
                    result.epoch,
                ),
                **payload,
            )
        save_checkpoint(output_dir / LAST_NAME, **payload)

    train(
        method,
        loader,
        epochs=config.train.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        start_epoch=start_epoch,
        amp=amp_enabled,
        momentum_start=config.train.momentum_start,
        momentum_end=config.train.momentum_end,
        log_every=config.train.log_every,
        on_epoch_end=on_epoch_end,
    )
    print(f"Done. Checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
