# Pretraining

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml
```

Output goes to `output.dir` (default `<checkpoint_root>/<run_id>`): a periodic
`sparc_coco_resnet18_ep0010.pth` every `save_every` epochs, `last.pth` every
epoch, and `config.yaml`. Re-running the same command resumes from `last.pth`
(`train.resume: auto`).

A checkpoint holds `encoder` (the transferable backbone, plain torchvision key
names), `model` (everything, for resuming), optimiser, scheduler, AMP scaler,
the NT-Xent memory bank, and the resolved config.

## The loop

[`engine/trainer.py`](../src/sparc/engine/trainer.py) is method-agnostic: per
step it schedules the momentum coefficient, calls `update_momentum` then
`training_step`, and handles AMP, stepping and logging. Objectives implement the
[`SSLMethod`](../src/sparc/methods/base.py) interface and own their own
criteria.

## Knobs

| Key | Default | Notes |
|---|---|---|
| `method.lambda_region` | 0.5 | SPARC only. 0 = MoCo v2 exactly |
| `method.region_pool_size` | 7 | grid regions are pooled onto; >7 upsamples the 7×7 map |
| `method.max_regions` | 16 | regions sampled per image |
| `method.temperature` | 0.5 | global NT-Xent (lightly's default; see method.md) |
| `method.dense_lambda` | 0.5 | DenseCL only |
| `train.amp` | true | mixed precision on CUDA |
| `data.limit` | null | cap the corpus — smoke tests only |

## Reproducibility

Seeds cover Python, NumPy and torch, and the DataLoader is given an explicit
generator and worker init function rather than relying on PyTorch's default
worker seeding. Two runs with the same seed on the same hardware match; across
GPUs or cuDNN versions expect differences within seed noise (≈0.4 mIoU points).
A run that was resumed does not reproduce an uninterrupted one's RNG stream.
