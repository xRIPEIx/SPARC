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

## Hardware

| Key | Default | Notes |
|---|---|---|
| `train.device` | `auto` | `auto` picks CUDA when available, else CPU. Also `cpu`, `cuda`, `cuda:1` |
| `train.gpu` | null | CUDA device index; `--set train.gpu=1`. `CUDA_VISIBLE_DEVICES` works too |
| `train.amp` | true | Mixed precision; only active on CUDA. Halves memory, no measurable effect on results |
| `train.batch_size` | 64 | ~18 GB at 224 px with AMP. See the note below before lowering it |
| `data.num_workers` | 8 | Dataloader processes; set to the CPU cores available to the job |

Nothing needs configuring on a single-GPU machine: `auto` finds it. Multi-GPU
training is deliberately not implemented (see the FAQ).

**Batch size is part of the experiment, not just a memory setting.** The
in-batch region loss draws its negatives from the batch, and the global term's
memory bank is refilled at batch granularity, so a smaller batch changes the
objective's negatives and the published numbers no longer apply. On a GPU that
cannot hold 64 at 224 px with AMP, prefer a smaller `data.input_size` for
exploratory work and say so when reporting; a 3g.20gb A100 MIG slice (20 GB)
is enough for the paper's setting.

The same keys exist for `sparc-eval` (`train.device`, `train.gpu`,
`train.amp`, `train.batch_size`, `train.eval_batch_size`, `data.num_workers`).
Detection at batch 2 needs ~10 GB; segmentation at batch 16 and 512 px crops
~14 GB.

## Reproducibility

Seeds cover Python, NumPy and torch, and the DataLoader is given an explicit
generator and worker init function rather than relying on PyTorch's default
worker seeding. Two runs with the same seed on the same hardware match; across
GPUs or cuDNN versions expect differences within seed noise (≈0.4 mIoU points).
A run that was resumed does not reproduce an uninterrupted one's RNG stream.
