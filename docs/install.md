# Install

```bash
git clone https://github.com/xRIPEIx/SPARC.git && cd SPARC
pip install -e ".[dev]"
pytest            # ~1 minute on CPU; no datasets, no network, no GPU needed
```

`pytest` passing is the install check. It exercises mask generation, all three
pretraining objectives, both downstream tasks and the reporting tools on
synthetic data.

## Extras

| Extra | Gives you |
|---|---|
| `timm` | Any [timm](https://github.com/huggingface/pytorch-image-models) model as a backbone via `backbone.name: timm:<model>` |
| `viz` | `sparc-report figure` (matplotlib) |
| `dev` | pytest, ruff |

```bash
pip install -e ".[dev,timm,viz]"
```

## Exact versions

`requirements-lock.txt` pins the environment the study ran in
(torch 2.10, torchvision 0.25, scikit-image 0.26.0). `pip install -r
requirements-lock.txt` reproduces it.

**scikit-image is pinned exactly in `pyproject.toml`, on purpose.** It *is* the
superpixel method: `slic()` output can change between releases, which would
silently change every downstream number. The pinned version regenerates the
study's archived masks identically; do not relax it without re-running that
check (see [reproduce.md](reproduce.md)).

## GPU

Pretraining and fine-tuning need CUDA; everything else runs on CPU. The study
used single A100 3g.20gb MIG slices (20 GB), with AMP on. Nothing requires more
than one GPU, and multi-GPU is deliberately not implemented: the published
numbers depend on a single-process batch of 64 and a 4096-entry memory bank.
