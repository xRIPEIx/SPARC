# Configuration

YAML, validated against typed dataclasses
([`config/schema.py`](../src/sparc/config/schema.py)) before anything is built,
so a typo is an error at second zero rather than a surprise an hour in.

## Composition

A config lists what it inherits in `defaults:`; later entries and the file's own
keys win:

```yaml
# configs/pretrain/sparc_coco_r18.yaml
defaults: [_base.yaml, ../backbone/resnet18.yaml]
method:
  name: sparc
  lambda_region: 0.5
```

## Overrides

Anything can be changed from the command line. Unknown keys are **rejected**:

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml \
               --set method.lambda_region=0.7 backbone.name=resnet50 train.seed=3
```

Values are parsed as YAML scalars, so `null`, `true`, numbers and lists mean
what they look like.

## Paths

`configs/paths.yaml` is the only place dataset locations live; see
[datasets.md](datasets.md). Precedence: `paths.local.yaml` > environment
variable > default. Paths are resolved into the config for the run but are
**not** saved into checkpoints — they describe the machine, not the experiment.

## Provenance

The fully-resolved config is printed at start, written to
`<output.dir>/config.yaml`, and embedded in every checkpoint under `"config"`.
A checkpoint therefore explains how it was made.

## Which file for what

| Directory | Contents |
|---|---|
| `configs/pretrain/` | one file per objective; `_base.yaml` holds the shared protocol |
| `configs/downstream/` | segmentation and detection; `_base.yaml` holds the frozen fine-tuning protocol |
| `configs/backbone/` | one file per architecture |
| `configs/superpixel/` | one file per mask set |
| `configs/sweeps/` | one file per ablation; expands to a manifest |
