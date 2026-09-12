# Downstream evaluation

```bash
sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
           --set model.ssl_ckpt=runs/checkpoints/sparc_lambda_0p5/last.pth \
                 experiment.config_id=sparc_lambda_0p5
sparc-eval --config configs/downstream/voc_det_frcnn_r18.yaml --set ...
```

Writes one result row to `results_csv` (default
`<results_root>/<task>/<run_id>.csv`) and, with `train.save_model=true`, keeps
the best fine-tuned model.

## The frozen protocol

| | Segmentation | Detection |
|---|---|---|
| Model | FCN head on the final stage | Faster R-CNN + FPN over all four stages |
| Train / val | VOC2012 train / val | VOC2012 Main train / val |
| Epochs | 40 | 10 |
| Batch | 16 | 2 |
| Optimiser | AdamW, lr 7.48e-5, wd 1.20e-4, cosine | same |
| Selection | best-epoch mIoU | best-epoch AP |
| Metrics | mIoU (confusion matrix over the whole val set), pixel acc. | COCO-style AP, AP50, AP75 |

The learning rate and weight decay were tuned **once**, by an Optuna search over
segmentation against a fixed checkpoint, and then applied to every arm and both
tasks. One protocol across arms is what makes the comparison measure the
pretraining objective rather than how much search each arm received.

## Initialisations

`model.init` is `ssl` (load `model.ssl_ckpt`), `supervised_imagenet`
(torchvision weights) or `random`.

Loading a checkpoint that does not fit the requested architecture is a **hard
error**: at least 95% of the backbone's parameters must match. A silent partial
load would leave most of the network random and still report a plausible
number. `model.allow_partial_load=true` overrides, deliberately.

## Any backbone

Both heads size themselves from the backbone's reported per-stage channels, so
`--set backbone.name=resnet50` or `timm:convnext_tiny` works unchanged. When
comparing arms on a new backbone, keep the normalisation layer the same across
`init` modes; torchvision's own resnet50 detection builder silently picks
frozen BatchNorm for pretrained weights and trainable BatchNorm otherwise.
