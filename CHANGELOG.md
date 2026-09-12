# Changelog

## v1.0.0 — paper release

First public release. Everything needed to reproduce the paper's tables and to
apply the method to new data:

- `sparc-masks`, `sparc-pretrain`, `sparc-eval`, `sparc-sweep`, `sparc-report`
- Backbone registry (any torchvision ResNet or timm model), method registry
  (SPARC, MoCo v2, DenseCL), superpixel registry (SLIC + two worked examples)
- Aggregated results for the three sweeps, and the tables and figures rendered
  from them
- Released weights: five encoders and two fine-tuned VOC models (see MODEL_ZOO.md)

This code is a restructuring of the implementation the study was run with,
verified bit-identical on a fixed batch (loss and gradients) and reproducing the
archived superpixel masks exactly.
