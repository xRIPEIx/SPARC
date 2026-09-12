# Extending SPARC

Four recipes. Each is one registry entry; none touches the trainer, the CLIs or
the config schema. Every recipe ends with the test you should add.

## A new backbone

Any torchvision ResNet is built in and **any timm model already works**:

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set backbone.name=timm:convnext_tiny
```

For a family the registry does not know, return a `BackboneSpec`
([`models/backbones/base.py`](../src/sparc/models/backbones/base.py)):

```python
from sparc.models.backbones import BackboneSpec, register_backbone

@register_backbone("mynet")
def build_mynet(*, name, imagenet_init=False, **_):
    net = MyNet(pretrained=imagenet_init)          # classification head removed
    return BackboneSpec(
        name=name,
        net=net,                                   # its state_dict is what gets released
        feature_dim=768,                           # channels of the final stage
        stage_channels=(96, 192, 384, 768),        # per stage, coarsest last — the FPN reads this
        stage_strides=(4, 8, 16, 32),
        make_extractor=lambda stages: MyNetStages(net, stages),   # image -> {"s1":..., "s4":...}
    )
```

`make_extractor` must return a module whose parameters are **shared** with
`net`, not copied, so momentum updates reach it. Register it via a plugin
module (`method.plugins: [my_package.backbones]`) or a `sparc.backbones` entry
point.

*Test:* `stage_channels` equals the channel count observed on a real forward
pass, and `make_extractor(("s4",))` output matches `net`'s own final feature
map. `tests/unit/test_backbone_registry.py` is the template.

## A new dataset

**Pretraining** needs only an image tree. Point `data.images` at it. For SPARC,
generate masks with `sparc-masks --set superpixel.images=<tree>` and point
`data.masks` at the output — the mask tree mirrors the image tree, one `.npy`
per image, matched by relative path
([`data/mask_dataset.py`](../src/sparc/data/mask_dataset.py)). No code.

**Downstream** on something other than VOC: implement a `Dataset` returning
`(image_tensor, mask_tensor)` for segmentation or `(image_tensor, target_dict)`
for detection — the torchvision detection contract — and a builder like
[`data/voc/segmentation.py`](../src/sparc/data/voc/segmentation.py). Set
`num_classes` when calling `build_downstream_model`.

*Test:* the alignment invariant — an image whose pixel values encode its own
labels must still agree with its mask after your transform.
`tests/unit/test_transforms_alignment.py` shows how.

## A new superpixel method

One function honouring the contract, one YAML:

```python
from sparc.data.superpixel import register_superpixel

@register_superpixel("quickshift")
def quickshift_mask(image_rgb, n_segments, *, kernel_size=3, max_dist=6.0):
    labels = skimage.segmentation.quickshift(image_rgb, kernel_size=kernel_size, max_dist=max_dist)
    return labels                                  # int [H, W], labels dense from 0
```

```yaml
# configs/superpixel/quickshift_n100.yaml
superpixel:
  method: quickshift
  n_segments: 100
  output: ${paths.superpixel_root}/quickshift_n100
```

Labels must start at 0 and be dense; label 0 is a real region downstream. If
your method has no direct segment-count knob, search a proxy parameter to land
near `n_segments` — `felzenszwalb` in
[`data/superpixel/skimage_methods.py`](../src/sparc/data/superpixel/skimage_methods.py)
is the worked example.

*Test:* `tests/unit/test_superpixel_methods.py` is parameterised over every
registered method; yours is picked up automatically.

## A new pretraining objective

Subclass [`SSLMethod`](../src/sparc/methods/base.py) and implement three
methods:

```python
from sparc.methods import SSLMethod, SSLBatch, StepOutput, register_method

@register_method("byol_region")
class BYOLRegion(SSLMethod):
    requires_region_masks = True                   # checked before any compute is spent

    def __init__(self, spec, *, proj_dim=128, **kw):
        super().__init__(spec)
        self.backbone = spec.final_extractor()
        ...

    def training_step(self, batch: SSLBatch) -> StepOutput:
        ...
        return StepOutput(loss=loss, metrics={"loss": float(loss)})

    def update_momentum(self, m: float) -> None: ...
    def encoder_state_dict(self): return self.backbone.state_dict()   # plain backbone keys
```

Constructor keyword arguments are filled from `method.*` config keys by name;
anything extra goes in `method.params`. Register from outside the repository
with `method.plugins: [my_package.methods]`.

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set method.name=byol_region
```

*Test:* `tests/unit/test_methods.py` treats every registered method uniformly
— finite scalar loss, gradients reach the backbone, frozen momentum copies
track the online ones, `encoder_state_dict` has plain keys.

## What deliberately does not extend

The per-item loops in `SuperpixelRegionPool` draw from `torch.randperm` once per
image. Vectorising them changes the RNG stream and therefore every number, by
an amount smaller than seed noise — invisible in results while silently
breaking reproduction. If you do it, put it behind a default-off flag and test
exact equivalence against the loop version over many random batches.
