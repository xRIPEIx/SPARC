#!/usr/bin/env python
"""The five-minute path: run a released fine-tuned model on your own images.

    python scripts/download_checkpoints.py seg_head
    python scripts/demo_predict.py --task segmentation \
        --weights checkpoints/sparc_lambda_0p5_voc_segmentation_resnet18.pth \
        --images path/to/some/jpegs --out demo_out

Writes one PNG per image: the photo with predicted VOC classes overlaid
(segmentation) or predicted boxes drawn (detection). CPU is fine.
No dataset is needed -- any JPEG or PNG will do.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from sparc.data.mask_dataset import IMAGE_SUFFIXES
from sparc.data.voc.classes import VOC_CLASSES
from sparc.eval.builders import build_downstream_model

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def palette(n: int = 21) -> np.ndarray:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.3, 1.0, size=(n, 3))
    p[0] = 0.0  # background
    return p


def load_model(task: str, weights: Path, device):
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    backbone = payload.get("backbone", "resnet18") if isinstance(payload, dict) else "resnet18"
    model, _ = build_downstream_model(task, backbone_name=backbone, init="random")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload.get("metrics") if isinstance(payload, dict) else None


@torch.no_grad()
def predict_segmentation(model, image: Image.Image, device, size: int = 520) -> Image.Image:
    x = TF.normalize(TF.to_tensor(TF.resize(image, [size])), MEAN, STD)[None].to(device)
    logits = model(x)["out"]
    logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = logits.argmax(1)[0].cpu().numpy()
    rgb = np.asarray(image).astype(np.float64) / 255.0
    colour = palette()[pred]
    keep = pred > 0
    out = rgb.copy()
    out[keep] = 0.45 * rgb[keep] + 0.55 * colour[keep]
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8)), sorted(
        {VOC_CLASSES[c] for c in np.unique(pred) if c}
    )


@torch.no_grad()
def predict_detection(model, image: Image.Image, device, threshold: float = 0.5):
    out = model([TF.to_tensor(image).to(device)])[0]
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    found = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"], strict=True):
        if score < threshold:
            continue
        x0, y0, x1, y1 = box.tolist()
        name = VOC_CLASSES[int(label)]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 210, 0), width=3)
        draw.text((x0 + 3, y0 + 2), f"{name} {float(score):.2f}", fill=(255, 210, 0))
        found.append(name)
    return canvas, sorted(set(found))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=("segmentation", "detection"), required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="a file or a directory")
    ap.add_argument("--out", type=Path, default=Path("demo_out"))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="detection score threshold")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metrics = load_model(a.task, a.weights, device)
    if metrics:
        print(f"loaded {a.weights.name} (VOC2012 val: {metrics})")

    paths = (
        [a.images]
        if a.images.is_file()
        else sorted(
            Path(d) / f
            for d, _, fs in os.walk(a.images)
            for f in fs
            if Path(f).suffix.lower() in IMAGE_SUFFIXES
        )[: a.limit]
    )
    if not paths:
        raise SystemExit(f"No images under {a.images}")
    a.out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        with Image.open(p) as im:
            image = im.convert("RGB")
        if a.task == "segmentation":
            result, found = predict_segmentation(model, image, device)
        else:
            result, found = predict_detection(model, image, device, a.threshold)
        dest = a.out / f"{p.stem}_{a.task}.png"
        result.save(dest)
        print(f"{dest}: {', '.join(found) or 'nothing above threshold'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
