"""`sparc-masks` -- generate a superpixel mask set for an image tree.

    sparc-masks --config configs/superpixel/slic_n100.yaml
    sparc-masks --config configs/superpixel/slic_n100.yaml --num-shards 32 --shard-index 7

Writes one .npy per image, mirroring the image tree, plus a metadata JSON
recording exactly how the set was made. Resumable: existing masks are skipped,
so a job that hits its time limit is re-run rather than restarted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from sparc.config.loader import load_config
from sparc.config.schema import SuperpixelRoot
from sparc.data.mask_dataset import IMAGE_SUFFIXES
from sparc.data.superpixel import (
    accepted_kwargs,
    compute_mask,
    list_superpixel_methods,
    write_meta,
)

DTYPES = {"auto": None, "uint8": np.uint8, "int16": np.int16, "int32": np.int32}


def iter_images(image_root: Path) -> list[Path]:
    """Sorted, so shards are stable across invocations and machines."""
    return sorted(
        Path(dirpath) / name
        for dirpath, _dirs, names in os.walk(image_root)
        for name in names
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    )


def cast_mask(mask: np.ndarray, dtype: str) -> np.ndarray:
    lo, hi = int(mask.min()), int(mask.max())
    if dtype == "auto":
        for candidate in (np.uint8, np.int16, np.int32):
            info = np.iinfo(candidate)
            if info.min <= lo and hi <= info.max:
                return mask.astype(candidate, copy=False)
        raise ValueError(f"labels [{lo}, {hi}] exceed int32")
    target = DTYPES[dtype]
    info = np.iinfo(target)
    if lo < info.min or hi > info.max:
        raise ValueError(f"Mask labels [{lo}, {hi}] do not fit dtype {dtype}")
    return mask.astype(target, copy=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="sparc-masks", description="Generate superpixel masks.")
    parser.add_argument("--config", help="A configs/superpixel/*.yaml file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--paths", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing masks.")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_methods:
        print("\n".join(list_superpixel_methods()))
        return 0
    if not args.config:
        raise SystemExit("--config is required (or pass --list-methods)")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    root = load_config(args.config, args.set, paths_file=args.paths, schema=SuperpixelRoot)
    if args.print_config:
        print(OmegaConf.to_yaml(root, resolve=True))
        return 0
    cfg = root.superpixel
    if cfg.dtype not in DTYPES:
        raise SystemExit(f"dtype must be one of {sorted(DTYPES)}")

    image_root, output_root = Path(cfg.images), Path(cfg.output)
    if not image_root.is_dir():
        raise SystemExit(f"Image root does not exist: {image_root}")

    kwargs = accepted_kwargs(
        cfg.method,
        {"compactness": cfg.compactness, "sigma": cfg.sigma, "min_size_floor": cfg.min_size_floor},
    )
    images = iter_images(image_root)
    if not images:
        raise SystemExit(f"No images under {image_root}")
    if cfg.limit:
        images = images[: int(cfg.limit)]
    shard = images[args.shard_index :: args.num_shards]

    output_root.mkdir(parents=True, exist_ok=True)
    if args.shard_index == 0:
        write_meta(
            output_root,
            {
                "mode": "precomputed_npy",
                "method": cfg.method,
                "n_segments": cfg.n_segments,
                "params": kwargs,
                "dtype": cfg.dtype,
                "image_root": str(image_root),
                "num_images": len(images),
            },
        )

    print(
        f"method={cfg.method} n_segments={cfg.n_segments} params={kwargs} "
        f"shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(images)} images "
        f"-> {output_root}",
        flush=True,
    )

    done = skipped = 0
    began = time.time()
    for i, image_path in enumerate(shard, start=1):
        out = output_root / image_path.relative_to(image_root).with_suffix(".npy")
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        mask = cast_mask(compute_mask(cfg.method, rgb, cfg.n_segments, **kwargs), cfg.dtype)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a job killed mid-write never leaves a truncated
        # .npy that the resume logic would then skip as "done".
        tmp = out.with_suffix(".npy.tmp")
        # Through a file handle, not a path: np.save(path) silently appends
        # ".npy" to any name that lacks it, which would turn this into
        # "x.npy.tmp.npy" and make the rename below fail.
        with tmp.open("wb") as handle:
            np.save(handle, mask)
        os.replace(tmp, out)
        done += 1
        if i % 500 == 0:
            rate = (time.time() - began) / max(done, 1)
            print(f"  {i}/{len(shard)} ({rate:.3f}s/image)", flush=True)

    print(f"done: {done} written, {skipped} already present, {time.time() - began:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
