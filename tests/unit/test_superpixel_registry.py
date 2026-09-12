"""Tests for the superpixel registry and the `sparc-masks` generator."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sparc.cli.masks import cast_mask, main
from sparc.data.superpixel import (
    accepted_kwargs,
    compute_mask,
    list_superpixel_methods,
    load_meta,
    register_superpixel,
)


class TestRegistry:
    def test_built_ins_are_registered(self):
        assert {"slic", "felzenszwalb", "compact_watershed"} <= set(list_superpixel_methods())

    def test_unknown_method_lists_what_exists(self):
        with pytest.raises(ValueError, match="slic"):
            compute_mask("quickshift", np.zeros((8, 8, 3), np.uint8), 4)

    def test_registering_a_new_method_is_one_decorator(self):
        """The extension path documented in docs/extending.md."""

        @register_superpixel("_test_grid")
        def grid(image_rgb, n_segments, cell=4):
            h, w = image_rgb.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            return (yy // cell) * ((w + cell - 1) // cell) + xx // cell

        mask = compute_mask("_test_grid", np.zeros((8, 8, 3), np.uint8), 4, cell=4)
        assert mask.shape == (8, 8) and mask.max() == 3
        assert accepted_kwargs("_test_grid", {"cell": 2, "compactness": 9}) == {"cell": 2}

    def test_accepted_kwargs_drops_none_and_foreign_keys(self):
        assert accepted_kwargs(
            "slic", {"compactness": 10.0, "sigma": None, "min_size_floor": 5}
        ) == {"compactness": 10.0}

    def test_contract_is_enforced(self):
        @register_superpixel("_test_bad")
        def bad(image_rgb, n_segments):
            return np.zeros((2, 2), np.float32)

        with pytest.raises(ValueError, match="label map"):
            compute_mask("_test_bad", np.zeros((8, 8, 3), np.uint8), 4)


class TestCastMask:
    def test_auto_picks_the_smallest_dtype(self):
        assert cast_mask(np.arange(100, dtype=np.int64), "auto").dtype == np.uint8
        assert cast_mask(np.arange(300, dtype=np.int64), "auto").dtype == np.int16
        assert cast_mask(np.arange(70000, dtype=np.int64), "auto").dtype == np.int32

    def test_values_survive_the_cast(self):
        m = np.arange(200, dtype=np.int64).reshape(10, 20)
        assert np.array_equal(cast_mask(m, "auto"), m)

    def test_explicit_dtype_that_cannot_hold_labels_raises(self):
        with pytest.raises(ValueError, match="do not fit"):
            cast_mask(np.arange(300), "uint8")


@pytest.fixture
def image_tree(tmp_path):
    root = tmp_path / "images"
    (root / "sub").mkdir(parents=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:40, 0:48]
    for i, rel in enumerate(["a.jpg", "b.png", "sub/c.jpg", "sub/d.jpg", "e.jpg"]):
        base = ((yy * 3 + xx * 2 + 40 * i) % 256).astype(np.float64)
        img = np.stack([base, base * 0.6 + 30, base * 0.3 + 90], -1) + rng.normal(0, 4, (40, 48, 3))
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / rel)
    return root


def _run(tmp_path, image_tree, out, **extra):
    settings = {
        "superpixel.images": str(image_tree),
        "superpixel.output": str(out),
        "superpixel.n_segments": "12",
    }
    settings.update(extra)
    argv = [
        "--config",
        "configs/superpixel/slic_n100.yaml",
        "--set",
        *[f"{k}={v}" for k, v in settings.items()],
    ]
    return argv


class TestMasksCli:
    def test_mirrors_the_image_tree_and_writes_metadata(self, tmp_path, image_tree):
        out = tmp_path / "masks"
        assert main(_run(tmp_path, image_tree, out)) == 0
        assert (out / "a.npy").is_file() and (out / "sub" / "c.npy").is_file()
        meta = load_meta(out)
        assert meta["method"] == "slic" and meta["n_segments"] == 12
        assert meta["num_images"] == 5
        m = np.load(out / "a.npy")
        assert m.shape == (40, 48) and m.dtype == np.uint8 and m.min() == 0

    def test_resume_skips_existing_masks(self, tmp_path, image_tree, capsys):
        out = tmp_path / "masks"
        main(_run(tmp_path, image_tree, out))
        capsys.readouterr()
        main(_run(tmp_path, image_tree, out))
        assert "0 written, 5 already present" in capsys.readouterr().out

    def test_shards_partition_the_corpus_exactly(self, tmp_path, image_tree):
        """Every image in exactly one shard: no gaps, no double work."""
        out = tmp_path / "masks"
        for i in range(3):
            main(_run(tmp_path, image_tree, out) + ["--num-shards", "3", "--shard-index", str(i)])
        assert len(list(out.rglob("*.npy"))) == 5

    def test_limit_caps_the_corpus(self, tmp_path, image_tree):
        out = tmp_path / "masks"
        main(_run(tmp_path, image_tree, out, **{"superpixel.limit": "2"}))
        assert len(list(out.rglob("*.npy"))) == 2

    def test_regenerated_masks_are_deterministic(self, tmp_path, image_tree):
        a, b = tmp_path / "a", tmp_path / "b"
        main(_run(tmp_path, image_tree, a))
        main(_run(tmp_path, image_tree, b))
        assert np.array_equal(np.load(a / "a.npy"), np.load(b / "a.npy"))
