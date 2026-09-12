# tools

Standalone scripts that are not part of the library API.

| Script | Purpose |
|---|---|
| `visualize_region_alignment.py` | Two augmented views with superpixel boundaries, and the regions matched across them painted in colour. The region loss's mechanism, made visible; also the quickest way to spot an image/mask misalignment. |

Everything that produces a number in the paper lives in the package instead
(`sparc-report`), so it is tested and versioned with the code.
