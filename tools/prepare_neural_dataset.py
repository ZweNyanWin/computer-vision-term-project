"""Normalise the four capture rings into one flat, COLMAP-ready directory.

The four shoot folders use overlapping filenames (``10.jpeg`` exists in three of
them) and a few malformed ones (``12jpeg.jpeg``), so they cannot simply be
concatenated. This writes ``data/neural_capture/all/<ring>_<NNN>.jpg`` as
*relative symlinks* onto the untouched originals: nothing is copied, cropped,
re-encoded or moved, so EXIF and pixels stay bit-identical and the shoot folders
remain the only source of truth.

Naming is deliberate, not cosmetic:

  * ``<ring>_`` prefixes disambiguate the collisions and keep each ring
    contiguous in sorted order.
  * ``high_`` sorts first, and the high ring is entirely main-lens (5712 px).
    MLX3D derives ONE integer downscale factor for the whole set from the first
    image in sorted order (``capture/pipeline.py:_auto_downscale``), so putting
    the largest sensor first is what keeps every training image at or under the
    preset's ``max_dim``. Sorting a 4032 px frame first would silently push the
    5712 px frames over it.

A manifest recording the SHA-256 of every original is written alongside, so the
dataset a result was computed from can be proved later.

    .venv-mlx3d/bin/python tools/prepare_neural_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

from PIL import Image, ExifTags

# Shoot folder -> short ring tag. Order here is documentation only; the output
# is sorted by tag so `high` leads regardless.
RINGS = {
    "High ring": "high",
    "Low ring": "low",
    "Middle ring": "mid",
    "Top": "top",
}

ACCEPTED = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


def natural_key(name: str):
    """Sort ``2.jpeg`` before ``10.jpeg`` and cope with ``12jpeg.jpeg``."""
    stem = Path(name).stem
    digits = re.findall(r"\d+", stem)
    return (int(digits[0]) if digits else 1 << 30, stem)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_of(path: Path) -> tuple[tuple[int, int], dict]:
    with Image.open(path) as im:
        size = im.size
        raw = im.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        try:
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in raw.get_ifd(0x8769).items()})
        except Exception:
            pass
    return size, tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="capture root holding the four ring folders")
    ap.add_argument("--out", default="data/neural_capture/all")
    ap.add_argument("--manifest", default="config/neural_manifest.csv")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    missing = [d for d in RINGS if not (data / d).is_dir()]
    if missing:
        print(f"error: missing ring folder(s): {', '.join(missing)}")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        if stale.is_symlink() or stale.is_file():
            stale.unlink()

    rows = []
    for folder, tag in sorted(RINGS.items(), key=lambda kv: kv[1]):
        src_dir = data / folder
        names = [
            n.name
            for n in src_dir.iterdir()
            if n.is_file() and not n.name.startswith("._") and n.suffix.lower() in ACCEPTED
        ]
        for i, name in enumerate(sorted(names, key=natural_key), start=1):
            src = src_dir / name
            dst = out / f"{tag}_{i:03d}.jpg"
            # Relative link, so moving the whole project does not break it.
            os.symlink(os.path.relpath(src, out), dst)
            size, t = exif_of(src)
            rows.append(
                {
                    "name": dst.name,
                    "ring": tag,
                    "index": i,
                    "original": str(src.relative_to(data.parent)),
                    "width": size[0],
                    "height": size[1],
                    "focal_mm": round(float(t["FocalLength"]), 4) if "FocalLength" in t else "",
                    "f_number": round(float(t["FNumber"]), 3) if "FNumber" in t else "",
                    "lens": t.get("LensModel", ""),
                    "iso": t.get("ISOSpeedRatings", ""),
                    "exposure_s": float(t["ExposureTime"]) if "ExposureTime" in t else "",
                    "sha256": sha256(src),
                }
            )

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    first = sorted(r["name"] for r in rows)[0]
    first_row = next(r for r in rows if r["name"] == first)
    print(f"linked            {len(rows)} images -> {out}")
    for tag in sorted({r['ring'] for r in rows}):
        n = sum(1 for r in rows if r["ring"] == tag)
        print(f"  {tag:<5}           {n}")
    print(f"manifest          {manifest}")
    print(f"first in sort     {first}  {first_row['width']}x{first_row['height']}  f={first_row['focal_mm']}mm")
    lenses = {(r["width"], r["height"], r["focal_mm"]) for r in rows}
    print(f"distinct optics   {len(lenses)}")
    for w_, h_, fmm in sorted(lenses, reverse=True):
        n = sum(1 for r in rows if (r["width"], r["height"], r["focal_mm"]) == (w_, h_, fmm))
        print(f"                  {n:>3} x {w_}x{h_} f={fmm}mm")
    if len(lenses) > 1:
        print(
            "\nNOTE: more than one physical lens is present, so this set must NOT be\n"
            "reconstructed with --ImageReader.single_camera 1. Run COLMAP with\n"
            "single_camera 0 (one camera per optic) and hand MLX3D the result via\n"
            "--poses existing."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
