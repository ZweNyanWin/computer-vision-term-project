"""Pre-flight check on a neural-capture photo folder.

Reads every image once and answers the two questions that decide whether the
shoot is usable, both of which are invisible without EXIF:

  1. Is this ONE camera?  MLX3D runs COLMAP with --ImageReader.single_camera 1
     (capture/colmap_wrap.py:71-86), so every frame must share pixel dimensions,
     focal length and aperture. A mixed set does not error - it silently
     rescales intrinsics per image and corrupts the reconstruction.

  2. Was exposure actually locked?  SIFT survives exposure drift, so COLMAP will
     report a healthy registration rate either way. Gaussian Splatting fits a
     photometric loss, so drift is baked in as haze.

Also catches the file-level traps: HEIC (excluded by MLX3D's extension
whitelist, so those frames vanish without a warning), AppleDouble stubs, and
motion blur.

Run with the MLX3D venv, which has both Pillow and OpenCV:

    .venv-mlx3d/bin/python tools/check_capture.py data/neural_capture/all
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import cv2
import numpy as np
from PIL import Image, ExifTags

# MLX3D's own whitelist, from capture/pipeline.py. Anything outside it is
# skipped silently by mlx3d-capture, so we flag it loudly here instead.
ACCEPTED = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
SILENTLY_DROPPED = {".heic", ".heif", ".dng", ".raw", ".mov", ".aae"}

# Below this variance-of-Laplacian a 1024-px-wide frame is soft enough that the
# wood grain SIFT relies on is gone. Calibrated against the existing ring, whose
# sharpest frames score in the hundreds.
BLUR_FLOOR = 60.0


def _exif(path: str) -> dict:
    """Flatten the main IFD and the Exif sub-IFD into one tag->value dict."""
    with Image.open(path) as im:
        size = im.size
        raw = im.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        try:
            sub = raw.get_ifd(0x8769)
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
        except Exception:
            pass
    return size, tags


def _blur_score(path: str) -> float:
    """Variance of the Laplacian, measured at a fixed width so the number is
    comparable between frames regardless of capture resolution."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return float("nan")
    h, w = img.shape
    if w != 1024:
        img = cv2.resize(img, (1024, max(1, round(h * 1024 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="directory of capture originals")
    ap.add_argument("--blur-floor", type=float, default=BLUR_FLOOR)
    ap.add_argument("--no-blur", action="store_true", help="skip the blur pass (much faster)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f"error: {args.folder} is not a directory")
        return 2

    names = sorted(os.listdir(args.folder))
    usable, dropped, applestubs = [], [], []
    for n in names:
        if n.startswith("._"):
            applestubs.append(n)
            continue
        ext = os.path.splitext(n)[1].lower()
        if ext in ACCEPTED:
            usable.append(n)
        elif ext in SILENTLY_DROPPED:
            dropped.append(n)

    problems = []

    print(f"folder            {args.folder}")
    print(f"usable images     {len(usable)}")
    if len(usable) < 3:
        problems.append(f"MLX3D needs at least 3 usable images; found {len(usable)}.")

    if dropped:
        problems.append(
            f"{len(dropped)} file(s) MLX3D will SKIP WITHOUT WARNING "
            f"(e.g. {dropped[0]}). Re-export as JPEG - set Formats to Most Compatible."
        )
    if applestubs:
        problems.append(
            f"{len(applestubs)} AppleDouble '._*' stub(s) present. Remove with: dot_clean '{args.folder}'"
        )

    sizes = collections.Counter()
    focal = collections.Counter()
    eq35 = collections.Counter()
    fnum = collections.Counter()
    lens = collections.Counter()
    iso, shutter = [], []

    for n in usable:
        p = os.path.join(args.folder, n)
        try:
            size, t = _exif(p)
        except Exception as exc:
            problems.append(f"{n}: unreadable ({exc})")
            continue
        sizes[size] += 1
        focal[round(float(t["FocalLength"]), 3) if "FocalLength" in t else None] += 1
        eq35[t.get("FocalLengthIn35mmFilm")] += 1
        fnum[round(float(t["FNumber"]), 2) if "FNumber" in t else None] += 1
        lens[t.get("LensModel")] += 1
        if "ISOSpeedRatings" in t:
            iso.append(t["ISOSpeedRatings"])
        if "ExposureTime" in t:
            shutter.append(float(t["ExposureTime"]))

    def report(label: str, counter: collections.Counter, fatal: bool) -> None:
        print(f"{label:<18}{len(counter)} distinct")
        for value, count in counter.most_common():
            print(f"                  {count:>4} x {value}")
        if len(counter) > 1:
            msg = f"{label.strip()} is not constant across the set ({len(counter)} values)."
            if fatal:
                problems.append(
                    msg + " COLMAP runs with --single_camera 1, so mixed values corrupt intrinsics."
                )
            else:
                problems.append(msg)

    print()
    report("pixel size", sizes, fatal=True)
    report("focal length", focal, fatal=True)
    report("35mm-equivalent", eq35, fatal=True)
    report("aperture", fnum, fatal=True)
    report("lens", lens, fatal=True)

    print()
    if iso:
        print(f"ISO               {min(iso)}-{max(iso)}")
        if len(set(iso)) > 1:
            problems.append(
                f"ISO varies ({min(iso)}-{max(iso)}); exposure was not locked. "
                "Survivable for COLMAP, but photometric drift is baked into the splat."
            )
    if shutter:
        print(f"shutter           1/{1/max(shutter):.0f} - 1/{1/min(shutter):.0f} s")
        if len(set(shutter)) > 1:
            problems.append(f"shutter varies across {len(set(shutter))} values; exposure was not locked.")
        if max(shutter) > 1 / 80:
            problems.append(
                f"slowest frame is 1/{1/max(shutter):.0f} s, below the 1/80 s handheld floor. "
                "Blur is unrecoverable; add light."
            )

    if not args.no_blur and usable:
        print()
        scores = []
        for n in usable:
            scores.append((_blur_score(os.path.join(args.folder, n)), n))
        scores.sort()
        finite = [s for s, _ in scores if np.isfinite(s)]
        if finite:
            print(f"sharpness         median {np.median(finite):.0f}, worst {finite[0]:.0f}")
            soft = [(s, n) for s, n in scores if np.isfinite(s) and s < args.blur_floor]
            if soft:
                problems.append(
                    f"{len(soft)} frame(s) below the sharpness floor ({args.blur_floor:.0f}): "
                    + ", ".join(n for _, n in soft[:8])
                    + ("..." if len(soft) > 8 else "")
                    + ". Re-shoot each from the same position; do not move the frog."
                )

    print()
    if problems:
        print(f"NOT READY - {len(problems)} problem(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("READY - one camera, locked exposure, no soft frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
