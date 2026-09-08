"""Convert the workshop frames to JPEG and measure what each angle actually shows.

The station's central claim - that one photograph gives you a relief and 45 give
you a closed surface - is not made in prose, it is shown as a number that changes
as the visitor turns the frog: how many triangles survive back-face culling at
this angle. Those counts come from `render3d.py`, the same renderer that drew the
frames, so the station cannot quote a figure its own pictures disagree with.

The counts are written to `workshop/frames/visibility.json` AND inlined into
`workshop/index.html`, because the station is opened from `file://`, where a
browser will not fetch a sibling JSON file. Re-running this keeps the two copies
in step; `tools/check_workshop.py` fails if they ever drift.

    .venv-depth/bin/python tools/measure_visibility.py
"""

from __future__ import annotations

import glob
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render3d import load_obj, normalize_mesh, render_frame  # noqa: E402

DEFAULT_FRAMES = 36
DEFAULT_PITCH = -5.0
MESHES = {"relief": "model3d/frog_real.obj", "mesh": "model3d/frog_combined.obj"}
JPEG_QUALITY = 88


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument(
        "--pitch", type=float, default=DEFAULT_PITCH,
        help="mesh pitch used by render3d.py; the measurement must match it",
    )
    args = ap.parse_args()
    if args.frames < 2:
        ap.error("--frames must be at least 2")

    for tag in ("relief", "mesh", "splat"):
        d = ROOT / "workshop" / "frames" / tag
        for p in sorted(glob.glob(str(d / "f_*.png"))):
            img = cv2.imread(p)
            if img is None:
                print(f"error: could not read {p}")
                return 2
            cv2.imwrite(p.replace(".png", ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            os.remove(p)
        n = len(glob.glob(str(d / "f_*.jpg")))
        print(f"  {tag:<7} {n} frames")
        if n != args.frames:
            print(f"error: expected {args.frames} frames in {d}, found {n}")
            return 2

    counts = {}
    for tag, obj in MESHES.items():
        v, uv, f, tp = load_obj(ROOT / obj)
        V = normalize_mesh(v)
        tex = cv2.imread(str(tp)) if tp else None
        visible = []
        for i in range(args.frames):
            # The frames are rendered as `--frames 37 --sweep 360` with the
            # duplicated +180 endpoint discarded (see run_workshop.sh), which
            # makes the schedule exactly -180 + 10i. Asking render3d.py for 36
            # frames instead would step 10.2857 degrees, and this count would
            # then belong to a different angle than the frame it labels.
            yaw = -180.0 + i * (360.0 / args.frames)
            _, n_vis = render_frame(
                V, uv, f, tex, width=128, height=128, yaw=yaw, pitch=args.pitch
            )
            visible.append(int(n_vis))
        counts[tag] = {"total_triangles": len(f), "visible": visible}
        print(f"  {tag:<7} {len(f):>6} triangles, visible {min(visible)}-{max(visible)}")

    out = ROOT / "workshop" / "frames" / "visibility.json"
    out.write_text(json.dumps(counts, indent=1) + "\n")
    print(f"wrote {out}")

    index = ROOT / "workshop" / "index.html"
    html = index.read_text()
    compact = json.dumps(counts, separators=(",", ":"))
    import re

    new, n_sub = re.subn(r"var vis = \{.*?\};", "var vis = " + compact + ";", html, count=1, flags=re.S)
    if n_sub != 1:
        print("error: could not find the inlined `var vis = {...};` in index.html")
        return 2
    index.write_text(new)
    print(f"inlined the same counts into {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
