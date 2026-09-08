"""Render a turntable orbit of the Gaussian splat for the workshop station.

`mlx3d-render` takes a look-at camera but no way to say "orbit the object that
this reconstruction is actually of", because the splat lives in COLMAP's
arbitrary world frame: there is no canonical up, no centre and no scale. All
three are recovered here the same way `select_holdout.py` recovers them - object
centre from the median sparse point, orbit axis from the pooled within-ring
scatter of the camera directions - so the orbit matches the one the photographer
actually walked rather than an axis guessed from the coordinate system.

Frames are composed tightly around the object. The reconstruction includes the
room it was shot in, so the committed frame set must be visually checked before
publication; the object-centric composition is presentation only and changes no
reported evaluation measurement.

    .venv-mlx3d/bin/python tools/render_orbit.py \\
        --splat model3d/gaussian/frog88_train_balanced/splat.ply \\
        --model model3d/gaussian/frog88_undist --out workshop/frames/splat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_holdout import spherical  # noqa: E402  (same axis fit, one definition)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray):
    """World-to-camera (R, t) in OpenCV convention: +Z forward, +Y down."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-8:  # looking straight along the axis; any perpendicular will do
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(right)
    right /= n
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ eye


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat", required=True)
    ap.add_argument("--model", required=True, help="COLMAP workspace giving the scene's frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--elevation", type=float, default=32.0, help="degrees above the object's horizon")
    ap.add_argument("--object-radius", type=float, default=1.0)
    ap.add_argument("--fill", type=float, default=0.82, help="fraction of the frame the object should span")
    ap.add_argument("--render-scale", type=float, default=2.0, help="supersample before the crop")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx3d.cameras import Camera
    from mlx3d.datasets import load_colmap
    from mlx3d.splatting import GaussianModel

    ds = load_colmap(args.model, images_dir="unused", load_images=False)
    centers = np.stack([np.asarray(c.camera_center) for c in ds.cameras])
    rings = [n.split("_")[0] for n in ds.image_names]
    _, _, obj, up, _ = spherical(centers, np.asarray(ds.points), rings)
    radius = float(np.median(np.linalg.norm(centers - obj, axis=1)))

    pts = np.asarray(ds.points)
    obj_pts = pts[np.linalg.norm(pts - obj, axis=1) < args.object_radius]
    # Radius of the object itself, robustly: the 95th percentile keeps a few
    # stray triangulations from widening the shot.
    r_obj = float(np.percentile(np.linalg.norm(obj_pts - obj, axis=1), 95))

    model = GaussianModel.load_ply(args.splat)
    n0 = model.num_gaussians
    finite = np.ones(n0, dtype=bool)
    for v in model.params.values():
        finite &= np.isfinite(np.asarray(v).reshape(n0, -1)).all(axis=1)
    if (~finite).sum():
        model.select(np.where(finite)[0])
        print(f"dropped {int((~finite).sum())} non-finite Gaussian(s)")

    # Orbit basis perpendicular to the recovered axis.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    big = int(args.size * args.render_scale)
    # Frame the object: half-angle it subtends from the orbit, mapped to `fill`
    # of the frame. Solved on the supersampled canvas, then cropped down.
    f = (args.fill * big / 2) / (r_obj / radius)
    el = np.radians(args.elevation)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("f_*.jpg"):
        old.unlink()

    print(f"object centre {np.round(obj, 3)}  orbit radius {radius:.2f}  object radius {r_obj:.2f}")
    print(f"rendering {args.frames} frames at {big}x{big} -> crop {args.size}x{args.size}")

    for i in range(args.frames):
        az = 2 * np.pi * i / args.frames
        eye = obj + radius * (np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up)
        R, t = look_at(eye, obj, up)
        cam = Camera(
            R=mx.array(R.astype(np.float32)), t=mx.array(t.astype(np.float32)),
            fx=f, fy=f, cx=big / 2, cy=big / 2, width=big, height=big,
        )
        img = model.render(cam)["image"]
        mx.eval(img)
        rgb = (np.clip(np.asarray(img), 0, 1) * 255 + 0.5).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (args.size, args.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"f_{i:03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  frame {i + 1}/{args.frames}  azimuth {np.degrees(az):5.0f}", end="\r")
    print(f"\nwrote {args.frames} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
