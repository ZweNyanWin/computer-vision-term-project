"""Render a turntable orbit of the Gaussian splat for the workshop station.

`mlx3d-render` takes a look-at camera but no way to say "orbit the object that
this reconstruction is actually of", because the splat lives in COLMAP's
arbitrary world frame: there is no canonical up, no centre and no scale. All
three are recovered here the same way `select_holdout.py` recovers them - object
centre from the median sparse point, orbit axis from the pooled within-ring
scatter of the camera directions - so the orbit matches the one the photographer
actually walked rather than an axis guessed from the coordinate system.

The fitted checkpoint contains the photographed room as well as the frog.  For
the object-only workshop view, this script keeps the dense connected cluster
near the recovered object centre, rejects pale background and very faint
Gaussians, and composites the result on black.  This is presentation-only
foreground isolation: the original checkpoint and all reported held-out
measurements remain untouched.

    .venv-mlx3d/bin/python tools/render_orbit.py \\
        --splat model3d/gaussian/frog88_train_balanced/splat.ply \\
        --model model3d/gaussian/frog88_undist --out workshop/frames/splat
"""

from __future__ import annotations

import argparse
from collections import deque
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_holdout import spherical  # noqa: E402  (same axis fit, one definition)


SH_C0 = 0.28209479177387814


def _largest_voxel_component(
    means: np.ndarray, candidates: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Return candidate rows in the largest 26-connected occupied voxel set.

    The frog is a dense connected cloud.  Room floaters that survive the
    distance/colour tests form much smaller islands, so this removes them
    without inventing an image-space silhouette or modifying any Gaussian.
    """
    if candidates.size == 0:
        return candidates
    keys = np.floor(means[candidates] / voxel_size).astype(np.int32)
    unseen = {tuple(row) for row in keys}
    largest: set[tuple[int, int, int]] = set()
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            x, y, z = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        neighbour = (x + dx, y + dy, z + dz)
                        if neighbour in unseen:
                            unseen.remove(neighbour)
                            component.add(neighbour)
                            queue.append(neighbour)
        if len(component) > len(largest):
            largest = component
    in_largest = np.fromiter(
        (tuple(row) in largest for row in keys), dtype=bool, count=len(keys)
    )
    return candidates[in_largest]


def foreground_indices(
    means: np.ndarray,
    opacity_logits: np.ndarray,
    sh_dc: np.ndarray,
    centre: np.ndarray,
    *,
    radius: float,
    min_opacity: float,
    min_saturation: int,
    dark_value: int,
    voxel_size: float,
    finite: np.ndarray | None = None,
    up: np.ndarray | None = None,
    floor_margin: float = 0.02,
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    """Select the local, opaque, wood/dark, connected object Gaussians."""
    n = len(means)
    if finite is None:
        finite = (
            np.isfinite(means).all(axis=1)
            & np.isfinite(opacity_logits)
            & np.isfinite(sh_dc).reshape(n, -1).all(axis=1)
        )

    spatial = finite & (np.linalg.norm(means - centre, axis=1) < radius)
    # Stable sigmoid of the raw 3DGS opacity logits.
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -80.0, 80.0)))
    opaque = spatial & (opacity >= min_opacity)

    # The zeroth-order spherical-harmonic coefficient is the view-independent
    # base colour.  Keep saturated wood OR dark details (notably the black
    # eyes/mouth); reject only bright, near-neutral platform/background splats.
    rgb = np.clip(0.5 + SH_C0 * sh_dc.reshape(n, 3), 0.0, 1.0)
    hsv = cv2.cvtColor(
        (rgb[:, None, :] * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2HSV
    )[:, 0, :]
    appearance = (hsv[:, 1] >= min_saturation) | (hsv[:, 2] <= dark_value)

    # A support surface is the awkward case for connected components: it
    # physically touches the frog.  Detect its horizontal band from rejected
    # pale Gaussians below the object centre, then clip just above that plane.
    # On scenes without a sufficiently strong pale band, no floor cut is made.
    floor_height: float | None = None
    above_floor = np.ones(n, dtype=bool)
    if up is not None:
        height = (means - centre) @ up
        pale_below = opaque & ~appearance & (height < -0.2 * radius)
        pale_heights = height[pale_below]
        if pale_heights.size >= 100:
            edges = np.arange(-radius, 0.0 + 0.0101, 0.01)
            counts, edges = np.histogram(pale_heights, bins=edges)
            peak = int(np.argmax(counts))
            if counts[peak] >= 25:
                floor_height = float((edges[peak] + edges[peak + 1]) / 2.0)
                above_floor = height > floor_height + floor_margin
    coloured = opaque & appearance & above_floor

    candidates = np.flatnonzero(coloured)
    connected = _largest_voxel_component(means, candidates, voxel_size)
    stats = {
        "total": n,
        "finite": int(finite.sum()),
        "spatial": int(spatial.sum()),
        "opaque": int(opaque.sum()),
        "appearance": int(coloured.sum()),
        "connected": int(connected.size),
        "floor_height": floor_height,
    }
    return connected, stats


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
    ap.add_argument(
        "--object-radius", type=float, default=1.0,
        help="world-space radius used both for framing and foreground isolation",
    )
    ap.add_argument(
        "--min-opacity", type=float, default=0.02,
        help="discard weaker translucent haze (activated opacity, 0..1)",
    )
    ap.add_argument(
        "--min-saturation", type=int, default=25,
        help="keep bright Gaussians at or above this HSV saturation (0..255)",
    )
    ap.add_argument(
        "--dark-value", type=int, default=160,
        help="also keep dark neutral details at or below this HSV value (0..255)",
    )
    ap.add_argument(
        "--voxel-size", type=float, default=0.025,
        help="world-space cell size for removing disconnected floaters",
    )
    ap.add_argument(
        "--floor-margin", type=float, default=0.02,
        help="clip this far above an automatically detected pale support plane",
    )
    ap.add_argument("--fill", type=float, default=0.82, help="fraction of the frame the object should span")
    ap.add_argument("--render-scale", type=float, default=2.0, help="supersample before the crop")
    args = ap.parse_args()

    if args.object_radius <= 0:
        ap.error("--object-radius must be positive")
    if not 0 <= args.min_opacity <= 1:
        ap.error("--min-opacity must be in [0, 1]")
    if not 0 <= args.min_saturation <= 255:
        ap.error("--min-saturation must be in [0, 255]")
    if not 0 <= args.dark_value <= 255:
        ap.error("--dark-value must be in [0, 255]")
    if args.voxel_size <= 0:
        ap.error("--voxel-size must be positive")
    if args.floor_margin < 0:
        ap.error("--floor-margin must be non-negative")

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
    if obj_pts.size == 0:
        raise ValueError("object radius contains no sparse points; increase --object-radius")
    # Radius of the object itself, robustly: the 95th percentile keeps a few
    # stray triangulations from widening the shot.
    r_obj = float(np.percentile(np.linalg.norm(obj_pts - obj, axis=1), 95))

    model = GaussianModel.load_ply(args.splat)
    n0 = model.num_gaussians
    finite = np.ones(n0, dtype=bool)
    for v in model.params.values():
        finite &= np.isfinite(np.asarray(v).reshape(n0, -1)).all(axis=1)
    keep, isolation = foreground_indices(
        np.asarray(model.params["means"]),
        np.asarray(model.params["opacities"]),
        np.asarray(model.params["sh_dc"]),
        obj,
        radius=args.object_radius,
        min_opacity=args.min_opacity,
        min_saturation=args.min_saturation,
        dark_value=args.dark_value,
        voxel_size=args.voxel_size,
        finite=finite,
        up=up,
        floor_margin=args.floor_margin,
    )
    if keep.size == 0:
        raise ValueError("foreground isolation removed every Gaussian")
    model.select(keep)
    print(
        "foreground isolation: "
        f"{isolation['total']:,} total -> {isolation['spatial']:,} local -> "
        f"{isolation['opaque']:,} opaque -> {isolation['appearance']:,} wood/dark -> "
        f"{isolation['connected']:,} connected"
    )
    if isolation["floor_height"] is not None:
        print(
            f"support plane: height {isolation['floor_height']:+.3f}; "
            f"clipped at {isolation['floor_height'] + args.floor_margin:+.3f}"
        )

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
    background = mx.zeros((3,), dtype=mx.float32)

    for i in range(args.frames):
        az = 2 * np.pi * i / args.frames
        eye = obj + radius * (np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up)
        R, t = look_at(eye, obj, up)
        cam = Camera(
            R=mx.array(R.astype(np.float32)), t=mx.array(t.astype(np.float32)),
            fx=f, fy=f, cx=big / 2, cy=big / 2, width=big, height=big,
        )
        img = model.render(cam, background=background)["image"]
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
