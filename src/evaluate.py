"""Hold-out evaluation of novel-view synthesis against real photographs.

The renderer has never been scored against anything real. This measures it the
only way that means something: withhold photographs the pipeline never sees,
synthesise the view at that angle from a photograph it *did* see, and compare.

Every synthesised view is scored twice against the same withheld photograph:

    render    the mesh reconstructed from the nearest captured photograph,
              rotated to the held-out angle
    baseline  the nearest captured photograph itself, unaltered -- what you
              would show if you simply snapped to the closest frame

The baseline column is what makes the render column mean anything. A render
that cannot beat "just show the nearest photograph" has not earned its
reconstruction stage.

Because a render and a photograph never share intrinsics, both are reduced to
their silhouette: segmented, cropped to the bounding box, and resized to a
common square. Both columns get identical treatment, so the comparison between
them is like for like even though the absolute values are conservative.

Usage
-----
    python src/evaluate.py --frames data --every 2 --depth-mode model --out output/full_e2
    python src/evaluate.py --frames data --every 4 --depth-mode model --out output/full_e4
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import psnr, ssim  # noqa: E402
from reconstruct import reconstruct_photo, segment_foreground  # noqa: E402
from render3d import load_obj, normalize_mesh, render_frame  # noqa: E402

ANGLE_NAME = re.compile(r"^(\d{1,3})$")


def discover_frames(folder: Path) -> dict[int, Path]:
    """Map turntable angle in degrees to its photograph.

    Filenames are the angle itself (``0.jpeg`` ... ``350.jpeg``), which is how
    the capture was recorded, so the angle needs no separate index file.
    """
    frames: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        match = ANGLE_NAME.match(path.stem)
        if match:
            frames[int(match.group(1))] = path
    if len(frames) < 4:
        raise SystemExit(
            f"ERROR: found {len(frames)} angle-named photographs in {folder}. "
            "Expected files named by degrees, e.g. 0.jpeg, 10.jpeg, ... 350.jpeg"
        )
    return frames


def angular_delta(from_deg: float, to_deg: float) -> float:
    """Signed shortest rotation from one angle to another, in (-180, 180]."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


def silhouette_crop(bgr: np.ndarray, size: int, mask: np.ndarray | None = None) -> np.ndarray | None:
    """Reduce an image to its object, centred in a fixed square.

    A render and a photograph share no intrinsics, so they are only comparable
    once the object is isolated and normalised for position and scale. Returns
    None when nothing was found to crop.
    """
    if mask is None:
        try:
            mask = segment_foreground(bgr)
        except (ValueError, cv2.error):
            return None
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None

    cropped = bgr[y0:y1, x0:x1]
    isolated = np.zeros_like(cropped)
    np.copyto(isolated, cropped, where=mask[y0:y1, x0:x1, None])

    # Pad to square before resizing so the aspect ratio is never distorted.
    height, width = isolated.shape[:2]
    side = max(height, width)
    square = np.zeros((side, side, 3), dtype=isolated.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = isolated
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def render_at(mesh: tuple, delta_deg: float, pitch: float, size: int) -> np.ndarray:
    vertices, uvs, faces, texture = mesh
    frame, _ = render_frame(
        vertices, uvs, faces, texture,
        width=size * 2, height=size * 2, yaw=delta_deg, pitch=pitch,
    )
    return frame


def build_mesh_for(path: Path, cache_dir: Path, depth_mode: str, relief: float, grid: int) -> tuple | None:
    """Reconstruct one photograph and load the mesh, caching by frame name."""
    base = cache_dir / path.stem
    if not base.with_suffix(".obj").exists():
        try:
            reconstruct_photo(
                path, base, depth_mode=depth_mode, relief=relief, grid=grid
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"  ! reconstruction failed for {path.name}: {exc}")
            return None
    vertices, uvs, faces, texture_path = load_obj(base.with_suffix(".obj"))
    texture = cv2.imread(str(texture_path)) if texture_path else None
    return normalize_mesh(vertices), uvs, faces, texture


def score_pair(reference: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    return psnr(reference, test), ssim(reference, test)


def calibrate_yaw_sign(cases, meshes, pitch, size, frames) -> float:
    """Decide which rotation direction matches the turntable, by measuring.

    The turntable turns the object; the renderer turns the mesh. Which sign
    lines the two up is a convention, and this project has already been bitten
    once by assuming one. So try both on a few pairs and keep the better.
    """
    scores = {}
    sample = cases[: min(4, len(cases))]
    for sign in (1.0, -1.0):
        total = []
        for held, captured in sample:
            mesh = meshes.get(captured)
            truth = silhouette_crop(cv2.imread(str(frames[held])), size)
            if mesh is None or truth is None:
                continue
            delta = sign * angular_delta(captured, held)
            crop = silhouette_crop(render_at(mesh, delta, pitch, size), size)
            if crop is None:
                continue
            total.append(psnr(truth, crop))
        scores[sign] = float(np.mean(total)) if total else -1.0
    best = max(scores, key=scores.get)
    print(
        f"yaw sign calibration: +1 -> {scores[1.0]:.2f} dB, "
        f"-1 -> {scores[-1.0]:.2f} dB; using {best:+.0f}"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=Path("data"), help="folder of angle-named photographs")
    parser.add_argument("--every", type=int, default=2, help="keep every Nth frame as the captured set")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="angles to drop entirely")
    parser.add_argument("--out", type=Path, default=Path("output/metrics"))
    parser.add_argument("--depth-mode", choices=("shape", "model"), default="shape")
    parser.add_argument("--relief", type=float, default=0.42)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--size", type=int, default=256, help="square side both images are reduced to")
    args = parser.parse_args()

    frames = discover_frames(args.frames)
    for angle in args.exclude:
        if frames.pop(angle, None) is not None:
            print(f"excluded {angle} deg")

    angles = sorted(frames)
    captured = angles[:: args.every]
    held_out = [a for a in angles if a not in set(captured)]
    if not held_out:
        raise SystemExit("ERROR: --every left nothing held out; nothing to score against")

    print(f"{len(angles)} photographs: {len(captured)} captured, {len(held_out)} held out")
    print(f"captured spacing: {360 / max(len(captured), 1):.1f} deg\n")

    cache = args.out / "meshes"
    cache.mkdir(parents=True, exist_ok=True)

    print("reconstructing the captured set...")
    started = time.perf_counter()
    meshes = {}
    for angle in captured:
        mesh = build_mesh_for(frames[angle], cache, args.depth_mode, args.relief, args.grid)
        if mesh is not None:
            meshes[angle] = mesh
    recon_seconds = time.perf_counter() - started
    print(f"  {len(meshes)}/{len(captured)} meshes in {recon_seconds:.1f}s\n")

    cases = []
    for held in held_out:
        nearest = min(meshes, key=lambda c: abs(angular_delta(c, held))) if meshes else None
        if nearest is not None:
            cases.append((held, nearest))
    if not cases:
        raise SystemExit("ERROR: no held-out frame had a usable reconstruction to compare against")

    sign = calibrate_yaw_sign(cases, meshes, args.pitch, args.size, frames)

    rows = []
    started = time.perf_counter()
    for held, nearest in cases:
        truth_bgr = cv2.imread(str(frames[held]))
        truth = silhouette_crop(truth_bgr, args.size)
        base = silhouette_crop(cv2.imread(str(frames[nearest])), args.size)
        if truth is None or base is None:
            print(f"  ! skipped {held} deg: could not isolate the object")
            continue

        delta = angular_delta(nearest, held)
        render = silhouette_crop(render_at(meshes[nearest], sign * delta, args.pitch, args.size), args.size)
        if render is None:
            print(f"  ! skipped {held} deg: render produced nothing to score")
            continue

        p_r, s_r = score_pair(truth, render)
        p_b, s_b = score_pair(truth, base)
        rows.append({
            "held_out_deg": held,
            "captured_deg": nearest,
            "delta_deg": round(delta, 1),
            "psnr_render_db": round(p_r, 4),
            "ssim_render": round(s_r, 5),
            "psnr_baseline_db": round(p_b, 4),
            "ssim_baseline": round(s_b, 5),
        })
        print(f"  {held:>3d} deg  from {nearest:>3d} ({delta:+.0f})   "
              f"render {p_r:6.2f} dB / {s_r:.3f}    baseline {p_b:6.2f} dB / {s_b:.3f}")
    render_seconds = time.perf_counter() - started

    if not rows:
        raise SystemExit("ERROR: nothing could be scored")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def column(key):
        return np.array([r[key] for r in rows], dtype=float)

    pr, pb = column("psnr_render_db"), column("psnr_baseline_db")
    sr, sb = column("ssim_render"), column("ssim_baseline")
    summary = {
        "n_scored": len(rows),
        "n_captured": len(meshes),
        "captured_spacing_deg": round(360 / max(len(meshes), 1), 2),
        "depth_mode": args.depth_mode,
        "psnr_render_mean_db": round(float(pr.mean()), 4),
        "psnr_render_median_db": round(float(np.median(pr)), 4),
        "ssim_render_mean": round(float(sr.mean()), 5),
        "psnr_baseline_mean_db": round(float(pb.mean()), 4),
        "psnr_baseline_median_db": round(float(np.median(pb)), 4),
        "ssim_baseline_mean": round(float(sb.mean()), 5),
        "psnr_gain_db": round(float(pr.mean() - pb.mean()), 4),
        "render_wins_psnr": int((pr > pb).sum()),
        "render_wins_ssim": int((sr > sb).sum()),
        "reconstruct_seconds": round(recon_seconds, 2),
        "render_score_seconds": round(render_seconds, 2),
    }
    summary_path = args.out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"\n{'':<12}{'PSNR (dB)':>12}{'SSIM':>10}")
    print(f"{'render':<12}{summary['psnr_render_mean_db']:>12.2f}{summary['ssim_render_mean']:>10.3f}")
    print(f"{'baseline':<12}{summary['psnr_baseline_mean_db']:>12.2f}{summary['ssim_baseline_mean']:>10.3f}")
    print(f"{'gain':<12}{summary['psnr_gain_db']:>+12.2f}"
          f"{summary['ssim_render_mean'] - summary['ssim_baseline_mean']:>+10.3f}")
    print(f"\nrender beats baseline at {summary['render_wins_psnr']}/{len(rows)} positions on PSNR, "
          f"{summary['render_wins_ssim']}/{len(rows)} on SSIM")

    # A difference of two means says nothing without its error bar. Each position
    # carries both scores, so the paired difference is the right statistic -- and
    # it is what stops a 0.03 dB gain being written up as a crossover.
    for label, render_col, baseline_col in (
        ("PSNR", "psnr_render_db", "psnr_baseline_db"),
        ("SSIM", "ssim_render", "ssim_baseline"),
    ):
        diff = column(render_col) - column(baseline_col)
        stderr = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("inf")
        margin = 1.96 * stderr
        verdict = "significant" if abs(diff.mean()) > margin else "NOT significant (interval spans zero)"
        print(f"  paired d{label}: {diff.mean():+.4f}  95% CI [{diff.mean()-margin:+.4f}, "
              f"{diff.mean()+margin:+.4f}]  {verdict}")
    if args.depth_mode == "shape":
        print("\nNOTE: depth is the synthetic shape proxy, not a learned prediction.")
        print("      These numbers measure segmentation, meshing and rendering only.")
    print(f"\nwrote {metrics_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
