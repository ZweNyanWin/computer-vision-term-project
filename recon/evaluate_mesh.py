"""Score one reconstructed mesh against the turntable photographs.

`src/evaluate.py` rebuilds a relief per captured frame. A photogrammetry mesh is
the opposite: one model covering every angle. So the protocol differs in exactly
one place -- there is nothing to hold out, because the mesh already saw all the
photographs during reconstruction. That makes this a *fit* measurement, not a
generalisation one, and it must be labelled as such: it is not comparable to the
held-out relief scores as though both were predictions of unseen views.

What it does establish is the thing the relief cannot do at all -- render a
plausible view from anywhere on the ring, including the far side.

Object Capture chooses its own world orientation, so the yaw that maps mesh
angle onto turntable angle is unknown. It is found by measurement, over a coarse
sweep, and reported. Same principle as everywhere else in this project: do not
assume a convention that can be measured.

    python recon/evaluate_mesh.py model3d/frog_mv.obj --out output/mesh_metrics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import discover_frames, silhouette_crop  # noqa: E402
from metrics import psnr, ssim  # noqa: E402
from render3d import load_obj, normalize_mesh, render_frame  # noqa: E402


def render(mesh, yaw: float, pitch: float, size: int) -> np.ndarray:
    vertices, uvs, faces, texture = mesh
    frame, _ = render_frame(
        vertices, uvs, faces, texture,
        width=size * 2, height=size * 2, yaw=yaw, pitch=pitch,
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--frames", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--exclude", type=int, nargs="*", default=[300])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output/mesh_metrics")
    parser.add_argument("--pitch", type=float, default=-15.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--coarse-step", type=float, default=15.0)
    args = parser.parse_args()

    frames = discover_frames(args.frames)
    for angle in args.exclude:
        frames.pop(angle, None)
    angles = sorted(frames)
    print(f"{len(angles)} photographs\n")

    vertices, uvs, faces, texture_path = load_obj(args.mesh)
    texture = cv2.imread(str(texture_path)) if texture_path else None
    mesh = (normalize_mesh(vertices), uvs, faces, texture)
    print(f"mesh: {len(vertices)} vertices, {len(faces)} triangles")

    truths = {}
    for angle in angles:
        crop = silhouette_crop(cv2.imread(str(frames[angle])), args.size)
        if crop is not None:
            truths[angle] = crop

    # Object Capture picks its own world frame; find the offset by measuring.
    sample = sorted(truths)[:: max(1, len(truths) // 6)]
    best_offset, best_score, best_sign = 0.0, -1.0, 1.0
    for sign in (1.0, -1.0):
        for offset in np.arange(0, 360, args.coarse_step):
            scores = []
            for angle in sample:
                crop = silhouette_crop(render(mesh, sign * angle + offset, args.pitch, args.size), args.size)
                if crop is not None:
                    scores.append(psnr(truths[angle], crop))
            if scores and np.mean(scores) > best_score:
                best_score, best_offset, best_sign = float(np.mean(scores)), float(offset), sign
    print(f"orientation found by sweep: yaw = {best_sign:+.0f} * angle + {best_offset:.0f} deg "
          f"({best_score:.2f} dB on {len(sample)} samples)\n")

    rows = []
    for angle in sorted(truths):
        crop = silhouette_crop(render(mesh, best_sign * angle + best_offset, args.pitch, args.size), args.size)
        if crop is None:
            continue
        p, s = psnr(truths[angle], crop), ssim(truths[angle], crop)
        rows.append({"angle_deg": angle, "psnr_db": round(p, 4), "ssim": round(s, 5)})
        print(f"  {angle:>3d} deg   {p:6.2f} dB   {s:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "mesh_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    p = np.array([r["psnr_db"] for r in rows])
    s = np.array([r["ssim"] for r in rows])
    summary = {
        "n_scored": len(rows),
        "protocol": "fit (all photographs were used to build the mesh)",
        "psnr_mean_db": round(float(p.mean()), 4),
        "psnr_median_db": round(float(np.median(p)), 4),
        "ssim_mean": round(float(s.mean()), 5),
        "mesh_vertices": len(vertices),
        "mesh_triangles": len(faces),
        "yaw_sign": best_sign,
        "yaw_offset_deg": best_offset,
    }
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"\nmesh vs photographs: {summary['psnr_mean_db']:.2f} dB, SSIM {summary['ssim_mean']:.3f}")
    print("NOTE: fit, not hold-out. Every photograph shown here also went into the")
    print("      reconstruction, so this is not comparable to the relief's held-out")
    print("      scores as a prediction of unseen views.")
    print(f"\nwrote {args.out / 'mesh_metrics.csv'}")


if __name__ == "__main__":
    main()
