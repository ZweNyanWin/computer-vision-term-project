"""Choose the held-out views from recovered camera geometry, not filename order.

`NEURAL_CAPTURE.md` asks for a 10-15% test set "spread across horizontal angle,
elevation and all sides". Filename order does not give that: the shoot's numbering
is inconsistent between rings and the operator's angular steps were uneven (the
pilot ring's hand-marked 10 degree steps came out with jumps of up to 42 degrees).
Picking every k-th filename would therefore leave clumps and gaps.

So the split is chosen from the COLMAP poses instead. That uses only recovered
camera position - no image content, no training signal - and it happens before any
Gaussian is fitted, so it cannot be tuned to flatter a result.

Geometry, all derived rather than assumed:

  * object centre = component-wise median of the sparse points (robust to the
    stray background points COLMAP always triangulates).
  * up axis = the axis about which the rings are circles. Each ring was walked at
    a roughly fixed elevation, so for the correct axis every camera in a ring has
    the same component along it; deviations from a ring's own mean are therefore
    perpendicular to it. The axis is the smallest eigenvector of the pooled
    within-ring scatter, which drives those deviations to zero.

    Averaging the object-to-camera directions instead - the obvious first guess -
    is biased by how many frames each ring happens to contain and by how far the
    operator stood, and on this capture it lands 6 degrees away and inflates the
    within-ring elevation spread by up to 3 degrees. The axis is not assumed: the
    eigenvalue gap says how well determined it is, and the script refuses to
    write a split if the rings are not actually circles about it.
  * azimuth = angle about that axis; elevation = angle above the plane normal
    to it.

Within each ring the cameras are sorted by azimuth and the held-out ones are
taken at evenly spaced positions around the circle, so the test set covers every
side of the frog at every elevation.

    .venv-mlx3d/bin/python tools/select_holdout.py model3d/gaussian/frog88
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np


def load_geometry(root: Path):
    from mlx3d.datasets import load_colmap

    ds = load_colmap(str(root), images_dir=str(root / "does-not-matter"), load_images=False)
    centers = np.stack([np.asarray(c.camera_center) for c in ds.cameras])
    points = np.asarray(ds.points)
    return ds.image_names, centers, points


def spherical(centers: np.ndarray, points: np.ndarray, rings: list[str]):
    """Return (azimuth_deg, elevation_deg, object_centre, up, eigenvalues)."""
    obj = np.median(points, axis=0)
    v = centers - obj
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)

    # For the true axis u, every camera in a ring shares one elevation, so
    # (unit_i - ring_mean) . u == 0. Pooling that scatter over all rings gives a
    # matrix whose smallest eigenvector is u, and whose eigenvalue gap says how
    # well the data pins it down.
    scatter = np.zeros((3, 3))
    for r in sorted(set(rings)):
        x = unit[np.array(rings) == r]
        x = x - x.mean(axis=0)
        scatter += x.T @ x
    eigvals, eigvecs = np.linalg.eigh(scatter)
    up = eigvecs[:, 0]
    if float(np.mean(unit @ up)) < 0:  # point it at the cameras, not away
        up = -up

    # Any two directions orthogonal to `up` give a consistent azimuth origin;
    # the absolute zero is arbitrary, only the ordering around the circle matters.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, up)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    elev = np.degrees(np.arcsin(np.clip(unit @ up, -1, 1)))
    azim = np.degrees(np.arctan2(v @ e2, v @ e1)) % 360.0
    return azim, elev, obj, up, eigvals


def evenly_spaced(n: int, k: int) -> list[int]:
    """k indices spread as evenly as possible over 0..n-1."""
    if k <= 0:
        return []
    return sorted({int(round(i * n / k)) % n for i in range(k)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", help="COLMAP workspace holding sparse/0")
    ap.add_argument("--out", default="config/neural_split.csv")
    ap.add_argument("--fraction", type=float, default=0.136, help="target held-out fraction")
    ap.add_argument("--manifest", default="config/neural_manifest.csv")
    args = ap.parse_args()

    root = Path(args.workspace)
    names, centers, points = load_geometry(root)
    rings = [n.split("_")[0] for n in names]
    azim, elev, obj, up, eigvals = spherical(centers, points, rings)

    order = ["low", "mid", "high", "top"]
    present = [r for r in order if r in set(rings)]

    # The axis is trustworthy exactly when the rings really are circles about it:
    # the smallest eigenvalue (scatter along the axis) must be far below the two
    # in-plane ones, and each ring's elevation must then be nearly constant.
    gap = float(eigvals[1] / max(eigvals[0], 1e-12))
    print(f"up-axis fit: within-ring scatter eigenvalues {np.round(eigvals, 3)}")
    print(f"             in-plane / along-axis ratio {gap:.1f}x")
    if gap < 5.0:
        print(
            "\nERROR: the camera positions do not form rings about any single axis\n"
            "(ratio below 5x), so azimuth would be meaningless. No split written."
        )
        return 1

    print("\nrecovered geometry by ring (degrees above the object's horizon):")
    worst = 0.0
    for r in present:
        sub = np.array([e for e, rr in zip(elev, rings) if rr == r])
        worst = max(worst, float(sub.std()))
        print(f"  {r:<5} n={len(sub):<3} mean {sub.mean():+6.1f}  sd {sub.std():5.2f}  "
              f"range {sub.min():+6.1f} to {sub.max():+6.1f}")
    if worst > 10.0:
        print(
            f"\nERROR: one ring's elevation varies by sd {worst:.1f} deg, so it is not a\n"
            "ring at a fixed elevation and the axis fit cannot be trusted. No split written."
        )
        return 1
    print(f"  every ring holds its elevation to sd <= {worst:.1f} deg, so the axis is sound.")

    # Nominal folder names are not evidence of elevation. Say so when they disagree
    # rather than silently reordering or failing.
    means = {r: float(np.mean([e for e, rr in zip(elev, rings) if rr == r])) for r in present}
    if [means[r] for r in present] != sorted(means[r] for r in present):
        by_height = sorted(present, key=lambda r: means[r])
        print(
            "\n  NOTE: measured elevation does not follow the folder names.\n"
            "  Actual order, lowest first: " + " < ".join(f"{r} ({means[r]:+.0f})" for r in by_height) +
            "\n  The rings are grouped as shot, so they still stratify the split, but the\n"
            "  capture covers fewer distinct elevations than the folder names imply."
        )
    print()

    total = len(names)
    target = max(1, round(args.fraction * total))
    # Allocate the budget across rings in proportion to ring size, largest
    # remainder first, so the shares sum to exactly `target`.
    sizes = {r: sum(1 for rr in rings if rr == r) for r in present}
    exact = {r: sizes[r] * target / total for r in present}
    alloc = {r: int(math.floor(exact[r])) for r in present}
    for r in sorted(present, key=lambda r: exact[r] - alloc[r], reverse=True):
        if sum(alloc.values()) >= target:
            break
        alloc[r] += 1

    sha = {}
    mpath = Path(args.manifest)
    if mpath.exists():
        with mpath.open() as f:
            sha = {r["name"]: r["sha256"] for r in csv.DictReader(f)}

    rows = []
    for r in present:
        idx = [i for i, rr in enumerate(rings) if rr == r]
        idx.sort(key=lambda i: azim[i])
        chosen = set(evenly_spaced(len(idx), alloc[r]))
        picked = [azim[idx[j]] for j in sorted(chosen)]
        gaps = np.diff(sorted(picked)) if len(picked) > 1 else np.array([])
        print(
            f"  {r:<5} hold out {alloc[r]}/{sizes[r]} at azimuth "
            + ", ".join(f"{a:.0f}" for a in sorted(picked))
            + (f"  (min gap {gaps.min():.0f} deg)" if gaps.size else "")
        )
        for j, i in enumerate(idx):
            rows.append(
                {
                    "name": names[i],
                    "ring": r,
                    "split": "holdout" if j in chosen else "train",
                    "azimuth_deg": round(float(azim[i]), 2),
                    "elevation_deg": round(float(elev[i]), 2),
                    "sha256": sha.get(names[i], ""),
                }
            )

    rows.sort(key=lambda r: r["name"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_hold = sum(1 for r in rows if r["split"] == "holdout")
    print(
        f"\nwrote {out}: {len(rows) - n_hold} train / {n_hold} holdout "
        f"({100 * n_hold / len(rows):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
