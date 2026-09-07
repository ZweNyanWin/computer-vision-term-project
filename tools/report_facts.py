"""Emit every fact `NEURAL_CAPTURE.md` says the report must record.

That section lists: image count, registered count, minimum and maximum camera
elevation actually achieved, camera model, whether exposure was locked and how,
lighting, MLX3D and MLX versions, resolution, iterations, training time, peak
memory, final Gaussian count and splat.ply size.

Each is read back out of the artifacts rather than transcribed by hand, so the
table in the report cannot drift from the run that produced it.

    .venv-mlx3d/bin/python tools/report_facts.py \\
        --full model3d/gaussian/frog88_undist \\
        --train model3d/gaussian/frog88_train \\
        --manifest config/neural_manifest.csv --split config/neural_split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def colmap_stats(sparse: Path) -> dict:
    r = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(sparse), "--log_target", "stdout"],
        capture_output=True, text=True,
    )
    text = r.stdout + r.stderr
    # model_analyzer writes through glog, so every line carries a
    # "I20260907 19:28:52.006549 0x20cbc2c40 model.cc:441] " prefix. Strip to the
    # last "] " before matching, or every key misses.
    keys = {
        "Cameras": "cameras",
        "Images": "images",
        "Registered images": "registered",
        "Points": "points",
        "Observations": "observations",
        "Mean track length": "mean_track_length",
        "Mean observations per image": "mean_obs_per_image",
        "Mean reprojection error": "mean_reproj_px",
    }
    out = {}
    for line in text.splitlines():
        body = line.rsplit("] ", 1)[-1].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        if key.strip() in keys:
            out[keys[key.strip()]] = value.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("--manifest", default="config/neural_manifest.csv")
    ap.add_argument("--split", default="config/neural_split.csv")
    ap.add_argument("--out", default="output/neural/report_facts.txt")
    ap.add_argument(
        "--holdout", default=None,
        help="holdout_summary.txt to fold in (default: alongside --out)",
    )
    args = ap.parse_args()

    L = []
    add = L.append

    with open(args.manifest) as f:
        man = list(csv.DictReader(f))
    optics = {}
    for r in man:
        optics.setdefault((r["width"], r["height"], r["focal_mm"], r["f_number"], r["lens"]), 0)
        optics[(r["width"], r["height"], r["focal_mm"], r["f_number"], r["lens"])] += 1
    iso = sorted({int(r["iso"]) for r in man if r["iso"]})
    exp = sorted({float(r["exposure_s"]) for r in man if r["exposure_s"]})

    add("CAPTURE")
    add(f"  photographs            {len(man)}")
    for tag in sorted({r['ring'] for r in man}):
        add(f"    {tag:<20} {sum(1 for r in man if r['ring'] == tag)}")
    add(f"  distinct optics        {len(optics)}")
    for (w, h, fmm, fn, lens), n in sorted(optics.items(), key=lambda kv: -kv[1]):
        add(f"    {n:>3} x {w}x{h}  {fmm} mm  f/{fn}  {lens}")
    add(f"  ISO                    {iso[0]}-{iso[-1]} ({len(iso)} distinct values) - NOT locked")
    add(f"  shutter                1/{1/exp[-1]:.0f}-1/{1/exp[0]:.0f} s ({len(exp)} distinct) - NOT locked")
    add(f"  exposure lock          none; AE/AF drifted across the session")

    add("")
    add("RECONSTRUCTION (COLMAP)")
    r = subprocess.run(["colmap", "--help"], capture_output=True, text=True)
    ver = (r.stdout + r.stderr).splitlines()[0].strip()
    add(f"  {ver}")
    for label, root in (("full model", args.full), ("training model", args.train)):
        if not root:
            continue
        sparse = Path(root) / "sparse" / "0"
        if not sparse.exists():
            continue
        st = colmap_stats(sparse)
        add(f"  {label}")
        for k in ("cameras", "images", "registered", "points", "observations",
                  "mean_track_length", "mean_obs_per_image", "mean_reproj_px"):
            if k in st:
                add(f"    {k:<22} {st[k]}")

    if Path(args.split).exists():
        with open(args.split) as f:
            sp = list(csv.DictReader(f))
        el = np.array([float(r["elevation_deg"]) for r in sp])
        add("")
        add("CAMERA GEOMETRY (recovered, relative to the object's horizon)")
        add(f"  elevation range        {el.min():+.1f} to {el.max():+.1f} degrees")
        for ring in ["low", "mid", "high", "top"]:
            sub = [float(r["elevation_deg"]) for r in sp if r["ring"] == ring]
            if sub:
                add(f"    {ring:<20} n={len(sub):<3} {min(sub):+6.1f} to {max(sub):+6.1f}  mean {np.mean(sub):+6.1f}")
        n_hold = sum(1 for r in sp if r["split"] == "holdout")
        add(f"  split                  {len(sp) - n_hold} train / {n_hold} held out "
            f"({100 * n_hold / len(sp):.1f}%)")

    add("")
    add("TRAINING (MLX3D)")
    import mlx.core as mx
    import mlx3d
    add(f"  mlx3d                  {getattr(mlx3d, '__version__', 'n/a')}")
    add(f"  mlx                    {mx.__version__}")
    add(f"  device                 {mx.default_device()}")
    for root in filter(None, [args.train]):
        summary = Path(root) / "capture.json"
        if summary.exists():
            js = json.loads(summary.read_text())
            for k, v in js.items():
                if isinstance(v, (str, int, float)):
                    add(f"  {k:<22} {v}")
        splat = Path(root) / "splat.ply"
        if splat.exists():
            add(f"  splat.ply              {splat.stat().st_size / 1e6:.1f} MB")
            from mlx3d.splatting import GaussianModel
            m = GaussianModel.load_ply(str(splat))
            add(f"  final gaussians        {m.num_gaussians:,}")
            add(f"  sh degree              {m.active_sh_degree}")

    hold = Path(args.holdout or (Path(args.out).parent / "holdout_summary.txt"))
    if hold.exists():
        add("")
        add("HELD-OUT RESULT")
        for line in hold.read_text().splitlines():
            add("  " + line)

    text = "\n".join(L)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
