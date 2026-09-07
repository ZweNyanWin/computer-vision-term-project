"""Build the training workspace: the same scene with the held-out views removed.

MLX3D trains on every image present in ``<out>/sparse/0``, so a held-out split has
to be enforced by giving it a model that does not contain the withheld views at
all. This derives that model from the full one with ``colmap image_deleter``
rather than re-running the mapper, which matters: re-running SfM on 76 images
would produce its own arbitrary world frame, and the withheld cameras could then
no longer be expressed in it. Deleting from the solved model leaves every
surviving pose bit-identical, so the withheld poses stay directly usable for
rendering at evaluation time.

What the split does and does not guarantee, stated precisely because the report
has to:

  * No held-out pixel reaches the trainer, and no Gaussian is ever fitted to one.
  * ``image_deleter`` also drops each withheld view's observations, so every
    point that survives into the initialisation is witnessed by training views.
  * Camera poses and the sparse points were nonetheless solved by bundle
    adjustment over all 88 photographs, so the withheld observations did
    influence the geometry. This is the standard 3DGS/Mip-NeRF-360 evaluation
    protocol - it is the only way the withheld cameras can be located in the
    training model's frame - and it is disclosed rather than designed away.

    .venv-mlx3d/bin/python tools/make_train_workspace.py \\
        --full model3d/gaussian/frog88_undist \\
        --out model3d/gaussian/frog88_train \\
        --split config/neural_split.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def read_bin_image_names(sparse: Path) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx3d.datasets.colmap import _read_images_bin

    meta = _read_images_bin(str(sparse / "images.bin"))
    return sorted(m["name"] for m in meta.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="workspace with sparse/0 + images/ for ALL views")
    ap.add_argument("--out", required=True, help="training workspace to create")
    ap.add_argument("--split", default="config/neural_split.csv")
    args = ap.parse_args()

    full = Path(args.full).resolve()
    out = Path(args.out).resolve()
    full_sparse = full / "sparse" / "0"
    full_images = full / "images"
    for p in (full_sparse / "images.bin", full_images):
        if not p.exists():
            print(f"error: missing {p}")
            return 2

    with open(args.split) as f:
        split = {r["name"]: r["split"] for r in csv.DictReader(f)}
    holdout = sorted(n for n, s in split.items() if s == "holdout")

    present = set(read_bin_image_names(full_sparse))
    missing = [n for n in holdout if n not in present]
    if missing:
        print(f"error: held-out view(s) not in the full model: {missing}")
        return 2

    if out.exists():
        shutil.rmtree(out)
    (out / "sparse").mkdir(parents=True)
    (out / "images").mkdir()

    names_file = out / "holdout_names.txt"
    names_file.write_text("\n".join(holdout) + "\n")

    cmd = [
        "colmap", "image_deleter",
        "--input_path", str(full_sparse),
        "--output_path", str(out / "sparse" / "0"),
        "--image_names_path", str(names_file),
    ]
    (out / "sparse" / "0").mkdir(parents=True)
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        return r.returncode

    kept = read_bin_image_names(out / "sparse" / "0")
    for name in kept:
        src = full_images / name
        if not src.exists():
            print(f"error: undistorted image missing: {src}")
            return 2
        os.symlink(os.path.relpath(src, out / "images"), out / "images" / name)

    leaked = sorted(set(kept) & set(holdout))
    print(f"full model      {len(present)} images")
    print(f"training model  {len(kept)} images  -> {out / 'sparse' / '0'}")
    print(f"withheld        {len(holdout)} images")
    print(f"leaked into training: {len(leaked)}  {'OK' if not leaked else leaked}")
    n_links = len(list((out / 'images').iterdir()))
    print(f"image links     {n_links}")
    if leaked or n_links != len(kept) or len(kept) + len(holdout) != len(present):
        print("FAILED consistency check")
        return 1
    print("consistency OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
