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
has to, and measured rather than assumed (the counts below are for this capture):

  * No held-out pixel reaches the trainer, and no Gaussian is ever fitted to one.
    The photometric loss is computed over the 76 training photographs only.
  * Every point surviving into the initialisation has at least two training
    observations, because ``image_deleter`` drops a point once its track falls
    below that.
  * It does NOT follow that the surviving points are independent of the withheld
    views. **14,877 of the 41,431 survivors (36%) were observed by a held-out
    view in the full reconstruction**, and their positions and colours were bundle
    adjusted using those observations. Deleting an observation afterwards does not
    undo its earlier influence: the kept points are bit-identical to the full
    model's, position and colour alike.
  * Nor were the 3,761 dropped points "seen only by held-out views". Only 49 had
    no training observation at all; the other 3,712 had exactly one, and fell
    below COLMAP's minimum track length of two.

So this is held-out **photometric** evaluation over a **shared SfM
initialisation**, which is the standard 3DGS / Mip-NeRF-360 protocol and is what
the numbers should be called. A stricter protocol is possible - reconstruct from
the 76 training images alone, then localise the withheld cameras into that fixed
model with COLMAP's image registration - and would be a different, separately
reported experiment. It is not what was run here.

    .venv-mlx3d/bin/python tools/make_train_workspace.py \\
        --full model3d/gaussian/frog88_undist \\
        --out model3d/gaussian/frog88_train \\
        --split config/neural_split.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

    # Stamp what this workspace was built from. run_neural.sh refuses to train in
    # a quality workspace whose stamp no longer matches, which is what stops a
    # rewritten split from being evaluated against a stale set of training images.
    def sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    provenance = {
        "split_csv": str(Path(args.split)),
        "split_sha256": sha256_of(Path(args.split)),
        "source_model": str(full_sparse),
        "source_sha256": {
            f: sha256_of(full_sparse / f)
            for f in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "train_images_sha256": hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest(),
        "n_train": len(kept),
        "n_holdout": len(holdout),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    leaked = sorted(set(kept) & set(holdout))
    print(f"full model      {len(present)} images")
    print(f"training model  {len(kept)} images  -> {out / 'sparse' / '0'}")
    print(f"withheld        {len(holdout)} images")
    print(f"leaked into training: {len(leaked)}  {'OK' if not leaked else leaked}")
    n_links = len(list((out / 'images').iterdir()))
    print(f"image links     {n_links}")
    print(f"provenance      {out / 'provenance.json'}")
    if leaked or n_links != len(kept) or len(kept) + len(holdout) != len(present):
        print("FAILED consistency check")
        return 1
    print("consistency OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
