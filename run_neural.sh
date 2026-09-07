#!/usr/bin/env bash
# End-to-end neural (Gaussian Splatting) reconstruction of the frog, from the
# four capture rings to a held-out score.
#
# Re-running is safe but "resumable" needs care, because the stages differ:
#
#   prepare / sfm / undistort / split   skip work already completed, guarded by
#                                       sentinels written only on success
#   train                               NOT resumable. MLX3D 0.3.0 has no
#                                       checkpoint resume - _stage_train always
#                                       rebuilds from GaussianModel.from_points -
#                                       so re-running restarts from iteration 0.
#                                       It refuses by default when splat.ply
#                                       exists; RETRAIN=1 overrides.
#   eval / facts                        cheap, always re-run
#
#   ./run_neural.sh prepare     normalise the four rings into one flat set
#   ./run_neural.sh sfm         COLMAP: features -> matches -> mapper
#   ./run_neural.sh undistort   remove lens distortion; PINHOLE cameras
#   ./run_neural.sh split       choose the held-out views from camera geometry
#   ./run_neural.sh train fast|balanced|best
#   ./run_neural.sh eval  fast|balanced|best
#   ./run_neural.sh facts       the table NEURAL_CAPTURE.md asks the report for
#   ./run_neural.sh all         prepare -> ... -> train balanced -> eval
#
# Why this is not just `mlx3d-capture data/neural_capture/all`:
#
#   1. The shoot used TWO physical lenses (48 frames on the 6.765 mm main camera
#      at 5712x4284, 40 on the 2.22 mm ultra-wide at 4032x3024 - the whole middle
#      ring plus three low frames). MLX3D runs COLMAP with
#      --ImageReader.single_camera 1 (capture/colmap_wrap.py:71-86), which forces
#      one shared intrinsic model. That does not error; it silently fits one
#      focal length to two lenses. This script runs COLMAP itself with
#      single_camera 0, giving one camera per optic, and hands MLX3D the result
#      with --poses existing.
#   2. MLX3D's rasteriser projects through the analytic EWA pinhole path, which
#      ignores distortion coefficients. Training on distorted pixels against a
#      pinhole projection bakes the mismatch into the Gaussians. So the images
#      are undistorted first and the cameras become PINHOLE, as in the reference
#      3DGS pipeline.
#   3. MLX3D trains on every view in the model and `mlx3d-eval` scores training
#      views. A held-out result needs the withheld views removed from the model
#      before training and rendered afterwards; that is what split/eval do.

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv-mlx3d/bin/python
IMAGES=data/neural_capture/all
SFM=model3d/gaussian/frog88
UNDIST=model3d/gaussian/frog88_undist
step="${1:-all}"
quality="${2:-balanced}"

# MLX3D reduces the whole set by ONE integer divisor, derived from the largest
# dimension of the first image in sorted order (capture/pipeline.py:_auto_downscale).
# Evaluation must render at exactly the resolution the splat was trained at, so
# rather than hardcode the number this recomputes it with the same formula against
# the same images. Hardcoding it would break silently the moment the capture,
# the undistorted output size or a preset changed - and a downscale mismatch does
# not error, it just scores the wrong thing.
downscale_for() {
  local dir="$1" quality="$2"
  $PY - "$dir" "$quality" <<'PYEOF'
import math, sys
from mlx3d.capture.pipeline import QUALITY_PRESETS
from mlx3d.capture.frames import list_images
from PIL import Image
with Image.open(list_images(sys.argv[1])[0]) as im:
    d = max(im.size)
print(max(1, math.ceil(d / QUALITY_PRESETS[sys.argv[2]].train_max_dim)))
PYEOF
}

do_prepare() {
  $PY tools/prepare_neural_dataset.py
  $PY tools/check_capture.py "$IMAGES" --no-blur || true   # advisory, not a gate
}

# Stage guards are sentinels written only after a stage exits successfully, not
# "does the output file exist". An interrupted matcher leaves a database behind
# that looks finished but holds a fraction of the pairs, and a mapper run on it
# quietly reconstructs less of the scene. Killing a re-run mid-match during this
# project's own development left exactly that: 319 verified pairs where a complete
# run has 1,789. Existence is not completion.
do_sfm() {
  mkdir -p "$SFM"

  if [ ! -f "$SFM/.extracted" ]; then
    colmap feature_extractor \
      --database_path "$SFM/database.db" --image_path "$IMAGES" \
      --ImageReader.camera_model OPENCV \
      --ImageReader.single_camera 0 \
      --FeatureExtraction.max_image_size 3200 \
      2>&1 | tee "$SFM/extract.log" | tail -2
    touch "$SFM/.extracted"
  else
    echo "features: already extracted (remove $SFM/.extracted to redo)"
  fi

  if [ ! -f "$SFM/.matched" ]; then
    colmap exhaustive_matcher \
      --database_path "$SFM/database.db" \
      --FeatureMatching.guided_matching 1 \
      2>&1 | tee "$SFM/match.log" | tail -2
    touch "$SFM/.matched"
  else
    echo "matches: already complete (remove $SFM/.matched to redo)"
  fi

  # Report what the graph actually contains, so a truncated database is visible
  # rather than silently feeding the mapper.
  $PY - "$SFM/database.db" <<'PYEOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
n = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
kp = db.execute("SELECT COUNT(*) FROM keypoints WHERE rows>0").fetchone()[0]
tv = db.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows>0").fetchone()[0]
print(f"match graph: {kp}/{n} images with features, "
      f"{tv} verified pairs of {n*(n-1)//2} possible ({100*tv/max(n*(n-1)//2,1):.1f}%)")
if kp != n:
    raise SystemExit(f"ERROR: only {kp} of {n} images have features.")
PYEOF

  if [ ! -f "$SFM/sparse/0/images.bin" ]; then
    mkdir -p "$SFM/sparse"
    colmap mapper --database_path "$SFM/database.db" --image_path "$IMAGES" \
      --output_path "$SFM/sparse" 2>&1 | tee "$SFM/mapper.log" | tail -2
  fi
  colmap model_analyzer --path "$SFM/sparse/0" --log_target stdout
}

do_undistort() {
  if [ ! -f "$UNDIST/sparse/0/images.bin" ]; then
    rm -rf "$UNDIST"
    colmap image_undistorter \
      --image_path "$IMAGES" --input_path "$SFM/sparse/0" \
      --output_path "$UNDIST" --output_type COLMAP
    # image_undistorter writes sparse/ directly; MLX3D's --poses existing wants
    # sparse/0, and so does load_colmap's preferred layout.
    mkdir -p "$UNDIST/sparse/0"
    mv "$UNDIST"/sparse/*.bin "$UNDIST/sparse/0/"
  fi
  colmap model_analyzer --path "$UNDIST/sparse/0" --log_target stdout
}

do_split() {
  $PY tools/select_holdout.py "$UNDIST"
  $PY tools/make_train_workspace.py --full "$UNDIST" \
      --out "model3d/gaussian/frog88_train" --split config/neural_split.csv
}

# Densification strategy. Vanilla 3DGS is the default and the reported result.
# METHOD=mcmc runs the same split and the same evaluation with MLX3D's MCMC
# strategy instead, which is the documented remedy for floaters - the failure this
# capture actually exhibits, in the one held-out view where the under-observed
# background is resolved as a dark blob in front of the camera. Both are reported.
METHOD="${METHOD:-vanilla}"
suffix() { [ "$METHOD" = vanilla ] && echo "$1" || echo "$1_$METHOD"; }

do_train() {
  local out="model3d/gaussian/frog88_train_$(suffix "$1")"
  local base="model3d/gaussian/frog88_train"
  [ -f "$base/provenance.json" ] || { echo "run '$0 split' first: no $base/provenance.json" >&2; exit 2; }

  if [ ! -d "$out" ]; then
    cp -R "$base" "$out"
  elif ! cmp -s "$base/provenance.json" "$out/provenance.json"; then
    # The split was recomputed after this workspace was built. Training on it now
    # would fit images the current split calls held out, and the evaluation would
    # report one of them as a novel view. Refuse rather than silently reuse.
    echo "ERROR: $out was built from a different split than $base." >&2
    echo "       Delete it to rebuild, or re-run '$0 split' if the split is stale." >&2
    diff <(cat "$base/provenance.json") <(cat "$out/provenance.json") | head -12 >&2
    exit 2
  fi

  # MLX3D has no checkpoint resume: _stage_train always rebuilds the model with
  # GaussianModel.from_points, so calling this again retrains from scratch over
  # the previous artifacts. Make that a decision rather than an accident.
  if [ -f "$out/splat.ply" ] && [ "${RETRAIN:-0}" != "1" ]; then
    echo "$out/splat.ply already exists - training is NOT resumable, so this would"
    echo "restart from iteration 0 and overwrite it. Set RETRAIN=1 to do that"
    echo "deliberately, or delete the directory for a clean run. Skipping."
    return 0
  fi

  # A second trainer writing the same directory would interleave checkpoints.
  if ! mkdir "$out/.lock" 2>/dev/null; then
    echo "ERROR: $out/.lock exists - another training run holds this workspace." >&2
    echo "       Remove it if no trainer is running." >&2
    exit 2
  fi
  trap 'rmdir "$out/.lock" 2>/dev/null || true' EXIT
  # --low-mem is mandatory on a 16 GB machine: without it max_gaussians is None
  # and growth is unbounded (capture/pipeline.py:230).
  #
  # Wrapped in /usr/bin/time -l because MLX3D records wall time but not memory,
  # and NEURAL_CAPTURE.md's "Record for the report" asks for peak memory. The
  # figure lands in <out>/time.txt as "maximum resident set size" in bytes.
  # tqdm writes its progress bar to stderr, and so does /usr/bin/time, so the two
  # cannot be split by stream. Capture everything to one log, then lift the
  # resource summary out of it.
  /usr/bin/time -l .venv-mlx3d/bin/mlx3d-capture "$out/images" \
    --poses existing --quality "$1" --method "$METHOD" --low-mem --no-viewer --out "$out" \
    2>&1 | tee "$out/train.log" | tail -20
  tr '\r' '\n' < "$out/train.log" \
    | grep -aE "real +[0-9]|maximum resident set size|peak memory footprint" \
    > "$out/time.txt" || true
  echo "resource summary -> $out/time.txt"
  cat "$out/time.txt"
  rmdir "$out/.lock" 2>/dev/null || true
  trap - EXIT
}

# Radius about the scene median that bounds the frog, in COLMAP's arbitrary units.
# Fixed from scale rather than from the pictures: the recovered camera radius is
# 4.48 units at a working distance of about 37 cm (CAPTURE protocol), so one unit
# is roughly 8 cm and 1.0 is a sphere of about 17 cm across the frog's centre -
# a 12 cm carving, comfortably bounded. Used only for the secondary object-region
# diagnostic; the headline number is full-frame and does not depend on it.
OBJECT_RADIUS=1.0

do_eval() {
  local out="model3d/gaussian/frog88_train_$(suffix "$1")"
  $PY tools/eval_holdout.py \
    --splat "$out/splat.ply" --model "$UNDIST" --images "$(cd "$UNDIST/images" && pwd)" \
    --split config/neural_split.csv --downscale "$(downscale_for "$out/images" "$1")" \
    --object-radius "$OBJECT_RADIUS" \
    --out "output/neural/$(suffix "$1")" --contact-sheet
}

case "$step" in
  prepare)   do_prepare ;;
  sfm)       do_sfm ;;
  undistort) do_undistort ;;
  split)     do_split ;;
  train)     do_train "$quality" ;;
  eval)      do_eval "$quality" ;;
  facts)     $PY tools/report_facts.py --full "$UNDIST" \
               --train "model3d/gaussian/frog88_train_$(suffix "$quality")" \
               --holdout "output/neural/$(suffix "$quality")/holdout_summary.txt" ;;
  all)       do_prepare; do_sfm; do_undistort; do_split
             do_train "$quality"; do_eval "$quality"
             $PY tools/report_facts.py --full "$UNDIST" \
               --train "model3d/gaussian/frog88_train_$(suffix "$quality")" \
               --holdout "output/neural/$(suffix "$quality")/holdout_summary.txt" ;;
  *) echo "unknown step: $step" >&2; exit 2 ;;
esac
