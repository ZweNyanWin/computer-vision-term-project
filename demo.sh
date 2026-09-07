#!/usr/bin/env bash
# Four-minute demonstration of the 31 August checkpoint.
#
#   ./demo.sh          run every step, pausing between them
#   ./demo.sh 3        run step 3 only
#   ./demo.sh check    verify every prerequisite and exit. Run this BEFORE the
#                      talk: it reports everything missing at once, with the
#                      command that rebuilds each, and never opens a window.
#   DEMO_PYTHON=...    interpreter to use. Defaults to .venv-depth/bin/python
#                      when that exists (steps 3, 4 and 6 need torch), else
#                      python3.
#   DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3
#                      warm the model cache once while online
#
# Each step is quick enough to run live. Nothing here trains or reconstructs
# from scratch: the slow stages (the hold-out sweep, photogrammetry) are read
# back from the results they wrote, and the commands that produced them are
# printed so the audience can see they were not run for the first time on stage.

# -e matters more than it looks. This script used to run with -u alone, so a
# step whose embedded Python raised FileNotFoundError printed a traceback, and
# the script carried on and finished with "done" and status 0. Four of the seven
# steps were failing that way on this machine and the exit code said everything
# was fine. A demo that cannot fail cannot be trusted to have worked.
set -euo pipefail
cd "$(dirname "$0")"

# Learned depth needs torch + transformers, which the system python3 does not
# have and which this project keeps out of it. `.venv-depth` is the dedicated
# environment for them (see README); prefer it automatically so the demo needs no
# ceremony, and let DEMO_PYTHON override for anything else.
if [ -n "${DEMO_PYTHON:-}" ]; then
  PY="$DEMO_PYTHON"
elif [ -x .venv-depth/bin/python ]; then
  PY=.venv-depth/bin/python
else
  PY=python3
fi

BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'
say()  { printf '\n%s>>> %s%s\n' "$BOLD" "$1" "$OFF"; }
note() { printf '%s    %s%s\n' "$DIM" "$1" "$OFF"; }
pause() { [ "${CHECK:-0}" = 1 ] && return 0; [ -n "${STEP:-}" ] || { printf '\n%s    [Enter]%s' "$DIM" "$OFF"; read -r _; }; }

STEP="${1:-}"
CHECK=0; [ "$STEP" = "check" ] && { CHECK=1; STEP=""; }
run_step() { [ "$CHECK" = 1 ] && return 1; [ -z "$STEP" ] || [ "$STEP" = "$1" ]; }

MISSING=0
fail() { printf '    %sMISSING%s %s\n              rebuild: %s\n' "$BOLD" "$OFF" "$1" "$2"; MISSING=1; }

# Check a prerequisite. In `check` mode every problem is collected and reported;
# in a normal run the first one stops the script rather than letting the talk
# proceed into a traceback.
need_file() {
  [ -e "$1" ] && return 0
  fail "$1" "$2"
  [ "$CHECK" = 1 ] || { printf '\n%sstopping: the step above cannot run.%s\n' "$BOLD" "$OFF"; exit 1; }
  return 1
}
need_module() {
  "$PY" -c "import $1" 2>/dev/null && return 0
  fail "python module '$1' (interpreter: $PY)" "$3"
  [ "$CHECK" = 1 ] || { printf '\n%sstopping: the step above cannot run.%s\n' "$BOLD" "$OFF"; exit 1; }
  return 1
}

# Everything the demo needs but cannot make for itself. Artifacts a step produces
# during the run (model3d/demo_live.*) are deliberately not listed: they are
# outputs, not prerequisites, and listing them would report a clean machine as
# broken.
if [ "$CHECK" = 1 ]; then
  say "Prerequisites"
  need_file "data/90.jpeg" "the capture is in data/; see CAPTURE.md" || true
  ls data/[0-9]*.jpeg >/dev/null 2>&1 || fail "the 36-frame ring in data/" "see CAPTURE.md"
  ls data/ring_high/*.jpeg >/dev/null 2>&1 || fail "data/ring_high/" "see CAPTURE.md"
  need_module torch "" "python3 -m venv .venv-depth && .venv-depth/bin/python -m pip install -r requirements-depth.txt" || true
  need_module transformers "" "python3 -m venv .venv-depth && .venv-depth/bin/python -m pip install -r requirements-depth.txt" || true
  need_file "output/full_e6/metrics.csv" "python3 src/evaluate.py --frames data --every 6 --depth-mode model" || true
  need_file "model3d/frog_real.obj" "DEMO_ALLOW_DOWNLOAD=1 $PY reconstruct.py data/90.jpeg --depth-mode model --out model3d/frog_real" || true
  need_file "model3d/frog_combined.obj" "./rebuild_object_capture.sh" || true
  need_file "recon/frog_combined.usdz" "./rebuild_object_capture.sh" || true
  need_file "outputs/frog_3d_turntable.mp4" "$PY render3d.py model3d/frog_combined.obj --frames 36 --sweep 360 --video --out outputs/frog_3d" || true
  need_file "output/neural/balanced/holdout_summary.txt" "./run_neural.sh all balanced" || true
  if [ "$MISSING" = 1 ]; then
    printf '\n%sNOT READY - rebuild the items above before presenting.%s\n' "$BOLD" "$OFF"
    exit 1
  fi
  printf '\n%sREADY - every prerequisite is present.%s\n' "$BOLD" "$OFF"
  exit 0
fi

# ---------------------------------------------------------------- 1. capture
if run_step 1; then
say "1. What was captured"
note "A closed turntable ring: the frog rotates 10 degrees at a time, camera fixed."
ls data/[0-9]*.jpeg | wc -l | xargs printf '    ring photographs: %s\n'
ls data/ring_high/*.jpeg | wc -l | xargs printf '    elevated photographs: %s\n'
pause
fi

# ----------------------------------------------------------- 2. segmentation
if run_step 2; then
say "2. Segmentation on every frame of the ring"
note "Brightness thresholding failed on real photographs. This runs on saturation."
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from reconstruct import segment_foreground, _corner_fraction
bad, fracs = [], []
for a in range(0, 360, 10):
    b = cv2.imread(f'data/{a:02d}.jpeg'); s = 900 / max(b.shape[:2])
    b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = segment_foreground(b); c = _corner_fraction(m.astype(np.uint8) * 255)
    fracs.append(m.mean())
    if c >= 0.02: bad.append(a)
print(f'    36 frames checked, foreground {100*min(fracs):.0f}-{100*max(fracs):.0f}%')
print(f'    frames with background in the corners: {len(bad)}')
PY
pause
fi

# ------------------------------------------------------- 3. learned depth
if run_step 3; then
say "3. Learned depth and a textured mesh, from one photograph"
note "Depth Anything V2 Small. No GPU: about three seconds on this laptop."
if need_module torch "" "python -m pip install -r requirements-depth.txt" \
   && need_module transformers "" "python -m pip install -r requirements-depth.txt"; then
  if [ "${DEMO_ALLOW_DOWNLOAD:-0}" = "1" ]; then
      "$PY" reconstruct.py data/90.jpeg --depth-mode model --out model3d/demo_live --relief 0.42 --grid 120
  else
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          "$PY" reconstruct.py data/90.jpeg --depth-mode model --out model3d/demo_live --relief 0.42 --grid 120
  fi
fi
pause
fi

# ------------------------------------------------------------ 4. novel views
if run_step 4; then
say "4. Synthesising views the camera never took"
need_file model3d/demo_live.obj "./demo.sh 3  (needs torch; see step 3)" \
  && "$PY" render3d.py model3d/demo_live.obj --frames 5 --sweep 70 --size 420 --out outputs/demo_live
note "Five viewpoints from a single photograph."
pause
fi

# --------------------------------------------------------- 5. measured result
if run_step 5; then
say "5. What the measurement says"
note "Produced earlier by: python3 src/evaluate.py --frames data --every 6 --depth-mode model"
python3 - <<'PY'
import csv, math
import numpy as np
print(f"    {'spacing':>8}{'n':>4}{'render':>9}{'baseline':>10}   SSIM difference (95% CI)")
for e, sp in ((2, 20), (4, 40), (6, 60), (9, 90)):
    rows = list(csv.DictReader(open(f'output/full_e{e}/metrics.csv')))
    s = next(csv.DictReader(open(f'output/full_e{e}/summary.csv')))
    d = np.array([float(r['ssim_render']) - float(r['ssim_baseline']) for r in rows])
    ci = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
    mark = '  <- clears its interval' if abs(d.mean()) > ci else ''
    print(f"    {sp:>7}d{len(rows):>4}{float(s['psnr_render_mean_db']):>9.2f}"
          f"{float(s['psnr_baseline_mean_db']):>10.2f}   {d.mean():+.4f} "
          f"({d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}){mark}")
print()
print('    Only the 60-degree SSIM difference clears its confidence interval.')
print('    No PSNR difference at any spacing does.')
PY
pause
fi

# ------------------------------------------------- 6. the relief's limitation
if run_step 6; then
say "6. Why one photograph is not enough"
note "Counting triangles that survive back-face culling as the camera swings behind."
need_file model3d/frog_real.obj \
  "DEMO_ALLOW_DOWNLOAD=1 $PY reconstruct.py data/90.jpeg --depth-mode model --out model3d/frog_real"
need_file model3d/frog_combined.obj \
  "./rebuild_object_capture.sh  (Apple Object Capture over the 45 ring/elevated photographs)"
"$PY" - <<'PY'
import sys; sys.path.insert(0, '.')
import cv2
from pathlib import Path
from render3d import load_obj, normalize_mesh, render_frame
for tag, p in (('relief, 1 photograph ', 'model3d/frog_real.obj'),
               ('mesh, 45 photographs ', 'model3d/frog_combined.obj')):
    v, uv, f, tp = load_obj(Path(p)); V = normalize_mesh(v)
    tex = cv2.imread(str(tp)) if tp else None
    front = render_frame(V, uv, f, tex, width=300, height=300, yaw=0, pitch=-12)[1]
    back = render_frame(V, uv, f, tex, width=300, height=300, yaw=180, pitch=-12)[1]
    print(f'    {tag} front {front:6d}   behind {back:6d}   ({100*back/len(f):5.1f}% of the surface)')
PY
pause
fi

# ------------------------------------------------------------ 7. the artefacts
if run_step 7; then
say "7. The reconstruction itself"
note "Apple Object Capture, 45 photographs, 80 s, no GPU and no cloud service."
need_file recon/frog_combined.usdz "./rebuild_object_capture.sh"
need_file outputs/frog_3d_turntable.mp4 \
  "$PY render3d.py model3d/frog_combined.obj --frames 36 --sweep 360 --video --out outputs/frog_3d"
if [ "$CHECK" = 0 ]; then
  open recon/frog_combined.usdz && note "opened - drag to rotate it"
  sleep 1
  open outputs/frog_3d_turntable.mp4 && note "and a full 360 turntable through our own renderer"
fi
fi

# ------------------------------------------------- 8. the neural reconstruction
if run_step 8; then
say "8. Multi-view: 88 photographs, Gaussian Splatting"
note "The frog stays still and the camera moves, so this is real structure-from-motion."
need_file output/neural/balanced/holdout_summary.txt "./run_neural.sh all balanced"
if [ "$CHECK" = 0 ]; then
  sed -n '/^HELD-OUT/,/^$/p;/^PAIRED/,/^$/p' output/neural/balanced/holdout_summary.txt | sed 's/^/    /'
  note "12 of 88 views withheld before training; every one beats the nearest photograph."
  note "Inspect it: mlx3d-view model3d/gaussian/frog88_train_balanced/splat.ply"
fi
pause
fi

printf '\n%sdone%s\n' "$BOLD" "$OFF"
