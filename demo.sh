#!/usr/bin/env bash
# Four-minute demonstration of the 31 August checkpoint.
#
#   ./demo.sh          run every step, pausing between them
#   ./demo.sh 3        run step 3 only
#   DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3
#                      warm the model cache once while online
#
# Each step is quick enough to run live. Nothing here trains or reconstructs
# from scratch: the slow stages (the hold-out sweep, photogrammetry) are read
# back from the results they wrote, and the commands that produced them are
# printed so the audience can see they were not run for the first time on stage.

set -u
cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'
say()  { printf '\n%s>>> %s%s\n' "$BOLD" "$1" "$OFF"; }
note() { printf '%s    %s%s\n' "$DIM" "$1" "$OFF"; }
pause() { [ -n "${STEP:-}" ] || { printf '\n%s    [Enter]%s' "$DIM" "$OFF"; read -r _; }; }

STEP="${1:-}"
run_step() { [ -z "$STEP" ] || [ "$STEP" = "$1" ]; }

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
if [ "${DEMO_ALLOW_DOWNLOAD:-0}" = "1" ]; then
    python3 reconstruct.py data/90.jpeg --depth-mode model --out model3d/demo_live --relief 0.42 --grid 120
else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 reconstruct.py data/90.jpeg --depth-mode model --out model3d/demo_live --relief 0.42 --grid 120
fi
pause
fi

# ------------------------------------------------------------ 4. novel views
if run_step 4; then
say "4. Synthesising views the camera never took"
python3 render3d.py model3d/demo_live.obj --frames 5 --sweep 70 --size 420 --out outputs/demo_live
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
python3 - <<'PY'
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
open recon/frog_combined.usdz 2>/dev/null && note "opened - drag to rotate it"
sleep 1
open outputs/frog_3d_turntable.mp4 2>/dev/null && note "and a full 360 turntable through our own renderer"
fi

printf '\n%sdone%s\n' "$BOLD" "$OFF"
