#!/usr/bin/env bash
# Build the workshop station: render every frame it shows, then bake the
# self-contained copy.
#
#   ./run_workshop.sh          rebuild everything (about 3 minutes)
#   ./run_workshop.sh frames   just the frames
#   ./run_workshop.sh bake     just the standalone file
#   ./run_workshop.sh serve    serve it at http://127.0.0.1:8765 and stop with ^C
#   ./run_workshop.sh check    verify the station is complete and consistent
#
# The station shows three reconstructions of the same frog turning through 360
# degrees, from 1, 45 and 88 photographs. Every frame is rendered AHEAD of time.
# The one- and 45-photo meshes go through this project's NumPy/OpenCV renderer;
# the 88-photo neural result goes through MLX3D's Gaussian rasteriser, driven by
# tools/render_orbit.py. Playback then needs no GPU, server or network, so bad
# venue wifi cannot change what the visitor sees.

set -euo pipefail
cd "$(dirname "$0")"

# render3d.py needs numpy + opencv; .venv-depth has them and is present whenever
# learned depth is installed. Fall back to python3, which also carries opencv.
if [ -x .venv-depth/bin/python ]; then PY=.venv-depth/bin/python; else PY=python3; fi
MLX=.venv-mlx3d/bin/python
SPLAT=model3d/gaussian/frog88_train_balanced/splat.ply
MODEL=model3d/gaussian/frog88_undist
FRAMES=36
MESH_PITCH=-5

say() { printf '\n>>> %s\n' "$1"; }

do_frames() {
  for f in model3d/frog_real.obj model3d/frog_combined.obj "$SPLAT"; do
    [ -e "$f" ] || { echo "missing $f" >&2
      echo "  frog_real.obj:     $PY reconstruct.py data/90.jpeg --depth-mode model --out model3d/frog_real" >&2
      echo "  frog_combined.obj: ./rebuild_object_capture.sh" >&2
      echo "  splat.ply:         ./run_neural.sh all balanced" >&2
      exit 2; }
  done

  # 37 frames, then drop the last. render3d.py spreads --frames evenly across
  # --sweep INCLUSIVE of both ends (np.linspace), so "--frames 36 --sweep 360"
  # steps 360/35 = 10.2857 degrees and ends at +180, which is the same view as
  # the -180 it started from. That gives a duplicated frame at the wrap and,
  # worse, an angle schedule that no simple formula reproduces - the visible-
  # surface count would then be measured for a different angle than the frame it
  # labels, by up to 10 degrees, silently. Asking for 37 makes the step exactly
  # 10 degrees; discarding the duplicate endpoint leaves a true 36-step cycle at
  # -180 + 10i, which is what tools/measure_visibility.py assumes and what the
  # splat orbit already uses.
  say "1/3  One photograph - the learned-depth relief"
  rm -rf workshop/frames/relief && mkdir -p workshop/frames/relief
  $PY render3d.py model3d/frog_real.obj --frames $((FRAMES + 1)) --sweep 360 \
     --pitch "$MESH_PITCH" --size 512 \
     --out workshop/frames/relief/f | tail -1
  rm -f workshop/frames/relief/f_036.png

  say "2/3  45 photographs - the Object Capture mesh"
  rm -rf workshop/frames/mesh && mkdir -p workshop/frames/mesh
  $PY render3d.py model3d/frog_combined.obj --frames $((FRAMES + 1)) --sweep 360 \
     --pitch "$MESH_PITCH" --size 512 \
     --out workshop/frames/mesh/f | tail -1
  rm -f workshop/frames/mesh/f_036.png

  say "3/3  88 photographs - the Gaussian splat"
  $MLX tools/render_orbit.py --splat "$SPLAT" --model "$MODEL" \
     --out workshop/frames/splat --frames "$FRAMES" --fill 0.90 | tail -1

  say "Converting to JPEG and measuring visible surface per angle"
  $PY tools/measure_visibility.py --frames "$FRAMES" --pitch "$MESH_PITCH"
}

do_bake() { say "Baking the self-contained station"; $PY tools/build_workshop.py; }

do_check() {
  say "Checking the station"
  $PY tools/check_workshop.py
}

case "${1:-all}" in
  frames) do_frames ;;
  bake)   do_bake ;;
  check)  do_check ;;
  serve)  echo "http://127.0.0.1:8765/  (^C to stop)"
          cd workshop && exec python3 -m http.server 8765 --bind 127.0.0.1 ;;
  all)    do_frames; do_bake; do_check
          say "Done"
          echo "  open workshop/index.html                       (needs frames/ beside it)"
          echo "  open workshop/frog-station-standalone.html     (one file, works anywhere)" ;;
  *) echo "unknown step: $1" >&2; exit 2 ;;
esac
