#!/usr/bin/env bash
# Rebuild the Apple Object Capture mesh and everything the demo reads from it.
#
#   ./rebuild_object_capture.sh
#
# `model3d/`, `recon/input_*/`, `recon/*.usdz` and the compiled tool are all
# gitignored, so a fresh clone -- or this machine, which lost them -- has none of
# it. `demo.sh check` reports what is missing and points here. Nothing downloads:
# Object Capture ships with macOS and runs locally.
#
# The 45 photographs are the closed 36-frame eye-level ring plus the 9 elevated
# frames, which is the same set the reported 25,008-vertex / 49,999-triangle mesh
# was built from. Detail is `medium` for the same reason as before: `full` and
# `raw` cost far more time for a mesh this evaluation never uses at that density.

set -euo pipefail
cd "$(dirname "$0")"

PY="${DEMO_PYTHON:-python3}"
INPUT=recon/input_combined
USDZ=recon/frog_combined.usdz
OBJ=model3d/frog_combined

say() { printf '\n>>> %s\n' "$1"; }

say "1. Staging the 45 photographs"
rm -rf "$INPUT"; mkdir -p "$INPUT"
n=0
for f in data/[0-9]*.jpeg; do cp "$f" "$INPUT/ring_$(basename "$f")"; n=$((n+1)); done
for f in data/ring_high/*.jpeg; do cp "$f" "$INPUT/high_$(basename "$f")"; n=$((n+1)); done
echo "    $n photographs in $INPUT"
[ "$n" -eq 45 ] || echo "    NOTE: expected 45; the reported mesh used 45."

say "2. Compiling the Object Capture driver"
if [ ! -x recon/photogrammetry ] || [ recon/photogrammetry.swift -nt recon/photogrammetry ]; then
  swiftc -O recon/photogrammetry.swift -o recon/photogrammetry
  echo "    built recon/photogrammetry"
else
  echo "    recon/photogrammetry is up to date"
fi

say "3. Reconstructing (about 80 s on the M1 Pro, no GPU service, no network)"
./recon/photogrammetry "$INPUT" "$USDZ" medium

say "4. Converting USDZ to the OBJ bundle render3d.py reads"
"$PY" recon/usdz_to_obj.py "$USDZ" --out "$OBJ"

say "5. Rendering the turntable the demo's step 7 opens"
"$PY" render3d.py "$OBJ.obj" --frames 36 --sweep 360 --video --out outputs/frog_3d

say "Done"
ls -la "$USDZ" "$OBJ.obj" outputs/frog_3d_turntable.mp4 2>/dev/null || true
