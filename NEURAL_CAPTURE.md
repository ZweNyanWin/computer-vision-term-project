# Neural capture protocol

This dataset is for the MLX3D Gaussian Splatting extension. It is separate from
the existing turntable ring used by the learned-depth and Object Capture
baselines.

## Scene geometry

- Keep the frog completely stationary and move the camera around it.
- Do not use the fixed-camera, rotating-frog ring as COLMAP/MLX3D input. Its
  stationary background and rotating subject describe contradictory camera
  motion.
- Keep the surface, markers, background, lighting, and every other visible
  object stationary for the whole shoot.

## Coverage

Capture about 90–120 sharp photographs with 60–80% overlap:

- 30–36 near eye level, named `low_000.jpg`, `low_010.jpg`, and so on.
- 30–36 from roughly 25–35 degrees above, named `mid_*.jpg`.
- 24–36 from roughly 50–60 degrees above, named `high_*.jpg`.
- Several extra top-oblique views, named `top_*.jpg`.

Store the originals at:

```text
data/neural_capture/all/
```

Lock focal length, focus, exposure, and white balance. Do not zoom between
frames. Disable portrait-mode blur, filters, and aggressive HDR. A mildly
textured stationary surface or small stationary markers help COLMAP recover
camera poses; avoid a completely featureless backdrop.

## Fast M1 Pro diagnostic

The diagnostic may use every image because it tests capture and pose quality,
not held-out accuracy:

```bash
source .venv-mlx3d/bin/activate
mlx3d-capture data/neural_capture/all \
  --quality fast \
  --low-mem \
  --no-viewer \
  --out model3d/gaussian/frog_m1_test
```

Before final training, require:

- About 90% or more images registered.
- Camera locations form coherent low, middle, and high paths around the frog.
- Front, sides, back, and top are present.
- No split clusters, duplicated frogs, or large dissolved regions appear.

If this gate fails, check blur and overlap, add intermediate views or stationary
texture, and recapture before running a longer optimization.

Inspect a successful result with:

```bash
mlx3d-view model3d/gaussian/frog_m1_test/splat.ply --fast
```

## Evaluation rule

Select and record an evenly distributed 10–15% test set before final training.
Spread it across horizontal angle, elevation, and all sides of the frog. The
held-out RGB images must not be used for Gaussian optimization.

MLX3D's training-view score is training fit, not novel-view accuracy. Final
claims require rendering the saved splat at held-out camera poses and comparing
those renders with the held-out photographs using `src/metrics.py`.

Both `data/` and `model3d/` are gitignored. Back up the photographs, split
manifest, COLMAP model, capture metadata, and final `splat.ply` separately.
