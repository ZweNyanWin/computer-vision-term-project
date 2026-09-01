# Thai Wooden Frog — Image-Based 3D Rendering

Novel-view synthesis of a Thai wooden frog (*kob mai*) from photographs, built on
OpenCV and NumPy with no external 3D renderer.

The physical frog has been photographed: five hero shots plus a closed 36-frame
turntable ring, all 36 of which segment cleanly. Every number quoted below comes
from those photographs, scored against frames the pipeline never saw.

`CLAUDE.md` is the authority on current state, what each result does and does not
support, and the constraints on this codebase. Read it before changing anything.

## Run the offline demo

```bash
conda activate cv
python run_progress_demo.py
```

This runs the complete pipeline on a clearly labelled **synthetic** proxy, so the
downstream stages can be shown without any photographs present. It is a smoke
test, not a result — nothing it prints is a measurement of the frog.

```text
outputs/progress_contact_sheet.png
```

Automated check:

```bash
python -m unittest discover -s tests -v
```

## Measured results

```bash
python src/evaluate.py --frames data --every 4 --exclude 300 --depth-mode model
python src/make_figures.py        # writes figures/
```

Frames are withheld from the ring, each held-out angle is rendered from the
nearest frame the pipeline *did* see, and both that render and the nearest
captured photograph are scored against the withheld one. The frame-switching
baseline is the point: a render that cannot beat "show the closest photograph"
has not earned its reconstruction stage.

Two claims are supported, and no more:

- **Learned depth beats the shape proxy** — mean render quality **+1.31 dB**,
  consistent across every spacing tested.
- **At 60° spacing the render is structurally closer to the withheld photograph
  than frame-switching is** — ΔSSIM **+0.024**, 95% CI [+0.014, +0.035], n=30.

**No PSNR difference is significant at any spacing**; every interval spans zero.
Frame 300° is excluded from every run because the operator's hand is in the shot.
The CSVs in `output/` are the only source for figures quoted in the report — see
`CLAUDE.md` for why intervals are reported rather than differences of means.

## What is implemented

- Otsu thresholding on **saturation**, morphology, largest-component selection,
  and contour fill — a real backdrop is never evenly lit, which defeats a
  brightness-only threshold
- learned monocular depth (Depth Anything V2 Small) and an analytic shape proxy
- depth orientation check, since the network predicts *inverse* depth and a raw
  prediction renders the frog hollow
- height-field mesh creation, textured OBJ and MTL export
- pinhole projection `x = K[R|t]X`, back-face culling, painter's depth ordering,
  affine texture mapping per visible triangle, Lambertian shading
- held-out PSNR/SSIM evaluation against a frame-switching baseline
- closed-surface reconstruction via Apple Object Capture — 25,008 vertices,
  49,999 triangles

## What remains pending

- turntable rings 2 and 3, at ~30° and ~60° camera elevation. Every result so
  far comes from a single eye-level ring
- the optional classifier and the workshop interface

## Learned depth on a real frog photo

Runs locally. Depth Anything V2 Small is a 25M-parameter model, so it needs no
GPU — on Apple silicon it uses MPS and takes about three seconds a frame. Weights
download on first use, so that run needs internet. `colab_depth.ipynb` does the
same thing on Colab for a machine without `torch`.

```bash
python -m pip install -r requirements-depth.txt
python reconstruct.py data/frog_front.jpg \
  --depth-mode model \
  --out model3d/frog \
  --relief 0.35 \
  --grid 120
```

Render a novel-view arc for the single-image relief:

```bash
python render3d.py model3d/frog.obj \
  --frames 9 \
  --yaw 0 \
  --sweep 80 \
  --video \
  --out outputs/frog
```

A single photograph produces only a front-facing relief, so a narrow arc is
honest. Render a full 360° turn only from the Object Capture mesh, which is a
closed surface.

## File guide

```text
run_progress_demo.py   one-command offline smoke test
reconstruct.py         segmentation, depth, mesh, OBJ export
render3d.py            novel-view renderer
colab_depth.ipynb      learned depth on Colab
src/evaluate.py        hold-out scoring against withheld photographs
src/metrics.py         PSNR and SSIM on NumPy/OpenCV
src/make_figures.py    writes figures/ from the CSVs
recon/                 Apple Object Capture: Swift driver, USDZ to OBJ, scoring
tests/                 offline verification
data/                  source photographs (gitignored)
model3d/               OBJ, MTL, texture, mask, and depth outputs
output/                hold-out metric CSVs - the source for report figures
outputs/               contact sheet and demo metrics
figures/               report figures
docs/                  progress reports and the presentation script
CAPTURE.md             photography protocol
```
