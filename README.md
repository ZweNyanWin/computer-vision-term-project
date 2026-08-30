# Thai Wooden Frog — Image-Based 3D Rendering

This folder contains the runnable prototype described in
`CSX4213_Progress_Report_3D_Rendering.docx`.

The real wooden frog has not been purchased or photographed yet. For today's
progress check, the code therefore creates a clearly labelled synthetic wooden
proxy and runs the complete downstream pipeline. It does **not** claim that the
temporary depth map or output is a result for the Thai wooden frog.

## Run today's progress demo

```bash
cd "/Users/zwenyanwin/Desktop/Computer Vision/Term_Project"
conda activate cv
python run_progress_demo.py
```

The main image to show is:

```text
outputs/progress_contact_sheet.png
```

It shows six checkpoint stages: synthetic input, segmentation mask, temporary
depth proxy, front rendering, and two novel views. The terminal also prints the
vertex count, triangle count, and rendering time. A short animation is written
to `outputs/progress_novel_views.mp4`. No notebook (`.ipynb`) is required.

Run the automated check if needed:

```bash
python -m unittest discover -s tests -v
```

## What is implemented now

- Otsu thresholding on saturation, morphology, largest-component selection, and
  contour fill — robust to an unevenly lit backdrop, which defeats a
  brightness-only threshold
- temporary analytic depth proxy for an offline demonstration
- height-field mesh creation
- textured OBJ and MTL export
- pinhole projection using `x = K[R|t]X`
- back-face culling and painter's depth ordering
- affine texture mapping for every visible triangle
- Lambertian shading
- progress contact sheet and JSON performance metrics

## What remains pending

- buying and photographing the real Thai wooden frog
- running learned monocular depth on that real photograph
- capturing the multi-view turntable image set
- structure-from-motion / complete closed-surface reconstruction
- the optional classifier and workshop interface

These pending items match the progress report and do not need to be finished for
today's checkpoint.

## Learned depth on a real frog photo

There is no `torch` on the project laptop, so the monocular depth stage runs on
Colab: open `colab_depth.ipynb`, upload a photograph, and it returns the OBJ
bundle. The notebook checks the depth orientation before meshing — Depth
Anything predicts inverse depth, which this pipeline would otherwise turn into a
hollow relief.

To run it locally instead, install the optional depth packages. Model weights
download the first time the command runs, so internet access is required then.

```bash
python -m pip install -r requirements-depth.txt
python reconstruct.py data/frog_front.jpg \
  --depth-mode model \
  --out model3d/frog \
  --relief 0.35 \
  --grid 120
```

Render a safe novel-view arc for the single-image relief:

```bash
python render3d.py model3d/frog.obj \
  --frames 9 \
  --yaw 0 \
  --sweep 80 \
  --video \
  --out outputs/frog
```

A single photograph produces only a front-facing relief, so a narrow viewing arc
is honest. A full 360-degree turntable should be rendered only after the later
multi-view reconstruction creates a complete surface.

## File guide

```text
run_progress_demo.py   one-command offline checkpoint
reconstruct.py         segmentation, depth preparation, mesh, OBJ export
render3d.py            novel-view renderer
tests/                 small offline verification
data/                  source photographs
model3d/               OBJ, MTL, texture, mask, and depth outputs
outputs/               rendered views, contact sheet, and metrics
```
