# Thai Wooden Frog — Image-Based 3D Rendering

Novel-view synthesis of a Thai wooden frog (*kob mai*) from photographs. The
single-view relief and closed-mesh paths use the project's OpenCV/NumPy renderer;
the optional neural comparison uses MLX3D's Gaussian rasteriser.

The physical frog has been photographed: five hero shots plus a closed 36-frame
turntable ring, all 36 of which segment cleanly. Every number quoted below comes
from those photographs, scored against frames the pipeline never saw.

`CLAUDE.md` is the authority on current state, what each result does and does not
support, and the constraints on this codebase. Read it before changing anything.

## Run the offline demo

Learned depth lives in its own environment; there is no conda on the machine this
was last run on.

```bash
python3 -m venv .venv-depth
.venv-depth/bin/python -m pip install -r requirements-depth.txt
DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3     # warms the Depth Anything cache once
```

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
python src/evaluate.py --frames data --every 4 --depth-mode model
python src/make_figures.py        # writes figures/
```

Frames are withheld from the ring, each held-out angle is rendered from the
nearest frame the pipeline *did* see, and both that render and the nearest
captured photograph are scored against the withheld one. The frame-switching
baseline is the point: a render that cannot beat "show the closest photograph"
has not earned its reconstruction stage.

Two claims are supported, and no more:

- **Learned depth improves on the shape proxy where the relief still has useful
  geometry** — render quality is **+1.66 dB at 20°** and **+1.64 dB at 40°**,
  but falls to **−0.10 dB at 90°**, where the relief has little of the object
  left to show.
- **At 60° spacing the render is structurally closer to the withheld photograph
  than frame-switching is** — ΔSSIM **+0.024**, 95% CI [+0.014, +0.035], n=30.

**No PSNR difference is significant at any spacing**; every interval spans zero.
Frame 300° was re-shot after the operator's hand entered the original; the
replacement is included throughout the full 36-frame results.
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
  49,999 triangles (rebuilt 7 September as 25,011 / 50,000; the engine is not
  deterministic, so the two runs are not the same mesh)
- multi-view Gaussian Splatting: COLMAP structure-from-motion over 88
  photographs of a stationary frog, then MLX3D training on Metal, with a
  held-out split fixed before training

## The workshop station

`workshop/index.html` is the visitor-facing piece: drag the frog to turn it, and
switch between the reconstruction built from **1**, **45** and **88**
photographs. As you turn it, the station shows how much of the model is actually
facing you — 4,502 of the relief's 4,576 triangles from the front, and **1** from
behind, against a steady half of the closed mesh's 50,000 from every angle. That
is the whole argument about viewpoint coverage, made by dragging rather than by
assertion.

Every frame is rendered ahead of time. The one- and 45-photo paths use this
project's renderer; the 88-photo path uses MLX3D's Gaussian rasteriser with an
orbit recovered by project code. Playback needs no GPU, server or network.

```bash
./run_workshop.sh          # render the frames and bake the standalone file
./run_workshop.sh check    # verify it is complete, consistent and offline
./run_workshop.sh serve    # http://127.0.0.1:8765
```

`workshop/frog-station-standalone.html` is the same station as a single roughly 4 MB
file with all 108 frames embedded — nothing has to travel with it.

## What remains pending

- optional extensions: a genuine second elevation band for the neural capture
  (it covers only +15° and +43°), a re-shoot with exposure locked, and the
  classifier

## Neural reconstruction — done and scored

88 photographs, stationary frog, moving camera, per `NEURAL_CAPTURE.md`. COLMAP
4.1.1 registered **88/88 into one model** at 1.117 px mean reprojection error;
MLX3D 0.3.0 trained 140,018 Gaussians in 22.9 minutes on the M1 Pro's GPU.

Twelve views were withheld before training, chosen from recovered camera geometry
and frozen in `config/neural_split.csv`. The splat was then rendered at each
withheld camera's own pose and intrinsics and scored against the withheld
photograph, with the nearest same-lens captured photograph as the baseline:

| | PSNR | SSIM |
|---|---|---|
| held out, render | **20.02 dB** | **0.7325** |
| held out, nearest-photograph baseline | 13.77 dB | 0.5281 |

ΔPSNR **+6.25 dB**, 95% CI [+4.54, +7.95]; ΔSSIM **+0.204**, 95% CI [+0.163,
+0.246]; n = 12, both p < 0.0001. The render wins on 12 of 12 views on both
metrics. The training-view score (26.13 dB) is fit, not accuracy, and is never
reported as the latter.

This capture used two lenses — 40 of the 88 frames are the ultra-wide digitally
cropped by iOS auto-macro — so COLMAP runs with one camera model per optic and
the images are undistorted before training. Running plain `mlx3d-capture` on the
folder would silently fit one focal length to both. `NEURAL_CAPTURE.md` records
that, the exposure drift, the thin elevation coverage, and the floater seen in
one held-out view.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-mlx3d
source .venv-mlx3d/bin/activate
python -m pip install -r requirements-mlx3d.txt

./run_neural.sh all balanced      # ~40 min end to end; stages skip completed
                                  # work, but training is NOT resumable
mlx3d-view model3d/gaussian/frog88_train_balanced/splat.ply
```

## Learned depth on a real frog photo

Runs locally. Depth Anything V2 Small is a 25M-parameter model, so it needs no
GPU — on Apple silicon it uses MPS and takes about three seconds a frame. Weights
download on first use, so that run needs internet. `colab_depth.ipynb` does the
same thing on Colab for a machine without `torch`.

```bash
python -m pip install -r requirements-depth.txt
python reconstruct.py data/90.jpeg \
  --depth-mode model \
  --out model3d/frog \
  --relief 0.35 \
  --grid 120
```

For the presentation demo, warm the cache once while online; normal demo runs
then use the cache without making network requests:

```bash
DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3
./demo.sh
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
docs/demo_script.md    current four-minute demonstration script
docs/presentation_script.md  archived pre-capture presentation script
docs/*.docx            progress reports
CAPTURE.md             photography protocol
NEURAL_CAPTURE.md      fixed-frog, moving-camera MLX3D protocol + capture record
run_neural.sh          end-to-end neural reconstruction and held-out scoring
tools/                 capture pre-flight, split selection, held-out evaluation
config/                source manifest and the frozen train/holdout split
output/neural/         held-out Gaussian Splatting metrics and contact sheets
requirements-mlx3d.txt pinned Apple Silicon neural environment
```
