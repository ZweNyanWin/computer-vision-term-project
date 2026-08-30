# CLAUDE.md — read this before doing anything

Context for anyone (human or Claude) picking up this repository.

## What this project is

CSX 4213 Computer Vision term project, Assumption University. Deliverable is a
**workshop on Thai arts and culture**, 22 September 2026. The project is worth
**50% of the course grade**.

**Subject:** the Thai wooden frog (*kob mai*) — a hardwood carving hollowed into a
resonator with a ridged back, played by scraping a beater across it. It is both a
woodcarving craft and a folk instrument.

**Task the teacher set:** 3D rendering — synthesising views of an object from
photographs, i.e. Szeliski Chapter 14, image-based rendering.

## Current state — read this before claiming anything works

| Component | State |
|---|---|
| Novel-view renderer (`render3d.py`) | **Done**, tested, measured |
| Reconstruction (`reconstruct.py`) | **Done**, both paths. Learned depth runs locally — `torch` is installed and Depth Anything V2 Small uses MPS, ~3 s a frame |
| Progress demo (`run_progress_demo.py`) | **Done** — end-to-end on a synthetic proxy |
| Unit test (`tests/test_pipeline.py`) | **Passing** |
| Real frog photographs | **Both shoots done** — 5 hero shots plus a closed 36-frame turntable ring in `data/`, all 36 segmenting cleanly |
| Hold-out evaluation (`src/evaluate.py`) | **Done and run** — see `output/metrics/` |
| Multi-view / structure-from-motion | **Not started** |
| Classifier (`scraper.py`, `training/train.py`) | **Not started.** Carried over from an earlier topic |
| Workshop station | **Not started** |

**The numbers currently in `outputs/` come from a synthetic proxy object, not the
real frog.** `run_progress_demo.py` generates a clearly-labelled synthetic wooden
shape so the downstream pipeline can be demonstrated before capture. Do not
present those figures as results for the frog, and do not let them drift into the
report as if they were.

## Pipeline

```
photograph → segmentation → depth → height-field mesh → novel-view rendering
```

- **Segmentation** — Otsu threshold, morphological open/close, largest connected
  component, contour fill. Otsu runs on **saturation, not brightness**, chosen
  automatically by whichever channel it separates more cleanly. A real backdrop
  is never evenly lit, so its grey levels span a wide range and a global
  brightness threshold cuts through the background instead of around the object;
  a neutral backdrop stays desaturated under any illumination while the wood
  keeps its hue. Still assumes a *plain* (unpatterned) background — that is
  controlled at capture time rather than solved in software.
- **Depth** — either a learned monocular network (Depth Anything V2, MiDaS
  fallback) or, for the demo, a synthetic shape proxy. Relative depth only, no
  metric scale.
- **Mesh** — regular grid over the silhouette, depth as height, two triangles per
  quad when all four corners are inside the mask, UVs from the source image so the
  photograph becomes the texture. Exports OBJ + MTL.
- **Rendering** — pinhole projection `x = K[R|t]X`, back-face culling on the sign
  of the projected signed area, far-to-near painter's ordering, per-triangle
  affine texture warp, Lambertian shading.

## Rules that matter

1. **Do not swap in a rendering engine.** No Blender, Open3D, pyrender, OpenGL.
   The projection, hidden-surface and shading stages being our own implementation
   is the point of the assignment. OpenCV and NumPy only.
2. **Do not invent results.** If the model has not been trained or the frog has
   not been photographed, the report says pending. This has been deliberate
   throughout.
3. **Single-image reconstruction gives a relief, not a closed surface.** It
   recovers what faces the camera and nothing behind. Say so; do not overclaim.
4. **Reconstruction from web images does not work.** Downloaded photos are
   different instances of the object, not multiple views of one. Structure-from-
   motion needs the same physical object across frames. This killed the previous
   topic's 3D plan — do not propose it again.

## Files

```
reconstruct.py        photo -> segmentation -> depth -> textured OBJ
render3d.py           OBJ -> novel views / turntable video
run_progress_demo.py  end-to-end demo on a synthetic proxy; writes outputs/
colab_depth.ipynb     Depth Anything V2 on Colab; returns the .obj bundle
src/evaluate.py       hold-out scoring against withheld photographs
src/metrics.py        PSNR and SSIM on NumPy/OpenCV, no scikit-image
tests/test_pipeline.py
CAPTURE.md            how to photograph the frog (turntable protocol)
scraper.py            dataset collection for the classifier (not started)
training/train.py     MobileNetV2 transfer learning (not started)
docs/                 progress report (.docx) and the presentation script
outputs/              contact sheet + metrics from the demo (evidence)
output/metrics/       hold-out CSVs — the only source for numbers in the paper
data/                 photographs — gitignored, stays local
```

## What the evaluation measured

Frames are withheld from the 36-frame ring, each held-out angle is rendered from
the nearest frame the pipeline *did* see, and both that render and the nearest
captured photograph are scored against the withheld one. The baseline column is
the point: a render that cannot beat "just show the closest photograph" has not
earned its reconstruction stage.

### Two claims are supported. Do not write more than these.

**1. Learned depth beats the shape proxy, decisively.** Mean render quality rises
+1.31 dB across every spacing tested. Large, consistent, not marginal.

**2. At 60° spacing the render is structurally closer to the withheld photograph
than frame-switching is:** ΔSSIM **+0.024, 95% CI [+0.014, +0.035]**, n=30,
t=4.56. This is the only per-spacing result that clears its own error bars.

Everything else is noise. Paired per-position differences, full 36-frame ring:

| spacing | n | ΔPSNR (95% CI) | ΔSSIM (95% CI) |
|---|---|---|---|
| 20° | 18 | −0.19 [−0.55, +0.16] | +0.009 [−0.002, +0.020] |
| 40° | 27 | −0.23 [−0.58, +0.13] | +0.008 [−0.005, +0.021] |
| 60° | 30 | +0.22 [−0.02, +0.45] | **+0.024 [+0.014, +0.035]** |
| 90° | 32 | −0.24 [−0.51, +0.03] | +0.003 [−0.008, +0.015] |

**No PSNR result is significant at any spacing.** Every interval spans zero.

### How this was caught, so it is not repeated

An earlier version of this file claimed learned depth "moves the PSNR crossover
to ~40°", from a +0.03 dB gain. Re-running with one extra photograph (frame 300,
re-shot) turned that same number into −0.23 dB. **One frame in 36 moved the
result by more than the effect being reported.** The claim was noise given a
narrative.

So: gains under ~0.3 dB here mean nothing, and any per-spacing claim needs its
confidence interval computed from the per-position paired differences in
`output/full_e*/metrics.csv` — not from a difference of two means.

PSNR and SSIM disagree for a reason worth stating in the paper: PSNR rewards
sharp pixels and the baseline is a *real photograph* at the wrong angle, while
SSIM compares structure and the render is at the right angle. For novel-view
synthesis, structure is the property that matters.

Every number here comes from one eye-level ring. Nothing has looked down on the
frog yet.

Frame 300° is excluded from every run — the operator's hand is in the shot. Say
so in the paper rather than quietly dropping it.

## Running it

```bash
conda activate cv
pip install -r requirements.txt          # numpy + opencv only
python run_progress_demo.py              # regenerates outputs/
python tests/test_pipeline.py            # unittest, not pytest

# once real photos exist
python reconstruct.py data/frog/single/front.jpg --out model3d/frog
python render3d.py model3d/frog.obj --frames 36 --sweep 360 --video --out outputs/frog
```

Learned depth needs `requirements-depth.txt` (torch + transformers). There is no
GPU on the dev machine — run that stage on Colab and bring the `.obj` back.

## Gotchas

- `render3d.py --sweep` defaults to **80°**, not 360. A "turntable" without
  `--sweep 360` renders a small arc.
- After reconstructing, always open `model3d/<name>_mask.png` and check the
  segmentation before trusting the mesh. A **plausible `foreground:` percentage
  is not proof the mask is right** — an inverted mask reported 70.6% on a photo
  where the frog occupied 31%, because it had latched onto the background. Look
  at the image. If the background leaked in, try `--segment-channel saturation`
  (or `gray`), then `--threshold N`.
- The morphology kernel in `_refine` is deliberately small. Enlarging it closes
  the hollow resonator cavity, but it also bridges the frog to the table it
  stands on when the two are similar in tone — tested and rejected.
- `data/**` is gitignored. Photographs stay local — do not commit ~112 turntable
  images.
- Y-up mesh vs Y-down image coordinates: `render3d.py` flips this internally. If
  a render comes out mirrored, that is where to look.
- **Depth convention: larger value = further from the camera.** `build_mesh`
  writes depth straight into the vertex z and `render3d` sits on the low-z side.
  Monocular networks predict the opposite (inverse depth, nearest scores
  highest), so a raw prediction renders the frog inside-out — hollow and
  unlit. `orient_depth` decides by measuring the object against the backdrop
  rather than trusting a checkpoint's convention. If a relief looks caved in,
  check this before anything else.

## Next step

Shoot rings 2 and 3, at ~30° and ~60° camera elevation. Every result so far
comes from a single eye-level ring, and the evaluation has now put a number on
what that costs: past ~60° of spacing the relief has nothing left to show. A
comparable project reconstructed its subject from one eye-level ring and the
roof came back as a hole for exactly this reason.

After that, multi-view reconstruction (COLMAP or Apple Object Capture) becomes
possible, and the same `src/evaluate.py` scores it against the same withheld
photographs — so the mesh and the relief compare on one footing.

## Why this subject

The frog was chosen deliberately for its capture properties: small, matte, rigid,
and obtainable. That means it can be placed on a turntable and photographed from
every angle, so a genuine multi-view reconstruction is achievable.

This is also why rule 4 above exists. An earlier plan depended on reconstructing
an object that could only be found in online photographs, and that does not work:
those are photographs of different instances, not multiple views of one object.
Owning the physical object is what makes the reconstruction possible.
