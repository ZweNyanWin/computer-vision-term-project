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
| Reconstruction (`reconstruct.py`) | **Done**, both paths, both running. Learned depth lives in `.venv-depth` (torch 2.14 on MPS, transformers 5.16) — there is no conda env on this machine, so `.venv-depth` replaces the `cv` env the older docs assumed. ~3 s a frame |
| Progress demo (`run_progress_demo.py`) | **Done** — end-to-end on a synthetic proxy |
| Presentation demo (`demo.sh`) | **Done — all eight steps run**, step 8 being the neural result. `./demo.sh check` reports READY. It previously printed `done` and exited 0 while four steps were broken; it now runs under `set -euo pipefail` and picks up `.venv-depth` automatically |
| Unit test (`tests/test_pipeline.py`) | **Passing** |
| Real frog photographs | **Three shoots done** — 5 hero shots, a closed 36-frame turntable ring, and 9 elevated photographs; plus the 88-photograph stationary-frog capture of 7 September (19 low / 37 mid / 24 high / 8 top). All in `data/`; all 36 ring frames segment cleanly |
| Hold-out evaluation (`src/evaluate.py`) | **Done and run** — full-ring runs are in `output/full_e*/` |
| Explicit reconstruction | **Done** — Apple Object Capture accepted all 45 ring/elevated photographs; 25,008 vertices, 49,999 triangles. Rebuilt 7 September with `./rebuild_object_capture.sh` after the artifacts were lost: 25,011 vertices, 50,000 triangles. **Object Capture is not deterministic**, so quote whichever run the report's figures come from and do not treat the two as the same mesh |
| Custom multi-view / structure-from-motion | **Done as a pipeline, not as our own solver** — COLMAP 4.1.1 registers 88/88 at 1.117 px through `run_neural.sh`. The SfM implementation is COLMAP's; ours is the two-camera handling, the split and the evaluation. Object Capture remains a separate black-box comparison |
| Neural multi-view reconstruction | **Done and held-out scored** — MLX3D 0.3.0 on Metal, 140,018 Gaussians in 22.9 min. Held-out novel view **20.02 dB / 0.7325 SSIM** against a nearest-photograph baseline of 13.77 dB / 0.5281, on 12 views withheld before training. See `output/neural/balanced/` |
| Classifier (`scraper.py`, `training/train.py`) | **Not started.** Carried over from an earlier topic |
| Workshop station | **Not started.** The splat is viewable now with `mlx3d-view model3d/gaussian/frog88_train_balanced/splat.ply`, but no workshop interface has been built |

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
NEURAL_CAPTURE.md     fixed-frog, moving-camera protocol for COLMAP + MLX3D,
                      and the record of what the 7 September shoot produced
run_neural.sh         the whole neural path: prepare -> sfm -> undistort ->
                      split -> train -> eval -> facts. Stages skip completed work;
                      training is NOT resumable and refuses to overwrite (RETRAIN=1)
rebuild_object_capture.sh  regenerates the gitignored Object Capture mesh,
                      usdz and turntable that demo.sh steps 6-7 read
.venv-depth/          torch + transformers for learned depth (gitignored)
.venv-mlx3d/          mlx3d + COLMAP tooling for the neural path (gitignored)
tools/prepare_neural_dataset.py  four ring folders -> one flat COLMAP-ready set
tools/check_capture.py           EXIF/blur pre-flight on a capture folder
tools/select_holdout.py          picks the held-out views from camera geometry
tools/make_train_workspace.py    removes the held-out views from the model
tools/eval_holdout.py            renders the withheld poses and scores them
tools/report_facts.py            the table NEURAL_CAPTURE.md asks the report for
config/neural_manifest.csv       every source photo, its optics and SHA-256
config/neural_split.csv          the frozen train/holdout split
output/neural/<quality>/         held-out metrics, summary and contact sheet
                      (sheets are cropped to the frog: the repo is public and the
                       frames are of a private home. Scores are full-frame)
scraper.py            dataset collection for the classifier (not started)
training/train.py     MobileNetV2 transfer learning (not started)
docs/                 progress report (.docx) and the presentation script
outputs/              contact sheet + metrics from the demo (evidence)
output/full_e*/       full-ring hold-out CSVs — the source for Table III
output/metrics/       earlier sweep and figure caches; check exclusions before reuse
data/                 photographs — gitignored, stays local
```

## What the evaluation measured

Frames are withheld from the 36-frame ring, each held-out angle is rendered from
the nearest frame the pipeline *did* see, and both that render and the nearest
captured photograph are scored against the withheld one. The baseline column is
the point: a render that cannot beat "just show the closest photograph" has not
earned its reconstruction stage.

### Two claims are supported. Do not write more than these.

**1. Learned depth improves on the shape proxy only while the relief still has
useful geometry.** Render quality rises +1.66 dB at 20° spacing and +1.64 dB at
40°, but falls to −0.10 dB at 90°, where the relief has little of the object
left to show. Do not describe this gain as consistent across spacings.

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

Every hold-out number here comes from one eye-level ring. Nine elevated
photographs were used by the explicit reconstruction, but they are not a closed
ring and do not enter the hold-out evaluation.

Frame 300° was re-shot after the operator's hand entered the original. The
replacement is included throughout the full 36-frame results.

## What the neural evaluation measured

Separate capture, separate protocol, separate claim. 88 photographs of a
**stationary** frog with a moving camera (`NEURAL_CAPTURE.md`), 12 of them
withheld before training, the Gaussian splat rendered at each withheld camera's
own recovered pose and intrinsics, and scored full-frame against the withheld
photograph. Baseline as always: the nearest photograph actually captured — here
restricted to the same lens, because this shoot used two.

### One claim is supported. Do not write more than this.

**The Gaussian splat renders a withheld viewpoint closer to the withheld
photograph than the nearest captured photograph is, on every view tested:**

| | mean | 95% CI | n | t | p |
|---|---|---|---|---|---|
| ΔPSNR | **+6.25 dB** | [+4.54, +7.95] | 12 | 8.06 | <0.0001 |
| ΔSSIM | **+0.204** | [+0.163, +0.246] | 12 | 10.94 | <0.0001 |

Held out: 20.02 dB / 0.7325. Baseline: 13.77 dB / 0.5281. 12 of 12 views win on
both metrics; worst case +2.69 dB and +0.106 SSIM. Source:
`output/neural/balanced/holdout_metrics.csv`.

### What this claim is not

- **Not comparable to the height-field results above.** Different capture,
  different object coverage, different baseline separation (8.7° here). Do not
  put the two in one table, and do not say Gaussian Splatting "beat" the relief
  pipeline — they were never measured against the same thing.
- **Not a training-view score.** MLX3D's own `mlx3d-eval` reports fit on training
  views (26.13 dB / 0.8232 here). The 6.1 dB gap between that and the held-out
  number is exactly why it must never be quoted as novel-view accuracy.
- **Not evidence the background reconstructs.** Scored inside the frog's own
  projected box (9.1% of frame) the render reaches 24.91 dB / 0.808; the rest of
  the frame is a room seen from too few angles and is largely smear. Report the
  full-frame number as the result and the object number as what it is made of.
- **Not free of the capture's faults.** Two lenses (40 of 88 frames are a
  digitally-cropped ultra-wide, iOS auto-macro), exposure never locked (1.36-stop
  spread), elevation covering only two bands and nothing above +53°. All measured,
  all in `NEURAL_CAPTURE.md`.
- **Not independent of the held-out views at initialisation.** SfM was solved over
  all 88 photographs before the split was applied, so **14,877 of the 41,431**
  points the model starts from (36%) were observed by a withheld view and bundle
  adjusted using it; the kept points are bit-identical to the full model's. No
  withheld pixel reaches the trainer, but the correct name for the result is
  *held-out photometric evaluation over a shared SfM initialisation*, not a
  reconstruction independent of the test views. Saying otherwise is wrong, and an
  earlier version of the docs did.
- **Not an untouched test set.** These same 12 views were used to compare vanilla
  against MCMC and to pick which to report. Both are published, and the choice
  changes neither number, but that is still model selection on the test set.
- **n = 12, one object, one session.** The intervals describe variation across
  twelve viewpoints of this frog. They say nothing about other objects or captures.

Already tested, do not redo: **MCMC densification** (`METHOD=mcmc`). It fixes the
one floater — `high_004` goes 15.53 → 21.03 dB — and reaches the same full-frame
held-out PSNR from a 9.3 MB file instead of 34.7 MB, but loses 4–5 dB of object
detail on the well-covered middle ring and wins only 5 of 12 views. MLX3D's MCMC
is fixed-budget, so it is also capacity-confounded (37,512 Gaussians vs 140,018).
Vanilla is the reported result; both runs are in `output/neural/`.

## Running it

**There is no conda on this machine**, so the `cv` environment the next block
names does not exist. The working equivalents are the system `python3` (numpy +
opencv, enough for the renderer and evaluation), `.venv-depth` (learned depth) and
`.venv-mlx3d` (the neural path). `./demo.sh check` reports what is missing.

```bash
# learned depth - what the `cv` env used to provide
python3 -m venv .venv-depth
.venv-depth/bin/python -m pip install -r requirements-depth.txt

conda activate cv                        # only if you have such an env
pip install -r requirements.txt          # numpy + opencv only
python run_progress_demo.py              # regenerates outputs/
python tests/test_pipeline.py            # unittest, not pytest

# real single-image relief
python reconstruct.py data/90.jpeg --depth-mode model --out model3d/frog
python render3d.py model3d/frog.obj --frames 9 --sweep 80 --video --out outputs/frog

# full turntable only from the closed Object Capture mesh
python render3d.py model3d/frog_combined.obj --frames 36 --sweep 360 --video --out outputs/frog_3d
```

The neural path is separate — its own venv, its own driver, every stage
re-runnable, and it goes end to end in about 40 minutes on the M1 Pro:

```bash
./run_neural.sh all balanced
```

Or one stage at a time: `prepare`, `sfm`, `undistort`, `split`,
`train <fast|balanced|best>`, `eval <quality>`, `facts`. `METHOD=mcmc` swaps the
densification strategy and writes to its own workspace, leaving the reported
vanilla result untouched.

**"Resumable" means different things per stage.** `prepare`/`sfm`/`undistort`/
`split` skip work already completed, guarded by sentinels written only on success.
**Training is not resumable at all** — MLX3D 0.3.0 rebuilds the model from
`GaussianModel.from_points` every time, so re-running restarts from iteration 0.
It now refuses when `splat.ply` exists; `RETRAIN=1` overrides deliberately. A
quality workspace built from a different split is refused outright, and `eval`
verifies the split against the model the splat was actually trained on. **Do not run plain `mlx3d-capture` on this dataset** —
it forces one camera model onto two lenses and trains a pinhole rasteriser on
distorted pixels; `NEURAL_CAPTURE.md` explains both.

```bash
mlx3d-view model3d/gaussian/frog88_train_balanced/splat.ply   # inspect the result
```

Learned depth needs `requirements-depth.txt` (torch + transformers). It runs
locally on Apple silicon through MPS in about three seconds per frame. The first
run downloads about 100 MB; warm the presentation cache with
`DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3`. Normal `demo.sh` runs cache-only so a poor
venue connection cannot stall it. `colab_depth.ipynb` remains a fallback for a
machine without `torch`.

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
- `data/**` is gitignored. The source photographs stay local; do not commit them.
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

The neural branch is finished and scored: capture, 88/88 registration, held-out
evaluation, and the limitations all recorded above and in `NEURAL_CAPTURE.md`.
What remains is the **workshop station** — the interactive piece the 22 September
deliverable actually is. `mlx3d-view` already serves the splat in a browser, so
the open question is what a visitor does with it, not whether it renders.

Two things would materially improve the reconstruction if there is time, in this
order:

1. **A second elevation band.** The capture has two, +15° and +43°, and nothing
   above +53°. That is the one axis where coverage is genuinely thin, and it is
   why the top of the frog is the weakest part of the model. A short ring shot
   properly from above — main lens, above 30 cm so auto-macro never engages —
   would cost an hour and is the highest-value addition.
2. **A re-shoot with exposure actually locked.** ISO and shutter drifted 1.36
   stops across this session, and Gaussian Splatting bakes that in as haze. The
   field card already says how; it was not followed.

Neither is required for the report. Both are honest improvements rather than
larger claims, and the current numbers stand without them.

Do not compare the Object Capture mesh against the 45 photographs that built it
as if it were prediction — such a score measures fit instead. The same rule is
why the neural result withholds 12 views before training rather than after.

## Why this subject

The frog was chosen deliberately for its capture properties: small, matte, rigid,
and obtainable. That means it can be placed on a turntable and photographed from
every angle, so a genuine multi-view reconstruction is achievable.

This is also why rule 4 above exists. An earlier plan depended on reconstructing
an object that could only be found in online photographs, and that does not work:
those are photographs of different instances, not multiple views of one object.
Owning the physical object is what makes the reconstruction possible.
