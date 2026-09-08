# Neural capture protocol

Dataset for the MLX3D Gaussian Splatting extension. Separate from the existing
turntable ring, which stays as the learned-depth and Object Capture baseline.

Every number below was measured on this machine (MacBook Pro 18,3, M1 Pro,
16 GB) and on the existing photographs, not taken from a tutorial.

## Why the pilot ring cannot be reused

Not for the reason usually given. The 36-frame ring **does** reconstruct: COLMAP
4.1.1 registers 36/36 with 0.617 px mean reprojection error, and the recovered
camera centres form a clean planar ring (planarity ratio 0.045, radius std/mean
0.07).

It works only because the backdrop is so featureless that it contributes no
verified matches. That is fragile, and it breaks under load: the identical
command on the **full-resolution** originals fails outright — *"No good initial
image pair found"*. At 4032 px the bland backdrop yields just enough features to
assert the camera never moved, which contradicts the rotating frog, and the
degenerate zero-baseline solution wins.

MLX3D passes the raw folder to COLMAP with no downscaling
(`capture/colmap_wrap.py:71-86`, `capture/pipeline.py:162`), so a fixed-camera
ring is exactly the configuration that fails. Move the camera; keep the frog
still.

For the report, the honest mechanism is: a fixed camera with a rotating object
embeds two rigid motions in one sequence; the static set is explained by a
zero-baseline homography where the essential matrix and translation are
undefined, and RANSAC's larger consensus set wins.

## What actually went wrong last time

Both faults are invisible without reading EXIF, and both were present:

| Fault | Evidence in the existing photographs |
|---|---|
| Exposure never locked | 9 distinct shutter speeds and ISO 1000/1250/1600 within one 36-frame ring |
| Lens/zoom changed | 35 mm-equivalent varies 24 / 35 / 37 / 49 / 64 across the project; the ring used the 2.22 mm f/2.2 ultra-wide with a digital crop, `ring_high/` used the 6.765 mm f/1.78 main camera |
| Too little light | 1/40 s at ISO 1250 — slow enough to blur handheld, noisy enough to hurt matching |

SIFT survives all of this, so COLMAP will still report a healthy registration
rate. Gaussian Splatting does not: it fits a photometric loss, so brightness and
colour drift are baked in as haze.

Two things you do **not** need to worry about, both measured:

- The frog is not low-texture — 8,236 to 15,862 SIFT keypoints per image. No
  chalk, no talc, no spray.
- The ridged back is not a repetition hazard — only ~1% of descriptors have a
  confusable twin, and Lowe's ratio test discards ambiguity rather than
  mismatching it.

## Field card

**Camera — set once, never touch again**

- Main lens at **1×**. Never 0.5×, never 2×, never pinch-zoom.
- Distance ≈ **2.5–3× the frog's length** (a 12 cm frog → 30–36 cm). The frog
  fills about a quarter of the frame width. This is correct; do not fill the
  frame.
- Stay above 30 cm so iOS never switches to the ultra-wide for auto-macro.
- **Do not crop, ever.** Cropping moves the principal point and silently
  corrupts intrinsics under `--single_camera 1`.

**Elevation, in centimetres** (at a 40 cm working distance, measured from the
frog's centre)

| Ring | Angle | Camera height above frog | Horizontal distance | Frames |
|---|---|---|---|---|
| Low | ~10° | 7 cm | 39 cm | 45 |
| Mid | ~30° | 20 cm | 35 cm | 45 |
| High | ~55° | 33 cm | 23 cm | 40 |
| Top-oblique | ~70° | — | — | 10 |

Total **140 frames**. The high ring is chest height at a table — no ladder.

140 is deliberate over-shooting. The pilot's hand-marked 10° steps came out with
a median of 8.6° but jumps of 42.4°, 29.9° and 27.7°, and one 21.3° reversal.
Gaps are cheap to prevent and impossible to fix afterwards. COLMAP cost at 140
frames is ~2 min extraction + ~11 min matching (measured at 0.69 s/image and
0.066 s/pair on your own full-resolution frames), so frame count is not the
constraint — under-shooting is.

## Before you start

**iPhone settings**

- Settings → Camera → Formats → **Most Compatible**. MLX3D's image whitelist
  excludes `.heic`/`.heif` entirely, so HEIC files are silently skipped.
  Converting afterwards is worse — several routes bake the rotation flag into
  the pixels and produce mixed image sizes, which corrupts intrinsics.
- Settings → Camera → **turn off Auto Macro**. This is the setting that caused
  the 2.22 mm ultra-wide frames.
- Turn off Live Photos, filters, and any Photographic Style.
- Landscape orientation, held the same way for all 140 frames.

**Locking exposure properly.** Tap-and-hold gives AE/AF LOCK, which pins focus
and the metering target — but it does **not** lock white balance and does not
pin a specific ISO/shutter pair. That is why the pilot drifted despite being
"locked". Use a manual app that fixes ISO, shutter, white balance (Kelvin) and
focus explicitly — Halide, ProCam or Lightroom Mobile, all of which write JPEG.
Set it once, before the first frame, for the whole session.

**Light.** Aim for 1/80 s or faster; that is the floor, because blur is
unrecoverable and noise is not. Priority order: no motion blur first, no
specular hotspot second, low ISO third. ISO 640–800 with a braced phone beats
ISO 400 under one hard lamp. Bring diffused sources *close* — 20–40 cm from the
frog, just outside the frame — rather than bouncing off a distant ceiling.
Close the curtains and use artificial light only: the session runs 60–120
minutes and passing cloud shifts exposure by 1–2 stops.

**Scene.** One rigid body: the frog, whatever it rests on, the mat and every
marker must never move relative to each other. Only the camera moves.

- Put the frog where you can walk a full circle — a stool or table in open
  floor, not against a wall. A 360° orbit needs about 1.5 m of clearance.
- Markers must each look *different* from every other, be matte, and be at least
  fingernail-sized. No identical coins, nothing metallic, no rice grains — they
  vanish at training resolution and identical objects are exactly the repetition
  hazard to avoid. Printed random text, a patterned cloth, mismatched bits of
  card or LEGO all work.
- Remove the beater stick. It is a second object and would fuse into the model.
- Decide the frog's position before the first frame and do not adjust anything
  afterwards — not the stool, not a lamp, not a marker.

## Shooting

Walk the circle. Stop, aim at the frog's centre, brace, shoot, check, step.
Never shoot while moving. Hold the frog at the same size in the frame — that is
the practical way to keep distance constant.

Do not rotate the frog when you reach the back. Walk behind it.

Your own shadow will cross the object somewhere on a 360° orbit; that is
unavoidable, not a discipline failure. Raise the lights and aim them down so the
shadow falls outside the frame.

Every 10–15 frames, pinch-zoom into one and check the wood grain is sharp. A
blurred frame is re-shot from the same position — never by moving the frog.

Shoot the whole set in one continuous session, including the frames that will
later become the test set. Do not shoot the held-out views as a separate pass.

## After the shoot

Copy originals — preserving EXIF — to `data/neural_capture/all/`, then verify
before running anything:

```bash
python tools/check_capture.py data/neural_capture/all
```

This must report a single value for pixel dimensions, focal length and aperture.
Any frame that disagrees is deleted and its position re-shot; the single-camera
assumption the whole run rests on is otherwise silently broken.

## Reconstruction on this machine

Measured trainer speed (M1 Pro, 16 GB, sh_degree 3, `low_memory=True`):

| Gaussians | Resolution | ms/iteration | Peak memory |
|---|---|---|---|
| 300 k | 1008×756 | 270 | — |
| 1.2 M | 1008×756 | 661 | 3.5 GB |
| 1.2 M | 1344×1008 | 973 | — |
| 2.5 M | 1344×1008 | 2216 | 6.4 GB |
| 4.0 M | 1344×1008 | 3766 | 10.2 GB |

So the presets cost, on this machine:

- `--quality fast` — the registration gate, ~10–15 min
- `--quality balanced` — 7,000 iterations at 1008×756, **40–80 min**. This is
  the preset that produces the reportable result.
- `--quality best` — 30,000 iterations at 1344×1008, **~8.1 hours** at the
  1.2 M cap. Overnight and unattended only, after `balanced` has already
  succeeded.

**`--low-mem` is mandatory here, not optional.** Without it `max_gaussians` is
`None` (`capture/pipeline.py:230`) so growth is unbounded; on a 16 GB machine
where macOS already holds 4–6 GB, an uncapped run reaches the wall, swaps, and
dies hours in. The 1.2 M cap is a hardware constraint to disclose in the report,
not a modelling choice — at the cap the densification threshold goes to infinity
so detail plateaus (`splatting/trainer.py:399-405`).

Filename order does not matter: with COLMAP installed, MLX3D always uses the
exhaustive matcher for a photo directory (sequential is gated on video input).

```bash
source .venv-mlx3d/bin/activate
mlx3d-capture data/neural_capture/all \
  --quality fast --low-mem --no-viewer \
  --out model3d/gaussian/frog_gate
```

Gate before spending 80 minutes: about 90% or more registered, camera centres
forming three coherent rings, front/sides/back/top all present, no split
clusters or duplicated frogs. MLX3D prints only the registered total, so a low
count means finding the gap yourself.

Inspect with:

```bash
mlx3d-view model3d/gaussian/frog_gate/splat.ply --fast
```

## Evaluation rule

Choose an evenly distributed 10–15% test set — 20 of 140, spread across
horizontal angle, elevation and all sides — and record it in a split manifest
before final training. 120 views reach the trainer.

`mlx3d-eval` computes PSNR, SSIM and L1 on **training** views. It has no
held-out split and no LPIPS. Its number is training fit, not novel-view
accuracy, and must never be reported as the latter. Held-out claims require
rendering the saved splat at the withheld camera poses and scoring against the
withheld photographs with `src/metrics.py`.

## What the 7 September capture actually produced

Everything below was measured on the 88 photographs that were shot, not planned.
The protocol above is unchanged and still correct; this section records where the
shoot departed from it, and what had to change downstream as a result.

**Headline.** 88 photographs — 19 low, 37 middle, 24 high, 8 top-oblique. COLMAP
4.1.1 registered **88/88 into a single model**: 45,192 points, 217,867
observations, mean track length 4.82, **mean reprojection error 1.117 px**. That
clears the 90% gate in `CLAUDE.md` outright.

### The shoot used two lenses, and the protocol predicted why

`check_capture.py` reports NOT READY on this set, correctly:

| | main camera | second group |
|---|---|---|
| frames | 48 | 40 — the whole middle ring, plus 3 low frames |
| pixels | 5712x4284 | 4032x3024 |
| EXIF focal | 6.765 mm f/1.78 | 2.22 mm f/2.2 |
| EXIF 35 mm-equivalent | 24 | **24** |
| measured horizontal FOV | 68.8 deg | **68.2 deg** |
| median distance to the frog | 5.52 units | 2.81 units |

The two groups report the same 35 mm-equivalent and measure the same field of
view, so the second group is not a wider view of the scene: it is the **ultra-wide
sensor digitally cropped back to the main lens' framing**. That is iOS auto-macro,
which engages below roughly 30 cm — and the second group was shot at 0.51x the
distance of the first. The protocol names this exact failure twice ("turn off Auto
Macro", "stay above 30 cm so iOS never switches to the ultra-wide"). The mechanism
is confirmed rather than inferred: EXIF, recovered focal length and recovered
standing distance all agree.

Nothing was re-shot. The frames are sharp and correctly exposed for their own
settings, and 40 of 88 is too much of the capture to discard. They are instead
handled honestly, which is what the rest of this section is about.

### Why `mlx3d-capture <folder>` is the wrong command for this dataset

The one-liner in "Reconstruction on this machine" would silently damage this
capture in two separate ways.

1. **One camera fitted to two lenses.** MLX3D runs COLMAP with
   `--ImageReader.single_camera 1` (`capture/colmap_wrap.py:71-86`). That does not
   error on a mixed set; it forces one focal length and one principal point across
   both optics. `run_neural.sh` runs COLMAP itself with `single_camera 0`, which
   gives one camera per optic — and COLMAP duly created exactly two, splitting the
   set 48/40 with no supervision.
2. **A pinhole rasteriser trained on distorted pixels.** MLX3D's default
   projection is the analytic EWA path, which ignores distortion coefficients
   entirely (`splatting/projection.py`, `splatting/render.py:64`). Training against
   OPENCV-distorted photographs therefore bakes the mismatch into the Gaussians. So
   the images are passed through `colmap image_undistorter` first, exactly as the
   reference 3DGS pipeline does; both cameras become PINHOLE with zero distortion,
   and MLX3D is handed the result with `--poses existing`.

### The rings are two elevation bands, not four

The camera positions were solved for, then the orbit axis was fitted to them: for
the true axis every camera in a ring shares one elevation, so the axis is the
smallest eigenvector of the pooled within-ring scatter. It is well determined here
— in-plane scatter exceeds along-axis scatter by 63.8x — and every ring holds its
elevation to within 6.2 deg standard deviation, which is the check that the fit is
real.

| ring | n | planned elevation | **measured** | sd |
|---|---|---|---|---|
| low | 19 | ~10 deg | **+14.8** | 5.2 |
| mid | 37 | ~30 deg | **+40.9** | 4.5 |
| high | 24 | ~55 deg | **+46.0** | 4.9 |
| top | 8 | ~70 deg | **+42.5** | 6.2 |

The middle, high and top sets sit within 6 deg of each other. The capture has one
low band near +15 deg and one broad upper band near +43 deg — **two** distinct
elevations, not the four the folder names imply, and no view above +53 deg. The
folder names are kept because they record how the shoot was walked, and they still
stratify the split usefully, but they are not evidence of elevation. Elevation
coverage is the one axis on which this capture is genuinely thin, and the report
should say so rather than repeat the planned figures.

### The held-out split

12 of 88 views (13.6%) are withheld, allocated across rings in proportion to their
size and spaced evenly around the azimuth circle within each: low 3, mid 5, high 3,
top 1, spanning 3 deg to 290 deg of azimuth. The split is chosen from recovered
camera *position* only — no image content, no training signal — and is fixed in
`config/neural_split.csv` with each photograph's SHA-256 before any Gaussian is
fitted.

The withheld views are then deleted from the sparse model with `colmap
image_deleter`, so MLX3D never sees them: 3,761 of the 45,192 points go with them,
leaving 41,431 to initialise from. Deleting from the solved model rather than
re-running SfM on 76 images keeps every surviving pose bit-identical, so the
withheld cameras stay directly usable for rendering.

**What the split does not claim.** This is the part most easily overstated, so it
is measured rather than described. Counts are reproducible from the two models
with the track data in `points3D.bin`:

| | |
|---|---|
| points surviving into training | 41,431 |
| …of those, observed by a withheld view in the full model | **14,877 (36%)** |
| points dropped with the withheld views | 3,761 |
| …that had **no** training observation at all | **49** |
| …that had exactly one, and so fell below COLMAP's minimum track length of 2 | 3,712 |
| kept points whose position or colour changed | **0** |

So two things are true at once. No withheld pixel reaches the trainer and no
Gaussian is fitted to one — the photometric loss runs over the 76 training
photographs only. But structure-from-motion was solved over all 88 before the
split was applied, so a third of the surviving points were bundle adjusted using
withheld observations, and deleting an observation afterwards does not undo that
influence. The kept points are bit-identical to the full model's.

The honest name for the result is therefore **held-out photometric evaluation over
a shared SfM initialisation**, which is the standard 3DGS / Mip-NeRF-360
arrangement. An earlier version of this section said the 3,761 dropped points were
"witnessed only by withheld views"; that was wrong, as the table shows — nearly all
of them simply lost their second observation.

A stricter protocol does exist, and the claim that joint reconstruction is the only
option was also wrong: COLMAP can register images into an existing reconstruction
([FAQ](https://colmap.github.io/faq.html#register-localize-new-images-into-an-existing-reconstruction)),
so one could reconstruct from the 76 training images alone and localise the
withheld cameras into that fixed model. That is a different experiment with its own
number, and it is not what was run here.

### What was actually run, and what it cost

All on the M1 Pro, 16 GB, with `./run_neural.sh`:

| stage | time |
|---|---|
| feature extraction, 88 images, OPENCV, `single_camera 0` | 4.8 min |
| exhaustive matching, guided, 3,828 pairs | 6.2 min |
| mapper | 3.3 min |
| `image_undistorter` | ~2 min |
| training, `--quality balanced --low-mem` | **22.9 min** |
| held-out evaluation, 88 renders + scoring | ~2 min |

The match graph is dense for a 360 deg orbit: **1,789 of 3,828 possible pairs
verified (46.7%)**, median 232 inliers. That figure is from the run that built the
reported model; re-running the matcher gives 1,786, because RANSAC is stochastic
and COLMAP is not seeded here. Differences of that size do not move the
reconstruction — the re-matched database still yields 45,192 points at 1.117215 px.

`--low-mem` caps growth at 1.2 M Gaussians. It never bound: the balanced run
finished at **140,018** Gaussians, 12% of the cap, so on this capture the cap is
not the limit on detail that the table above warns it can be. The limit here is
view coverage, not memory.

### Held-out result

12 withheld views, rendered at their own recovered poses and intrinsics and scored
full-frame against the withheld photographs. The baseline is the nearest training
photograph **taken with the same lens**, at a mean separation of 8.7 deg.

| | PSNR | SSIM |
|---|---|---|
| training fit (**not a result** — the splat saw these pixels) | 26.13 dB | 0.8232 |
| **held out, render** | **20.02 dB** | **0.7325** |
| held out, nearest-photograph baseline | 13.77 dB | 0.5281 |

Paired per-view differences, render minus baseline:

| | mean | 95% CI | n | t | p |
|---|---|---|---|---|---|
| ΔPSNR | **+6.25 dB** | [+4.54, +7.95] | 12 | 8.06 | <0.0001 |
| ΔSSIM | **+0.204** | [+0.163, +0.246] | 12 | 10.94 | <0.0001 |

Both intervals clear zero comfortably, and the result is unanimous: the render
beats the nearest real photograph on **12 of 12** views on both metrics, worst case
+2.69 dB and +0.106 SSIM. That is a categorically stronger result than the
height-field pipeline reached on the eye-level ring, where only ΔSSIM at 60 deg
cleared its error bars. The two are **not comparable** — different capture,
different object coverage, different baseline separation — and must not be placed
in one table as if they were.

The 6.1 dB gap between training fit and held-out performance is the honest cost of
generalisation, and it is why `mlx3d-eval`'s training-view number must never be
quoted as a novel-view result.

The `fast` preset is close behind. Its gate run reached 19.28 dB / 0.7178 held out
— already ΔPSNR +5.49 and ΔSSIM +0.200 over the baseline — in 6.5 minutes against
`balanced`'s 22.9. Moving to `balanced` bought +0.74 dB on average and improved 11
of 12 views.

That gain **cannot be attributed to longer training**. The presets change three
things at once: 3,000 → 7,000 iterations, 951x709 → 1141x851, and SH degree 2 → 3,
which together also took the model from 78,888 to 140,018 Gaussians. Separating
those would need one-factor-at-a-time runs that were not done. "More of everything
helps a little" is the whole claim the two runs support.

### Where it fails, and why

Two failure modes, both worth reporting rather than hiding.

A note on the evidence first, because it bears on what the contact sheets can and
cannot show. `output/neural/*/holdout_contact_sheet.jpg` is **cropped to the frog**
— the frames are of a room in a private home and this repository is public. That
crop is presentation only: every dB and SSIM figure quoted anywhere is computed on
the **whole photograph**, and the sheets label their numbers `full` for that reason.
The consequence is that the sheets are good evidence for the first failure below
being *absent* on the object, and no evidence at all for the second, which happens
in the part of the frame the crop removes. That one is carried by the numbers.
`tools/eval_holdout.py --sheet-pad` widens the window — large values open it out to
essentially the whole frame (4:3 aspect is enforced, so a few pixels stay cropped) — if you want to look at the backgrounds locally. Do not commit
what that produces.

**The room is reconstructed far worse than the frog.** Scoring only inside the
frog's own projected bounding box — 9.1% of the frame, defined from the sparse
geometry rather than drawn by hand, and applied identically to render, photograph
and baseline — gives 24.91 dB / 0.808 for the render against 12.33 dB / 0.214 for
the baseline. So most of the full-frame error is background: metres away, seen
across a narrow range of angles, and resolved as smear. The frog, which is what the
workshop is about, is reconstructed well. The full-frame number stays the headline
result; this one says what it is made of.

**Under-constrained background produces floaters.** In `high_004` the balanced run
put a dark, semi-transparent blob in front of the camera, dropping the whole frame
to 15.53 dB from the fast run's 20.34 dB and pulling mean brightness to 102.9
against the photograph's 123.2. It is the one view that got worse with more
capacity, and the mechanism is standard 3DGS behaviour: a region observed from too
few angles is cheaply explained by a large translucent Gaussian near the camera.
Unlocked exposure was checked as a cause and ruled out — that view sits +0.01 stops
from its ring's mean.

### Testing the floater: MCMC densification

The floater has a documented remedy, so it was tested rather than asserted. The
same split, the same evaluation, the same preset, with `METHOD=mcmc` — MLX3D's
MCMC relocation instead of vanilla clone/split/prune.

It fixed the floater, and only the floater:

| | vanilla | MCMC |
|---|---|---|
| Gaussians | 140,018 | **37,512** |
| `splat.ply` | 34.7 MB | **9.3 MB** |
| training time | 22.9 min | 19.5 min |
| training fit | 26.13 dB | 23.48 dB |
| **held out, full frame** | **20.02 dB / 0.7325** | 19.98 dB / 0.7227 |
| held out, object region | **24.91 dB / 0.808** | 23.68 dB / 0.692 |
| ΔPSNR vs baseline | +6.25 [+4.54, +7.95] | +6.21 [+4.13, +8.29] |
| `high_004`, the floater view | 15.53 dB | **21.03 dB** |

`high_004` gains **+5.50 dB** — by far the largest single change in either
direction, and the object region there improves too. That is the mechanism
confirmed: a fixed Gaussian budget cannot spend capacity on a large translucent
blob in an under-observed region.

It is not free. MCMC wins on only 5 of 12 views full-frame and 3 of 12 in the
object region, and loses 4-5 dB of object detail on the well-covered middle ring,
where vanilla's extra capacity is doing real work. The two are also **not a clean
A/B**: MLX3D's MCMC is fixed-budget (`splatting/model.py:relocate_mcmc` keeps N
constant), so it trained with 37,512 Gaussians against vanilla's 140,018. The
comparison is capacity-confounded and is reported as such.

**Vanilla stays the reported result.** MCMC is worth knowing about for two
reasons: it is the fix if floaters ever dominate, and it reaches the same
full-frame held-out PSNR from a 9.3 MB file — a quarter the size — which is the
number that will matter when the splat has to load in a browser at the workshop.

Two honest caveats on this comparison. It is **one paired run**, so it shows that
the floater went away under MCMC, not that fixed budget is the *mechanism* — the
two runs also differ in Gaussian count, and either could explain it. And choosing
vanilla on the strength of these scores is model selection on the held-out set;
both runs are reported precisely so the choice is visible rather than hidden.

### Limitations to state in the report

1. **Two lenses, and the second is a cropped ultra-wide.** Handled with per-optic
   intrinsics and undistortion, not ignored — but 40 of 88 frames come from the
   lower-resolution sensor at half the standing distance.
2. **Exposure was never locked.** ISO 250-640 and 1/48-1/59 s, a 1.36-stop spread
   across the session. Gaussian Splatting fits a photometric loss, so that drift is
   baked into the model as haze.
3. **The slowest frames are 1/48 s**, below the 1/80 s handheld floor in the field
   card, and five frames sit under the sharpness floor. Nothing was re-shot.
4. **Elevation coverage is thin.** Two bands, +15 deg and +43 deg, nothing above
   +53 deg, and nothing below +8 deg. The frog's underside is unobserved, as the
   section below describes.
5. **Poses and sparse points were solved over all 88 photographs**, the standard
   3DGS protocol; 36% of the surviving points were bundle adjusted using withheld
   observations, though no withheld pixel reached the trainer. Call the result
   held-out photometric evaluation over a shared SfM initialisation.
6. **The 12 views were also used to compare alternatives.** The vanilla-versus-MCMC
   comparison, and the choice to report vanilla, were made by looking at these same
   held-out scores. That is model selection on the test set. It is a weak form of
   it — both runs are reported in full, and the choice does not change either
   number — but the split stopped being a wholly untouched test set the moment it
   was used to pick between two models, and a genuinely clean confirmation would
   need views withheld from this decision too.
7. **n = 12, one object, one session.** The intervals describe how much the
   render-minus-baseline difference varies **across these twelve viewpoints of this
   frog**. They are not evidence about other objects, other captures, or Gaussian
   Splatting in general, and they are computed from per-view paired differences
   with a t distribution rather than from a difference of two means.
8. **Two of the three exported splats contained non-finite Gaussians** — 3 rows in
   `fast`, 146 in MCMC, with NaN in position, scale and rotation; the reported
   `balanced` artifact has none. MLX3D's compactor selects on opacity and
   importance and has no finite-value filter, so they were exported. The evaluator
   now drops them and records the count; re-running every evaluation with that gate
   reproduced all three metrics CSVs byte-for-byte, which shows those Gaussians were
   already invisible to the tile binner rather than quietly contributing. They are
   still invalid as portable viewer assets.

## The bottom of the frog

The base is in contact with the support surface and is observed by no camera.
Suggested report wording:

> The base of the object was in contact with the support surface throughout
> capture and is therefore observed by no camera. Geometry in that region is
> unconstrained by the data; the reconstruction is valid for viewpoints above
> the support plane, and renderings from below are extrapolation rather than
> reconstruction. This is a property of single-session object-on-table capture,
> not a failure of the method.

That wording is honest, and it is what the current model supports. But if the
brief asks for a genuinely closed object — top **and** underside — the section
below is how to shoot it. Read it instead of improvising, because the obvious
approach is the one that fails.

## Shooting the whole sphere: top and bottom

### Why you cannot just flip it over

Turn the frog upside down for a second session and COLMAP sees a room that did
not move and an object that jumped. It resolves that contradiction the wrong way:
the background is large, textured and static, so it wins the registration, and the
upside-down views get placed as if the *camera* went under the floor. You get one
model with a second frog fused into it at the wrong orientation, or two models
that will not merge. The two reconstructions also carry independent scale, so
stitching them afterwards is a second problem on top of the first.

There are two ways out. Take the first if the frog will balance.

### Method A — lift it off the table (recommended)

Put the frog on a support **much narrower than the frog itself**, so the camera can
see underneath in the same session. A small jar, a spice bottle, a bottle neck, a
cardboard tube — 8 to 15 cm tall, and narrower than the frog's footprint. Fix it
down with Blu-Tack or museum putty so it cannot slip or rotate; a wobble mid-session
is the same failure as flipping it.

Now the only unobserved geometry is the contact patch, a couple of square
centimetres, instead of the entire base. One session, one rigid scene, no merging,
no scale problem.

Two things this changes about the room:

- **Background.** Low shots point the camera upward, and a blank ceiling gives
  COLMAP nothing to match — the same featureless-backdrop failure that killed the
  full-resolution pilot ring. Put something patterned where the low shots will look:
  a printed sheet, a patterned cloth, a bookshelf in frame. Clutter is your friend
  here.
- **Light.** The underside will be in shadow. Put a white sheet of paper or card on
  the floor under the support to bounce light up into it, or add a low lamp. Do not
  fix this by raising ISO after you have started; exposure is locked for the whole
  session.

**The rings.** Elevation is measured from the frog's own centre, so 0 degrees is
level with it and negative is looking up from below.

| Ring | Elevation | Frames | Notes |
|---|---|---|---|
| Top-down | +85 deg | 6 | almost straight down |
| High | +60 deg | 16 | |
| Upper-mid | +35 deg | 24 | |
| Level | 0 deg | 24 | phone at the frog's own height |
| Lower-mid | −35 deg | 24 | crouch, or lower the stool |
| Low | −60 deg | 16 | phone near the floor, angled up |
| Bottom-up | −85 deg | 6 | phone flat on the floor, lens up |

About **116 frames**. The two extreme rings are small because a 6-frame circle at
85 degrees already covers that cap; the middle rings carry the reconstruction.

Put the support on a stool at roughly chest height and the negative rings become
comfortable rather than acrobatic — that is the whole reason for the stool.

### Method B — two sessions, if it will not balance

Only if the frog cannot be supported. The trick is to make the background move
*with* the object, so there is no static scene to mis-register against.

1. Glue or putty the frog onto a small board — a book, a placemat, a sheet of stiff
   card — and cover the board with **rich, non-repeating** markers. Printed random
   text works; identical coins do not.
2. Shoot everything above the board.
3. Lift board and frog **together**, turn the whole assembly over, and shoot again.
   The frog never moves relative to the board.
4. Shoot against a plain, featureless surround so the room contributes no matches
   at all — the opposite of the advice in Method A, and the reason Method A is
   easier.

Now the marker constellation is rigid with respect to the frog in both sessions,
and COLMAP can register them into one model. The board's own face is lost where it
touches the frog, which is fine.

### Settings — the same field card, and it was not followed last time

Everything in **Field card** and **Before you start** above still applies. Three of
them are the reason the 7 September capture came out the way it did, so they are
worth repeating:

- **Stay above 30 cm.** Auto-macro silently switched 40 of the 88 frames to the
  ultra-wide sensor. Turn Auto Macro off in Settings → Camera as well.
- **Lock exposure with a manual app**, not tap-and-hold. ISO drifted 250–640 and
  shutter 1/48–1/59 across the last session, and Gaussian Splatting bakes that in
  as haze.
- **1/80 s or faster**, and Most Compatible so the files are JPEG.

### Check the coverage before you trust it

Both checks already exist, and the second is the one that would have caught the
last capture's real weakness:

```bash
.venv-mlx3d/bin/python tools/check_capture.py data/neural_capture/all   # optics, exposure, blur
./run_neural.sh sfm && ./run_neural.sh undistort && ./run_neural.sh split
```

`split` prints the elevation actually achieved per ring. Last time it revealed that
the four folders were really two bands, +15 and +43 degrees, with nothing above
+53 — the folder names were not evidence. Read that table before shooting again;
it will tell you whether you got the top and bottom you were aiming for, in
degrees, from the reconstruction itself.

### If there is only time for one thing

Do Method A. The top of the current model is its weakest region — nothing was shot
above +53 degrees — so the elevated-support sphere fixes the measured weakness and
the missing underside in the same session.

## Record for the report

Image count, registered count, minimum and maximum camera elevation actually
achieved, camera model, whether exposure was locked and how, lighting, MLX3D and
MLX versions, resolution, iterations, training time, peak memory, final Gaussian
count and `splat.ply` size.

Every one of those is read back out of the artifacts rather than transcribed, so
the table in the report cannot drift from the run that produced it:

```bash
./run_neural.sh facts balanced     # -> output/neural/report_facts.txt
```

Two entries on that list need a caveat. **Lighting** is not recoverable from the
files and has to be written down by whoever shot it. **Peak memory** was not
instrumented for the vanilla run — `run_neural.sh` now wraps training in
`/usr/bin/time -l` and writes `<workspace>/time.txt`, but that was added
afterwards. It did capture the MCMC run: **3.45 GB peak footprint** (3.21 GiB), 0.95 GiB
maximum resident, for 7,000 iterations at 1141x851 with 41,431 Gaussians. Memory
was never the binding constraint either way — the vanilla run finished at 140,018
Gaussians against a 1.2 M cap, and the measured table above puts 1.2 M at 3.5 GB
on this 16 GB machine.

Both `data/` and `model3d/` are gitignored, so the photographs, the COLMAP models
and `splat.ply` are **not** in the repository and must be backed up separately.
What *is* tracked, and is enough to reproduce and to audit the result:
`config/neural_manifest.csv` (every source photograph with its optics and
SHA-256), `config/neural_split.csv` (the frozen split), `output/neural/*/` (the
metrics, summaries and contact sheets) and `run_neural.sh` with `tools/`.

The repository is public, so the tracked contact sheets are cropped to the frog.
Nothing else in `output/neural/` contains imagery. If more of the capture is ever
committed, check first what is in frame — the shoot was done in a private home,
and `data/**` is gitignored for that reason as much as for size.
