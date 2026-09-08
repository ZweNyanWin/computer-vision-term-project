# Four-minute demonstration — 8 September 2026 machine-learning checkpoint

Use the saved results and the offline workshop. Do not train or rebuild frames
on stage: the balanced run took 22.9 minutes and MLX3D 0.3.0 cannot resume a
partly completed training run.

## Preflight before class

From Terminal:

    cd "/Users/zwenyanwin/Desktop/computer-vision-term-project"
    ./demo.sh check
    ./run_workshop.sh check
    open workshop/frog-station-standalone.html

Both checks must print **READY**. Leave the terminal and workshop open. As an
optional visual fallback, also pre-open:

    open output/neural/balanced/holdout_contact_sheet.jpg

## 0:00–0:25 — What changed

Show the frog or the workshop title.

> "Last week I had a learned single-image depth relief and a conventional
> 45-photo mesh. After the teacher asked for a machine-learning method, I built
> a multi-view 3D Gaussian Splatting pipeline. For this capture the frog stays
> still and the camera moves around it."

The key research question is whether the learned representation can synthesize a
withheld viewpoint more faithfully than simply displaying the nearest photograph.

## 0:25–0:55 — Show the workflow

Run:

    sed -n '17,24p' run_neural.sh

Point to the six stages: prepare, SfM, undistort, split, train, evaluate/facts.

> "COLMAP is the geometric front end: it estimates camera intrinsics, poses and
> sparse structure. MLX3D is the learning stage: it optimizes the positions,
> shapes, opacity and view-dependent appearance of the 3D Gaussians from the
> training photographs."

Do not call COLMAP itself machine learning, and do not describe 3DGS as a
watertight mesh.

## 0:55–1:20 — State the capture facts

> "There are 88 photographs: 19 low, 37 middle, 24 high and 8 top. EXIF revealed
> two phone optics, so I calibrated two camera models and undistorted both
> groups. COLMAP registered 88 out of 88 images at 1.117-pixel mean reprojection
> error. I froze 76 views for Gaussian fitting and 12 for photometric
> evaluation."

The folder names are shooting groups, not four distinct elevation rings.
Recovered geometry contains one low band near +15 degrees and one broad upper
band near +43 degrees.

## 1:20–2:30 — Compare 1, 45 and 88 photographs

Use the browser workshop. Click **1 photograph** and drag near the rear view;
the relief becomes a sliver. Without changing the angle, click **45
photographs** to show the closed mesh. Then click **88 photographs** and drag
through several views.

> "One photograph gives only a textured relief. Forty-five photographs give a
> closed, printable photogrammetry mesh. Eighty-eight photographs train a learned
> Gaussian appearance representation. The balanced Gaussian run used 7,000
> iterations and took 22.9 minutes on the M1 Pro. The evaluated checkpoint has
> 140,018 Gaussians. For this display only, 58,981 connected foreground
> Gaussians are isolated and composited on black; the evaluated model and scores
> are unchanged."

This wording matters: the room was isolated during presentation rendering, not
removed from the trained checkpoint.

## 2:30–3:20 — Show the held-out result

Return to Terminal and run:

    ./demo.sh 8

> "At each held-out camera pose I compare the Gaussian render with the real
> withheld photograph. The baseline is the nearest retained photograph from the
> same lens, averaging 8.7 degrees away. Full-frame performance is 20.02 dB
> PSNR and 0.7325 SSIM for the render, versus 13.77 dB and 0.5281 for the
> baseline. The paired improvements are +6.248 dB and +0.2044 SSIM, both
> p less than 0.0001. The render wins all 12 views on both metrics."

Do not call the 26.13 dB training score accuracy; it is fit to observed views.

## 3:20–3:45 — Show qualitative evidence

Scroll the workshop to **Does it actually work?** Point to the withheld-view
comparison and the shared-SfM caveat. If the browser is difficult to read, show
the pre-opened contact sheet.

> "The 12 test photographs contributed no pixels to the Gaussian loss. However,
> camera poses and sparse initialization were first solved with all 88
> photographs, so this is held-out photometric evaluation over a shared SfM
> initialization, not a fully independent reconstruction."

## 3:45–4:00 — Limits and conclusion

> "The remaining limitations are exposure drift, only two effective elevation
> bands, no underside, one object and one capture session, and later use of the
> same 12 views when comparing vanilla and MCMC variants. The requested
> machine-learning workflow is complete for this checkpoint; broader capture and
> a training-only SfM study would strengthen future work."

Stop there.

## Do not run live

- **./run_neural.sh all balanced** — long and training is not resumable.
- **./run_workshop.sh** with no argument — rebuilds 108 frames.
- **mlx3d-view .../splat.ply** — the raw checkpoint includes the learned room.
- **./demo.sh** with no step — it runs the older geometric story.

## Fallback commands

    open workshop/index.html
    open workshop/frames/splat/f_009.jpg
    open output/neural/balanced/holdout_contact_sheet.jpg
    sed -n '/^HELD-OUT/,/^$/p;/^PAIRED/,/^$/p' output/neural/balanced/holdout_summary.txt

## Likely questions

**Why does the raw splat look flesh-like or cloudy?**  
The optimization learned the room and pale support together with the frog.
Unconstrained background geometry and exposure drift become translucent
Gaussians. The workshop selects the connected frog cloud for display without
changing the scored checkpoint.

**Is this really machine learning if COLMAP is used?**  
Yes. COLMAP estimates cameras and sparse initialization. MLX3D performs
thousands of gradient-based optimization steps to learn the Gaussian scene
parameters that reproduce the training images.

**Were the test photographs completely unseen?**  
Their pixels were excluded from Gaussian fitting, but all 88 views contributed
to the initial SfM solution. Say "held-out photometric evaluation over a shared
SfM initialization."

**Can the Gaussian model be 3D-printed?**  
No. It is an appearance cloud, not a watertight surface. Use the 45-photo mesh
when printable geometry is required.

**Do the four folder names mean four elevation rings?**  
No. Recovered camera geometry shows two effective bands and no view above
+53.1 degrees.
