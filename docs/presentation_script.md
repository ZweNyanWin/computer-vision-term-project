# 4-minute progress presentation — speaking script

A walkthrough of the report: the subject, the method, and current status.

---

## 0:00 – 0:40 · Subject

> "The subject is the Thai wooden frog, *kob mai* — a hardwood carving hollowed
> into a resonator, with a ridged back played by scraping a beater across it. It
> is simultaneously a woodcarving craft and a folk instrument, which is what makes
> it appropriate for the workshop.
>
> We also chose it for its geometry. It is small, rigid, matte, and has pronounced
> surface relief in the ridges. Those properties matter for reconstruction — matte
> surfaces reconstruct far more reliably than glossy ones, and the ridges give
> dense structure to match against.
>
> For context, our original topic was taken by another team, so this is a change
> of subject. The pipeline carried over unchanged."

---

## 0:40 – 2:40 · Method

**→ Table I, then Section III**

> "The report describes a four-stage pipeline from photographs to synthesised
> views.
>
> **Stage one, acquisition and segmentation.** The object is isolated from the
> background by Otsu thresholding, morphological closing and opening, then
> largest-connected-component selection with contour filling. It assumes a plain
> background, which we control at capture time rather than solve in software.
>
> **Stage two, depth estimation.** A monocular depth network — Depth Anything V2,
> with MiDaS as fallback — predicts a relative depth map from a single photograph.
> The output is normalised and Gaussian-smoothed, since the raw prediction carries
> high-frequency noise that would otherwise appear as surface roughness in the
> mesh.
>
> **Stage three, surface reconstruction.** We lay a regular grid over the
> segmented region and emit one vertex per sample inside the silhouette, taking
> depth as height. Each group of four neighbouring vertices becomes two triangles,
> provided all four fall inside the mask. Texture coordinates come directly from
> the source image, so the photograph itself becomes the texture map. It exports
> as OBJ with an MTL material.
>
> **Stage four, rendering** — the main contribution. Vertices are transformed into
> camera space and projected through a pinhole model, with intrinsics built from a
> chosen field of view. Hidden surfaces are handled in two passes: back-face
> culling on the sign of the projected signed area, then depth-sorted drawing.
> Because the proxy is a height field there is no interpenetrating geometry, so
> depth sorting is sufficient and a per-pixel z-buffer is unnecessary. Each
> visible triangle is textured by an affine warp between its texture coordinates
> and its projected coordinates, then shaded with a Lambertian term computed from
> the camera-space face normal.
>
> The renderer is OpenCV and NumPy only — no rendering engine. That was
> deliberate: the projection, hidden-surface and shading models are implemented
> rather than called."

---

## 2:40 – 3:20 · Limitation and the planned extension

> "The significant limitation is that single-image reconstruction yields a relief,
> not a closed surface — it recovers what faces the camera and nothing behind it.
>
> The planned extension is a turntable capture: ten-degree increments across three
> camera elevations, roughly a hundred images, fixed lighting and locked exposure,
> then structure-from-motion. That produces a complete surface, and it drops
> straight into the same renderer without changing anything downstream.
>
> That is the other reason for this object — it is physically obtainable and small
> enough to put on a turntable. Reconstruction from found images is not viable,
> because those are different instances of the object rather than multiple views
> of one."

---

## 3:20 – 3:50 · Status

**→ Table III, then Table II**

> "On status — the renderer is complete and measured. On a test object of 8,813
> vertices and 17,088 triangles it renders at roughly 1.1 seconds per frame, and a
> 24-frame turntable in 27 seconds, single-threaded on CPU. Culling removes about
> eight percent of triangles at forty degrees of yaw. Three defects were found and
> corrected during validation: a Y-axis convention mismatch between mesh and image
> coordinates, a default camera distance that clipped the object, and a sign error
> in the diffuse term.
>
> Reconstruction is implemented but the depth network has not been run. Capture,
> the classifier and the workshop station have not started. The report marks all
> of those pending rather than estimating them."

---

## 3:50 – 4:00 · Question

> "One point we would like your input on: whether to prioritise a full multi-view
> reconstruction of a single object, or a faster single-image path that
> generalises to whatever a visitor brings to the table."

---

## If you run long

Cut in this order:

1. The three defects (status section)
2. Stages one and two — name them, keep the detail on three and four
3. The segmentation method detail

Never cut: stage four, the relief limitation, or the closing question.

## Likely questions

**"Why not photogrammetry from the start?"**
> "It is the intended path — the object was chosen to make a turntable session
> possible. Single-image reconstruction is what runs today, which let the renderer
> be built and validated before capture."

**"How reliable is monocular depth?"**
> "It is relative, not metric — no absolute scale, so the relief parameter is set
> by inspection. Multi-view would give scale up to a similarity transform."

**"Why implement the renderer rather than use an existing one?"**
> "So the projection, hidden-surface and shading stages are our own
> implementation. It is roughly 200 lines on top of OpenCV."

**"Where is the training component?"**
> "Not started. The scraper and transfer-learning script carried over. We would
> like to confirm the categories with you before committing to a dataset."
