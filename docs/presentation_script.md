# 4-minute progress presentation — speaking script

Covers the two items requested: the selected topic, and the 3D rendering pipeline
(Chapter 14). Have the report open; the **→** marks point at it.

---

## 0:00 – 0:30 · Opening

> "Two things to cover, both from your brief: the topic we've selected in Thai
> arts and culture, and the computer vision pipeline for 3D rendering.
>
> One change first — our original topic, the Khon masks, was already taken by
> another team, so we've moved to a new subject. The pipeline we'd built carried
> over unchanged, which I'll come back to."

---

## 0:30 – 1:15 · The selected topic

**→ title page / Section I-A**

> "Our subject is the Thai wooden frog — *kob mai*.
>
> It's a frog carved from a single piece of hardwood, hollowed out into a
> resonating chamber, with a row of ridges cut across the back. You draw the
> wooden beater along the ridges and it makes a croaking rasp. So it's a scraped
> idiophone — the same family as the guiro — and at the same time a piece of Thai
> woodcarving. It's sold throughout Thai handicraft markets and visitors
> recognise it immediately.
>
> We also chose it for technical reasons. It's small, rigid, matte, and it has
> strong surface relief in those carved ridges. Most importantly we can actually
> obtain one and put it on a turntable — which our previous subject never allowed."

---

## 1:15 – 2:45 · The pipeline (Chapter 14)

**→ Section II-A, then Table I**

> "Chapter 14 is image-based rendering — synthesising new views of a scene from
> photographs, instead of modelling and shading it from scratch.
>
> The chapter organises these methods by how much geometry they use. At one end,
> light field and lumigraph rendering store a dense sample of rays and interpolate
> between them, with no geometric model at all. In the middle, view interpolation
> uses implicit geometry — correspondence, or per-pixel depth. At the other end,
> you recover an explicit surface and texture it from the source photographs.
>
> We're at the explicit-geometry end, and that's deliberate. Light field methods
> need hundreds of densely sampled views, which isn't realistic with a handheld
> camera. An explicit textured proxy renders convincingly from very few photos.
>
> Our pipeline has four stages.
>
> One, acquisition and segmentation — Otsu thresholding, morphology, largest
> connected component.
>
> Two, depth estimation — a monocular depth network gives a relative depth map
> from a single photograph.
>
> Three, surface reconstruction — that depth map becomes a triangulated
> height-field mesh, textured with the original photo, exported as an OBJ.
>
> Four, novel-view rendering, the Chapter 14 stage — project with x equals K,
> bracket R t, X; remove hidden surfaces by back-face culling and the painter's
> algorithm; texture each triangle with an affine warp; shade with Lambertian
> n dot l.
>
> All of that is course material. We used OpenCV and NumPy only — no rendering
> engine."

---

## 2:45 – 3:20 · What is actually built

**→ Table III, then Table II**

> "The renderer is complete and measured. On a test object — 8,813 vertices,
> 17,088 triangles — it renders a frame in about 1.1 seconds, and a 24-frame
> turntable in 27 seconds, single-threaded on CPU. Back-face culling takes it
> from 17,000 triangles down to about 15,700 at forty degrees of yaw.
>
> We validated it end to end and found and fixed three bugs: a vertical mirroring
> from a Y-up versus Y-down convention mismatch, a default camera distance that
> cropped the object, and a sign error in the Lambertian term.
>
> To be clear, that's a **test** object. We fixed the frog as our subject only
> recently, so there's no frog imagery yet — Table II lists everything that hasn't
> been started."

---

## 3:20 – 4:00 · Next steps, then a question

> "Next: get a physical frog, shoot a turntable set — ten-degree steps at three
> camera heights, about 108 images — and run structure-from-motion on it. That
> gives a complete surface instead of the single-sided relief a single photo can
> produce, and it's the dense input the stronger image-based rendering methods
> assume.
>
> One question. For the workshop, would you rather we prioritise a full
> multi-view reconstruction of one frog, or a faster single-image pipeline that
> works on any object a visitor brings to the table?"

---

## If you are running long

Cut, in this order:

1. The three bugs (2:45 section) — keep only "validated end to end"
2. The middle of the IBR spectrum — say "from no geometry, through implicit, to
   explicit; we're at the explicit end"
3. Stages 1–3 detail — name them, describe only stage 4

Never cut: the frog justification, stage 4, or the closing question.

## Likely questions

**"Why not photogrammetry from the start?"**
> "That's exactly where we're heading — the frog was chosen because it makes a
> turntable session possible. Single-image reconstruction is what runs today, so
> the renderer could be built and tested before capture."

**"Is monocular depth accurate?"**
> "It's relative, not metric — no absolute scale. That's why the relief parameter
> is set by inspection right now, and why multi-view is the next stage."

**"Why not use Blender or Open3D?"**
> "We wanted the projection, hidden-surface and shading stages to be the course's
> own models rather than library calls. It's about 200 lines on top of OpenCV."

**"Where's the classifier / training component?"**
> "Not started — the scraper and the transfer-learning script carried over from
> the previous topic. We'd like to confirm the categories with you before we
> spend an afternoon cleaning a dataset."
