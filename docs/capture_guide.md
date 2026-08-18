# Capture guide — photographing the wooden frog

The quality of the reconstruction is decided here, not in the code. A careless
capture session cannot be rescued by better software; a careful one makes the
rest of the pipeline easy.

Two capture modes are described. **Mode A** is the minimum needed to run the
current single-image pipeline. **Mode B** is the turntable set that unlocks
multi-view reconstruction, which is the planned next stage.

---

## Mode A — single photograph (works with the pipeline today)

You need one good frontal photograph.

**Background.** Plain and strongly contrasting with the wood. A sheet of dark
card or a matte black cloth is ideal — the segmentation step thresholds
brightness, so a busy background is the main cause of failure.

**Lighting.** Soft and even, from slightly above and to one side. Two desk lamps
through baking paper works. Avoid direct flash: it produces a specular hotspot on
the varnish, and the depth network reads a blown highlight as a surface feature.

**Camera.** Phone is fine. Fill the frame with the frog, leaving a small margin.
Shoot at eye level with the object, not looking down at it. Turn HDR off if you
can — it flattens exactly the shading cues the depth network relies on.

**Sanity check.** Run `depth_to_mesh.py` and open the `_depth.png` preview. The
frog should be brighter than its background and the ridges should be visible as
banding. If the depth map is uniform, the lighting was too flat.

---

## Mode B — turntable set (for multi-view reconstruction)

This is the session worth doing properly, because it removes the biggest
limitation of the current pipeline: a single photo yields only a relief, never
the far side of the object.

**Setup**

- Put the frog on something you can rotate in repeatable steps. A lazy Susan
  works; so does a plate on a sheet of paper with angle marks drawn every 10°.
- Mark the table so the object returns to the same spot if it is knocked.
- Fix the camera on a tripod or a stack of books. **The camera must not move.**
- Keep lighting fixed to the room, not to the turntable — the light must not
  rotate with the object.

**Shooting**

1. Rotate in steps of **10°**, one frame per step → 36 images per ring.
2. Repeat at three camera heights: roughly **level with the frog**, **30° above**,
   and **60° above**. That gives about 108 images.
3. Add a few frames looking almost straight down, to close the top.
4. Keep focus and exposure locked for the whole session. On a phone: tap and hold
   to lock AE/AF before you start.

**Rules that matter**

- Consecutive frames must overlap heavily — 10° steps guarantee this.
- Never change zoom mid-session. Changing focal length changes the intrinsics.
- No moving shadows. If your own shadow sweeps across the object as you rotate
  it, the reconstruction will contain that shadow as geometry.
- Matte objects reconstruct better than glossy ones. If the frog is heavily
  varnished, a light dusting of cornflour on an inconspicuous area is the
  standard trick — but only if the object is yours to treat that way.

**Processing**

Feed the image set to COLMAP or Meshroom for structure-from-motion and dense
reconstruction. Both are free. The output mesh drops straight into `render3d.py`
in place of the single-image relief.

---

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Segmentation grabs the background | busy or low-contrast backdrop | plain dark card; or `--threshold N` |
| Mesh looks flat | flat, frontal lighting | light from one side, raise `--relief` |
| Bumpy, noisy surface | blown highlights from flash | diffuse the light, no direct flash |
| SfM fails to register images | too few views, or the camera moved | 10° steps, tripod, locked exposure |
| Reconstruction has baked-in shadows | light rotated with the object | fix lights to the room |

---

## What to record for the report

- Number of images, angular step, number of rings
- Camera model, focal length, whether exposure was locked
- Lighting arrangement, background material
- For SfM: how many images registered successfully out of the total

That last figure is a genuine, quotable result — registration rate is the
standard way to report how well a capture session went.
