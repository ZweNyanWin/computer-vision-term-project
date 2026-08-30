# Photographing the wooden frog

Two shoots in one session. **Shoot A** is five photos that feed the pipeline that
already runs. **Shoot B** is the turntable set for multi-view reconstruction.

Do A first and process it before starting B — if the background or lighting is
wrong you want to find out after five photos, not after a hundred.

---

## Setup

| Item | What to use |
|---|---|
| Background | Plain **dark** cloth or matte black card. The frog is pale wood; segmentation thresholds brightness, so contrast is what makes it work. |
| Surface | Sheet of paper on a table, so you can mark angles on it |
| Lighting | Two lamps, each through baking paper or a white t-shirt. One slightly left and above, one filling from the right. |
| Camera | Phone is fine. It must sit on something rigid — tripod, or a stack of books. |
| Turntable | Lazy Susan, or a plate you rotate by hand |

**Remove the beater stick** for every reconstruction photo. It occludes the ridges
and it is a second object — it would end up fused into the mesh. Shoot it
separately for `data/frog/context/`.

**Camera settings.** Turn HDR **off** — it flattens the shading the depth step
reads. No flash: it blows a specular highlight on the varnish, which reconstructs
as a dent. Lock focus and exposure (tap and hold on the frog until AE/AF lock
appears). Lock white balance if your camera app allows it. Do not zoom at any
point after this.

---

## Shoot A — five photos, straight away

Frog centred, filling about 70–80% of the frame, camera at the frog's eye level.

1. `front.jpg` — directly facing the front of the frog
2. `front_left.jpg` — rotate the frog ~20° left
3. `front_right.jpg` — ~20° right
4. `three_quarter.jpg` — ~45°, showing the ridged back and the side together
5. `top.jpg` — camera raised ~45°, looking down at the ridges

Into `data/frog/single/`.

**Then process one immediately:**

```bash
python reconstruct.py data/frog/single/front.jpg --out model3d/frog_front
```

Open `model3d/frog_front_mask.png`. The frog should be solid white on black. If
the background leaked in, fix it now — darker cloth, or pass `--threshold 90`
and adjust the number until the mask is clean. Only then continue.

---

## Shoot B — the turntable set

The camera does not move. The frog rotates. This is the opposite of walking
around the object, and it matters: a fixed camera with fixed lights means the
lighting on the object stays consistent as it turns.

**Mark your angles.** Draw a circle on the paper, mark **36 ticks 10° apart**
(every tick = 1/36 of the circle). Put the frog at the centre. Mark its starting
orientation so you can return to it.

**Ring 1 — camera level with the frog** → `data/frog/turntable/ring_low/`

1. Position the camera level with the frog. Lock exposure and focus.
2. Photograph. Rotate the frog one tick (10°). Photograph. Repeat.
3. 36 photographs, back to the start.

**Ring 2 — camera ~30° above** → `ring_mid/`
Raise the camera, aim down at the frog. Re-lock exposure. Repeat the 36 steps.

**Ring 3 — camera ~60° above** → `ring_high/`
Steeper still, looking well down onto the back. Repeat the 36 steps.

Finish with 3–4 frames from almost directly overhead, into `ring_high/` — these
close the top of the model.

That is ~112 photographs. Budget 30–40 minutes.

**Rules that decide whether this works**

- The lights belong to the room, not the turntable. If a lamp turns with the
  frog, the shading rotates with it and reconstruction fails.
- Your own shadow must not sweep across the object as you reach in to rotate it.
  Stand to the side; rotate, step back, then shoot.
- Do not re-lock exposure mid-ring. Only between rings.
- If you knock the frog off centre, put it back on its start mark and redo that
  ring.

---

## Where the files go

```
data/frog/
  single/      Shoot A - 5 photos, used by reconstruct.py today
  turntable/
    ring_low/    36 photos, camera level
    ring_mid/    36 photos, camera ~30 degrees above
    ring_high/   36 photos + overhead, camera ~60 degrees above
  context/     the frog with its beater, in the hand, being played
```

These folders are gitignored — the photographs stay on your machine, not in the
repository.

---

## After the shoot

Single-image path, works now:

```bash
python reconstruct.py data/frog/single/front.jpg --out model3d/frog
python render3d.py model3d/frog.obj --frames 36 --sweep 360 --video --out outputs/frog
```

Multi-view path: load `data/frog/turntable/` into COLMAP or Meshroom, export the
mesh, and render it with the same `render3d.py`.

**Record for the report:** number of images, angular step, number of rings, camera
model, whether exposure was locked, background and lighting used. For the
multi-view run, how many images registered out of the total — that figure is the
standard way to report how well a capture session went.
