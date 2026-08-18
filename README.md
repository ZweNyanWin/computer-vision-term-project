# Image-Based 3D Rendering of the Thai Wooden Frog (Kob Mai)

CSX 4213 Computer Vision term project — Workshop on Thai Arts and Culture,
22 September 2026.

The subject is the Thai wooden frog (**กบไม้**, *kob mai*): a frog carved from a
single piece of hardwood, hollowed into a resonating chamber, with ridges cut
across its back. Drawing the wooden beater along the ridges produces a croaking
rasp, so it is a scraped idiophone in the same family as the guiro — and at the
same time a piece of Thai woodcarving.

The goal is **novel-view synthesis**: photograph the object, recover a textured
geometric proxy, then render it from viewpoints that were never photographed.
This follows Szeliski Chapter 14, *Image-Based Rendering*, and sits at the
explicit-geometry end of that spectrum.

## Pipeline

```
photographs ──► segmentation ──► depth estimation ──► surface reconstruction ──► novel-view rendering
                 (Otsu +           (Depth Anything      (height-field mesh,        (K[R|t]X, culling,
                  morphology)       V2 / MiDaS)          textured OBJ)              painter's, texture,
                                                                                    Lambertian shading)
```

| Stage | Method | Szeliski ch. |
|-------|--------|--------------|
| Acquisition, segmentation | Otsu, morphology, largest component | 2, 3 |
| Depth estimation | monocular network (pretrained) | 12 |
| Surface reconstruction | height-field mesh, OBJ + MTL | 13 |
| Projection | `x = K [R\|t] X` | 2 |
| Hidden-surface removal | back-face culling + painter's algorithm | 14 |
| Texture mapping | per-triangle affine warp | 3, 14 |
| Shading | Lambertian `n · l` | 2 |

No rendering engine is used — the renderer is built from the course's own camera,
transformation and reflectance models, on top of OpenCV and NumPy only.

## Setup

```bash
conda activate cv                # opencv + numpy
pip install -r requirements.txt  # torch/transformers only needed for depth
```

`render3d.py` runs anywhere. `depth_to_mesh.py` needs PyTorch, so run that stage
on Google Colab (free GPU) and bring the `.obj` back.

## Usage

**1 — reconstruct a mesh from one photo**

```bash
python depth_to_mesh.py data/frog/frog_front.jpg --out model3d/frog
# -> model3d/frog.obj  .mtl  .png  _depth.png
```

Useful flags: `--relief 0.2–0.5` (depth exaggeration), `--threshold N` if Otsu
picks the wrong side of the histogram, `--no-segment` to mesh the whole frame.

**2 — render novel views**

```bash
# single view
python render3d.py model3d/frog.obj --yaw 40 --pitch -10 --out renders/frog40

# turntable video
python render3d.py model3d/frog.obj --frames 36 --video --out renders/frog
```

Inspect a mesh by hand at <https://3dviewer.net> — drag in the `.obj`, `.mtl`
and `.png` together.

**3 — classification component** (`scraper.py` → clean → `training/train.py`)

MobileNetV2 transfer learning over Thai carved wooden handicraft categories, so
the system can name the object before reconstructing it. Classes are provisional;
see the TODO in `scraper.py`.

## Status

| Component | Status |
|-----------|--------|
| Novel-view renderer | done, measured |
| Depth-to-mesh reconstruction | implemented; depth network not yet run |
| Frog image capture | not started |
| Multi-view / turntable set | planned — see `docs/capture_guide.md` |
| Classifier | not started |
| Workshop station | not started |

Measured on a test object: 8,813 vertices / 17,088 triangles, 1.1 s per 560×560
frame, 24-frame turntable in 26.7 s, single-threaded CPU.

## Why this object

Unlike a museum artefact, the wooden frog is small, rigid, matte, cheap and
obtainable. It can be put on a turntable and photographed from every angle, which
makes genuine **multi-view** capture possible — and multi-view is what lifts the
reconstruction from a single-sided relief to a complete surface. That capture
protocol is in [`docs/capture_guide.md`](docs/capture_guide.md).

## Layout

```
depth_to_mesh.py       single image -> textured relief mesh
render3d.py            mesh -> novel views / turntable video
scraper.py             dataset collection for the classifier
training/train.py      MobileNetV2 transfer learning
data/                  input images (gitignored)
model3d/               reconstructed meshes (gitignored)
renders/               rendered output (gitignored)
docs/                  progress report, capture guide
```

## Team

- Zwe Nyan Win
- (teammate)
