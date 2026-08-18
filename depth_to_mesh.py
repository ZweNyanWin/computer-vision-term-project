"""Single-image 3D reconstruction: object photo -> textured relief mesh (.obj).

Recovers the geometric proxy that render3d.py then renders from novel viewpoints.
A depth map is estimated from ONE photograph (Chapter 12, monocular depth
estimation) and converted into a height-field mesh (Chapter 13).

The result is a relief: it represents the surface facing the camera, not the far
side of the object. That is inherent to single-image reconstruction. For a full
surface, capture a turntable set and run structure-from-motion instead - see
docs/capture_guide.md - then feed that mesh straight to render3d.py.

Pipeline:
    photo -> monocular depth -> foreground segmentation -> smoothing
          -> height-field grid mesh -> textured .obj / .mtl

Run (Colab free GPU recommended; torch is not installed locally):
    pip install torch transformers opencv-python
    python depth_to_mesh.py data/frog/frog_front.jpg --out model3d/frog

Outputs, written next to --out:
    <name>.obj   textured relief mesh
    <name>.mtl   material referencing the texture
    <name>.png   texture (the cropped source photo)
    <name>_depth.png   depth map preview, for the report
"""

import argparse
import os

import cv2
import numpy as np


# ----------------------------------------------------------------- depth ----
def estimate_depth(bgr):
    """Return a float32 relative-depth map in [0,1]; larger = nearer the camera.

    Tries Depth Anything V2 first, falls back to MiDaS. Both are pretrained:
    we are *reconstructing* a mesh with them, not training a 3D model.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        from transformers import pipeline
        from PIL import Image

        print("depth model: Depth Anything V2 (transformers)")
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf")
        depth = np.asarray(pipe(Image.fromarray(rgb))["depth"], dtype=np.float32)
    except Exception as exc:                                  # noqa: BLE001
        print(f"Depth Anything unavailable ({exc.__class__.__name__}), trying MiDaS")
        import torch

        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        tf = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        with torch.no_grad():
            pred = model(tf(rgb).to(device))
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)

    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        raise SystemExit("depth map is flat - the model failed on this image")
    return (depth - lo) / (hi - lo)


# ---------------------------------------------------------- segmentation ----
def foreground_mask(bgr, manual_thresh=None):
    """Binary mask of the object, assuming a reasonably plain background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if manual_thresh is not None:
        _, binary = cv2.threshold(blur, manual_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # the object should be the bright side; flip if Otsu chose the background
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    if binary[[0, -1], :].mean() > 127 or binary[:, [0, -1]].mean() > 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # keep only the largest connected component, then fill its holes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled > 0


# ------------------------------------------------------------------ mesh ----
def build_mesh(depth, keep, relief=0.35, grid=220):
    """Height-field mesh over `keep`. Returns (vertices, uvs, faces 1-indexed)."""
    h, w = depth.shape
    step = max(1, int(round(max(h, w) / grid)))
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)

    d = cv2.GaussianBlur(depth, (0, 0), 2.0)
    aspect = w / h

    index = -np.ones((len(ys), len(xs)), np.int64)
    verts, uvs = [], []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            if not keep[y, x]:
                continue
            index[r, c] = len(verts)
            # centre on the origin; x spans [-aspect/2, aspect/2], y spans [-.5,.5]
            verts.append((((x / w) - 0.5) * aspect,
                          (0.5 - (y / h)),
                          float(d[y, x]) * relief))
            uvs.append((x / w, 1.0 - y / h))

    faces = []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            a, b = index[r, c], index[r, c + 1]
            cc, dd = index[r + 1, c], index[r + 1, c + 1]
            if min(a, b, cc, dd) < 0:
                continue                      # quad touches the background
            # +1: OBJ indices are 1-based. CCW winding so normals face +z.
            faces.append((a + 1, cc + 1, dd + 1))
            faces.append((a + 1, dd + 1, b + 1))

    return np.array(verts, np.float32), np.array(uvs, np.float32), faces


def write_obj(path_base, verts, uvs, faces, texture_name):
    name = os.path.basename(path_base)
    with open(path_base + ".obj", "w") as f:
        f.write("# relief mesh - generated by depth_to_mesh.py\n")
        f.write(f"mtllib {name}.mtl\no {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uvs:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        f.write(f"usemtl {name}\n")
        for a, b, c in faces:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    with open(path_base + ".mtl", "w") as f:
        f.write(f"newmtl {name}\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
                f"Ks 0.000 0.000 0.000\nd 1.0\nillum 1\nmap_Kd {texture_name}\n")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="source photograph of the object")
    ap.add_argument("--out", default="model3d/object", help="output path base")
    ap.add_argument("--relief", type=float, default=0.35,
                    help="depth exaggeration; 0.2 flat, 0.5 pronounced")
    ap.add_argument("--grid", type=int, default=220, help="mesh resolution")
    ap.add_argument("--threshold", type=int, default=None,
                    help="manual background threshold 0-255 (default: Otsu)")
    ap.add_argument("--no-segment", action="store_true",
                    help="mesh the whole frame instead of segmenting")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    scale = 900 / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"image: {bgr.shape[1]}x{bgr.shape[0]}")

    depth = estimate_depth(bgr)

    if args.no_segment:
        keep = np.ones(depth.shape, bool)
    else:
        keep = foreground_mask(bgr, args.threshold)
        pct = 100 * keep.mean()
        print(f"foreground: {pct:.1f}% of pixels")
        if pct < 5 or pct > 95:
            print("  WARNING: segmentation looks wrong. Try --threshold N "
                  "or --no-segment, and check the _depth.png preview.")

    verts, uvs, faces = build_mesh(depth, keep, args.relief, args.grid)
    if len(faces) == 0:
        raise SystemExit("no faces produced - segmentation likely failed")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    texture = os.path.basename(args.out) + ".png"
    cv2.imwrite(args.out + ".png", bgr)
    write_obj(args.out, verts, uvs, faces, texture)

    vis = (depth * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis[~keep] = 0
    cv2.imwrite(args.out + "_depth.png", vis)

    print(f"vertices {len(verts)}  faces {len(faces)}")
    print(f"wrote {args.out}.obj / .mtl / .png / _depth.png")
    print("view it: https://3dviewer.net  (drag in the .obj, .mtl and .png together)")


if __name__ == "__main__":
    main()
