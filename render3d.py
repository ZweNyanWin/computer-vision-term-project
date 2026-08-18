"""Render a textured 3D mesh from novel viewpoints (image-based rendering).

Second half of the 3D component. `depth_to_mesh.py` reconstructs the geometry;
this script renders it. Everything is done with the camera model from the course
rather than a game engine, so each stage maps onto lecture material:

    perspective projection   x = K [R|t] X          (Ch. 2, image formation)
    camera intrinsics K      focal length, principal point
    hidden-surface removal   back-face culling + painter's algorithm
    texture mapping          per-triangle affine warp (Ch. 3, geometric transforms)
    shading                  Lambertian, intensity proportional to n . l

Only OpenCV and NumPy are needed - no renderer dependency, runs in the `cv` env.

Run:
    python render3d.py model3d/frog.obj --out renders/frog --frames 36 --video
    python render3d.py model3d/frog.obj --yaw 30 --pitch -10          # single view
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np


# ------------------------------------------------------------------- obj ----
def load_obj(path):
    """Parse a textured OBJ. Returns (vertices, uvs, faces, texture_path)."""
    verts, uvs, faces = [], [], []
    mtl = None
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "vt":
            uvs.append([float(x) for x in parts[1:3]])
        elif parts[0] == "f":
            idx = []
            for p in parts[1:4]:
                bits = p.split("/")
                idx.append((int(bits[0]) - 1,
                            int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1))
            faces.append(idx)
        elif parts[0] == "mtllib":
            mtl = os.path.join(os.path.dirname(path), parts[1])

    tex = None
    if mtl and os.path.exists(mtl):
        for line in open(mtl):
            if line.startswith("map_Kd"):
                tex = os.path.join(os.path.dirname(mtl), line.split(maxsplit=1)[1].strip())
    if tex is None or not os.path.exists(tex):
        guess = glob.glob(os.path.splitext(path)[0] + ".png")
        tex = guess[0] if guess else None

    return (np.array(verts, np.float64), np.array(uvs, np.float64),
            faces, tex)


# ---------------------------------------------------------------- camera ----
def rotation(yaw_deg, pitch_deg):
    """Object rotation: yaw about +Y, then pitch about +X."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rx @ Ry


def intrinsics(w, h, fov_deg):
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


# ------------------------------------------------------------------ draw ----
def render(verts, uvs, faces, texture, w, h, yaw, pitch, dist, fov,
           bg=(18, 18, 20), shading="texture", ambient=0.38):
    """Render one view. Painter's algorithm: far triangles first."""
    R = rotation(yaw, pitch)
    t = np.array([0.0, 0.0, dist])
    K = intrinsics(w, h, fov)

    # mesh is Y-up; OpenCV image coordinates are Y-down, so flip on the way in
    flip = np.diag([1.0, -1.0, 1.0])
    cam = verts @ (flip @ R).T + t             # object -> camera space
    proj = (cam @ K.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = proj[:, :2] / proj[:, 2:3]       # perspective divide
    pts = np.nan_to_num(pts)

    frame = np.full((h, w, 3), bg, np.uint8)
    if texture is not None:
        th, tw = texture.shape[:2]

    fi = np.array([[a for a, _ in f] for f in faces])
    ti = np.array([[b for _, b in f] for f in faces])

    depth = cam[fi, 2].mean(axis=1)            # per-face camera depth
    order = np.argsort(-depth)                 # far to near

    light = np.array([0.30, -0.40, -1.0])   # toward the light, slightly up-left
    light /= np.linalg.norm(light)

    drawn = 0
    for k in order:
        vi = fi[k]
        if np.any(cam[vi, 2] <= 1e-6):         # behind the camera
            continue
        dst = pts[vi].astype(np.float32)

        # back-face culling via signed area of the projected triangle
        area = ((dst[1, 0] - dst[0, 0]) * (dst[2, 1] - dst[0, 1]) -
                (dst[2, 0] - dst[0, 0]) * (dst[1, 1] - dst[0, 1]))
        if area >= 0:          # flipped handedness: front faces wind clockwise
            continue

        x0, y0 = np.floor(dst.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue

        # Lambertian shade from the camera-space face normal
        a, b, c = cam[vi]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            shade = ambient
        else:
            nu = n / ln
            if nu[2] > 0:                 # make the normal face the camera
                nu = -nu
            shade = max(float(nu @ light), 0.0)
        shade = np.clip(ambient + (1 - ambient) * shade, 0, 1)

        local = dst - np.array([x0, y0], np.float32)
        bw, bh = x1 - x0, y1 - y0
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(mask, local.astype(np.int32), 255, cv2.LINE_8)
        if not mask.any():
            continue

        if shading == "texture" and texture is not None and ti[k][0] >= 0:
            src = (uvs[ti[k]] * [tw - 1, th - 1]).astype(np.float32)
            src[:, 1] = (th - 1) - src[:, 1]          # OBJ uv origin is bottom-left
            M = cv2.getAffineTransform(src, local)
            patch = cv2.warpAffine(texture, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        else:
            colour = (200, 200, 205)
            if texture is not None and ti[k][0] >= 0:
                uv = uvs[ti[k]].mean(axis=0)
                px = int(np.clip(uv[0] * (tw - 1), 0, tw - 1))
                py = int(np.clip((1 - uv[1]) * (th - 1), 0, th - 1))
                colour = texture[py, px].tolist()
            patch = np.full((bh, bw, 3), colour, np.uint8)

        patch = (patch.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
        roi = frame[y0:y1, x0:x1]
        np.copyto(roi, patch, where=mask[..., None].astype(bool))
        drawn += 1

    return frame, drawn


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj")
    ap.add_argument("--out", default="renders/view")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--frames", type=int, default=1,
                    help=">1 sweeps yaw a full turn (turntable)")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--sweep", type=float, default=360.0, help="turntable arc in degrees")
    ap.add_argument("--dist", type=float, default=3.6)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--shading", choices=["texture", "flat"], default="texture")
    ap.add_argument("--video", action="store_true", help="also write an mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    verts, uvs, faces, tex_path = load_obj(args.obj)
    texture = cv2.imread(tex_path) if tex_path else None
    print(f"mesh: {len(verts)} vertices, {len(faces)} faces")
    print(f"texture: {tex_path if texture is not None else 'none (flat shading)'}")

    # normalise: centre the mesh and scale it into a unit box
    centre = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts = verts - centre
    extent = np.abs(verts).max()
    if extent > 0:
        verts = verts / extent

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    w = h = args.size
    writer = None
    if args.video and args.frames > 1:
        writer = cv2.VideoWriter(args.out + "_turntable.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    t0 = time.time()
    for i in range(args.frames):
        yaw = args.yaw if args.frames == 1 else args.yaw + args.sweep * i / args.frames
        frame, drawn = render(verts, uvs, faces, texture, w, h,
                              yaw, args.pitch, args.dist, args.fov,
                              shading=args.shading)
        if args.frames == 1:
            path = args.out + ".png"
        else:
            path = f"{args.out}_{i:03d}.png"
        cv2.imwrite(path, frame)
        if writer is not None:
            writer.write(frame)
        print(f"  frame {i+1}/{args.frames}  yaw={yaw:6.1f}  {drawn} triangles  -> {path}")

    if writer is not None:
        writer.release()
        print("wrote", args.out + "_turntable.mp4")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
