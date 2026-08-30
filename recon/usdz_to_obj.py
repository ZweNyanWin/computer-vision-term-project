"""Convert an Object Capture .usdz into the OBJ bundle render3d.py already reads.

Object Capture writes USD; the rest of this project speaks OBJ + MTL. Converting
means the photogrammetry mesh goes through the *same* renderer and the *same*
hold-out evaluation as the single-image relief, so the two compare on one footing
instead of by eye.

Uses `usdcat` (shipped with macOS) to flatten the archive to text, then reads the
arrays out of it. No USD Python bindings required.

    python recon/usdz_to_obj.py recon/frog_ring_low.usdz --out model3d/frog_mv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np


def usda_text(usdz: Path) -> str:
    result = subprocess.run(
        ["usdcat", str(usdz)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout:
        raise SystemExit(f"ERROR: usdcat could not read {usdz}\n{result.stderr[:400]}")
    return result.stdout


def array(text: str, declaration: str) -> str | None:
    """Return the bracketed body of a named USD array attribute."""
    match = re.search(re.escape(declaration) + r"\s*=\s*\[(.*?)\]", text, re.S)
    return match.group(1) if match else None


def floats(body: str, width: int) -> np.ndarray:
    values = np.fromstring(body.replace("(", " ").replace(")", " ").replace(",", " "), sep=" ")
    return values.reshape(-1, width)


def integers(body: str) -> np.ndarray:
    return np.fromstring(body.replace(",", " "), sep=" ").astype(np.int64)


def extract_texture(usdz: Path, destination: Path) -> Path | None:
    """Pull the base-colour map out of the archive, ignoring the AO and normal maps."""
    with zipfile.ZipFile(usdz) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        colour = [n for n in names if "_tex" in n.lower()] or [
            n for n in names if not any(k in n.lower() for k in ("_ao", "_norm"))
        ]
        if not colour:
            return None
        data = archive.read(colour[0])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    # 8k textures make the renderer crawl for no visible gain at these sizes.
    longest = max(image.shape[:2])
    if longest > 4096:
        scale = 4096 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(destination), image)
    return destination


def convert(usdz: Path, out_base: Path) -> dict:
    text = usda_text(usdz)

    points_body = array(text, "point3f[] points")
    indices_body = array(text, "int[] faceVertexIndices")
    counts_body = array(text, "int[] faceVertexCounts")
    if not (points_body and indices_body and counts_body):
        raise SystemExit("ERROR: the USD file has no mesh arrays where expected")

    points = floats(points_body, 3)
    face_indices = integers(indices_body)
    counts = integers(counts_body)
    if not np.all(counts == 3):
        raise SystemExit("ERROR: expected a triangulated mesh; found non-triangle faces")

    st_body = array(text, "texCoord2f[] primvars:st")
    st_index_body = array(text, "int[] primvars:st:indices")
    uvs = floats(st_body, 2) if st_body else None
    st_indices = integers(st_index_body) if st_index_body else None

    # 'st' is faceVarying: one UV per face corner, so it is indexed by corner
    # position rather than by vertex. Written out in that same order.
    triangles = face_indices.reshape(-1, 3)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    obj_path = out_base.with_suffix(".obj")
    mtl_path = out_base.with_suffix(".mtl")
    texture_path = out_base.with_suffix(".png")
    name = out_base.name

    have_uv = uvs is not None and st_indices is not None and len(st_indices) == len(face_indices)
    texture = extract_texture(usdz, texture_path)

    with obj_path.open("w", encoding="utf-8") as obj:
        obj.write(f"# Object Capture mesh converted from {usdz.name}\n")
        if texture:
            obj.write(f"mtllib {mtl_path.name}\n")
        obj.write(f"o {name}\n")
        for x, y, z in points:
            obj.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if have_uv:
            for u, v in uvs:
                obj.write(f"vt {u:.6f} {v:.6f}\n")
        if texture:
            obj.write(f"usemtl {name}\n")
        for face in range(len(triangles)):
            a, b, c = triangles[face] + 1
            if have_uv:
                ta, tb, tc = st_indices[face * 3 : face * 3 + 3] + 1
                obj.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
            else:
                obj.write(f"f {a} {b} {c}\n")

    if texture:
        mtl_path.write_text(
            f"newmtl {name}\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
            f"Ks 0.000 0.000 0.000\nd 1.0\nillum 1\nmap_Kd {texture_path.name}\n",
            encoding="utf-8",
        )

    return {
        "vertices": len(points),
        "triangles": len(triangles),
        "uvs": len(uvs) if have_uv else 0,
        "obj": obj_path,
        "texture": texture,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usdz", type=Path)
    parser.add_argument("--out", type=Path, default=Path("model3d/frog_mv"))
    args = parser.parse_args()

    result = convert(args.usdz, args.out)
    print(f"vertices:  {result['vertices']}")
    print(f"triangles: {result['triangles']}")
    print(f"uvs:       {result['uvs']}")
    print(f"texture:   {result['texture']}")
    print(f"wrote:     {result['obj']}")


if __name__ == "__main__":
    main()
