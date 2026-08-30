"""Render a textured OBJ mesh from novel viewpoints using OpenCV and NumPy.

Implemented stages: pinhole projection x = K[R|t]X, back-face culling,
far-to-near painter's ordering, per-triangle affine texture mapping, and
Lambertian shading. No external 3D renderer is used.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]], Path | None]:
    """Load vertices, texture coordinates, triangulated faces, and texture path."""
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[tuple[int, int]]] = []
    material_path: Path | None = None

    with path.open("r", encoding="utf-8", errors="replace") as obj:
        for raw_line in obj:
            parts = raw_line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(value) for value in parts[1:4]])
            elif parts[0] == "vt" and len(parts) >= 3:
                uvs.append([float(value) for value in parts[1:3]])
            elif parts[0] == "mtllib" and len(parts) >= 2:
                material_path = path.parent / " ".join(parts[1:])
            elif parts[0] == "f" and len(parts) >= 4:
                polygon: list[tuple[int, int]] = []
                for token in parts[1:]:
                    indices = token.split("/")
                    vertex_index = _obj_index(indices[0], len(vertices))
                    uv_index = -1
                    if len(indices) > 1 and indices[1]:
                        uv_index = _obj_index(indices[1], len(uvs))
                    polygon.append((vertex_index, uv_index))
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ has no renderable mesh: {path}")

    texture_path: Path | None = None
    if material_path and material_path.exists():
        with material_path.open("r", encoding="utf-8", errors="replace") as material:
            for raw_line in material:
                parts = raw_line.split(maxsplit=1)
                if parts and parts[0] == "map_Kd" and len(parts) == 2:
                    texture_path = material_path.parent / parts[1].strip()
                    break
    if texture_path is None:
        candidate = path.with_suffix(".png")
        texture_path = candidate if candidate.exists() else None

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(uvs, dtype=np.float64),
        faces,
        texture_path,
    )


def normalize_mesh(vertices: np.ndarray) -> np.ndarray:
    """Center a mesh and scale its largest half-extent to one unit."""
    centered = vertices - (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    extent = float(np.abs(centered).max())
    return centered / extent if extent > 0 else centered


def rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)
    rotate_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float64,
    )
    rotate_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float64,
    )
    return rotate_x @ rotate_y


def intrinsics(width: int, height: int, field_of_view: float) -> np.ndarray:
    focal = (width / 2.0) / np.tan(np.radians(field_of_view) / 2.0)
    return np.array(
        [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )


def render_frame(
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: list[list[tuple[int, int]]],
    texture: np.ndarray | None,
    width: int = 480,
    height: int = 480,
    yaw: float = 0.0,
    pitch: float = -5.0,
    distance: float = 3.8,
    field_of_view: float = 40.0,
    ambient: float = 0.45,
) -> tuple[np.ndarray, int]:
    """Render one frame and return the image plus the drawn-triangle count."""
    transform = rotation(yaw, pitch)
    translation = np.array([0.0, 0.0, distance])
    camera_matrix = intrinsics(width, height, field_of_view)

    # OBJ geometry is Y-up while OpenCV images are Y-down.
    coordinate_flip = np.diag([1.0, -1.0, 1.0])
    camera_vertices = vertices @ (coordinate_flip @ transform).T + translation
    homogeneous = camera_vertices @ camera_matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    projected = np.nan_to_num(projected)

    background = (25, 25, 28)
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    vertex_indices = np.asarray([[vertex for vertex, _ in face] for face in faces])
    uv_indices = np.asarray([[uv for _, uv in face] for face in faces])
    face_depth = camera_vertices[vertex_indices, 2].mean(axis=1)
    drawing_order = np.argsort(-face_depth)

    light = np.array([0.30, -0.40, -1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    texture_height, texture_width = texture.shape[:2] if texture is not None else (0, 0)

    drawn = 0
    for face_number in drawing_order:
        indices = vertex_indices[face_number]
        if np.any(camera_vertices[indices, 2] <= 1e-6):
            continue
        destination = projected[indices].astype(np.float32)
        signed_area = (
            (destination[1, 0] - destination[0, 0]) * (destination[2, 1] - destination[0, 1])
            - (destination[2, 0] - destination[0, 0]) * (destination[1, 1] - destination[0, 1])
        )
        if signed_area >= 0:
            continue

        x0, y0 = np.floor(destination.min(axis=0)).astype(int)
        x1, y1 = np.ceil(destination.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue

        point_a, point_b, point_c = camera_vertices[indices]
        normal = np.cross(point_b - point_a, point_c - point_a)
        normal_length = float(np.linalg.norm(normal))
        if normal_length < 1e-12:
            diffuse = 0.0
        else:
            unit_normal = normal / normal_length
            if unit_normal[2] > 0:
                unit_normal = -unit_normal
            diffuse = max(float(unit_normal @ light), 0.0)
        shade = float(np.clip(ambient + (1.0 - ambient) * diffuse, 0.0, 1.0))

        local = destination - np.array([x0, y0], dtype=np.float32)
        box_width, box_height = x1 - x0, y1 - y0
        mask = np.zeros((box_height, box_width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local.astype(np.int32), 255, cv2.LINE_8)
        if not mask.any():
            continue

        face_uvs = uv_indices[face_number]
        if texture is not None and np.all(face_uvs >= 0):
            source = (uvs[face_uvs] * [texture_width - 1, texture_height - 1]).astype(np.float32)
            source[:, 1] = (texture_height - 1) - source[:, 1]
            affine = cv2.getAffineTransform(source, local)
            patch = cv2.warpAffine(
                texture,
                affine,
                (box_width, box_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            patch = np.full((box_height, box_width, 3), (175, 185, 205), dtype=np.uint8)

        patch = np.clip(patch.astype(np.float32) * shade, 0, 255).astype(np.uint8)
        region = frame[y0:y1, x0:x1]
        np.copyto(region, patch, where=mask[..., None].astype(bool))
        drawn += 1

    return frame, drawn


def render_views(
    obj_path: Path,
    output_dir: Path,
    yaws: list[float],
    pitch: float = -5.0,
    size: int = 480,
) -> tuple[list[Path], list[int], float]:
    """Render a list of yaw angles and write numbered PNG files."""
    vertices, uvs, faces, texture_path = load_obj(obj_path)
    vertices = normalize_mesh(vertices)
    texture = cv2.imread(str(texture_path)) if texture_path else None
    if texture_path and texture is None:
        raise ValueError(f"could not read texture: {texture_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    triangle_counts: list[int] = []
    started = time.perf_counter()
    for index, yaw in enumerate(yaws):
        frame, drawn = render_frame(
            vertices, uvs, faces, texture, width=size, height=size, yaw=yaw, pitch=pitch
        )
        path = output_dir / f"view_{index:02d}_yaw_{yaw:+05.1f}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"could not write {path}")
        paths.append(path)
        triangle_counts.append(drawn)
    elapsed = time.perf_counter() - started
    return paths, triangle_counts, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/render"), help="output filename prefix")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=-5.0)
    parser.add_argument("--sweep", type=float, default=80.0, help="total yaw arc for multiple frames")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise SystemExit("ERROR: --frames must be at least 1")
    if args.size < 64:
        raise SystemExit("ERROR: --size must be at least 64")

    try:
        vertices, uvs, faces, texture_path = load_obj(args.obj)
        vertices = normalize_mesh(vertices)
        texture = cv2.imread(str(texture_path)) if texture_path else None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.frames == 1:
        yaws = [args.yaw]
    else:
        start = args.yaw - args.sweep / 2.0
        yaws = np.linspace(start, start + args.sweep, args.frames).tolist()

    writer = None
    video_path = args.out.parent / f"{args.out.name}_turntable.mp4"
    if args.video and args.frames > 1:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.size, args.size)
        )
        if not writer.isOpened():
            raise SystemExit("ERROR: OpenCV could not create the MP4 video")

    started = time.perf_counter()
    try:
        for index, yaw in enumerate(yaws):
            frame, drawn = render_frame(
                vertices,
                uvs,
                faces,
                texture,
                width=args.size,
                height=args.size,
                yaw=yaw,
                pitch=args.pitch,
            )
            output_path = (
                args.out.with_suffix(".png")
                if args.frames == 1
                else args.out.parent / f"{args.out.name}_{index:03d}.png"
            )
            cv2.imwrite(str(output_path), frame)
            if writer is not None:
                writer.write(frame)
            print(f"frame {index + 1}/{args.frames}: yaw={yaw:+.1f}, {drawn} triangles -> {output_path}")
    finally:
        if writer is not None:
            writer.release()

    print(f"mesh: {len(vertices)} vertices, {len(faces)} triangles")
    print(f"texture: {texture_path if texture is not None else 'none'}")
    if args.video and args.frames > 1:
        print(f"video: {video_path}")
    print(f"elapsed: {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
