"""Run the complete offline checkpoint demo with one command.

The script creates a clearly labelled synthetic wooden-object proxy because the
real Thai wooden frog has not been purchased yet. It then demonstrates:

1. Otsu segmentation and morphology
2. a synthetic relative-depth proxy (temporary, not a learned prediction)
3. height-field mesh reconstruction and OBJ/MTL export
4. novel-view projection, culling, texture mapping, and Lambertian shading
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from reconstruct import reconstruct_photo
from render3d import render_views


ROOT = Path(__file__).resolve().parent


def create_demo_photo(path: Path, width: int = 640, height: int = 480) -> Path:
    """Create a deterministic frog-like wooden proxy on a plain background."""
    rng = np.random.default_rng(4213)
    canvas = np.full((height, width, 3), (226, 229, 232), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.ellipse(mask, (315, 265), (205, 105), -4, 0, 360, 255, cv2.FILLED)
    cv2.ellipse(mask, (500, 247), (72, 70), -8, 0, 360, 255, cv2.FILLED)
    cv2.ellipse(mask, (175, 330), (100, 42), -16, 0, 360, 255, cv2.FILLED)
    cv2.ellipse(mask, (440, 350), (110, 40), 12, 0, 360, 255, cv2.FILLED)
    cv2.circle(mask, (528, 178), 24, 255, cv2.FILLED)

    yy, xx = np.mgrid[0:height, 0:width]
    grain = 13 * np.sin(xx / 18.0 + 0.7 * np.sin(yy / 31.0))
    grain += rng.normal(0, 4.0, size=(height, width))
    base = np.empty_like(canvas, dtype=np.float32)
    base[..., 0] = 48 + 0.50 * grain
    base[..., 1] = 91 + 0.85 * grain
    base[..., 2] = 142 + grain
    wood = np.clip(base, 0, 255).astype(np.uint8)
    canvas[mask > 0] = wood[mask > 0]

    # Carved back ridges and a simple eye make the source easy to explain.
    for x in range(245, 401, 20):
        cv2.line(canvas, (x, 187), (x + 4, 292), (31, 57, 89), 5, cv2.LINE_AA)
        cv2.line(canvas, (x + 6, 190), (x + 10, 289), (76, 128, 182), 2, cv2.LINE_AA)
    canvas[mask == 0] = (226, 229, 232)
    cv2.circle(canvas, (535, 176), 10, (12, 16, 20), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(canvas, (538, 173), 3, (235, 235, 235), cv2.FILLED, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "SYNTHETIC PROXY - real frog photo pending",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (60, 60, 65),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write {path}")
    return path


def labelled_panel(path: Path, label: str, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read panel image: {path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (size[0], 34), (20, 20, 23), cv2.FILLED)
    cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def write_contact_sheet(panel_paths: list[tuple[Path, str]], output_path: Path) -> None:
    panels = [labelled_panel(path, label) for path, label in panel_paths]
    rows = []
    for start in range(0, len(panels), 3):
        row = panels[start : start + 3]
        while len(row) < 3:
            row.append(np.full_like(panels[0], (25, 25, 28)))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"could not write {output_path}")


def write_progress_video(view_paths: list[Path], output_path: Path, fps: float = 3.0) -> None:
    """Write a short forward-and-back MP4 from the rendered novel views."""
    frames = [cv2.imread(str(path)) for path in view_paths]
    if any(frame is None for frame in frames):
        raise ValueError("could not read one or more rendered views for the video")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError("OpenCV could not create the progress MP4")
    try:
        sequence = frames + frames[-2:0:-1]
        for frame in sequence:
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    data_dir = ROOT / "data"
    model_dir = ROOT / "model3d"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = create_demo_photo(data_dir / "demo_proxy.png")
    model_base = model_dir / "demo_proxy"
    reconstruction = reconstruct_photo(
        source_path,
        model_base,
        depth_mode="shape",
        relief=0.42,
        grid=120,
    )

    yaws = [-35.0, -18.0, 0.0, 18.0, 35.0]
    view_paths, triangle_counts, elapsed = render_views(
        reconstruction["paths"]["obj"], output_dir / "views", yaws, pitch=-6.0, size=420
    )

    contact_sheet = output_dir / "progress_contact_sheet.png"
    write_contact_sheet(
        [
            (source_path, "1  Synthetic source (frog pending)"),
            (reconstruction["paths"]["mask"], "2  Otsu + morphology mask"),
            (reconstruction["paths"]["depth"], "3  Temporary depth proxy"),
            (view_paths[0], "4  Novel view: yaw -35 deg"),
            (view_paths[2], "5  Rendered front view"),
            (view_paths[-1], "6  Novel view: yaw +35 deg"),
        ],
        contact_sheet,
    )
    video_path = output_dir / "progress_novel_views.mp4"
    write_progress_video(view_paths, video_path)

    metrics = {
        "status": "progress-check prototype; real frog capture pending",
        "depth_method": reconstruction["depth_method"],
        "image_size": reconstruction["image_size"],
        "foreground_percent": round(float(reconstruction["foreground_percent"]), 2),
        "mesh_vertices": reconstruction["vertices"],
        "mesh_triangles": reconstruction["faces"],
        "view_yaws_degrees": yaws,
        "triangles_drawn_per_view": triangle_counts,
        "render_seconds_total": round(elapsed, 3),
        "render_seconds_per_view": round(elapsed / len(yaws), 3),
        "video": str(video_path),
    }
    metrics_path = output_dir / "progress_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\nPROGRESS DEMO COMPLETE")
    print("----------------------")
    print("The input and depth are synthetic placeholders; no frog result is claimed.")
    print(f"Mesh: {metrics['mesh_vertices']} vertices, {metrics['mesh_triangles']} triangles")
    print(f"Rendered: {len(yaws)} views in {elapsed:.2f} seconds")
    print(f"Show this image: {contact_sheet}")
    print(f"Optional video: {video_path}")
    print(f"Metrics: {metrics_path}")
    print(f"OBJ model: {reconstruction['paths']['obj']}")


if __name__ == "__main__":
    main()
