"""Create a textured height-field mesh from one object photograph.

The real project path uses a pretrained monocular depth model. The ``shape``
mode is an offline progress-check substitute: it constructs a deterministic
relative-depth proxy from the segmented silhouette so every later stage can be
demonstrated before the wooden frog has been purchased.

Examples
--------
Offline checkpoint:
    python reconstruct.py data/demo_proxy.png --depth-mode shape \
        --out model3d/demo_proxy

Real frog photo (later; requires requirements-depth.txt):
    python reconstruct.py data/frog_front.jpg --depth-mode model \
        --out model3d/frog
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _corner_fraction(binary: np.ndarray) -> float:
    """Fraction of the four corner patches that a binary image marks as set.

    The background normally occupies the image corners, so a mask that fills
    them has almost certainly latched onto the background instead of the object.
    """
    corner_size = max(2, min(binary.shape) // 20)
    corners = np.concatenate(
        [
            binary[:corner_size, :corner_size].ravel(),
            binary[:corner_size, -corner_size:].ravel(),
            binary[-corner_size:, :corner_size].ravel(),
            binary[-corner_size:, -corner_size:].ravel(),
        ]
    )
    return float(np.mean(corners > 0))


def _otsu_separability(channel: np.ndarray) -> float:
    """Return Otsu's between-class variance ratio for one channel, in [0, 1].

    This is the quantity Otsu's method maximises. A high value means the
    channel splits into two well-separated populations, which is what makes it
    a good feature to segment on; a low value means the split is arbitrary.
    """
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    probability = histogram / max(histogram.sum(), 1.0)
    levels = np.arange(256)
    mean = float((probability * levels).sum())
    variance = float((probability * (levels - mean) ** 2).sum())
    if variance <= 0:
        return 0.0

    split, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    split = int(split)
    weight_low = float(probability[: split + 1].sum())
    weight_high = float(probability[split + 1 :].sum())
    if weight_low <= 0 or weight_high <= 0:
        return 0.0

    mean_low = float((probability[: split + 1] * levels[: split + 1]).sum()) / weight_low
    mean_high = float((probability[split + 1 :] * levels[split + 1 :]).sum()) / weight_high
    return weight_low * weight_high * (mean_high - mean_low) ** 2 / variance


def _refine(binary: np.ndarray) -> np.ndarray:
    """Morphological cleanup, largest-component selection, and contour fill.

    The kernel is deliberately small. A larger one closes the carved cavity but
    also bridges the object to nearby surfaces of a similar tone, which silently
    fuses the table into the silhouette.
    """
    kernel_size = max(3, int(round(min(binary.shape) / 80)) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("segmentation found no foreground object")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _mask_from_channel(channel: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Threshold one channel and return the cleaned foreground mask."""
    if threshold is None:
        _, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

    # Try both polarities and keep whichever leaves the corners emptier.
    candidates = (binary, cv2.bitwise_not(binary))
    chosen = min(candidates, key=lambda c: (_corner_fraction(c), np.mean(c > 0)))
    return _refine(chosen)


def segment_foreground(
    bgr: np.ndarray,
    threshold: int | None = None,
    channel: str = "auto",
) -> np.ndarray:
    """Return a boolean mask for the largest object on a plain background.

    Brightness alone is fragile: an unevenly lit backdrop spans a wide range of
    grey levels, so a single global threshold cuts through the background rather
    than between background and object. Saturation is far more stable here,
    because a neutral backdrop stays desaturated under any illumination while
    the wood keeps its warm hue. ``auto`` therefore segments on whichever of the
    two channels Otsu separates more cleanly, then falls back to the other if
    the winner's mask covers the image corners.
    """
    if threshold is not None and not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if channel not in ("auto", "saturation", "gray", "colorful"):
        raise ValueError(f"unknown segmentation channel: {channel}")

    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)

    # Saturation alone fails when the backdrop is dark: an underexposed cloth
    # picks up a colour cast and reads as saturated as the wood. Shot from above
    # the frame holds three regions, not two -- dark backdrop, bright table, and
    # the object -- and no single-channel threshold separates three things.
    # Saturation times value does, because only the object is both saturated and
    # bright: the backdrop is saturated but dark, the table bright but grey.
    colorful = cv2.GaussianBlur(
        (hsv[..., 1].astype(np.float32) * hsv[..., 2] / 255.0).astype(np.uint8), (5, 5), 0
    )

    # A manual threshold is a grey level by definition, so it pins the channel.
    if threshold is not None or channel == "gray":
        return _mask_from_channel(gray, threshold) > 0
    if channel == "saturation":
        return _mask_from_channel(saturation) > 0
    if channel == "colorful":
        return _mask_from_channel(colorful) > 0

    ordered = sorted(
        (("saturation", saturation), ("colorful", colorful), ("gray", gray)),
        key=lambda item: _otsu_separability(item[1]),
        reverse=True,
    )
    fallback: np.ndarray | None = None
    for _, candidate in ordered:
        mask = _mask_from_channel(candidate)
        if _corner_fraction(mask) < 0.05:
            return mask > 0
        if fallback is None:
            fallback = mask
    return fallback > 0


def shape_proxy_depth(mask: np.ndarray) -> np.ndarray:
    """Make a deterministic relative-depth proxy for the offline demo.

    This is deliberately not presented as learned depth. It approximates a
    rounded relief using distance from the silhouette boundary and adds shallow
    carved ridges so mesh generation and rendering can be exercised today.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak <= 0:
        raise ValueError("cannot create depth from an empty mask")

    depth = np.sqrt(distance / peak)
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)

    ridges = np.zeros_like(depth, dtype=np.float32)
    for centre in np.linspace(0.36, 0.62, 9):
        ridges += np.exp(-((x_norm - centre) ** 2) / (2 * 0.006**2))
    back_band = np.exp(-((y_norm - 0.48) ** 2) / (2 * 0.16**2))
    depth = depth + 0.10 * ridges * back_band * mask
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.1)
    depth[~mask] = 0.0

    valid = depth[mask]
    depth = (depth - float(valid.min())) / max(float(valid.max() - valid.min()), 1e-6)
    depth[~mask] = 0.0
    return depth.astype(np.float32)


def orient_depth(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Put a depth map into this pipeline's convention: small values are near.

    ``build_mesh`` writes the depth value straight into the vertex z coordinate
    and ``render3d`` projects with the camera on the low-z side, so a *larger*
    value sits *further* from the camera. Monocular networks normally predict
    the opposite — inverse depth, where the nearest surface scores highest — and
    which way round a given checkpoint reports is not something to assume.

    So measure it instead. The photographed object stands in front of its
    backdrop, so whichever side of the mask carries the smaller values is the
    near side. If the object reads as further away than the background the map
    is inverted, and the relief would come out hollow.

    Returns the oriented map and whether it had to be flipped. Falls back to
    returning the input unchanged when the mask leaves nothing to compare.
    """
    if not mask.any() or mask.all():
        return depth, False

    object_depth = float(np.median(depth[mask]))
    background_depth = float(np.median(depth[~mask]))
    if object_depth <= background_depth:
        return depth, False
    return (float(depth.max()) - depth).astype(np.float32), True


def estimate_model_depth(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Estimate relative depth with pretrained Depth Anything V2.

    Depth Anything predicts inverse depth, so the raw map is oriented against
    ``mask`` before being returned; see :func:`orient_depth`.
    """
    try:
        from PIL import Image
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "learned depth dependencies are missing; run "
            "'python -m pip install -r requirements-depth.txt'"
        ) from exc

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print("Loading pretrained Depth Anything V2 Small...")
    estimator = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    result = estimator(Image.fromarray(rgb))
    depth = np.asarray(result["depth"], dtype=np.float32)
    if depth.shape != bgr.shape[:2]:
        depth = cv2.resize(
            depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    low, high = float(depth.min()), float(depth.max())
    if high - low < 1e-6:
        raise RuntimeError("the depth model returned a flat depth map")
    depth = ((depth - low) / (high - low)).astype(np.float32)

    if mask is not None:
        depth, flipped = orient_depth(depth, mask)
        if flipped:
            print("depth map was inverse depth; flipped so near surfaces are near")
    return depth


def load_depth_image(path: Path, size: tuple[int, int], invert: bool = False) -> np.ndarray:
    """Load a grayscale depth image and normalize it to [0, 1]."""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"could not read depth image: {path}")
    if raw.shape[::-1] != size:
        raw = cv2.resize(raw, size, interpolation=cv2.INTER_CUBIC)
    depth = raw.astype(np.float32) / 255.0
    return 1.0 - depth if invert else depth


def build_mesh(
    depth: np.ndarray,
    keep: np.ndarray,
    relief: float = 0.35,
    grid: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Convert a masked depth map into vertices, UVs, and OBJ-style faces."""
    if depth.shape != keep.shape:
        raise ValueError("depth map and mask must have the same shape")
    if grid < 10:
        raise ValueError("grid must be at least 10")

    height, width = depth.shape
    step = max(1, int(round(max(height, width) / grid)))
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    smooth = cv2.GaussianBlur(depth, (0, 0), 1.5)
    aspect = width / height

    index = -np.ones((len(ys), len(xs)), np.int64)
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if not keep[y, x]:
                continue
            index[row, col] = len(vertices)
            vertices.append(
                (
                    ((x / width) - 0.5) * aspect,
                    0.5 - (y / height),
                    float(smooth[y, x]) * relief,
                )
            )
            uvs.append((x / width, 1.0 - y / height))

    faces: list[tuple[int, int, int]] = []
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            top_left = index[row, col]
            top_right = index[row, col + 1]
            bottom_left = index[row + 1, col]
            bottom_right = index[row + 1, col + 1]
            if min(top_left, top_right, bottom_left, bottom_right) < 0:
                continue
            # OBJ indices start at 1. This winding faces the source camera.
            faces.append((top_left + 1, bottom_left + 1, bottom_right + 1))
            faces.append((top_left + 1, bottom_right + 1, top_right + 1))

    if not faces:
        raise ValueError("mesh has no faces; check segmentation or use a finer grid")
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        faces,
    )


def write_obj_bundle(
    output_base: Path,
    source_bgr: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: list[tuple[int, int, int]],
) -> dict[str, Path]:
    """Write OBJ, MTL, texture, mask, and depth-preview files."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    name = output_base.name
    obj_path = output_base.with_suffix(".obj")
    mtl_path = output_base.with_suffix(".mtl")
    texture_path = output_base.with_suffix(".png")
    mask_path = output_base.parent / f"{name}_mask.png"
    depth_path = output_base.parent / f"{name}_depth.png"

    with obj_path.open("w", encoding="utf-8") as obj:
        obj.write("# Textured relief generated by reconstruct.py\n")
        obj.write(f"mtllib {mtl_path.name}\no {name}\n")
        for x, y, z in vertices:
            obj.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            obj.write(f"vt {u:.6f} {v:.6f}\n")
        obj.write(f"usemtl {name}\n")
        for a, b, c in faces:
            obj.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    mtl_path.write_text(
        f"newmtl {name}\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {texture_path.name}\n",
        encoding="utf-8",
    )

    if not cv2.imwrite(str(texture_path), source_bgr):
        raise OSError(f"could not write {texture_path}")
    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    preview = cv2.applyColorMap(np.clip(depth * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    preview[~mask] = 0
    cv2.imwrite(str(depth_path), preview)
    return {
        "obj": obj_path,
        "mtl": mtl_path,
        "texture": texture_path,
        "mask": mask_path,
        "depth": depth_path,
    }


def reconstruct_photo(
    image_path: Path,
    output_base: Path,
    depth_mode: str = "model",
    depth_image: Path | None = None,
    invert_depth: bool = False,
    threshold: int | None = None,
    no_segment: bool = False,
    relief: float = 0.35,
    grid: int = 100,
    segment_channel: str = "auto",
) -> dict[str, object]:
    """Run segmentation, depth preparation, mesh creation, and export."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")

    scale = 900 / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = (
        np.ones(bgr.shape[:2], dtype=bool)
        if no_segment
        else segment_foreground(bgr, threshold, segment_channel)
    )

    if depth_image is not None:
        depth = load_depth_image(depth_image, (bgr.shape[1], bgr.shape[0]), invert_depth)
        method = "provided depth image"
    elif depth_mode == "shape":
        depth = shape_proxy_depth(mask)
        method = "synthetic shape proxy (progress demo only)"
    elif depth_mode == "model":
        depth = estimate_model_depth(bgr, mask)
        method = "Depth Anything V2 Small (pretrained)"
    else:
        raise ValueError(f"unknown depth mode: {depth_mode}")

    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.0)
    depth[~mask] = 0.0
    vertices, uvs, faces = build_mesh(depth, mask, relief=relief, grid=grid)
    paths = write_obj_bundle(output_base, bgr, depth, mask, vertices, uvs, faces)
    return {
        "image_size": (bgr.shape[1], bgr.shape[0]),
        "foreground_percent": 100.0 * float(mask.mean()),
        "depth_method": method,
        "vertices": len(vertices),
        "faces": len(faces),
        "paths": paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="object photograph")
    parser.add_argument("--out", type=Path, default=Path("model3d/object"), help="output path without extension")
    parser.add_argument("--depth-mode", choices=("model", "shape"), default="model")
    parser.add_argument("--depth-image", type=Path, help="use an existing grayscale depth map")
    parser.add_argument("--invert-depth", action="store_true", help="invert a provided depth image")
    parser.add_argument("--threshold", type=int, help="manual grayscale threshold, 0-255")
    parser.add_argument(
        "--segment-channel",
        choices=("auto", "saturation", "colorful", "gray"),
        default="auto",
        help="channel to segment on; auto picks whichever Otsu separates best",
    )
    parser.add_argument("--no-segment", action="store_true", help="mesh the complete frame")
    parser.add_argument("--relief", type=float, default=0.35, help="depth exaggeration")
    parser.add_argument("--grid", type=int, default=100, help="approximate samples across the longest image side")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = reconstruct_photo(
            args.image,
            args.out,
            depth_mode=args.depth_mode,
            depth_image=args.depth_image,
            invert_depth=args.invert_depth,
            threshold=args.threshold,
            no_segment=args.no_segment,
            relief=args.relief,
            grid=args.grid,
            segment_channel=args.segment_channel,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    width, height = result["image_size"]
    print(f"image: {width}x{height}")
    print(f"foreground: {result['foreground_percent']:.1f}%")
    print(f"depth: {result['depth_method']}")
    print(f"mesh: {result['vertices']} vertices, {result['faces']} triangles")
    print(f"wrote: {result['paths']['obj']}")


if __name__ == "__main__":
    main()
