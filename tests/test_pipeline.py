"""Small offline checks for the reconstruction and rendering code."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reconstruct import (  # noqa: E402
    orient_depth,
    reconstruct_photo,
    segment_foreground,
)
from render3d import load_obj, normalize_mesh, render_frame  # noqa: E402
from run_progress_demo import create_demo_photo  # noqa: E402


def unevenly_lit_object(width: int = 300, height: int = 400) -> np.ndarray:
    """A warm object on a neutral backdrop lit unevenly from one side.

    This reproduces the real capture condition that broke brightness-only
    segmentation: the backdrop sweeps across a wide range of grey levels, so
    part of it is brighter than the object and a global grey threshold cuts
    through the background instead of around the object.
    """
    gradient = np.linspace(40, 210, width, dtype=np.float32)
    backdrop = np.repeat(gradient[None, :], height, axis=0)
    image = np.dstack([backdrop, backdrop, backdrop]).astype(np.uint8)

    # Mid-toned warm wood: not the brightest thing in the frame, but by far
    # the most saturated.
    cv2.ellipse(
        image, (width // 2, height // 2), (width // 4, height // 3),
        0, 0, 360, (46, 96, 165), cv2.FILLED,
    )
    return image


class SegmentationTest(unittest.TestCase):
    def test_uneven_backdrop_does_not_invert_the_mask(self) -> None:
        image = unevenly_lit_object()
        mask = segment_foreground(image)
        height, width = mask.shape

        # The object sits in the middle and the backdrop owns the corners.
        self.assertTrue(mask[height // 2, width // 2], "object centre was excluded")
        for y, x in ((5, 5), (5, width - 5), (height - 5, 5), (height - 5, width - 5)):
            self.assertFalse(mask[y, x], f"backdrop corner ({y}, {x}) was included")
        self.assertLess(mask.mean(), 0.5, "mask covers more than half the frame")

    def test_explicit_channels_agree_on_a_clean_image(self) -> None:
        image = unevenly_lit_object()
        auto = segment_foreground(image, channel="auto")
        saturation = segment_foreground(image, channel="saturation")
        self.assertTrue(np.array_equal(auto, saturation))


class DepthOrientationTest(unittest.TestCase):
    """The renderer treats a larger depth value as further from the camera."""

    def setUp(self) -> None:
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[10:30, 10:30] = True

    def test_inverse_depth_is_flipped(self) -> None:
        # What a monocular network predicts: the near object scores highest.
        depth = np.where(self.mask, 0.9, 0.1).astype(np.float32)
        oriented, flipped = orient_depth(depth, self.mask)
        self.assertTrue(flipped)
        self.assertLess(oriented[self.mask].mean(), oriented[~self.mask].mean())

    def test_true_depth_is_left_alone(self) -> None:
        # Already in the pipeline's convention: the near object scores lowest.
        depth = np.where(self.mask, 0.1, 0.9).astype(np.float32)
        oriented, flipped = orient_depth(depth, self.mask)
        self.assertFalse(flipped)
        self.assertTrue(np.array_equal(oriented, depth))

    def test_degenerate_mask_is_passed_through(self) -> None:
        depth = np.linspace(0, 1, 1600, dtype=np.float32).reshape(40, 40)
        for mask in (np.zeros((40, 40), bool), np.ones((40, 40), bool)):
            oriented, flipped = orient_depth(depth, mask)
            self.assertFalse(flipped)
            self.assertTrue(np.array_equal(oriented, depth))


class PipelineTest(unittest.TestCase):
    def test_demo_reconstructs_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = create_demo_photo(root / "source.png", width=320, height=240)
            result = reconstruct_photo(
                image_path,
                root / "mesh" / "proxy",
                depth_mode="shape",
                relief=0.4,
                grid=36,
            )

            self.assertGreater(result["vertices"], 100)
            self.assertGreater(result["faces"], 100)
            self.assertTrue(result["paths"]["obj"].exists())
            self.assertTrue(result["paths"]["depth"].exists())

            vertices, uvs, faces, texture_path = load_obj(result["paths"]["obj"])
            texture = cv2.imread(str(texture_path))
            frame, drawn = render_frame(
                normalize_mesh(vertices),
                uvs,
                faces,
                texture,
                width=160,
                height=160,
                yaw=20,
            )
            background = np.array([25, 25, 28], dtype=np.uint8)
            changed_pixels = np.any(frame != background, axis=2).sum()
            self.assertGreater(drawn, 50)
            self.assertGreater(changed_pixels, 500)


if __name__ == "__main__":
    unittest.main()
