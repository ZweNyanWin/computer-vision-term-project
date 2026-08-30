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

from reconstruct import reconstruct_photo  # noqa: E402
from render3d import load_obj, normalize_mesh, render_frame  # noqa: E402
from run_progress_demo import create_demo_photo  # noqa: E402


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
