"""Offline checks for the held-out evaluation's reporting helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Same arrangement as tests/test_pipeline.py: the suite is run as
# `python tests/test_eval_holdout.py` (unittest, not pytest), so the project root
# is not on sys.path unless we put it there. Without this the import fails with
# ModuleNotFoundError and the test never runs at all.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval_holdout import format_p  # noqa: E402


class PValueFormattingTest(unittest.TestCase):
    def test_tiny_nonzero_probability_is_not_rounded_to_zero(self):
        self.assertEqual(format_p(0.00009), "p<0.0001")

    def test_regular_probability_keeps_four_decimals(self):
        self.assertEqual(format_p(0.12345), "p=0.1235")

    def test_undefined_probability_is_explicit(self):
        self.assertEqual(format_p(math.nan), "p=nan")


if __name__ == "__main__":
    unittest.main()
