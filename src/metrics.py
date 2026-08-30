"""PSNR and SSIM, implemented directly on NumPy and OpenCV.

Written out rather than imported from scikit-image so the project keeps the
two-dependency footprint the renderer already lives under, and because both
metrics are course material in their own right.

SSIM follows Wang et al. (2004): an 11x11 Gaussian window, sigma 1.5, and the
standard stabilising constants. The windowed statistics are evaluated only where
the window fits entirely inside the image, which is what "valid" means for a
sliding window and what reference implementations do by default.
"""

from __future__ import annotations

import cv2
import numpy as np

WINDOW_SIZE = 11
WINDOW_SIGMA = 1.5


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB, or infinity for identical images."""
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")
    error = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    if error <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / error))


def _windowed(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Mean structural similarity over the image, in [-1, 1]."""
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    if min(reference.shape) < WINDOW_SIZE:
        raise ValueError(f"images must be at least {WINDOW_SIZE}px on a side")

    a = reference.astype(np.float64)
    b = test.astype(np.float64)
    column = cv2.getGaussianKernel(WINDOW_SIZE, WINDOW_SIGMA)
    kernel = column @ column.T

    mu_a, mu_b = _windowed(a, kernel), _windowed(b, kernel)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = _windowed(a * a, kernel) - mu_aa
    sigma_bb = _windowed(b * b, kernel) - mu_bb
    sigma_ab = _windowed(a * b, kernel) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    similarity = numerator / denominator

    # Keep only positions where the window sat entirely inside the image.
    pad = WINDOW_SIZE // 2
    return float(similarity[pad:-pad, pad:-pad].mean())
