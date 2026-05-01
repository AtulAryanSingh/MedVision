"""
processing/filters.py

What this module does:
  Provides two classical spatial filters widely used in medical image
  pre-processing:
    apply_gaussian – noise-reduction via Gaussian convolution (SciPy)
    apply_sobel    – edge detection via Sobel gradient magnitude (OpenCV)

Why it exists:
  Separating filters from other processing operations keeps this file
  focused and makes each function independently testable and reusable.

Dependencies: OpenCV, NumPy, SciPy
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply a Gaussian blur to reduce noise.

    What it does:
      Convolves each 2-D plane of *arr* with a Gaussian kernel of standard
      deviation *sigma*.  For colour images the kernel is applied to each
      channel independently; the depth/colour axis is not blurred.

    Why it exists:
      Gaussian smoothing is the standard pre-processing step before edge
      detection or segmentation to suppress high-frequency noise while
      preserving large-scale structures.

    Parameters
    ----------
    arr   : np.ndarray – 2-D (H, W) or 3-D (H, W, C) float32 array
    sigma : float      – Gaussian kernel standard deviation (default 2.0)

    Returns
    -------
    np.ndarray – smoothed float32 array, same shape as *arr*.
    """
    if arr.ndim == 2:
        sigma_arg: float | tuple = sigma
    else:
        # Only blur spatial axes; avoid cross-channel blurring
        sigma_arg = (sigma, sigma, 0)
    blurred = gaussian_filter(arr.astype(np.float32), sigma=sigma_arg)
    return np.clip(blurred, arr.min(), arr.max())


def apply_sobel(arr: np.ndarray) -> np.ndarray:
    """
    Detect edges using Sobel gradient operators.

    What it does:
      Converts *arr* to greyscale if necessary, computes horizontal (Gx) and
      vertical (Gy) Sobel gradients, and returns the normalised gradient
      magnitude as a float32 array in [0, 255].

    Why it exists:
      Sobel edge detection highlights structural boundaries (tissue borders,
      lesions) without requiring a trained model.

    Parameters
    ----------
    arr : np.ndarray – 2-D or 3-D float32 array

    Returns
    -------
    np.ndarray – float32 edge-magnitude image (H × W).
    """
    # Work on a normalised uint8 greyscale copy
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    if u8.ndim == 3:
        grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    else:
        grey = u8

    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val * 255.0
    return mag.astype(np.float32)
