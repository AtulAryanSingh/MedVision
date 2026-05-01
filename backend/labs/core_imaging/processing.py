"""
labs/core_imaging/processing.py

What this module does:
  Provides three image-processing operations used by the Core Imaging Lab:
    • apply_kmeans   – segment an image into k colour clusters
    • apply_gaussian – smooth an image with a Gaussian blur
    • apply_sobel    – detect edges using Sobel gradient operators

Why it exists as a separate module:
  Keeping processing logic isolated from route handlers makes it easy to unit-
  test each function independently and to swap or extend algorithms without
  touching the HTTP layer.  Future labs can also import these utilities if
  needed.

Dependencies: OpenCV (cv2), NumPy, SciPy
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ── K-Means clustering ────────────────────────────────────────────────────────

def apply_kmeans(image: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Segment the image into *k* colour clusters using OpenCV's K-Means.

    What it does:
      Reshapes the image into a flat list of pixels, runs K-Means to find
      *k* representative colours, then replaces every pixel with its cluster
      centroid colour.  The result is a posterised / colour-quantised image
      that highlights dominant regions.

    Why it exists:
      K-Means colour segmentation is a classic exploratory step in medical
      imaging to separate tissue regions by intensity or colour without any
      prior labelling.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H × W × 3) as returned by cv2.imread / cv2.imdecode.
    k : int
        Number of clusters (default 4).

    Returns
    -------
    np.ndarray
        Colour-quantised BGR image with the same shape as *image*.
    """
    # Reshape to a 2-D array of pixels (N_pixels × 3) and convert to float32
    # because cv2.kmeans requires float32 input.
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    # Termination criteria: stop after 10 iterations OR when centres move < 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Run K-Means; KMEANS_RANDOM_CENTERS gives fast, reproducible-enough results
    _, labels, centres = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Replace each pixel with its cluster centre colour
    centres = np.uint8(centres)
    segmented = centres[labels.flatten()]
    return segmented.reshape(image.shape)


# ── Gaussian blur ─────────────────────────────────────────────────────────────

def apply_gaussian(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply a Gaussian blur to the image using SciPy's gaussian_filter.

    What it does:
      Convolves each colour channel independently with a Gaussian kernel of
      standard deviation *sigma*, reducing high-frequency noise.

    Why it exists:
      Gaussian smoothing is a fundamental pre-processing step that reduces
      sensor noise before downstream analysis (edge detection, segmentation,
      feature extraction).

    Parameters
    ----------
    image : np.ndarray
        BGR image (H × W × 3).
    sigma : float
        Standard deviation of the Gaussian kernel (default 2.0).

    Returns
    -------
    np.ndarray
        Smoothed image with the same dtype and shape as *image*.
    """
    # gaussian_filter operates channel-wise when sigma is a scalar and the
    # image is 3-D; we pass sigma=(sigma, sigma, 0) to avoid blurring across
    # colour channels.
    blurred = gaussian_filter(image.astype(np.float32), sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ── Sobel edge detection ──────────────────────────────────────────────────────

def apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in the image using Sobel gradient operators.

    What it does:
      Converts the image to greyscale, computes the horizontal (Gx) and
      vertical (Gy) Sobel gradients, then returns the combined gradient
      magnitude as a greyscale image.

    Why it exists:
      Sobel edge detection highlights structural boundaries in a scan or
      image—useful for identifying tissue borders, lesions, or other
      anatomical landmarks without any machine-learning model.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H × W × 3).

    Returns
    -------
    np.ndarray
        Single-channel (greyscale) edge magnitude image (uint8, H × W).
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients in x and y directions (64-bit float to avoid overflow)
    grad_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude = sqrt(Gx² + Gy²), normalised to [0, 255]
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val * 255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude
