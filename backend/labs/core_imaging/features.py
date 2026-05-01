"""
labs/core_imaging/features.py

What this module does:
  Computes basic statistical and distributional features from a greyscale
  representation of an image:
    • compute_mean      – pixel intensity mean
    • compute_std       – pixel intensity standard deviation
    • compute_histogram – intensity histogram (bins + counts)
    • extract_all       – convenience wrapper that returns all three

Why it exists as a separate module:
  Isolating feature extraction from route handlers makes the functions
  independently testable and reusable by future labs or analysis pipelines.
  It also keeps the route file focused on HTTP concerns only.

Dependencies: NumPy, OpenCV
"""

import cv2
import numpy as np
from typing import Any


# ── Per-feature helpers ───────────────────────────────────────────────────────

def compute_mean(grey: np.ndarray) -> float:
    """
    Return the mean pixel intensity of a greyscale image.

    What it does:
      Computes the arithmetic mean of all pixel values in the (H × W) array.

    Why it exists:
      The mean intensity gives a quick overall brightness reading of the
      region of interest and can flag exposure or contrast problems.

    Parameters
    ----------
    grey : np.ndarray
        Greyscale image (H × W), dtype uint8.

    Returns
    -------
    float
        Mean intensity in [0, 255].
    """
    return float(np.mean(grey))


def compute_std(grey: np.ndarray) -> float:
    """
    Return the standard deviation of pixel intensities in a greyscale image.

    What it does:
      Computes the population standard deviation across all pixels.

    Why it exists:
      High std dev indicates high contrast or textured regions; low std dev
      suggests uniform areas.  It is a cheap proxy for image quality or
      tissue heterogeneity.

    Parameters
    ----------
    grey : np.ndarray
        Greyscale image (H × W), dtype uint8.

    Returns
    -------
    float
        Standard deviation in [0, 127.5].
    """
    return float(np.std(grey))


def compute_histogram(grey: np.ndarray, bins: int = 256) -> dict[str, list[Any]]:
    """
    Compute the intensity histogram of a greyscale image.

    What it does:
      Divides the [0, 255] intensity range into *bins* equal-width buckets and
      counts the pixels that fall in each bucket.  Returns both the bin-edge
      midpoints and the counts so the caller can plot or serialise the result
      directly.

    Why it exists:
      A histogram is the most informative single summary of the intensity
      distribution.  It reveals bimodal distributions (e.g., background vs.
      tissue), clipping, and overall dynamic range – all without looking at
      spatial information.

    Parameters
    ----------
    grey : np.ndarray
        Greyscale image (H × W), dtype uint8.
    bins : int
        Number of histogram bins (default 256 = one bin per intensity level).

    Returns
    -------
    dict with keys:
        "bins"   – list of *bins* float values (bin-centre x-coordinates)
        "counts" – list of *bins* int values (pixel counts per bin)
    """
    counts, bin_edges = np.histogram(grey.flatten(), bins=bins, range=(0, 256))
    # Use bin centres (midpoints between consecutive edges) as x-axis labels
    bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    return {
        "bins": bin_centres,
        "counts": counts.tolist(),
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def extract_all(image: np.ndarray) -> dict[str, Any]:
    """
    Extract mean, std dev, and histogram from a BGR image in one call.

    What it does:
      Converts *image* to greyscale, then calls compute_mean, compute_std,
      and compute_histogram, collecting all results into a single dict.

    Why it exists:
      The route handler only needs one call; this keeps route code concise.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H × W × 3).

    Returns
    -------
    dict with keys: "mean", "std_dev", "histogram"
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean": compute_mean(grey),
        "std_dev": compute_std(grey),
        "histogram": compute_histogram(grey),
    }
