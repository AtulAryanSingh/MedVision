"""
processing/histogram.py

What this module does:
  Provides histogram computation, cumulative distribution function (CDF)
  calculation, and CDF-based automatic thresholding.

Why it exists:
  Histogram analysis is the most fundamental way to understand the intensity
  distribution of a medical image.  CDF-based thresholding is a principled,
  data-driven alternative to manually chosen thresholds.

Dependencies: NumPy
"""

from typing import Any, Dict, Tuple

import numpy as np


# ── Histogram ─────────────────────────────────────────────────────────────────

def compute_histogram(
    arr: np.ndarray, bins: int = 256
) -> Dict[str, Any]:
    """
    Compute the intensity histogram of *arr*.

    What it does:
      Flattens the array (treating all channels / slices the same) and
      counts pixels into *bins* equal-width buckets spanning the actual
      intensity range.  Returns both bin-centre x-coordinates and counts
      so the caller can plot or serialise directly.

    Why it exists:
      A histogram is the primary summary of the intensity distribution and
      is used by both the Processing Lab (visualisation) and CDF thresholding.

    Parameters
    ----------
    arr  : np.ndarray – any shape, float32
    bins : int        – number of histogram bins (default 256)

    Returns
    -------
    dict with keys: "bins" (list of float), "counts" (list of int),
                    "intensity_min", "intensity_max"
    """
    flat = arr.flatten()
    lo, hi = float(flat.min()), float(flat.max())
    counts, edges = np.histogram(flat, bins=bins, range=(lo, hi))
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {
        "bins": centres,
        "counts": counts.tolist(),
        "intensity_min": lo,
        "intensity_max": hi,
    }


def compute_cdf(histogram: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the cumulative distribution function from a pre-computed histogram.

    What it does:
      Calculates the normalised CDF (values in [0, 1]) from the histogram
      counts and attaches it to a copy of the histogram dict.

    Why it exists:
      The CDF expresses what fraction of pixels have intensity ≤ each bin
      value, which is used for threshold selection and histogram equalisation.

    Parameters
    ----------
    histogram : dict – as returned by compute_histogram()

    Returns
    -------
    dict with original histogram keys plus "cdf" (list of float in [0, 1]).
    """
    counts = np.array(histogram["counts"], dtype=np.float64)
    total = counts.sum()
    cdf = (counts.cumsum() / total).tolist() if total > 0 else [0.0] * len(counts)
    return {**histogram, "cdf": cdf}


# ── CDF-based threshold ───────────────────────────────────────────────────────

def apply_cdf_threshold(arr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Binarise *arr* at the intensity value corresponding to *percentile* of
    the CDF.

    What it does:
      Computes the histogram, finds the intensity bin where the CDF first
      reaches (percentile / 100), then returns a uint8 binary image where
      pixels above that threshold are 255 and the rest are 0.

    Why it exists:
      CDF-based thresholding automatically adapts to the intensity range of
      the image, producing consistent results across modalities without
      manual parameter tuning.

    Parameters
    ----------
    arr        : np.ndarray – 2-D float32 array
    percentile : float      – CDF percentile to use as threshold (0–100)

    Returns
    -------
    np.ndarray – binary uint8 image (0 or 255), same H×W as *arr*.
    """
    flat = arr.flatten()
    threshold = float(np.percentile(flat, percentile))
    binary = (arr >= threshold).astype(np.uint8) * 255
    return binary
