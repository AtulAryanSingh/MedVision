"""
analysis/statistics.py

What this module does:
  Statistical utilities for normalising, summarising, and comparing medical
  image arrays:
    normalize        – min-max, z-score, or histogram-equalisation
    summarise        – compact statistics dict (mean, std, range, quartiles)
    compare_arrays   – side-by-side stats + correlation for two arrays

Why it exists:
  Centralising normalisation and comparison here ensures consistent
  definitions across the feature extraction, analysis, and report modules.

Dependencies: NumPy, SciPy
"""

from typing import Any, Dict, Literal

import cv2
import numpy as np
from scipy.stats import pearsonr


NormMethod = Literal["minmax", "zscore", "equalize"]


def normalize(arr: np.ndarray, method: NormMethod = "minmax") -> np.ndarray:
    """
    Normalise *arr* using the specified method.

    What it does:
      • minmax   – scale linearly to [0, 1]
      • zscore   – subtract mean and divide by std (result in ≈[-3, 3])
      • equalize – histogram equalisation (uint8 output, greyscale only)

    Why it exists:
      Different downstream tasks (display, ML, comparison) require different
      normalisation strategies; a single function with a *method* parameter
      avoids duplicating the logic.

    Parameters
    ----------
    arr    : np.ndarray  – float32 input array
    method : str         – one of "minmax", "zscore", "equalize"

    Returns
    -------
    np.ndarray – normalised float32 array (same shape as *arr*).
    """
    arr = arr.astype(np.float32)
    if method == "minmax":
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    if method == "zscore":
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-8:
            return np.zeros_like(arr)
        return (arr - mu) / sigma

    if method == "equalize":
        from core.loader import normalise_to_uint8
        u8 = normalise_to_uint8(arr)
        grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
        eq = cv2.equalizeHist(grey)
        return eq.astype(np.float32)

    raise ValueError(f"Unknown normalisation method: {method!r}")


def summarise(arr: np.ndarray) -> Dict[str, Any]:
    """
    Return a compact statistics summary dict for *arr*.

    What it does:
      Computes mean, std, min, max, median, and interquartile range over
      the flattened array.

    Why it exists:
      Both the comparison and report functions need a canonical statistics
      snapshot; extracting it here avoids repeated computation.
    """
    flat = arr.flatten().astype(np.float64)
    q1, med, q3 = np.percentile(flat, [25, 50, 75])
    return {
        "mean": float(np.mean(flat)),
        "std_dev": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(med),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
    }


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    """
    Compare two arrays: per-array statistics + Pearson correlation.

    What it does:
      1. Generates a summarise() snapshot for each array.
      2. Computes the Pearson correlation on the common number of samples
         (arrays may have different sizes; they are sampled to the shorter
         one for correlation).
      3. Computes the mean absolute difference (MAD) on matched samples.

    Why it exists:
      Side-by-side quantitative comparison is the first step in evaluating
      whether two scans or two processing results are similar or different.

    Returns
    -------
    dict with keys: scan_a, scan_b (each a summarise() dict),
                    pearson_r, pearson_p, mean_abs_diff.
    """
    flat1 = arr1.flatten().astype(np.float64)
    flat2 = arr2.flatten().astype(np.float64)

    # Align lengths for correlation
    n = min(len(flat1), len(flat2))
    rng = np.random.default_rng(0)
    idx1 = rng.choice(len(flat1), n, replace=False)
    idx2 = rng.choice(len(flat2), n, replace=False)
    s1, s2 = flat1[idx1], flat2[idx2]

    if n > 1:
        r, p = pearsonr(s1, s2)
    else:
        r, p = 0.0, 1.0

    return {
        "scan_a": summarise(arr1),
        "scan_b": summarise(arr2),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_abs_diff": float(np.mean(np.abs(s1 - s2))),
    }
