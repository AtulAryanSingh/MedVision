"""
features/extractor.py

What this module does:
  Computes a comprehensive statistical feature vector from a 2-D intensity
  array:
    • mean, std, min, max intensity
    • skewness and kurtosis
    • percentiles (10th, 25th, 50th, 75th, 90th)
    • Shannon entropy (from normalised histogram)
    • pixel distribution statistics
    • basic shape descriptors (bounding-box coverage, effective radius)

Why it exists:
  Extracting a named feature vector from raw pixel data provides the
  numerical inputs needed for comparison, normalisation, and reporting,
  without requiring the ML or analysis layers to perform their own ad-hoc
  statistics.

Dependencies: NumPy, SciPy
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(arr: np.ndarray) -> Dict[str, Any]:
    """
    Extract a comprehensive feature vector from *arr*.

    What it does:
      1. Reduces the array to a 1-D float64 intensity vector.
      2. Computes first-order statistics (mean, std, min, max).
      3. Computes higher-order moments (skewness, kurtosis).
      4. Computes selected percentiles.
      5. Estimates Shannon entropy from the normalised histogram.
      6. Computes a downsampled (64-bin) histogram for downstream charting.
      7. Adds simple shape descriptors derived from the binary foreground.

    Why it exists:
      A single call that returns all features avoids scattering statistics
      computations across the API layer and ensures consistent results
      whether the caller is the feature endpoint, the analysis endpoint,
      or a comparison function.

    Parameters
    ----------
    arr : np.ndarray
        2-D or 3-D float32 array (colour channels collapsed to greyscale).

    Returns
    -------
    dict with keys:
        mean, std_dev, intensity_min, intensity_max,
        skewness, kurtosis, percentile_10/25/50/75/90,
        entropy, nonzero_fraction,
        histogram (64-bin chart data: {bins, counts}),
        shape_descriptors
    """
    # ── Reduce to 1-D ────────────────────────────────────────────────────────
    from core.loader import normalise_to_uint8
    import cv2

    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    flat = grey.flatten().astype(np.float64)

    # ── First-order stats ─────────────────────────────────────────────────────
    mean_val   = float(np.mean(flat))
    std_val    = float(np.std(flat))
    min_val    = float(np.min(flat))
    max_val    = float(np.max(flat))

    # ── Higher-order moments ──────────────────────────────────────────────────
    skew_val = float(skew(flat))
    kurt_val = float(kurtosis(flat))          # Fisher definition (excess kurtosis)

    # ── Percentiles ───────────────────────────────────────────────────────────
    p10, p25, p50, p75, p90 = [
        float(np.percentile(flat, p)) for p in (10, 25, 50, 75, 90)
    ]

    # ── Shannon entropy ───────────────────────────────────────────────────────
    counts, _ = np.histogram(flat, bins=256, range=(0, 256))
    probs = counts / (counts.sum() + 1e-12)
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # ── Non-zero fraction ─────────────────────────────────────────────────────
    nonzero_frac = float(np.count_nonzero(flat) / len(flat))

    # ── 64-bin histogram for charting ─────────────────────────────────────────
    chart_counts, chart_edges = np.histogram(flat, bins=64, range=(0, 256))
    chart_centres = ((chart_edges[:-1] + chart_edges[1:]) / 2).tolist()

    # ── Shape descriptors ─────────────────────────────────────────────────────
    # Binarise at Otsu threshold
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_pixels = int(np.count_nonzero(binary))
    total_pixels = int(binary.size)
    coverage = float(foreground_pixels / total_pixels) if total_pixels else 0.0

    # Effective radius: radius of a circle with the same area as foreground
    effective_radius = float(np.sqrt(foreground_pixels / np.pi)) if foreground_pixels > 0 else 0.0

    return {
        "mean": mean_val,
        "std_dev": std_val,
        "intensity_min": min_val,
        "intensity_max": max_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "percentile_10": p10,
        "percentile_25": p25,
        "percentile_50": p50,
        "percentile_75": p75,
        "percentile_90": p90,
        "entropy": entropy,
        "nonzero_fraction": nonzero_frac,
        "histogram": {
            "bins": chart_centres,
            "counts": chart_counts.tolist(),
        },
        "shape_descriptors": {
            "foreground_pixels": foreground_pixels,
            "total_pixels": total_pixels,
            "foreground_coverage": coverage,
            "effective_radius_px": effective_radius,
        },
    }
