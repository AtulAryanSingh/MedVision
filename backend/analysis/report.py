"""
analysis/report.py

What this module does:
  Assembles a structured analysis report for a single uploaded image by
  aggregating metadata, feature statistics, and (optionally) clustering
  results into a single serialisable dict.

Why it exists:
  The report endpoint needs a single function that assembles all available
  information about an image without duplicating logic from other modules.

Dependencies: NumPy (via imported modules), datetime
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def generate_report(
    image_id: str,
    metadata: Dict[str, Any],
    features: Optional[Dict[str, Any]] = None,
    cluster_info: Optional[Dict[str, Any]] = None,
    processing_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build and return a structured analysis report for *image_id*.

    What it does:
      1. Records the generation timestamp.
      2. Includes the full image metadata (shape, modality, spacing, etc.).
      3. Summarises features if provided (mean, std, entropy, etc.).
      4. Summarises clustering results if provided (k, centres, counts).
      5. Adds a human-readable interpretation of key metrics.
      6. Lists processing operations applied.

    Why it exists:
      A report function that accepts optional sections allows the API to
      return useful output even when the user hasn't run all analysis steps.

    Parameters
    ----------
    image_id           : str   – UUID of the uploaded image
    metadata           : dict  – as stored in the image cache
    features           : dict  – as returned by features/extractor.py
    cluster_info       : dict  – as stored in the image cache under "cluster"
    processing_history : list  – list of processing steps applied

    Returns
    -------
    dict – fully structured, JSON-serialisable report.
    """
    report: Dict[str, Any] = {
        "report_id": f"rpt-{image_id[:8]}",
        "image_id": image_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "image_info": {
            "filename": metadata.get("filename", "unknown"),
            "file_type": metadata.get("file_type", "unknown"),
            "modality": metadata.get("modality", "unknown"),
            "shape": metadata.get("shape", []),
            "is_3d": metadata.get("is_3d", False),
            "spacing_mm": metadata.get("spacing", [1.0, 1.0]),
            "intensity_range": {
                "min": metadata.get("intensity_min", 0),
                "max": metadata.get("intensity_max", 255),
            },
        },
        "processing_pipeline": processing_history or [],
        "feature_summary": None,
        "cluster_summary": None,
        "interpretation": [],
    }

    # ── Feature summary ───────────────────────────────────────────────────────
    if features:
        report["feature_summary"] = {
            "mean_intensity": features.get("mean"),
            "std_deviation": features.get("std_dev"),
            "min_intensity": features.get("intensity_min"),
            "max_intensity": features.get("intensity_max"),
            "skewness": features.get("skewness"),
            "kurtosis": features.get("kurtosis"),
            "entropy_bits": features.get("entropy"),
            "nonzero_fraction": features.get("nonzero_fraction"),
            "foreground_coverage": (
                features.get("shape_descriptors", {}).get("foreground_coverage")
            ),
            "effective_radius_px": (
                features.get("shape_descriptors", {}).get("effective_radius_px")
            ),
        }

    # ── Cluster summary ───────────────────────────────────────────────────────
    if cluster_info:
        report["cluster_summary"] = {
            "k": cluster_info.get("k"),
            "cluster_centers": cluster_info.get("centers"),
            "cluster_counts": cluster_info.get("cluster_counts"),
            "dominant_cluster": (
                int(np.argmax(cluster_info["cluster_counts"]))
                if cluster_info.get("cluster_counts")
                else None
            ),
        }

    # ── Auto-interpretation hints ─────────────────────────────────────────────
    hints = []
    if features:
        entropy = features.get("entropy", 0.0)
        if entropy < 3.0:
            hints.append("Low entropy — image appears nearly uniform; check exposure or windowing.")
        elif entropy > 6.5:
            hints.append("High entropy — image is rich in detail or contains significant noise.")

        skew = features.get("skewness", 0.0)
        if abs(skew) > 1.5:
            direction = "bright" if skew > 0 else "dark"
            hints.append(
                f"Distribution is skewed towards {direction} pixels (skewness={skew:.2f}); "
                "consider normalisation before analysis."
            )

        cov = features.get("shape_descriptors", {}).get("foreground_coverage", 0.5)
        if cov < 0.1:
            hints.append("Very little foreground detected; the image may be mostly background.")
        elif cov > 0.9:
            hints.append("Nearly all pixels classified as foreground; threshold may need adjustment.")

    if cluster_info and cluster_info.get("k"):
        hints.append(
            f"KMeans segmented the image into {cluster_info['k']} clusters. "
            "Review the ML Lab tab for the visual segmentation result."
        )

    if not hints:
        hints.append("No notable issues detected. Review the Feature Explorer for detailed statistics.")

    report["interpretation"] = hints

    return report
