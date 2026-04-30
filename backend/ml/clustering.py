"""
ml/clustering.py

What this module does:
  Provides KMeans-based intensity/colour segmentation of medical images.
  Returns the segmented (colour-quantised) image, cluster centres, and
  per-cluster pixel counts.

Why it exists:
  KMeans clustering is a standard exploratory technique for separating
  tissue regions by intensity without labelled training data.

Dependencies: NumPy, OpenCV, scikit-learn
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(
    arr: np.ndarray,
    k: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Cluster pixels of *arr* into *k* groups by intensity and return the
    segmented image and cluster statistics.

    What it does:
      1. Normalises *arr* to uint8 greyscale.
      2. Reshapes to (N_pixels, 1) feature matrix.
      3. Fits sklearn KMeans with *k* clusters.
      4. Reconstructs a colour-coded segmented image where each cluster is
         filled with a distinct HSV colour.
      5. Returns the segmented image as float32, plus centres and counts.

    Why it exists:
      Colour quantisation by KMeans visually separates intensity bands
      (e.g., bone, soft tissue, air in CT) which is useful for exploratory
      analysis and teaching.

    Parameters
    ----------
    arr          : np.ndarray – 2-D or 3-D float32 image
    k            : int        – number of clusters (2–16)
    random_state : int        – for reproducibility

    Returns
    -------
    dict with keys:
        segmented_image : np.ndarray (H × W × 3, float32)
        centers         : list of float – sorted cluster centre intensities
        cluster_counts  : list of int   – pixel count per cluster
        k               : int
    """
    from core.loader import normalise_to_uint8
    k = max(2, min(int(k), 16))

    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8

    # KMeans expects (N, n_features)
    pixels = grey.flatten().reshape(-1, 1).astype(np.float32)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(pixels)
    centres_raw = km.cluster_centers_.flatten()

    # Sort clusters by intensity so label 0 = darkest
    order = np.argsort(centres_raw)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    centres_sorted = centres_raw[order]

    # Build colour-coded segmented image
    h, w = grey.shape
    colour_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for cluster_idx in range(k):
        hue = int(cluster_idx / k * 179)
        hsv = np.array([[[hue, 220, 210]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        mask = (labels.reshape(h, w) == cluster_idx)
        colour_seg[mask] = bgr

    # Per-cluster counts
    counts = [int(np.sum(labels == i)) for i in range(k)]

    return {
        "segmented_image": colour_seg.astype(np.float32),
        "centers": [float(c) for c in centres_sorted],
        "cluster_counts": counts,
        "k": k,
    }
