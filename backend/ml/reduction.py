"""
ml/reduction.py

What this module does:
  PCA-based dimensionality reduction for visualising the pixel-intensity
  feature space of a medical image in 2-D (or 3-D).

Why it exists:
  PCA projection lets users inspect whether image regions form separable
  clusters in feature space and confirms that KMeans boundaries are
  meaningful, all without requiring a labelled dataset.

Dependencies: NumPy, scikit-learn
"""
from __future__ import annotations
from typing import Any, Dict, List

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    arr: np.ndarray,
    n_components: int = 2,
    n_samples: int = 5000,
    k_labels: np.ndarray | None = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Apply PCA to a random sample of pixels and return a 2-D scatter dataset.

    What it does:
      1. Normalises to uint8 greyscale.
      2. Builds a feature matrix: for each sampled pixel, the feature vector
         is [intensity, row_norm, col_norm] (3 features) – simple but enough
         to show spatial-intensity structure.
      3. Standardises features to zero mean / unit variance.
      4. Fits PCA with *n_components* principal components.
      5. Returns the projected points in a format ready for a scatter chart.

    Why it exists:
      A pure-intensity PCA (1-D feature) would collapse to a histogram.
      Including normalised spatial coordinates gives 3-D input so the 2-D
      projection is visually informative.

    Parameters
    ----------
    arr          : np.ndarray  – 2-D or 3-D float32 image
    n_components : int         – number of output PCA axes (2 or 3)
    n_samples    : int         – maximum pixels to sample (for performance)
    k_labels     : np.ndarray  – optional flat cluster label array (H*W)
    random_state : int

    Returns
    -------
    dict with keys:
        points              : list of {x, y, cluster} dicts
        explained_variance  : list of float (fraction per component)
        n_components        : int
        n_samples_used      : int
    """
    from core.loader import normalise_to_uint8
    n_components = max(2, min(int(n_components), 3))

    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    h, w = grey.shape
    flat = grey.flatten().astype(np.float32)

    # Build row/col normalised coordinates for every pixel
    rows = np.repeat(np.arange(h), w).astype(np.float32) / h
    cols = np.tile(np.arange(w), h).astype(np.float32) / w
    X = np.column_stack([flat, rows, cols])  # (N, 3)

    # Random sample
    rng = np.random.default_rng(random_state)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X[idx]
    labels_sample = k_labels[idx] if k_labels is not None else np.zeros(n, dtype=int)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    projected = pca.fit_transform(X_scaled)

    points: List[Dict[str, Any]] = [
        {"x": float(projected[i, 0]), "y": float(projected[i, 1]), "cluster": int(labels_sample[i])}
        for i in range(n)
    ]

    return {
        "points": points,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "n_components": n_components,
        "n_samples_used": n,
    }
