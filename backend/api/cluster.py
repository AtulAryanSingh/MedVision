"""
api/cluster.py

What this module does:
  POST /cluster – run KMeans clustering and PCA dimensionality reduction
  on an uploaded image, caching and returning the results.

Why it exists:
  Clustering and PCA are compute-intensive steps that belong in their own
  endpoint so the frontend ML Lab tab can trigger them independently.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import numpy as np

from api import find_uploaded_file, load_metadata, update_cache
from core.loader import array_to_base64_png, get_slice_2d, load_image
from ml.clustering import run_kmeans
from ml.reduction import run_pca

router = APIRouter()


class ClusterRequest(BaseModel):
    """Request body for the cluster endpoint."""
    image_id: str
    k: int = Field(4, ge=2, le=16, description="Number of KMeans clusters")
    n_samples: int = Field(5000, ge=100, le=20000, description="PCA sample size")


@router.post("/cluster", summary="Run KMeans clustering + PCA on an image")
async def cluster_image(req: ClusterRequest):
    """
    Segment *image_id* with KMeans and project pixels to 2-D via PCA.

    What it does:
      1. Loads the image (middle slice for 3-D).
      2. Runs KMeans with *k* clusters → segmented colour image + stats.
      3. Runs PCA on a pixel sample, using KMeans labels as colour coding.
      4. Caches both results and returns them in one JSON payload.

    Returns
    -------
    JSON: { image_id, k, segmented_image (base64 PNG),
            centers, cluster_counts, pca: { points, explained_variance } }
    """
    load_metadata(req.image_id)  # 404 guard
    path = find_uploaded_file(req.image_id)
    arr, _ = load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    # ── KMeans ────────────────────────────────────────────────────────────────
    km_result = run_kmeans(arr, k=req.k)

    # Flat label array for PCA colour coding
    import cv2
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    pixels = grey.flatten().reshape(-1, 1).astype(np.float32)

    from sklearn.cluster import KMeans
    km_for_labels = KMeans(n_clusters=req.k, random_state=42, n_init="auto")
    all_labels = km_for_labels.fit_predict(pixels)

    # Sort labels by centre intensity (consistent with run_kmeans)
    order = np.argsort(km_for_labels.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    all_labels = np.array([remap[l] for l in all_labels])

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_result = run_pca(arr, n_components=2, n_samples=req.n_samples, k_labels=all_labels)

    # ── Cache & return ────────────────────────────────────────────────────────
    cluster_cache = {
        "k": km_result["k"],
        "centers": km_result["centers"],
        "cluster_counts": km_result["cluster_counts"],
    }
    update_cache(req.image_id, {"cluster": cluster_cache})

    return JSONResponse({
        "image_id": req.image_id,
        "k": km_result["k"],
        "segmented_image": "data:image/png;base64," + array_to_base64_png(km_result["segmented_image"]),
        "centers": km_result["centers"],
        "cluster_counts": km_result["cluster_counts"],
        "pca": {
            "points": pca_result["points"],
            "explained_variance": pca_result["explained_variance"],
            "n_samples_used": pca_result["n_samples_used"],
        },
    })
