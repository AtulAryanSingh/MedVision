"""
api/features.py

What this module does:
  POST /features – extract and return the full statistical feature vector
  for an uploaded image, caching the result for later use by the report.

Why it exists:
  Separating feature extraction into its own endpoint allows the frontend
  Feature Explorer tab to request only features without also triggering
  image processing or clustering.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api import find_uploaded_file, load_metadata, update_cache
from core.loader import get_slice_2d, load_image
from features.extractor import extract_features

router = APIRouter()


class FeaturesRequest(BaseModel):
    """Request body for the features endpoint."""
    image_id: str


@router.post("/features", summary="Extract statistical feature vector from an image")
async def get_features(req: FeaturesRequest):
    """
    Compute and return the feature vector for *image_id*.

    What it does:
      1. Loads the image.
      2. Extracts the middle axial slice for 3-D volumes.
      3. Calls features.extractor.extract_features().
      4. Caches the result and returns it as JSON.

    Returns
    -------
    JSON: { image_id, mean, std_dev, intensity_min/max, skewness, kurtosis,
            percentiles, entropy, nonzero_fraction, histogram, shape_descriptors }
    """
    load_metadata(req.image_id)  # 404 guard
    path = find_uploaded_file(req.image_id)
    arr, _ = load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    feat = extract_features(arr)
    update_cache(req.image_id, {"features": feat})

    return JSONResponse({"image_id": req.image_id, **feat})
