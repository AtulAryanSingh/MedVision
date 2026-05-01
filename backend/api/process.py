"""
api/process.py

What this module does:
  POST /process – run a selected processing operation on an uploaded image
  and return the result as a base64 PNG string along with any computed
  metadata (e.g., connected-component stats).

  Supported processing_type values:
    gaussian           – Gaussian blur (param: sigma, default 2.0)
    sobel              – Sobel edge detection
    cdf_threshold      – CDF-based binarisation (param: percentile, default 95)
    erosion            – morphological erosion (param: kernel_size, default 5)
    dilation           – morphological dilation (param: kernel_size, default 5)
    opening            – morphological opening  (param: kernel_size, default 5)
    closing            – morphological closing  (param: kernel_size, default 5)
    connected_components – label regions + draw bounding boxes
    bounding_boxes     – draw bounding boxes only

Why it exists:
  A single "process" endpoint with a type selector keeps the API surface
  minimal; adding a new operation later means adding a function to the
  processing package and one entry in the dispatch table below.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import find_uploaded_file, load_metadata, update_cache
from core.loader import array_to_base64_png, get_slice_2d, load_image
import processing.filters as flt
import processing.morphology as morph
from processing.histogram import apply_cdf_threshold, compute_histogram

router = APIRouter()


class ProcessRequest(BaseModel):
    """Request body for the process endpoint."""
    image_id: str
    processing_type: str
    sigma: float = Field(2.0, ge=0.1, le=20.0, description="Gaussian sigma")
    kernel_size: int = Field(5, ge=1, le=31, description="Morphological kernel size")
    percentile: float = Field(95.0, ge=1.0, le=99.9, description="CDF threshold percentile")
    threshold: float = Field(128.0, ge=0.0, le=255.0, description="Binary threshold (0-255)")


_ALLOWED_TYPES = {
    "gaussian", "sobel", "cdf_threshold",
    "erosion", "dilation", "opening", "closing",
    "connected_components", "bounding_boxes",
}


@router.post("/process", summary="Run a processing operation on an uploaded image")
async def process_image(req: ProcessRequest):
    """
    Apply *processing_type* to the uploaded image identified by *image_id*.

    What it does:
      1. Validates the processing_type.
      2. Loads the image array.
      3. For 3-D volumes, extracts the middle axial slice for processing.
      4. Dispatches to the appropriate processing function.
      5. Returns the result as a base64 PNG plus any metadata (e.g. component list).
      6. Records the processing step in the image cache for the report.

    Returns
    -------
    JSON: { image_id, processing_type, result_image (base64 PNG),
            histogram_before, histogram_after, extra_meta }
    """
    if req.processing_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown processing_type '{req.processing_type}'. Allowed: {sorted(_ALLOWED_TYPES)}",
        )

    load_metadata(req.image_id)  # raises 404 if not found
    path = find_uploaded_file(req.image_id)
    arr, _ = load_image(path)

    # For 3-D volumes work on the middle axial slice
    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    extra_meta: Dict[str, Any] = {}

    # ── Dispatch ──────────────────────────────────────────────────────────────
    ptype = req.processing_type
    if ptype == "gaussian":
        result = flt.apply_gaussian(arr, sigma=req.sigma)
    elif ptype == "sobel":
        result = flt.apply_sobel(arr)
    elif ptype == "cdf_threshold":
        result = apply_cdf_threshold(arr, percentile=req.percentile).astype("float32")
    elif ptype == "erosion":
        result = morph.apply_erosion(arr, kernel_size=req.kernel_size)
    elif ptype == "dilation":
        result = morph.apply_dilation(arr, kernel_size=req.kernel_size)
    elif ptype == "opening":
        result = morph.apply_opening(arr, kernel_size=req.kernel_size)
    elif ptype == "closing":
        result = morph.apply_closing(arr, kernel_size=req.kernel_size)
    elif ptype == "connected_components":
        colour_img, components = morph.label_connected_components(arr, threshold=req.threshold)
        result = colour_img.astype("float32")
        extra_meta["components"] = components
        extra_meta["n_components"] = len(components)
    elif ptype == "bounding_boxes":
        result = morph.draw_bounding_boxes(arr, threshold=req.threshold)
    else:
        raise HTTPException(status_code=400, detail="Unreachable processing_type.")

    # Record in cache (for report)
    update_cache(req.image_id, {
        "last_processing": {
            "type": ptype,
            "params": {
                "sigma": req.sigma,
                "kernel_size": req.kernel_size,
                "percentile": req.percentile,
                "threshold": req.threshold,
            },
        }
    })

    return JSONResponse({
        "image_id": req.image_id,
        "processing_type": ptype,
        "result_image": "data:image/png;base64," + array_to_base64_png(result),
        "histogram_before": compute_histogram(arr, bins=64),
        "histogram_after": compute_histogram(result, bins=64),
        "extra_meta": extra_meta,
    })
