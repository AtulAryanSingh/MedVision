"""
api/report.py

What this module does:
  GET /report/{image_id} – aggregate all available cached analysis results
  for an image and return a structured JSON report.

Why it exists:
  The report endpoint is the final aggregation point; keeping it here
  separates the HTTP layer from the report-assembly logic in analysis/report.py.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api import find_uploaded_file, load_metadata
from core.loader import get_slice_2d, load_image
from analysis.report import generate_report
from features.extractor import extract_features

router = APIRouter()


@router.get("/report/{image_id}", summary="Generate a structured analysis report for an image")
async def get_report(image_id: str):
    """
    Return a full structured analysis report for *image_id*.

    What it does:
      1. Loads the cached metadata (raises 404 if not found).
      2. Uses cached features if available; otherwise computes them on-the-fly.
      3. Uses cached cluster info if available.
      4. Calls analysis.report.generate_report() to assemble the report.
      5. Returns the complete report as JSON.

    Returns
    -------
    JSON: structured report dict (see analysis/report.py for full schema).
    """
    meta = load_metadata(image_id)

    # Use cached features, or compute them now
    features = meta.get("features")
    if features is None:
        path = find_uploaded_file(image_id)
        arr, _ = load_image(path)
        if arr.ndim == 3 and arr.shape[0] > 4:
            arr = get_slice_2d(arr, axis=0)
        features = extract_features(arr)

    cluster_info = meta.get("cluster")
    last_processing = meta.get("last_processing")
    processing_history = [last_processing] if last_processing else []

    report = generate_report(
        image_id=image_id,
        metadata=meta,
        features=features,
        cluster_info=cluster_info,
        processing_history=processing_history,
    )
    return JSONResponse(report)
