"""
api/export.py

What this module does:
  Downloadable export endpoints:
    GET /export/{image_id}/png  – current image as PNG
    GET /export/{image_id}/npy  – full array as .npy (base64 JSON)
    GET /export/{image_id}/csv  – per-component metrics as CSV
                                   (area in px and mm² using voxel spacing)

Why it exists:
  Researchers and clinicians need to export processed results in standard
  formats for use in external tools, Python notebooks, and spreadsheets.
"""

import base64
import csv
import io

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from api import find_uploaded_file, load_metadata
from core.loader import async_load_image, get_slice_2d, normalise_to_uint8
from processing.morphology import label_connected_components

router = APIRouter()


# ── PNG export ────────────────────────────────────────────────────────────────

@router.get("/export/{image_id}/png", summary="Download image as PNG")
async def export_png(image_id: str):
    """
    Return the uploaded image (middle axial slice for 3-D) as a PNG download.

    What it does:
      Loads the array, takes the middle axial slice for 3-D volumes,
      normalises to uint8, PNG-encodes, and returns as a binary response.
    """
    load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = await async_load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    u8 = normalise_to_uint8(arr)
    if u8.ndim == 3 and u8.shape[2] == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    elif u8.ndim == 3:
        u8 = u8[:, :, 0]

    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="medvision_{image_id[:8]}.png"'
        },
    )


# ── NPY export ────────────────────────────────────────────────────────────────

@router.get("/export/{image_id}/npy", summary="Download full array as .npy (base64 JSON)")
async def export_npy(image_id: str):
    """
    Return the full NumPy array as a base64-encoded .npy blob inside JSON.

    What it does:
      Loads the array in its original dimensionality (2-D or 3-D), saves it
      to an in-memory .npy buffer, and returns the base64-encoded bytes so
      the client can reconstruct it with np.load(io.BytesIO(base64.b64decode(...))).
    """
    load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = await async_load_image(path)

    buf = io.BytesIO()
    np.save(buf, arr)
    npy_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_id": image_id,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "npy_b64": npy_b64,
    }


@router.get("/export/{image_id}/npy/stream", summary="Download full array as streamed .npy")
async def export_npy_stream(image_id: str):
    """
    Stream the full NumPy array as a binary .npy download.

    This endpoint avoids base64 JSON overhead for large arrays while keeping
    /export/{image_id}/npy available for backward compatibility.
    """
    load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = await async_load_image(path)

    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    def _iter_chunks(chunk_size: int = 1024 * 256):
        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(
        _iter_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="medvision_{image_id[:8]}.npy"'},
    )


# ── CSV metrics export ────────────────────────────────────────────────────────

@router.get("/export/{image_id}/csv", summary="Download connected-component metrics as CSV")
async def export_csv(image_id: str):
    """
    Compute connected components and return per-component metrics as CSV.

    What it does:
      1. Loads the image (middle axial slice for 3-D).
      2. Runs connected-component analysis with threshold=128.
      3. Computes real-world area in mm² using stored voxel spacing.
      4. Returns a CSV with label, area_px, area_mm2, centroid, and bounding box.

    Why it exists:
      Exporting ROI measurements with physical units (mm²) is required for
      clinical quantification and downstream statistical analysis.
    """
    meta = load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = await async_load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    spacing = meta.get("spacing", [1.0, 1.0, 1.0])
    voxel_area_mm2 = float(spacing[0]) * float(spacing[1])

    _, components = label_connected_components(arr, threshold=128.0)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "label", "area_px", "area_mm2",
        "cx_px", "cy_px",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    ])
    for c in components:
        writer.writerow([
            c["label"],
            c["area"],
            round(c["area"] * voxel_area_mm2, 3),
            round(c["center"]["cx"], 2),
            round(c["center"]["cy"], 2),
            c["bounding_box"]["x"],
            c["bounding_box"]["y"],
            c["bounding_box"]["w"],
            c["bounding_box"]["h"],
        ])

    return PlainTextResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="medvision_metrics_{image_id[:8]}.csv"'
        },
    )
