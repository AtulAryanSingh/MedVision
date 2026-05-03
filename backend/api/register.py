"""
api/register.py

What this module does:
  POST /register – apply a 2-D geometric transform (translation, rotation,
  zoom/rescale, or a combined affine) to an uploaded image and return the
  resampled result as a base64 PNG.

  Supported transform_type values:
    translate   – shift image by (shift_x, shift_y) pixels
    rotate      – rotate by angle_deg degrees around the image centre
    zoom        – rescale by zoom_x and zoom_y factors
    affine      – full 2×2 matrix + translation (advanced)

Why it exists:
  Registration and resampling are standard preprocessing steps before
  multi-image analysis (e.g., aligning serial scans or atlas registration).
  Keeping it as an independent endpoint means the frontend can call it on
  demand without triggering any other pipeline step.

Interpolation rules:
  - order=3 (bicubic) for regular images
  - order=0 (nearest-neighbour) for binary masks / label maps
"""

from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy.ndimage import affine_transform, rotate, zoom

from api import find_uploaded_file, load_metadata
from core.loader import array_to_base64_png, get_slice_2d, load_image, normalise_to_uint8
from processing.histogram import compute_histogram

router = APIRouter()

_ALLOWED_TRANSFORMS = {"translate", "rotate", "zoom", "affine"}


class RegisterRequest(BaseModel):
    """Request body for the register endpoint."""

    image_id: str
    transform_type: str = Field("rotate", description="One of: translate | rotate | zoom | affine")
    # translate
    shift_x: float = Field(0.0, description="Horizontal shift in pixels (translate)")
    shift_y: float = Field(0.0, description="Vertical shift in pixels (translate)")
    # rotate
    angle_deg: float = Field(0.0, description="Counter-clockwise rotation angle in degrees")
    # zoom
    zoom_x: float = Field(1.0, ge=0.1, le=10.0, description="Horizontal zoom factor (>1 = enlarge)")
    zoom_y: float = Field(1.0, ge=0.1, le=10.0, description="Vertical zoom factor (>1 = enlarge)")
    # interpolation
    is_mask: bool = Field(False, description="Use nearest-neighbour interpolation (for binary masks)")
    # affine (row-major 2×2, then tx, ty)
    matrix: Optional[List[float]] = Field(
        None,
        description="2×2 row-major transform matrix as 4 floats [m00,m01,m10,m11] (affine only)",
    )
    tx: float = Field(0.0, description="X translation for affine mode")
    ty: float = Field(0.0, description="Y translation for affine mode")


@router.post("/register", summary="Apply a geometric transform (registration/resampling) to an image")
async def register_image(req: RegisterRequest):
    """
    Apply a 2-D geometric transform to the uploaded image.

    What it does:
      1. Validates the transform_type.
      2. Loads the image (middle axial slice for 3-D volumes).
      3. Applies the requested transform with the appropriate interpolation order.
      4. Returns the resampled image as a base64 PNG plus before/after histograms.

    Returns
    -------
    JSON: { image_id, transform_type, result_image (base64 PNG),
            histogram_before, histogram_after, output_shape }
    """
    if req.transform_type not in _ALLOWED_TRANSFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown transform_type '{req.transform_type}'. Allowed: {sorted(_ALLOWED_TRANSFORMS)}",
        )

    load_metadata(req.image_id)
    path = find_uploaded_file(req.image_id)
    arr, _ = load_image(path)

    # For 3-D volumes, work on the middle axial slice
    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    # Collapse colour to greyscale for processing
    u8 = normalise_to_uint8(arr)
    if u8.ndim == 3:
        grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        grey = arr.astype(np.float32)

    order = 0 if req.is_mask else 3

    ttype = req.transform_type

    if ttype == "translate":
        # shift = [row_shift, col_shift] → [shift_y, shift_x]
        result = affine_transform(
            grey,
            matrix=np.eye(2),
            offset=[-req.shift_y, -req.shift_x],
            order=order,
            mode="constant",
            cval=0.0,
        )

    elif ttype == "rotate":
        result = rotate(
            grey,
            angle=req.angle_deg,
            reshape=False,
            order=order,
            mode="constant",
            cval=0.0,
        )

    elif ttype == "zoom":
        zoomed = zoom(grey, zoom=(req.zoom_y, req.zoom_x), order=order)
        # Crop or pad back to original size so the output shape is predictable
        h, w = grey.shape
        zh, zw = zoomed.shape
        result = np.zeros_like(grey)
        cy, cx = min(h, zh), min(w, zw)
        result[:cy, :cx] = zoomed[:cy, :cx]

    elif ttype == "affine":
        if req.matrix is None or len(req.matrix) != 4:
            raise HTTPException(
                status_code=400,
                detail="For 'affine' transform_type, supply 'matrix' as exactly 4 floats [m00,m01,m10,m11].",
            )
        mat = np.array(req.matrix).reshape(2, 2)
        result = affine_transform(
            grey,
            matrix=mat,
            offset=[req.ty, req.tx],
            order=order,
            mode="constant",
            cval=0.0,
        )
    else:
        raise HTTPException(status_code=400, detail="Unreachable transform_type.")

    result = result.astype(np.float32)

    return JSONResponse({
        "image_id": req.image_id,
        "transform_type": ttype,
        "result_image": "data:image/png;base64," + array_to_base64_png(result),
        "histogram_before": compute_histogram(grey, bins=64),
        "histogram_after": compute_histogram(result, bins=64),
        "output_shape": list(result.shape),
    })
