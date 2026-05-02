"""
api/mpr.py

What this module does:
  GET /mpr/{image_id} – return MPR (Multi-Planar Reconstruction) slices with
  geometrically-correct aspect ratios computed from voxel spacing, FOV
  dimensions, and optional window/level contrast adjustment.

Why it exists:
  The basic /preview endpoint normalises to screen pixels without honouring
  voxel spacing, so non-isotropic volumes appear stretched.  MPR with correct
  aspect ratios is essential for accurate clinical review.
"""

from typing import Optional
import base64

import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api import find_uploaded_file, load_metadata
from core.loader import load_image, get_slice_2d, normalise_to_uint8

router = APIRouter()


# ── Window / level ────────────────────────────────────────────────────────────

def apply_window_level(
    arr: np.ndarray, window_center: float, window_width: float
) -> np.ndarray:
    """
    Apply window/level (contrast windowing) to *arr*.

    What it does:
      Clips the array to [center - width/2, center + width/2] then linearly
      rescales to [0, 255] float32.

    Why it exists:
      CT images span thousands of Hounsfield units; viewing them without
      windowing compresses almost all contrast into a few grey levels.
    """
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.float32)


# ── Slice encoder with aspect-ratio correction ────────────────────────────────

def _slice_to_b64(
    slc: np.ndarray,
    pixel_size_row: float,
    pixel_size_col: float,
    max_dim: int = 512,
) -> str:
    """
    Encode *slc* as a base64 PNG resampled to the physical aspect ratio.

    What it does:
      Computes the physical extent (in mm) of each axis, scales so the longer
      physical extent maps to *max_dim* pixels while preserving the correct
      aspect ratio, then PNG-encodes and base64-encodes the result.

    Why it exists:
      Without this correction, slices from volumes with non-isotropic voxels
      (common in CT: e.g. 0.5 mm in-plane, 3 mm slice thickness) appear
      compressed or stretched.
    """
    u8 = normalise_to_uint8(slc)
    if u8.ndim == 3:
        u8 = u8[:, :, 0]

    h, w = u8.shape
    phys_h = max(h * abs(pixel_size_row), 1e-6)
    phys_w = max(w * abs(pixel_size_col), 1e-6)

    scale = max_dim / max(phys_h, phys_w)
    new_h = max(1, int(round(phys_h * scale)))
    new_w = max(1, int(round(phys_w * scale)))
    resized = cv2.resize(u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ok, buf = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/mpr/{image_id}", summary="Multi-Planar Reconstruction with correct aspect ratio")
async def get_mpr(
    image_id: str,
    axial_idx: Optional[int] = Query(None, description="Axial slice index (axis 0)"),
    coronal_idx: Optional[int] = Query(None, description="Coronal slice index (axis 1)"),
    sagittal_idx: Optional[int] = Query(None, description="Sagittal slice index (axis 2)"),
    window_center: Optional[float] = Query(None, description="Window/level centre value"),
    window_width: Optional[float] = Query(None, description="Window/level width"),
    max_dim: int = Query(512, ge=64, le=2048, description="Max output dimension in pixels"),
):
    """
    Return MPR slices with geometrically-correct aspect ratios.

    What it does:
      1. Loads the image and its stored voxel-spacing metadata.
      2. Optionally applies window/level contrast adjustment.
      3. Extracts axial / coronal / sagittal slices at the requested indices.
      4. Resamples each slice to the correct physical aspect ratio.
      5. Returns base64 PNGs plus FOV dimensions in mm.

    Returns
    -------
    JSON:
      image_id, is_3d, shape, spacing_mm [z,y,x],
      fov_mm { z_mm, y_mm, x_mm },
      slice_indices { axial, coronal, sagittal },
      axial / coronal / sagittal (base64 PNG data-URIs)
    """
    meta = load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = load_image(path)

    shape = arr.shape
    is_3d = meta.get("is_3d", False)
    spacing = meta.get("spacing", [1.0, 1.0, 1.0])
    while len(spacing) < 3:
        spacing.append(1.0)
    sp_z, sp_y, sp_x = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # Apply window/level if requested
    if window_center is not None and window_width is not None and window_width > 0:
        arr = apply_window_level(arr, window_center, window_width)

    if is_3d and arr.ndim >= 3:
        ax_i  = axial_idx    if axial_idx    is not None else shape[0] // 2
        cor_i = coronal_idx  if coronal_idx  is not None else shape[1] // 2
        sag_i = sagittal_idx if sagittal_idx is not None else shape[2] // 2
    else:
        ax_i = cor_i = sag_i = 0

    # Slice orientations and their row/col spacings:
    #   axial    (axis 0): rows=y (sp_y), cols=x (sp_x)
    #   coronal  (axis 1): rows=z (sp_z), cols=x (sp_x)
    #   sagittal (axis 2): rows=z (sp_z), cols=y (sp_y)
    axial_slc    = get_slice_2d(arr, axis=0, index=ax_i)
    coronal_slc  = get_slice_2d(arr, axis=1, index=cor_i)
    sagittal_slc = get_slice_2d(arr, axis=2, index=sag_i)

    axial_b64    = _slice_to_b64(axial_slc,    sp_y, sp_x, max_dim)
    coronal_b64  = _slice_to_b64(coronal_slc,  sp_z, sp_x, max_dim)
    sagittal_b64 = _slice_to_b64(sagittal_slc, sp_z, sp_y, max_dim)

    # FOV in mm
    if is_3d and arr.ndim >= 3:
        fov_mm = {
            "z_mm": round(shape[0] * sp_z, 2),
            "y_mm": round(shape[1] * sp_y, 2),
            "x_mm": round(shape[2] * sp_x, 2),
        }
    else:
        h, w = shape[:2]
        fov_mm = {"z_mm": round(h * sp_z, 2), "y_mm": round(h * sp_y, 2), "x_mm": round(w * sp_x, 2)}

    return JSONResponse({
        "image_id": image_id,
        "is_3d": is_3d,
        "shape": list(shape),
        "spacing_mm": [sp_z, sp_y, sp_x],
        "fov_mm": fov_mm,
        "slice_indices": {"axial": ax_i, "coronal": cor_i, "sagittal": sag_i},
        "axial":    axial_b64,
        "coronal":  coronal_b64,
        "sagittal": sagittal_b64,
    })
