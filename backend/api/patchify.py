"""
api/patchify.py

What this module does:
  POST /patchify – divide a 3-D volume into fixed-size overlapping or
  non-overlapping patches and return patch metadata plus a base64-encoded
  .npz blob ready for download.

Why it exists:
  Patch extraction is the standard pre-processing step before training 3-D
  CNNs (U-Net, 3-D CNN classifiers).  Running it server-side lets every
  client — browser, Colab, CLI — use the same logic.
"""

import base64
import io

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import find_uploaded_file, load_metadata
from core.loader import load_image

router = APIRouter()


class PatchifyRequest(BaseModel):
    """Request body for the patchify endpoint."""
    image_id: str
    patch_size: int = Field(32, ge=8, le=256, description="Cubic patch side length in voxels")
    stride: int = Field(16, ge=1, le=256, description="Stride (voxels); overlap = patch_size − stride")


@router.post("/patchify", summary="Extract 3-D patches from a volume")
async def patchify_volume(req: PatchifyRequest):
    """
    Extract cubic patches from a 3-D volume.

    What it does:
      1. Validates the upload is a 3-D volume.
      2. Slides a cubic window of side *patch_size* across the volume with
         step *stride* in each dimension.
      3. Stacks all patches into an (N, P, P, P) float32 array and saves
         it together with patch coordinates and voxel spacing to an in-memory
         .npz file.
      4. Returns patch metadata and the base64-encoded .npz for download.

    Returns
    -------
    JSON:
      image_id, volume_shape, patch_size, stride, n_patches, patches_shape,
      spacing_mm, npz_b64 (base64-encoded .npz containing arrays:
        'patches'  – float32 (N, P, P, P)
        'coords'   – int32   (N, 3) [z, y, x top-left corner]
        'spacing'  – float32 (3,) voxel spacing in mm)
    """
    meta = load_metadata(req.image_id)
    if not meta.get("is_3d", False):
        raise HTTPException(status_code=422, detail="Patchify requires a 3-D volume.")

    path = find_uploaded_file(req.image_id)
    arr, _ = load_image(path)

    if arr.ndim != 3:
        raise HTTPException(status_code=422, detail="Volume must be exactly 3-D (D×H×W).")

    D, H, W = arr.shape
    ps = req.patch_size
    st = req.stride

    patches, coords = [], []
    for z in range(0, D - ps + 1, st):
        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                patches.append(arr[z:z + ps, y:y + ps, x:x + ps])
                coords.append([z, y, x])

    if not patches:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Volume {D}×{H}×{W} is too small for patch size {ps}. "
                f"All dimensions must be ≥ {ps} voxels."
            ),
        )

    patches_arr = np.stack(patches, axis=0).astype(np.float32)
    coords_arr  = np.array(coords, dtype=np.int32)
    spacing     = np.array(meta.get("spacing", [1.0, 1.0, 1.0]), dtype=np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, patches=patches_arr, coords=coords_arr, spacing=spacing)
    npz_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "image_id":     req.image_id,
        "volume_shape": [D, H, W],
        "patch_size":   ps,
        "stride":       st,
        "n_patches":    len(patches),
        "patches_shape": list(patches_arr.shape),
        "spacing_mm":   meta.get("spacing", [1.0, 1.0, 1.0]),
        "npz_b64":      npz_b64,
    })
