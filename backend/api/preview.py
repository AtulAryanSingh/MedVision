"""
api/preview.py

What this module does:
  GET /preview/{image_id} – return base64-encoded PNG slices for the
  axial, sagittal, and coronal orientations of the uploaded image.

  For 2-D images, all three orientations return the same image.
  For 3-D volumes (DICOM, NIfTI) each orientation returns the slice at
  the requested index (defaulting to the middle).

Why it exists:
  Visualising slices is the first action users take after uploading;
  keeping the preview logic here keeps the route clean.
"""

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api import find_uploaded_file, load_metadata
from core.loader import array_to_base64_png, get_slice_2d, load_image

router = APIRouter()


@router.get("/preview/{image_id}", summary="Get axial / sagittal / coronal slice previews")
async def get_preview(
    image_id: str,
    axial_idx: Optional[int] = Query(None, description="Axial slice index (axis 0 for 3-D)"),
    coronal_idx: Optional[int] = Query(None, description="Coronal slice index (axis 1 for 3-D)"),
    sagittal_idx: Optional[int] = Query(None, description="Sagittal slice index (axis 2 for 3-D)"),
):
    """
    Return slice previews as base64 PNG strings.

    What it does:
      1. Loads the image into a NumPy array.
      2. Extracts a 2-D slice for each orientation using core.loader.get_slice_2d().
      3. Normalises and encodes each slice as a base64 PNG string.
      4. Returns all three in a JSON payload along with shape / is_3d info.

    Returns
    -------
    JSON: { axial, sagittal, coronal, shape, is_3d, slice_indices }
         where axial/sagittal/coronal are base64 PNG data-URI strings.
    """
    meta = load_metadata(image_id)
    path = find_uploaded_file(image_id)
    arr, _ = load_image(path)

    shape = arr.shape
    is_3d = meta.get("is_3d", False)

    # Determine default middle indices for 3-D volumes
    if is_3d and arr.ndim >= 3:
        ax_i  = axial_idx    if axial_idx    is not None else shape[0] // 2
        cor_i = coronal_idx  if coronal_idx  is not None else shape[1] // 2
        sag_i = sagittal_idx if sagittal_idx is not None else shape[2] // 2
    else:
        ax_i = cor_i = sag_i = 0

    axial_slice    = get_slice_2d(arr, axis=0, index=ax_i)
    coronal_slice  = get_slice_2d(arr, axis=1, index=cor_i)
    sagittal_slice = get_slice_2d(arr, axis=2, index=sag_i)

    return JSONResponse({
        "image_id": image_id,
        "is_3d": is_3d,
        "shape": list(shape),
        "slice_indices": {"axial": ax_i, "coronal": cor_i, "sagittal": sag_i},
        "axial":    "data:image/png;base64," + array_to_base64_png(axial_slice),
        "coronal":  "data:image/png;base64," + array_to_base64_png(coronal_slice),
        "sagittal": "data:image/png;base64," + array_to_base64_png(sagittal_slice),
    })
