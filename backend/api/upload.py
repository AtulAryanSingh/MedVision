"""
api/upload.py

What this module does:
  POST /upload – accept a medical image file (DICOM, NIfTI, PNG, JPG),
  validate it, persist it to disk, load it with the core loader to extract
  metadata, and return the metadata + image_id.

Why it exists:
  Upload is the entry point for every other operation; keeping its logic
  isolated here makes it easy to add new accepted formats in the future.
"""

import os
import uuid

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api import find_uploaded_file, get_upload_dir, save_metadata
from core.loader import load_image

router = APIRouter()

# Accepted MIME types and extensions
_ALLOWED_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "application/dicom", "application/octet-stream",
}
_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".gz"}


@router.post("/upload", summary="Upload a medical image for analysis")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept a multipart medical image upload, validate and persist it.

    What it does:
      1. Validates the file extension (MIME type is optional hint only).
      2. Reads the raw bytes.
      3. Saves the file to  data/uploads/<image_id>.<ext>.
      4. Loads it with core.loader.load_image() to extract metadata.
      5. Persists metadata to  data/cache/<image_id>.json.
      6. Returns JSON with image_id and all extracted metadata.

    Returns
    -------
    JSON: image_id, filename, file_type, shape, ndim, is_3d, modality,
          intensity_min, intensity_max, spacing, extra_meta, size_bytes.
    """
    filename = file.filename or "upload"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Handle .nii.gz double extension
    if filename.lower().endswith(".nii.gz"):
        ext = ".gz"

    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed: {sorted(_ALLOWED_EXTS)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_id = str(uuid.uuid4())
    save_path = os.path.join(get_upload_dir(), f"{image_id}{ext}")

    with open(save_path, "wb") as fh:
        fh.write(raw)

    # Load with the core loader to validate and extract metadata
    try:
        arr, meta = load_image(save_path)
    except Exception as exc:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=f"Could not parse image: {exc}")

    meta["image_id"] = image_id
    meta["filename"] = filename
    meta["size_bytes"] = len(raw)
    save_metadata(image_id, meta)

    return JSONResponse({**meta, "image_id": image_id})
