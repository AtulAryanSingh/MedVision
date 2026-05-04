"""
api/upload.py

What this module does:
  POST /upload        – accept a single medical image (DICOM, NIfTI, PNG,
                        JPG), validate it, persist it to disk, load it with
                        the core loader to extract metadata, and return the
                        metadata + image_id.

  POST /upload-series – accept multiple .dcm files that form one DICOM
                        series, save them to a UUID-named directory, then
                        load the full 3-D volume via load_dicom_series().

Why it exists:
  Upload is the entry point for every other operation; keeping its logic
  isolated here makes it easy to add new accepted formats in the future.
"""

import os
import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api import find_uploaded_file, get_upload_dir, save_metadata
from core.loader import load_dicom_series, load_image

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


@router.post("/upload-series", summary="Upload a DICOM series (multiple .dcm files)")
async def upload_dicom_series(files: List[UploadFile] = File(...)):
    """
    Accept multiple .dcm files that form a single DICOM series.

    What it does:
      1. Validates that every uploaded file has a .dcm extension.
      2. Generates one UUID for the entire series.
      3. Creates  data/uploads/<image_id>/  and saves each file as
         <index:04d>.dcm  (index-based naming to avoid path-traversal via
         user-supplied filenames).
      4. Calls core.loader.load_dicom_series() which performs a
         header-only pass for sorting and then fills a pre-allocated 3-D
         array one slice at a time.
      5. Persists metadata to  data/cache/<image_id>.json.
      6. Returns the same JSON schema as /upload plus n_slices.

    Returns
    -------
    JSON: image_id, file_type="dicom_series", shape, ndim, is_3d,
          modality, intensity_min, intensity_max, spacing, extra_meta
          (with n_slices, series_instance_uid), size_bytes, n_slices.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Validate all files are .dcm
    for f in files:
        name = f.filename or ""
        _, ext = os.path.splitext(name)
        if ext.lower() != ".dcm":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"All files must be DICOM (.dcm). "
                    f"Received: '{name}'"
                ),
            )

    image_id = str(uuid.uuid4())
    series_dir = os.path.join(get_upload_dir(), image_id)
    os.makedirs(series_dir, exist_ok=True)

    total_bytes = 0
    try:
        for idx, f in enumerate(files):
            raw = await f.read()
            if len(raw) == 0:
                continue
            total_bytes += len(raw)
            # Save with a safe index-based name to avoid path traversal
            dest = os.path.join(series_dir, f"{idx:04d}.dcm")
            with open(dest, "wb") as fh:
                fh.write(raw)
    except Exception as exc:
        # Clean up on failure
        import shutil
        shutil.rmtree(series_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Failed to save series: {exc}")

    # Load the series into a 3-D volume
    try:
        arr, meta = load_dicom_series(series_dir)
    except Exception as exc:
        import shutil
        shutil.rmtree(series_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Could not load DICOM series: {exc}")

    meta["image_id"] = image_id
    meta["filename"] = f"{len(files)} DICOM files"
    meta["size_bytes"] = total_bytes
    save_metadata(image_id, meta)

    return JSONResponse({**meta, "image_id": image_id})
