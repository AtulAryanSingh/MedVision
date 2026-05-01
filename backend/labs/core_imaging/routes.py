"""
labs/core_imaging/routes.py

What this module does:
  Defines the three HTTP endpoints for the Core Imaging Lab:

    POST /api/core-imaging/upload
      – Accept a multipart PNG/JPG upload, validate it, persist it under
        data/uploads/ with a UUID-based image_id, and return metadata.

    POST /api/core-imaging/process
      – Accept an image_id and a processing_type (kmeans | gaussian | sobel),
        apply the chosen algorithm, and return the result as a PNG byte stream.

    GET  /api/core-imaging/features/{image_id}
      – Accept an image_id and return JSON containing mean, std dev, and
        the intensity histogram (bins + counts).

Why it exists as a separate module:
  Keeping route definitions in their own file separates HTTP concerns from
  business logic (processing.py, features.py).  The router can be imported
  and mounted by main.py without tight coupling.
"""

import os
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse

from labs.core_imaging.processing import apply_gaussian, apply_kmeans, apply_sobel
from labs.core_imaging.features import extract_all

# ── Router setup ──────────────────────────────────────────────────────────────

router = APIRouter()

# Directory where uploaded originals are persisted.
# os.path.dirname(__file__) resolves to   backend/labs/core_imaging/
# We go three levels up to reach the   backend/   package root.
UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed MIME types – keep to PNG and JPEG for this lab.
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Valid processing operations exposed to the frontend.
ProcessingType = Literal["kmeans", "gaussian", "sobel"]


# ── Helper ────────────────────────────────────────────────────────────────────

def _validate_image_id(image_id: str) -> str:
    """
    Validate that *image_id* is a well-formed UUID and return a canonical,
    sanitised representation derived from the parsed UUID object.

    What it does:
      Parses *image_id* as a UUID.  If parsing succeeds, the canonical string
      is produced from the UUID object itself (not from the raw input), which
      breaks the taint chain used by static-analysis tools.  Raises HTTP 400
      if the value is not a valid UUID.

    Why it exists:
      All image_ids are generated internally by uuid.uuid4(), so any value
      that cannot be parsed as a UUID did not originate from a legitimate
      upload.  Rejecting such values before any file-system operation prevents
      path-traversal attacks.

    Returns
    -------
    str
        The canonical lower-case UUID string (e.g. ``"3fa85f64-5717-…"``).
    """
    try:
        # Reconstruct the string from the UUID object to break the taint chain –
        # the returned value comes from uuid.UUID, not from user input.
        return str(uuid.UUID(image_id))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid image_id format.")


def _load_image(image_id: str) -> np.ndarray:
    """
    Load a previously uploaded image from disk as a BGR NumPy array.

    What it does:
      Validates *image_id* as a well-formed UUID, derives a safe canonical
      string from the parsed UUID object, then scans UPLOAD_DIR for a matching
      file and decodes it with OpenCV.

    Why it exists:
      Centralises the file-lookup + decode logic so upload, process, and
      features endpoints all use the same code path.

    Raises
    ------
    HTTPException 400
        If image_id is not a valid UUID (guards against path traversal).
    HTTPException 404
        If no file with the given image_id is found.
    HTTPException 422
        If the file cannot be decoded as an image.
    """
    # Sanitise: safe_id is derived from uuid.UUID, not from raw user input.
    safe_id = _validate_image_id(image_id)

    upload_dir = os.path.realpath(UPLOAD_DIR)
    for ext in ALLOWED_EXTENSIONS:
        # Build path using the sanitised (UUID-object-derived) stem.
        candidate = os.path.realpath(os.path.join(upload_dir, f"{safe_id}{ext}"))
        # Belt-and-suspenders: ensure the resolved path stays within UPLOAD_DIR.
        if not candidate.startswith(upload_dir + os.sep):
            raise HTTPException(status_code=400, detail="Invalid image_id format.")
        if os.path.isfile(candidate):
            img = cv2.imread(candidate)
            if img is None:
                raise HTTPException(status_code=422, detail="Stored file is not a valid image.")
            return img
    raise HTTPException(status_code=404, detail=f"Image not found.")


def _encode_to_png(image: np.ndarray) -> bytes:
    """
    Encode a NumPy image array to PNG bytes.

    What it does:
      Uses cv2.imencode to compress the array as PNG and returns the raw bytes.

    Why it exists:
      Both the process endpoint and potential future endpoints need to return
      image bytes; extracting this keeps each call-site concise.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image to PNG.")
    return buffer.tobytes()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload an image for analysis")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept a multipart PNG/JPG upload and store it for later processing.

    What it does:
      1. Validates the file's content-type and extension.
      2. Reads the raw bytes and attempts to decode them with OpenCV to confirm
         the payload is a valid image (not just a renamed non-image).
      3. Saves the file to UPLOAD_DIR as  <uuid><original_ext>.
      4. Returns a JSON body with the generated image_id and basic metadata.

    Why it exists:
      Provides the entry point for all subsequent operations; every other
      endpoint is keyed on the image_id returned here.
    """
    # ── 1. Validate content-type ──────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()

    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Allowed: PNG, JPEG.",
        )

    # ── 2. Read and verify the image bytes ────────────────────────────────────
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="File could not be decoded as an image.")

    # ── 3. Persist to disk ────────────────────────────────────────────────────
    # Normalise extension: default to .png when the content-type is clearer
    # than the filename extension.
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".png"
    image_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    # ── 4. Return metadata ────────────────────────────────────────────────────
    return JSONResponse({
        "image_id": image_id,
        "filename": file.filename,
        "content_type": content_type or f"image/{ext.lstrip('.')}",
        "size_bytes": len(raw_bytes),
        "width": width,
        "height": height,
        "channels": channels,
    })


@router.post("/process", summary="Apply a processing algorithm to an uploaded image")
async def process_image(
    image_id: str = Form(...),
    processing_type: ProcessingType = Form(...),
):
    """
    Apply the requested processing algorithm and return the result as PNG.

    What it does:
      1. Loads the original image identified by *image_id*.
      2. Dispatches to the appropriate processing function based on
         *processing_type* (kmeans | gaussian | sobel).
      3. Encodes the result as PNG bytes and streams them back to the caller.

    Why it exists:
      Providing a single "process" endpoint with a type selector keeps the
      API surface small; extending it later means adding a new function in
      processing.py and a new entry in the dispatch dict here.

    Returns
    -------
    PNG image bytes (Content-Type: image/png).
    """
    img = _load_image(image_id)

    # Dispatch table – maps processing_type string to (function, kwargs)
    dispatch = {
        "kmeans": (apply_kmeans, {}),
        "gaussian": (apply_gaussian, {}),
        "sobel": (apply_sobel, {}),
    }

    func, kwargs = dispatch[processing_type]
    result = func(img, **kwargs)

    png_bytes = _encode_to_png(result)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/features/{image_id}", summary="Extract basic statistical features from an image")
async def get_features(image_id: str):
    """
    Compute and return mean, standard deviation, and intensity histogram.

    What it does:
      1. Loads the original image identified by *image_id*.
      2. Calls extract_all() which converts to greyscale and computes the three
         features.
      3. Returns the result as a JSON body.

    Why it exists:
      Feature extraction is a distinct analytical concern from processing; a
      dedicated endpoint lets the frontend (or any API consumer) request only
      the statistics it needs without also receiving image bytes.

    Returns
    -------
    JSON with keys: image_id, mean, std_dev, histogram {bins, counts}.
    """
    img = _load_image(image_id)
    features = extract_all(img)
    return JSONResponse({"image_id": image_id, **features})
