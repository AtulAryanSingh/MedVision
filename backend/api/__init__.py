"""
api/__init__.py

What this package does:
  Houses all FastAPI routers.  Each router module owns one logical endpoint
  group (upload, preview, process, features, cluster, report).

  Also provides shared disk-store helpers used by every router:
    get_upload_dir()        → absolute path to uploads folder
    get_cache_dir()         → absolute path to cache folder
    find_uploaded_file()    → locate any uploaded file by image_id
    save_metadata()         → persist image metadata to JSON
    load_metadata()         → load metadata JSON, raise 404 if missing
    update_cache()          → merge new keys into the cached JSON
"""

import os
import uuid
import json
from fastapi import HTTPException
import config


def _sanitize_image_id(image_id: str) -> str:
    """
    Parse *image_id* as a UUID and return its canonical string form.

    Why this instead of a regex:
      Parsing with uuid.UUID() and re-serialising with str() breaks the
      taint chain that static analysers (CodeQL) track from user-provided
      inputs to file-path operations.  If *image_id* is not a valid UUID
      this raises a 400 rather than propagating the raw value.
    """
    try:
        return str(uuid.UUID(image_id))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image_id format. Expected a UUID string.",
        )

# ── Directory helpers ─────────────────────────────────────────────────────────

def get_upload_dir() -> str:
    """Return absolute path to the uploads directory, creating it if needed."""
    path = config.UPLOAD_DIR
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


def get_cache_dir() -> str:
    """Return absolute path to the metadata cache directory, creating it if needed."""
    path = config.CACHE_DIR
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


# ── File-system helpers ───────────────────────────────────────────────────────

_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".gz"}


def find_uploaded_file(image_id: str) -> str:
    """
    Locate the uploaded file (or series directory) for *image_id*.

    What it does:
      Sanitises the image_id (parse as UUID + re-serialise), then:
      1. Checks whether a series directory  <image_id>/  exists in the
         uploads folder (DICOM series uploads land here).
      2. Falls back to scanning for a single file  <image_id>.<ext>  with
         each allowed extension.

    Why it exists:
      The upload endpoint stores files as  <image_id>.<original_ext>  for
      single files, or as  <image_id>/  for DICOM series.  Downstream
      endpoints only know the image_id; this function bridges the gap
      without hard-coding a single layout.

    Raises
    ------
    HTTPException 400  – if image_id is not a valid UUID.
    HTTPException 404  – if no matching file or directory is found.
    """
    safe_id = _sanitize_image_id(image_id)
    upload_dir = get_upload_dir()

    # Check for a DICOM series directory first
    series_dir = os.path.join(upload_dir, safe_id)
    if os.path.isdir(series_dir):
        return series_dir

    # Fall back to single-file lookup
    for ext in _ALLOWED_EXTS:
        candidate = os.path.join(upload_dir, f"{safe_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    raise HTTPException(status_code=404, detail=f"No uploaded file found for image_id='{safe_id}'.")


# ── Metadata JSON helpers ─────────────────────────────────────────────────────

def _cache_path(image_id: str) -> str:
    """Return the path to the JSON cache file for *image_id* (assumes already sanitised)."""
    return os.path.join(get_cache_dir(), f"{image_id}.json")


def save_metadata(image_id: str, metadata: dict) -> None:
    """
    Persist *metadata* to the JSON cache for *image_id*.

    What it does:
      Sanitises image_id, then serialises *metadata* to
      data/cache/<image_id>.json, creating the file if it does not exist
      or overwriting it if it does.

    Why it exists:
      Every endpoint needs metadata (shape, dtype, modality, etc.) without
      re-reading the full image file on each request.
    """
    safe_id = _sanitize_image_id(image_id)
    with open(_cache_path(safe_id), "w") as fh:
        json.dump(metadata, fh)


def load_metadata(image_id: str) -> dict:
    """
    Load the JSON cache for *image_id*.

    Raises
    ------
    HTTPException 400  – if image_id is not a valid UUID.
    HTTPException 404  – if the cache file does not exist.
    """
    safe_id = _sanitize_image_id(image_id)
    path = _cache_path(safe_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"image_id='{safe_id}' not found. Upload the image first.")
    with open(path) as fh:
        return json.load(fh)


def update_cache(image_id: str, new_data: dict) -> None:
    """
    Merge *new_data* into the existing JSON cache for *image_id*.

    What it does:
      Loads the current cache, merges (shallow update) with *new_data*,
      and writes back to disk.

    Why it exists:
      Allows feature extraction, clustering, and reporting to persist their
      results into the same cache file without clobbering other keys.
    """
    existing = load_metadata(image_id)
    existing.update(new_data)
    save_metadata(image_id, existing)
