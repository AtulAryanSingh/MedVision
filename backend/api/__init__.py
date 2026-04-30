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
import json
from fastapi import HTTPException

# ── Directory helpers ─────────────────────────────────────────────────────────

def get_upload_dir() -> str:
    """Return absolute path to the uploads directory, creating it if needed."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


def get_cache_dir() -> str:
    """Return absolute path to the metadata cache directory, creating it if needed."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


# ── File-system helpers ───────────────────────────────────────────────────────

_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".gz"}


def find_uploaded_file(image_id: str) -> str:
    """
    Locate the uploaded file for *image_id* in the uploads directory.

    What it does:
      Scans the uploads directory for a file whose stem equals *image_id*
      and whose extension is in the allowed set.

    Why it exists:
      The upload endpoint stores files as  <image_id>.<original_ext>, but
      downstream endpoints only know the image_id.  This function bridges
      that gap without hard-coding a single extension.

    Raises
    ------
    HTTPException 404  – if no matching file is found.
    """
    upload_dir = get_upload_dir()
    for ext in _ALLOWED_EXTS:
        candidate = os.path.join(upload_dir, f"{image_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    raise HTTPException(status_code=404, detail=f"No uploaded file found for image_id='{image_id}'.")


# ── Metadata JSON helpers ─────────────────────────────────────────────────────

def _cache_path(image_id: str) -> str:
    """Return the path to the JSON cache file for *image_id*."""
    return os.path.join(get_cache_dir(), f"{image_id}.json")


def save_metadata(image_id: str, metadata: dict) -> None:
    """
    Persist *metadata* to the JSON cache for *image_id*.

    What it does:
      Serialises *metadata* to  data/cache/<image_id>.json, creating the
      file if it does not exist or overwriting it if it does.

    Why it exists:
      Every endpoint needs metadata (shape, dtype, modality, etc.) without
      re-reading the full image file on each request.
    """
    with open(_cache_path(image_id), "w") as fh:
        json.dump(metadata, fh)


def load_metadata(image_id: str) -> dict:
    """
    Load the JSON cache for *image_id*.

    Raises
    ------
    HTTPException 404  – if the cache file does not exist.
    """
    path = _cache_path(image_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"image_id='{image_id}' not found. Upload the image first.")
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
