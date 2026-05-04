"""
core/loader.py

What this module does:
  Unified interface for loading medical images from disk into NumPy arrays.
  Supports DICOM series directories, single DICOM (.dcm), NIfTI (.nii /
  .nii.gz), and standard 2-D formats (PNG, JPG).  Also provides
  slice-extraction and base64-PNG encoding helpers used by the API preview
  and process endpoints.

Why it exists:
  Centralising all format-specific parsing here keeps every other module
  agnostic about how the data was stored on disk.

Dependencies: OpenCV, NumPy, pydicom (optional), nibabel (optional)
"""

import base64
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Canonical axis helper ─────────────────────────────────────────────────────
# Project-wide convention:
#   - 3-D volumes are represented in NumPy as (Z, Y, X) == (D, H, W)
#   - spacing is stored in metadata as [sp_z, sp_y, sp_x] (mm)
#
# This makes all downstream consumers (FOV, MPR, patchify, etc.) consistent.

def _canon_xyz_from_array_and_spacing(
    arr: np.ndarray,
    spacing: List[float],
    *,
    spacing_order: str,
) -> Dict[str, Any]:
    """
    Convert shape + spacing into strict (sizeX,sizeY,sizeZ) and (spacingX,spacingY,spacingZ),
    and return canonical spacing_zyx = [sp_z, sp_y, sp_x].

    Parameters
    ----------
    arr : np.ndarray
        Loaded array. For 3-D volumes this project uses (Z,Y,X).
    spacing : list[float]
        Spacing values as read from the file header.
    spacing_order : str
        - "zyx": spacing is [sp_z, sp_y, sp_x]
        - "xyz": spacing is [sp_x, sp_y, sp_z]  (common in some libraries / headers)

    Returns
    -------
    dict with:
      sizeX,sizeY,sizeZ (ints),
      spacingX,spacingY,spacingZ (floats),
      spacing_zyx (list[float] == [sp_z, sp_y, sp_x])
    """
    sp = [float(x) for x in (spacing or [])]
    while len(sp) < 3:
        sp.append(1.0)

    order = spacing_order.lower().strip()
    if order == "zyx":
        sp_z, sp_y, sp_x = sp[0], sp[1], sp[2]
    elif order == "xyz":
        sp_x, sp_y, sp_z = sp[0], sp[1], sp[2]
    else:
        raise ValueError(f"Unknown spacing_order='{spacing_order}' (expected 'zyx' or 'xyz')")

    # Determine whether this is a true 3-D volume (Z,Y,X) vs color image (H,W,C)
    is_color_2d = (arr.ndim == 3 and arr.shape[2] in (1, 3, 4))
    if arr.ndim >= 3 and not is_color_2d:
        sizeZ, sizeY, sizeX = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    else:
        # Treat 2-D images as a single-slice volume for consistent metadata math
        sizeZ, sizeY, sizeX = 1, int(arr.shape[0]), int(arr.shape[1])

    # Basic sanity
    if not np.isfinite([sp_x, sp_y, sp_z]).all():
        sp_x = sp_x if np.isfinite(sp_x) else 1.0
        sp_y = sp_y if np.isfinite(sp_y) else 1.0
        sp_z = sp_z if np.isfinite(sp_z) else 1.0
    if abs(sp_x) < 1e-12: sp_x = 1.0
    if abs(sp_y) < 1e-12: sp_y = 1.0
    if abs(sp_z) < 1e-12: sp_z = 1.0

    return {
        "sizeX": sizeX,
        "sizeY": sizeY,
        "sizeZ": sizeZ,
        "spacingX": float(sp_x),
        "spacingY": float(sp_y),
        "spacingZ": float(sp_z),
        "spacing_zyx": [float(sp_z), float(sp_y), float(sp_x)],
    }


# ── Public entry point ────────────────────────────────────────────────────────

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a medical image from *path* and return (array, metadata).

    What it does:
      - If *path* is a directory it is treated as a DICOM series and
        delegated to load_dicom_series().
      - Otherwise inspects the file extension to route to the correct
        format-specific loader.

    Why it exists:
      All API endpoints need a consistent, format-agnostic way to obtain a
      NumPy array from whatever the user uploaded.

    Parameters
    ----------
    path : str
        Absolute path to the image file or series directory on disk.

    Returns
    -------
    array : np.ndarray
        float32 array of shape (H, W), (H, W, C), or (D, H, W) depending
        on the format.
    metadata : dict
        Keys: file_type, shape, dtype_str, ndim, intensity_min,
              intensity_max, spacing, modality, is_3d, extra_meta.
    """
    if os.path.isdir(path):
        return load_dicom_series(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return _load_dicom(path)
    if ext in (".nii", ".gz"):
        return _load_nifti(path)
    return _load_standard(path)


# ── DICOM series loader ───────────────────────────────────────────────────────

def load_dicom_series(directory: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a directory of DICOM files as a 3-D volume.

    Canonical output:
      volume shape: (Z, Y, X)
      spacing meta: [sp_z, sp_y, sp_x]  in mm

    What it does:
      1. Scans *directory* for .dcm files.
      2. Header-only pass (stop_before_pixels=True) to extract
         SeriesInstanceUID, InstanceNumber, and ImagePositionPatient.
      3. Groups slices by SeriesInstanceUID; selects the largest group.
      4. Sorts slices by InstanceNumber (primary) or
         ImagePositionPatient[2] z-coordinate (fallback).
      5. Pre-allocates a float32 (D, H, W) array and fills it one slice at
         a time so only one slice of pixel data is in memory at once.
      6. Derives voxel spacing: PixelSpacing (y, x) + slice gap from
         ImagePositionPatient or SliceThickness.
      7. Passes spacing through the canonical helper so metadata is always
         stored as [z,y,x] and downstream MPR/FOV are correct.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom: pip install pydicom")

    # ── 1. Collect .dcm paths ─────────────────────────────────────────────
    dcm_paths: List[str] = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".dcm")
    ])
    if not dcm_paths:
        raise ValueError(f"No .dcm files found in directory: {directory}")

    # ── 2. Header-only pass ───────────────────────────────────────────────
    headers: List[Tuple[str, Any]] = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            headers.append((p, ds))
        except Exception:
            continue

    if not headers:
        raise ValueError("No readable DICOM files found in directory.")

    # ── 3. Group by SeriesInstanceUID, keep largest group ─────────────────
    groups: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    for p, ds in headers:
        uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        groups[uid].append((p, ds))

    uid, group = max(groups.items(), key=lambda kv: len(kv[1]))

    # ── 4. Sort slices ────────────────────────────────────────────────────
    def _sort_key(item: Tuple[str, Any]) -> Tuple[int, float]:
        _, ds = item
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return (0, float(inst))
            except Exception:
                pass
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            try:
                return (1, float(ipp[2]))
            except Exception:
                pass
        return (2, 0.0)

    group.sort(key=_sort_key)
    sorted_paths = [p for p, _ in group]

    # ── 5. Derive spacing from geometry (DICOM: PixelSpacing = [Y,X]) ─────
    first_ds = group[0][1]

    ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    try:
        sp_y, sp_x = float(ps[0]), float(ps[1])
    except Exception:
        sp_y, sp_x = 1.0, 1.0

    # Prefer actual inter-slice distance from IPP if available; else fall back
    if len(group) > 1:
        ipp0 = getattr(group[0][1], "ImagePositionPatient", None)
        ipp1 = getattr(group[1][1], "ImagePositionPatient", None)
        if ipp0 is not None and ipp1 is not None:
            try:
                sp_z = abs(float(ipp1[2]) - float(ipp0[2]))
            except Exception:
                sp_z = 0.0
        else:
            sp_z = 0.0
    else:
        sp_z = 0.0

    if sp_z < 1e-6:
        # Prefer SpacingBetweenSlices when present; else SliceThickness; else 1.0
        try:
            sp_z = float(getattr(first_ds, "SpacingBetweenSlices", 0.0) or 0.0)
        except Exception:
            sp_z = 0.0
        if sp_z < 1e-6:
            try:
                sp_z = float(getattr(first_ds, "SliceThickness", 1.0) or 1.0)
            except Exception:
                sp_z = 1.0
        if sp_z < 1e-6:
            sp_z = 1.0

    # spacing candidate is already [Z,Y,X] for project convention
    spacing_candidate_zyx = [sp_z, sp_y, sp_x]

    # ── 6. Pre-allocate and fill slice by slice ───────────────────────────
    first_full = pydicom.dcmread(sorted_paths[0])
    first_slice = _apply_rescale(first_full)
    H, W = first_slice.shape[:2]
    D = len(sorted_paths)

    volume = np.zeros((D, H, W), dtype=np.float32)
    volume[0] = first_slice

    for i, path in enumerate(sorted_paths[1:], start=1):
        ds_full = pydicom.dcmread(path)
        sl = _apply_rescale(ds_full)
        if sl.shape[:2] == (H, W):
            volume[i] = sl
        else:
            # Resize inconsistent slice to match first (rare edge case)
            volume[i] = cv2.resize(sl, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 7. Canonicalize spacing to [Z,Y,X] ────────────────────────────────
    canon = _canon_xyz_from_array_and_spacing(volume, spacing_candidate_zyx, spacing_order="zyx")
    spacing = canon["spacing_zyx"]

    modality = str(getattr(first_ds, "Modality", "unknown"))
    meta = _base_meta(volume, "dicom_series", spacing, modality)
    meta["extra_meta"] = {
        "patient_id":          str(getattr(first_ds, "PatientID", "")),
        "study_date":          str(getattr(first_ds, "StudyDate", "")),
        "series_description":  str(getattr(first_ds, "SeriesDescription", "")),
        "series_instance_uid": uid,
        "n_slices":            D,
        "window_center":       str(getattr(first_ds, "WindowCenter", "")),
        "window_width":        str(getattr(first_ds, "WindowWidth", "")),
        # Helpful strict axis mapping for downstream debugging/UI if desired
        "size_x":              canon["sizeX"],
        "size_y":              canon["sizeY"],
        "size_z":              canon["sizeZ"],
        "spacing_x":           canon["spacingX"],
        "spacing_y":           canon["spacingY"],
        "spacing_z":           canon["spacingZ"],
    }
    meta["n_slices"] = D
    return volume, meta


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_standard(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a 2-D standard image (PNG, JPG, BMP) with OpenCV.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not read: {path}")

    if raw.ndim == 3 and raw.shape[2] in (3, 4):
        arr = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB if raw.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    else:
        arr = raw
    arr = arr.astype(np.float32)

    return arr, _base_meta(arr, "png_jpg", [1.0, 1.0], "unknown")


def _load_dicom(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a single DICOM file with pydicom and apply rescale slope/intercept.

    Critical fix:
      Parse spacing as [SliceThickness(or SpacingBetweenSlices), PixelSpacing[0], PixelSpacing[1]]
      == [Z, Y, X] for this project's canonical metadata.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom: pip install pydicom")

    ds = pydicom.dcmread(path)
    arr = _apply_rescale(ds)

    # DICOM PixelSpacing is [row, col] == [Y, X]
    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    try:
        sp_y, sp_x = float(ps[0]), float(ps[1])
    except Exception:
        sp_y, sp_x = 1.0, 1.0

    # Z spacing: prefer SpacingBetweenSlices if present; else SliceThickness
    try:
        sp_z = float(getattr(ds, "SpacingBetweenSlices", 0.0) or 0.0)
    except Exception:
        sp_z = 0.0
    if sp_z < 1e-6:
        try:
            sp_z = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        except Exception:
            sp_z = 1.0
    if sp_z < 1e-6:
        sp_z = 1.0

    spacing_candidate_zyx = [sp_z, sp_y, sp_x]
    canon = _canon_xyz_from_array_and_spacing(arr, spacing_candidate_zyx, spacing_order="zyx")
    spacing = canon["spacing_zyx"]

    meta = _base_meta(arr, "dicom", spacing, str(getattr(ds, "Modality", "unknown")))
    meta["extra_meta"] = {
        "patient_id":         str(getattr(ds, "PatientID", "")),
        "study_date":         str(getattr(ds, "StudyDate", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "window_center":      str(getattr(ds, "WindowCenter", "")),
        "window_width":       str(getattr(ds, "WindowWidth", "")),
        "size_x":             canon["sizeX"],
        "size_y":             canon["sizeY"],
        "size_z":             canon["sizeZ"],
        "spacing_x":          canon["spacingX"],
        "spacing_y":          canon["spacingY"],
        "spacing_z":          canon["spacingZ"],
    }
    return arr, meta


def _apply_rescale(ds) -> np.ndarray:
    """
    Extract pixel data from a pydicom Dataset and apply HU rescaling.
    """
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def _load_nifti(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a NIfTI image (.nii or .nii.gz) with nibabel.

    Critical fix:
      nibabel zooms are typically (X, Y, Z) while we transpose array to (Z, Y, X).
      Therefore we must reorder spacing to match the transposed array before storing
      canonical metadata as [Z, Y, X].
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel: pip install nibabel")

    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)

    zooms = img.header.get_zooms()
    # zooms are usually (x, y, z)
    sp_x, sp_y, sp_z = (list(zooms[:3]) + [1.0, 1.0, 1.0])[:3]

    # nibabel returns (x, y, z); reorder to (z, y, x) = (D, H, W)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))

    # spacing_source is xyz, canonical helper returns spacing_zyx
    canon = _canon_xyz_from_array_and_spacing(arr, [sp_x, sp_y, sp_z], spacing_order="xyz")
    spacing = canon["spacing_zyx"]

    meta = _base_meta(arr, "nifti", spacing, "MRI")
    meta["extra_meta"] = {
        "affine": img.affine.tolist(),
        "size_x": canon["sizeX"],
        "size_y": canon["sizeY"],
        "size_z": canon["sizeZ"],
        "spacing_x": canon["spacingX"],
        "spacing_y": canon["spacingY"],
        "spacing_z": canon["spacingZ"],
    }
    return arr, meta


def _base_meta(arr: np.ndarray, file_type: str, spacing: list, modality: str) -> Dict[str, Any]:
    """Build the standard metadata dict common to all formats."""
    # is_3d: volumetric if ndim==3 AND the last axis > 4 (i.e. not a colour channel dim)
    is_3d = arr.ndim == 3 and arr.shape[-1] > 4
    return {
        "file_type": file_type,
        "shape": list(arr.shape),
        "dtype_str": "float32",
        "ndim": int(arr.ndim),
        "is_3d": bool(is_3d),
        "intensity_min": float(arr.min()),
        "intensity_max": float(arr.max()),
        # Canonical spacing is always [sp_z, sp_y, sp_x] for volumes (Z,Y,X)
        "spacing": spacing,
        "modality": modality,
        "extra_meta": {},
    }


# ── Slice helpers ─────────────────────────────────────────────────────────────

def get_slice_2d(arr: np.ndarray, axis: int, index: Optional[int] = None) -> np.ndarray:
    """
    Extract a 2-D slice from *arr* along *axis* at position *index*.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        # 2-D colour image stored as (H, W, C) — return greyscale projection
        return arr[:, :, 0] if arr.shape[2] == 1 else arr

    # True volumetric (D, H, W)
    dim_size = arr.shape[axis]
    idx = dim_size // 2 if index is None else int(np.clip(index, 0, dim_size - 1))
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


# ── Display helpers ───────────────────────────────────────────────────────────

def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalise *arr* to [0, 255] uint8 for display or encoding.
    """
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def array_to_base64_png(arr: np.ndarray) -> str:
    """
    Encode a 2-D or 3-channel NumPy array as a base64-encoded PNG string.
    """
    u8 = normalise_to_uint8(arr)
    if u8.ndim == 3 and u8.shape[2] == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    elif u8.ndim == 3:
        u8 = u8[:, :, 0]  # collapse to single channel

    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
