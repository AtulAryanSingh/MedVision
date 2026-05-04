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
# This keeps FOV/MPR logic consistent everywhere.

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
        - "xyz": spacing is [sp_x, sp_y, sp_z]  (common in NIfTI headers)

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
        sizeZ, sizeY, sizeX = 1, int(arr.shape[0]), int(arr.shape[1])

    # Sanity
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


# ── DICOM geometry helpers ────────────────────────────────────────────────────

def _try_float_list(x: Any, n: int) -> Optional[List[float]]:
    """Try to parse a DICOM multi-value into a list[float] of length >= n."""
    if x is None:
        return None
    try:
        xs = [float(v) for v in x]
        return xs if len(xs) >= n else None
    except Exception:
        return None


def _dicom_slice_position_scalar(ds) -> Tuple[Optional[float], bool]:
    """
    Return a scalar that increases along the slice axis.

    If ImageOrientationPatient is present, use dot(IPP, normal) where normal is
    cross(row_cosines, col_cosines). This is robust for oblique series.

    If IOP is missing/unusable, fall back to IPP[2] (assumes axial-ish data).

    Returns
    -------
    (pos_scalar, has_geometry)
      - has_geometry=True means we used normal projection
    """
    ipp = _try_float_list(getattr(ds, "ImagePositionPatient", None), 3)
    if ipp is None:
        return None, False

    iop = _try_float_list(getattr(ds, "ImageOrientationPatient", None), 6)
    if iop is None:
        return float(ipp[2]), False

    row = np.array(iop[:3], dtype=np.float64)
    col = np.array(iop[3:6], dtype=np.float64)
    nrm = np.cross(row, col)
    nrm_norm = float(np.linalg.norm(nrm))
    if nrm_norm < 1e-8:
        return float(ipp[2]), False

    nrm = nrm / nrm_norm
    return float(np.dot(np.array(ipp, dtype=np.float64), nrm)), True


def _dicom_get_pixel_spacing_yx(ds) -> Tuple[float, float]:
    """
    DICOM PixelSpacing is [row_spacing, col_spacing] == [Y, X] in mm.
    """
    ps = getattr(ds, "PixelSpacing", None)
    try:
        if ps is not None and len(ps) >= 2:
            return float(ps[0]), float(ps[1])
    except Exception:
        pass
    return 1.0, 1.0


def _dicom_get_spacing_z(ds) -> float:
    """
    Get Z spacing for single-slice DICOM where inter-slice spacing can't be derived.
    Preference:
      SpacingBetweenSlices > SliceThickness > 1.0
    """
    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        try:
            v = float(getattr(ds, tag, 0.0) or 0.0)
            if v > 1e-6:
                return v
        except Exception:
            pass
    return 1.0


# ── Public entry point ────────────────────────────────────────────────────────

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a medical image from *path* and return (array, metadata).
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

    Key robustness:
      - Uses ImageOrientationPatient + ImagePositionPatient to compute the true slice
        normal and sort slices by dot(IPP, normal) when possible (oblique-safe).
      - Derives sp_z from the robust median of consecutive slice position deltas.
      - Falls back to tags (SpacingBetweenSlices / SliceThickness) if geometry is absent.
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

    # ── 4. Sort slices (prefer geometry) ──────────────────────────────────
    pos_items: List[Tuple[str, Any, Optional[float], bool, Any]] = []
    for p, ds in group:
        inst = getattr(ds, "InstanceNumber", None)
        pos, has_geom = _dicom_slice_position_scalar(ds)
        pos_items.append((p, ds, pos, has_geom, inst))

    geom_count = sum(1 for _, _, pos, has_geom, _ in pos_items if pos is not None and has_geom)
    use_geom = geom_count >= max(2, int(0.7 * len(pos_items)))

    if use_geom:
        # Sort by projected position along slice normal; stable tie-breaker by SOPInstanceUID
        pos_items.sort(key=lambda t: (
            t[2] is None,
            t[2],
            str(getattr(t[1], "SOPInstanceUID", "")),
        ))
    else:
        # Fallback: InstanceNumber primary; then position scalar (which may be IPP[2]); then stable UID
        def _fallback_key(t):
            _, ds, pos, _, inst = t
            if inst is not None:
                try:
                    return (0, float(inst))
                except Exception:
                    pass
            if pos is not None:
                return (1, float(pos))
            return (2, 0.0)

        pos_items.sort(key=_fallback_key)

    sorted_paths = [p for (p, _, _, _, _) in pos_items]
    sorted_pos = [pos for (_, _, pos, _, _) in pos_items if pos is not None]

    # ── 5. Derive spacing (Y,X from PixelSpacing; Z from geometry or tags) ─
    first_ds = pos_items[0][1]
    sp_y, sp_x = _dicom_get_pixel_spacing_yx(first_ds)

    sp_z = 0.0
    if len(sorted_pos) >= 2:
        diffs = np.diff(np.array(sorted_pos, dtype=np.float64))
        diffs = np.abs(diffs)
        diffs = diffs[diffs > 1e-6]  # remove duplicates/zeros
        if diffs.size > 0:
            sp_z = float(np.median(diffs))

    if sp_z <= 1e-6:
        sp_z = _dicom_get_spacing_z(first_ds)

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
            # NOTE: resizing slices changes physical geometry; consider skipping/erroring in strict mode.
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
    }
    meta["n_slices"] = D
    return volume, meta


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_standard(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a 2-D standard image (PNG, JPG, BMP) with OpenCV.

    For 2-D, we provide a safe canonical spacing_zyx = [1,1,1] so downstream
    logic that expects len(spacing)>=3 never breaks.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not read: {path}")

    if raw.ndim == 3 and raw.shape[2] in (3, 4):
        arr = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB if raw.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    else:
        arr = raw
    arr = arr.astype(np.float32)

    spacing_candidate_zyx = [1.0, 1.0, 1.0]
    canon = _canon_xyz_from_array_and_spacing(arr, spacing_candidate_zyx, spacing_order="zyx")
    spacing = canon["spacing_zyx"]

    return arr, _base_meta(arr, "png_jpg", spacing, "unknown")


def _load_dicom(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a single DICOM file with pydicom and apply rescale slope/intercept.

    Critical axis mapping:
      - PixelSpacing -> (Y,X)
      - SliceThickness/SpacingBetweenSlices -> Z
      - Store metadata spacing canonically as [Z,Y,X]
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom: pip install pydicom")

    ds = pydicom.dcmread(path)
    arr = _apply_rescale(ds)

    sp_y, sp_x = _dicom_get_pixel_spacing_yx(ds)
    sp_z = _dicom_get_spacing_z(ds)

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

    Canonical output in this project:
      - array order: (Z, Y, X)
      - spacing metadata: [sp_z, sp_y, sp_x]

    Robust spacing:
      - Prefer affine-derived voxel sizes (handles rotation/shear/sign)
      - Fall back to header zooms if affine is missing/weird
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel: pip install nibabel")

    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)

    # --- 1) get source voxel sizes in NIfTI's (X,Y,Z) axis order ---
    sp_x = sp_y = sp_z = 1.0
    try:
        A = np.array(img.affine, dtype=np.float64)
        # voxel sizes are norms of the first 3 columns (ignoring translation row)
        sp_x = float(np.linalg.norm(A[:3, 0]))
        sp_y = float(np.linalg.norm(A[:3, 1]))
        sp_z = float(np.linalg.norm(A[:3, 2]))
    except Exception:
        zooms = list(img.header.get_zooms())
        sp_x = float(zooms[0]) if len(zooms) > 0 else 1.0
        sp_y = float(zooms[1]) if len(zooms) > 1 else 1.0
        sp_z = float(zooms[2]) if len(zooms) > 2 else 1.0

    # sanitize
    if not np.isfinite(sp_x) or sp_x <= 1e-6: sp_x = 1.0
    if not np.isfinite(sp_y) or sp_y <= 1e-6: sp_y = 1.0
    if not np.isfinite(sp_z) or sp_z <= 1e-6: sp_z = 1.0

    # --- 2) reorder array to project convention (Z,Y,X) ---
    # nibabel arrays are typically (X,Y,Z)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))  # (Z, Y, X)

    # --- 3) store spacing canonically as [Z,Y,X] ---
    spacing_candidate_zyx = [sp_z, sp_y, sp_x]
    canon = _canon_xyz_from_array_and_spacing(arr, spacing_candidate_zyx, spacing_order="zyx")
    spacing = canon["spacing_zyx"]

    meta = _base_meta(arr, "nifti", spacing, "MRI")
    meta["extra_meta"] = {"affine": img.affine.tolist()}
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

    axis: 0=axial (Z), 1=coronal (Y), 2=sagittal (X) for (Z,Y,X) volumes.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        # 2-D colour image stored as (H, W, C) — return greyscale projection
        return arr[:, :, 0] if arr.shape[2] == 1 else arr

    # True volumetric (Z, Y, X)
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
