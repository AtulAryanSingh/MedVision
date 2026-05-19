import os
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from analysis.report import generate_report
from api import find_uploaded_file, load_metadata
from api.cluster import ClusterRequest
from api.patchify import PatchifyRequest
from api.register import RegisterRequest
from core.loader import array_to_base64_png, async_load_image, get_slice_2d, normalise_to_uint8
from features.extractor import extract_features
from jobs.orchestrator import create_job, get_job, get_result, request_cancel, start_job
from ml.clustering import run_kmeans
from ml.reduction import run_pca
from processing.histogram import compute_histogram

router = APIRouter()


def _ok(message: str, data: dict[str, Any], status_code: int = 200):
    return JSONResponse({"status": "ok", "message": message, "data": data}, status_code=status_code)


def _exports_dir() -> str:
    from api import get_cache_dir

    path = os.path.join(get_cache_dir(), "exports")
    os.makedirs(path, exist_ok=True)
    return path


async def _patchify_worker(
    req: PatchifyRequest,
    job_id: str,
    update_progress: Callable[[int, str], None],
    is_canceled: Callable[[], bool],
):
    meta = load_metadata(req.image_id)
    if not meta.get("is_3d", False):
        raise ValueError("Patchify requires a 3-D volume.")

    path = find_uploaded_file(req.image_id)
    arr, _ = await async_load_image(path)

    if arr.ndim != 3:
        raise ValueError("Volume must be exactly 3-D (D×H×W).")

    D, H, W = arr.shape
    ps = req.patch_size
    st = req.stride

    patches = []
    coords = []

    total_z = max(1, (D - ps + st) // st)
    for zi, z in enumerate(range(0, D - ps + 1, st)):
        if is_canceled():
            return {"canceled": True}
        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                patches.append(arr[z : z + ps, y : y + ps, x : x + ps])
                coords.append([z, y, x])
        update_progress(int((zi + 1) / total_z * 85), "Extracting patches")

    if not patches:
        raise ValueError(
            f"Volume {D}x{H}x{W} is too small for patch size {ps}. "
            f"All dimensions must be ≥ {ps} voxels."
        )

    patches_arr = np.stack(patches, axis=0).astype(np.float32)
    coords_arr = np.array(coords, dtype=np.int32)
    spacing = np.array(meta.get("spacing", [1.0, 1.0, 1.0]), dtype=np.float32)

    npz_path = os.path.join(_exports_dir(), f"patches_{job_id}.npz")
    np.savez_compressed(npz_path, patches=patches_arr, coords=coords_arr, spacing=spacing)
    update_progress(95, "Persisting result")

    return {
        "image_id": req.image_id,
        "volume_shape": [D, H, W],
        "patch_size": ps,
        "stride": st,
        "n_patches": len(patches),
        "patches_shape": list(patches_arr.shape),
        "spacing_mm": meta.get("spacing", [1.0, 1.0, 1.0]),
        "npz_path": npz_path,
    }


async def _cluster_worker(
    req: ClusterRequest,
    _job_id: str,
    update_progress: Callable[[int, str], None],
    is_canceled: Callable[[], bool],
):
    load_metadata(req.image_id)
    path = find_uploaded_file(req.image_id)
    arr, _ = await async_load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    update_progress(20, "Running KMeans")
    km_result = run_kmeans(arr, k=req.k)

    if is_canceled():
        return {"canceled": True}

    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    pixels = grey.flatten().reshape(-1, 1).astype(np.float32)

    from sklearn.cluster import KMeans

    km_for_labels = KMeans(n_clusters=req.k, random_state=42, n_init="auto")
    all_labels = km_for_labels.fit_predict(pixels)
    order = np.argsort(km_for_labels.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    all_labels = np.array([remap[label] for label in all_labels])

    update_progress(70, "Running PCA")
    pca_result = run_pca(arr, n_components=2, n_samples=req.n_samples, k_labels=all_labels)

    return {
        "image_id": req.image_id,
        "k": km_result["k"],
        "segmented_image": "data:image/png;base64," + array_to_base64_png(km_result["segmented_image"]),
        "centers": km_result["centers"],
        "cluster_counts": km_result["cluster_counts"],
        "pca": {
            "points": pca_result["points"],
            "explained_variance": pca_result["explained_variance"],
            "n_samples_used": pca_result["n_samples_used"],
        },
    }


async def _report_worker(
    image_id: str,
    _job_id: str,
    update_progress: Callable[[int, str], None],
    is_canceled: Callable[[], bool],
):
    meta = load_metadata(image_id)
    features = meta.get("features")

    if features is None:
        path = find_uploaded_file(image_id)
        arr, _ = await async_load_image(path)
        if arr.ndim == 3 and arr.shape[0] > 4:
            arr = get_slice_2d(arr, axis=0)
        update_progress(40, "Extracting features")
        features = extract_features(arr)

    if is_canceled():
        return {"canceled": True}

    cluster_info = meta.get("cluster")
    last_processing = meta.get("last_processing")
    processing_history = [last_processing] if last_processing else []

    update_progress(80, "Assembling report")
    report = generate_report(
        image_id=image_id,
        metadata=meta,
        features=features,
        cluster_info=cluster_info,
        processing_history=processing_history,
    )
    return report


async def _register_worker(
    req: RegisterRequest,
    _job_id: str,
    update_progress: Callable[[int, str], None],
    _is_canceled: Callable[[], bool],
):
    from scipy.ndimage import affine_transform, rotate, zoom

    load_metadata(req.image_id)
    path = find_uploaded_file(req.image_id)
    arr, _ = await async_load_image(path)

    if arr.ndim == 3 and arr.shape[0] > 4:
        arr = get_slice_2d(arr, axis=0)

    u8 = normalise_to_uint8(arr)
    if u8.ndim == 3:
        grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        grey = arr.astype(np.float32)

    order = 0 if req.is_mask else 3
    ttype = req.transform_type

    update_progress(25, "Applying transform")
    if ttype == "translate":
        result = affine_transform(
            grey,
            matrix=np.eye(2),
            offset=[-req.shift_y, -req.shift_x],
            order=order,
            mode="constant",
            cval=0.0,
        )
    elif ttype == "rotate":
        result = rotate(grey, angle=req.angle_deg, reshape=False, order=order, mode="constant", cval=0.0)
    elif ttype == "zoom":
        zoomed = zoom(grey, zoom=(req.zoom_y, req.zoom_x), order=order)
        h, w = grey.shape
        zh, zw = zoomed.shape
        result = np.zeros_like(grey)
        cy, cx = min(h, zh), min(w, zw)
        result[:cy, :cx] = zoomed[:cy, :cx]
    elif ttype == "affine":
        if req.matrix is None or len(req.matrix) != 4:
            raise ValueError("For 'affine' transform_type, supply 'matrix' as 4 floats.")
        mat = np.array(req.matrix).reshape(2, 2)
        result = affine_transform(
            grey,
            matrix=mat,
            offset=[req.ty, req.tx],
            order=order,
            mode="constant",
            cval=0.0,
        )
    else:
        raise ValueError("Unknown transform_type.")

    result = result.astype(np.float32)

    update_progress(90, "Finalizing output")
    return {
        "image_id": req.image_id,
        "transform_type": ttype,
        "result_image": "data:image/png;base64," + array_to_base64_png(result),
        "histogram_before": compute_histogram(grey, bins=64),
        "histogram_after": compute_histogram(result, bins=64),
        "output_shape": list(result.shape),
    }


@router.post("/jobs/patchify", summary="Create async patchify job")
async def create_patchify_job(req: PatchifyRequest):
    job = create_job("patchify", req.model_dump())
    start_job(job["job_id"], lambda p, c: _patchify_worker(req, job["job_id"], p, c))
    return _ok("Patchify job created", get_job(job["job_id"]), status_code=202)


@router.post("/jobs/cluster", summary="Create async clustering job")
async def create_cluster_job(req: ClusterRequest):
    job = create_job("cluster", req.model_dump())
    start_job(job["job_id"], lambda p, c: _cluster_worker(req, job["job_id"], p, c))
    return _ok("Cluster job created", get_job(job["job_id"]), status_code=202)


@router.post("/jobs/report/{image_id}", summary="Create async report generation job")
async def create_report_job(image_id: str):
    job = create_job("report", {"image_id": image_id})
    start_job(job["job_id"], lambda p, c: _report_worker(image_id, job["job_id"], p, c))
    return _ok("Report job created", get_job(job["job_id"]), status_code=202)


@router.post("/jobs/register", summary="Create async registration job")
async def create_register_job(req: RegisterRequest):
    job = create_job("register", req.model_dump())
    start_job(job["job_id"], lambda p, c: _register_worker(req, job["job_id"], p, c))
    return _ok("Registration job created", get_job(job["job_id"]), status_code=202)


@router.get("/jobs/{job_id}", summary="Get async job status")
async def get_job_status(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _ok("Job status", job)


@router.post("/jobs/{job_id}/cancel", summary="Cancel async job")
async def cancel_job(job_id: str):
    try:
        job = request_cancel(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _ok("Cancel signal accepted", job)


@router.get("/jobs/{job_id}/result", summary="Get async job result")
async def get_job_output(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job is not complete (status={job['status']}).")

    return _ok("Job result", get_result(job_id))


@router.get("/jobs/{job_id}/result/stream", summary="Stream async job result artifact when available")
async def stream_job_result(job_id: str):
    try:
        result = get_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    npz_path = result.get("npz_path")
    if not npz_path:
        raise HTTPException(status_code=404, detail="No streamable artifact for this job.")
    if not os.path.isfile(npz_path):
        raise HTTPException(status_code=404, detail="Result artifact is missing.")

    return FileResponse(
        path=npz_path,
        media_type="application/octet-stream",
        filename=os.path.basename(npz_path),
    )
