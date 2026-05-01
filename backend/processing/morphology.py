"""
processing/morphology.py

What this module does:
  Morphological image processing operations and connected-component
  analysis:
    apply_erosion              – shrink bright foreground regions
    apply_dilation             – expand bright foreground regions
    apply_opening              – remove small bright noise blobs
    apply_closing              – fill small dark holes in bright regions
    label_connected_components – binarise then label each distinct region
    draw_bounding_boxes        – draw coloured rectangles around each region
    compute_center_of_mass     – find the intensity-weighted centroid

Why it exists:
  Morphological operations are essential preprocessing steps in medical
  image segmentation for cleaning binary masks and isolating ROIs.

Dependencies: OpenCV, NumPy, SciPy
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage


# ── Morphological structuring element ────────────────────────────────────────

def _kernel(size: int) -> np.ndarray:
    """Return a square uint8 kernel of side length *size*."""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1  # OpenCV requires odd kernel size
    return np.ones((size, size), dtype=np.uint8)


# ── Basic morphological operations ───────────────────────────────────────────

def apply_erosion(arr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Erode *arr*: shrink bright (foreground) regions.

    What it does:
      Applies a min-filter with a square structuring element.  Each output
      pixel is the minimum of its neighbourhood, effectively removing thin
      protrusions and small bright blobs.

    Why it exists:
      Erosion is used to eliminate small spurious high-intensity regions and
      to separate touching structures before connected-component analysis.
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    result = cv2.erode(grey, _kernel(kernel_size))
    return result.astype(np.float32)


def apply_dilation(arr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Dilate *arr*: expand bright (foreground) regions.

    What it does:
      Applies a max-filter with a square structuring element.  Each output
      pixel is the maximum of its neighbourhood.

    Why it exists:
      Dilation fills small gaps in segmentation masks and connects nearby
      disconnected regions.
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    result = cv2.dilate(grey, _kernel(kernel_size))
    return result.astype(np.float32)


def apply_opening(arr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).

    What it does:
      Removes small bright structures that are smaller than the kernel while
      preserving the overall shape of larger objects.

    Why it exists:
      Opening is the standard denoising operation for binary masks — it
      eliminates isolated foreground pixels without shrinking large regions.
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    k = _kernel(kernel_size)
    result = cv2.morphologyEx(grey, cv2.MORPH_OPEN, k)
    return result.astype(np.float32)


def apply_closing(arr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).

    What it does:
      Fills small dark holes inside bright regions without enlarging the
      external boundary significantly.

    Why it exists:
      Closing is used to consolidate incomplete segmentation masks where
      the foreground has small internal gaps.
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    k = _kernel(kernel_size)
    result = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, k)
    return result.astype(np.float32)


# ── Connected components ──────────────────────────────────────────────────────

def label_connected_components(
    arr: np.ndarray,
    threshold: float = 128.0,
    min_area: int = 50,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Binarise *arr* at *threshold* and label each connected component.

    What it does:
      1. Converts to greyscale and binarises at *threshold*.
      2. Uses OpenCV connectedComponentsWithStats to label each region.
      3. Filters out components smaller than *min_area* pixels.
      4. Returns a colour-coded label image and a list of component stats.

    Why it exists:
      Connected-component labelling is the first step in object counting,
      region-of-interest (ROI) extraction, and lesion characterisation.

    Returns
    -------
    label_image : np.ndarray
        uint8 colour image (H × W × 3) with each component in a distinct
        colour.
    components : list of dict
        Each entry has: label, area, bounding_box (x, y, w, h),
        center (cx, cy).
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8

    # Binarise
    _, binary = cv2.threshold(grey, int(threshold), 255, cv2.THRESH_BINARY)

    # Label components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Colour map: use HSV hues spread evenly across components
    h, w = grey.shape
    colour_image = np.zeros((h, w, 3), dtype=np.uint8)
    components: List[Dict[str, Any]] = []

    for i in range(1, n):  # skip background (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Assign a unique hue
        hue = int((i / max(n - 1, 1)) * 179)
        hsv_colour = np.array([[[hue, 255, 220]]], dtype=np.uint8)
        bgr_colour = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0, 0]
        colour_image[labels == i] = bgr_colour

        components.append({
            "label": int(i),
            "area": area,
            "bounding_box": {
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            },
            "center": {"cx": float(centroids[i, 0]), "cy": float(centroids[i, 1])},
        })

    return colour_image, components


def draw_bounding_boxes(arr: np.ndarray, threshold: float = 128.0) -> np.ndarray:
    """
    Draw bounding boxes around all connected components in *arr*.

    What it does:
      Calls label_connected_components, then draws a rectangle for each
      component on top of the (greyscale-converted) original image.

    Why it exists:
      Visualising bounding boxes is a quick sanity-check for whether the
      segmentation is picking up the correct structures.
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    canvas = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    _, components = label_connected_components(arr, threshold=threshold)
    for comp in components:
        bb = comp["bounding_box"]
        cv2.rectangle(
            canvas,
            (bb["x"], bb["y"]),
            (bb["x"] + bb["w"], bb["y"] + bb["h"]),
            (0, 255, 80),  # bright green
            2,
        )
        cx, cy = int(comp["center"]["cx"]), int(comp["center"]["cy"])
        cv2.circle(canvas, (cx, cy), 4, (0, 200, 255), -1)

    return canvas.astype(np.float32)


# ── Centre of mass ────────────────────────────────────────────────────────────

def compute_center_of_mass(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute the intensity-weighted centroid of the 2-D array.

    What it does:
      Uses scipy.ndimage.center_of_mass to find the (row, col) position
      whose surrounding pixel intensities are balanced.

    Why it exists:
      The centre of mass gives a single representative coordinate for the
      dominant bright region, useful for lesion localisation and alignment.

    Returns
    -------
    dict with keys "row" and "col" (float coordinates).
    """
    from core.loader import normalise_to_uint8
    u8 = normalise_to_uint8(arr)
    grey = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    row, col = ndimage.center_of_mass(grey.astype(np.float64))
    return {"row": float(row), "col": float(col)}
