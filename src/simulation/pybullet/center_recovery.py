"""Windowed top-surface depth recovery for 2D-to-3D grasp centre correction.

The single-pixel depth backprojection used in Stage 6A hits the visible
surface of the target, which for an oblique camera is often a side face
rather than the top face.  This module replaces the single-pixel sample
with a principled windowed search: within a small patch around the 2-D
grasp centre, the shallowest depth among pixels that belong to the
target segmentation mask is taken as the top-surface depth.  The
original 2-D centre pixel co-ordinates are preserved; only the depth
value is corrected before backprojection.

The algorithm is frozen so that geometry and multi-head CNN backends
share exactly the same recovery rule, making the subsequent physical
execution comparison fair.
"""

from __future__ import annotations

import math

import numpy as np

from src.simulation.pybullet.backprojection import (
    backproject_pixel,
)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

CENTER_RECOVERY_WINDOW_SIZE: int = 5
"""Odd window side-length in pixels for the target-mask-filtered search."""

CENTER_RECOVERY_PROTOCOL: str = "windowed_min_depth_target_mask_v1"
"""Protocol string recorded in every plan that uses this recovery rule."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recover_center_via_windowed_depth(
    center_x: float,
    center_y: float,
    depth_m: np.ndarray,
    segmentation: np.ndarray,
    target_body_id: int,
    width: int,
    height: int,
    view_matrix: tuple[float, ...] | list[float],
    projection_matrix: tuple[float, ...] | list[float],
    near: float,
    far: float,
    *,
    window_size: int = CENTER_RECOVERY_WINDOW_SIZE,
) -> tuple[tuple[int, int], float, tuple[float, float, float]]:
    """Recover a world point by sampling the shallowest target depth in a window.

    The 2-D grasp centre pixel is kept unchanged; only the depth value
    is replaced by the minimum (closest-to-camera) finite depth found
    among neighbouring pixels that belong to *target_body_id* in the
    PyBullet segmentation image.

    Parameters
    ----------
    center_x, center_y:
        Continuous 2-D grasp centre in image co-ordinates (0-indexed).
    depth_m:
        Metric depth image (H×W float array).
    segmentation:
        PyBullet segmentation image (H×W int array).  Body ids are
        extracted via ``value & ((1 << 24) - 1)`` when the raw value is
        non-negative.
    target_body_id:
        Expected PyBullet body unique id of the target object.
    width, height:
        Image dimensions in pixels.
    view_matrix, projection_matrix:
        PyBullet view and projection matrices (16 floats each).
    near, far:
        Camera clipping planes in metres.
    window_size:
        Odd integer ≥ 1; side-length of the square search window.

    Returns
    -------
    (sampled_pixel, corrected_depth_m, corrected_world_xyz)
        *sampled_pixel* is the (column, row) of the pixel whose depth was
        selected.  *corrected_depth_m* is that depth in metres.
        *corrected_world_xyz* is the world co-ordinate obtained by
        backprojecting the original grasp-centre pixel with the corrected
        depth.

    Raises
    ------
    ValueError
        If no valid target pixel is found inside the window, or if the
        window size is not a positive odd integer.
    """
    # -- validation ---------------------------------------------------------
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise ValueError("grasp centre must be finite")
    if not (0.0 <= center_x <= width - 1 and 0.0 <= center_y <= height - 1):
        raise ValueError("grasp centre must be inside image")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    if target_body_id < 0:
        raise ValueError("target_body_id must be non-negative")

    depth = np.asarray(depth_m)
    seg = np.asarray(segmentation)
    if depth.ndim != 2 or depth.shape != (height, width):
        raise ValueError("depth_m must be a (height, width) float array")
    if seg.ndim != 2 or seg.shape != (height, width):
        raise ValueError("segmentation must be a (height, width) int array")

    # -- nearest pixel for the original grasp centre ------------------------
    centre_col = math.floor(center_x + 0.5)
    centre_row = math.floor(center_y + 0.5)

    half = window_size // 2

    # -- search window ------------------------------------------------------
    best_pixel: tuple[int, int] | None = None
    best_depth: float | None = None

    # Tolerance for clipping-plane comparison (matching backprojection.py)
    clip_tolerance = float(np.finfo(np.float32).eps) * max(1.0, abs(near), abs(far))

    for dr in range(-half, half + 1):
        row = centre_row + dr
        if not 0 <= row < height:
            continue
        for dc in range(-half, half + 1):
            col = centre_col + dc
            if not 0 <= col < width:
                continue

            raw_seg = int(seg[row, col])
            if raw_seg < 0:
                continue
            seg_id = raw_seg & ((1 << 24) - 1)
            if seg_id != target_body_id:
                continue

            d = float(depth[row, col])
            if not math.isfinite(d):
                continue
            if d <= near + clip_tolerance or d >= far - clip_tolerance:
                continue

            if best_depth is None or d < best_depth:
                best_depth = d
                best_pixel = (col, row)

    if best_pixel is None or best_depth is None:
        raise ValueError(
            f"no valid target pixel (body {target_body_id}) found in "
            f"{window_size}x{window_size} window around ({centre_col}, {centre_row})"
        )

    # -- backproject with the original centre pixel and corrected depth -----
    point = backproject_pixel(
        centre_col,
        centre_row,
        best_depth,
        width,
        height,
        view_matrix,
        projection_matrix,
        near,
        far,
    )

    corrected_world: tuple[float, float, float] = point.world_xyz
    return best_pixel, best_depth, corrected_world
