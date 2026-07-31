"""Depth backprojection for the PyBullet perception study.

Matrix and depth conventions follow the public PyBullet quickstart guide:
https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
The audit design and implementation are project-specific.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DepthSample:
    """One nearest-pixel metric-depth sample."""

    column: int
    row: int
    depth_m: float


@dataclass(frozen=True)
class BackprojectedPoint:
    """One point represented in camera and world coordinates."""

    camera_xyz: tuple[float, float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ReprojectedPoint:
    """A world point projected back into the camera image."""

    pixel_x: float
    pixel_y: float
    depth_m: float


def sample_nearest_depth(
    depth_m: np.ndarray,
    center_x: float,
    center_y: float,
    near: float,
    far: float,
) -> DepthSample:
    """Sample metric depth with deterministic half-up pixel rounding."""

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise ValueError("grasp centre must be finite")
    height, width = depth.shape
    if not (
        0.0 <= center_x <= width - 1
        and 0.0 <= center_y <= height - 1
    ):
        raise ValueError("grasp centre must be inside image")

    column = math.floor(center_x + 0.5)
    row = math.floor(center_y + 0.5)
    value = float(depth[row, column])
    clip_tolerance = float(np.finfo(np.float32).eps) * max(
        1.0,
        abs(near),
        abs(far),
    )
    if (
        not math.isfinite(value)
        or value <= near + clip_tolerance
        or value >= far - clip_tolerance
    ):
        raise ValueError(
            "sampled depth must be finite and inside clip planes"
        )
    return DepthSample(column=column, row=row, depth_m=value)


def metric_depth_to_buffer(
    depth_m: float,
    near: float,
    far: float,
) -> float:
    """Invert PyBullet's perspective depth linearization formula."""

    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    if not math.isfinite(depth_m) or not near <= depth_m <= far:
        raise ValueError("depth must be finite and inside clip planes")
    return float((far - far * near / depth_m) / (far - near))


def _matrix4(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 16:
        raise ValueError(f"{name} matrix must contain 16 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} matrix must contain finite values")
    return array.reshape((4, 4), order="F")


def _invert(matrix: np.ndarray, name: str) -> np.ndarray:
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} matrix must be invertible") from exc
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"inverse {name} matrix must be finite")
    return inverse


def _divide_homogeneous(
    point: np.ndarray,
    name: str,
) -> tuple[float, float, float]:
    if point.shape != (4,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} homogeneous point must be finite")
    if math.isclose(float(point[3]), 0.0, abs_tol=1e-12):
        raise ValueError(f"{name} homogeneous W must be non-zero")
    xyz = point[:3] / point[3]
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} Cartesian point must be finite")
    return tuple(float(value) for value in xyz)


def backproject_pixel(
    column: int,
    row: int,
    depth_m: float,
    width: int,
    height: int,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
    near: float,
    far: float,
) -> BackprojectedPoint:
    """Recover camera and world coordinates for one sampled pixel."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= column < width or not 0 <= row < height:
        raise ValueError("sampled pixel must be inside image")
    view = _matrix4(view_matrix, "view")
    projection = _matrix4(projection_matrix, "projection")
    depth_buffer = metric_depth_to_buffer(depth_m, near, far)
    clip = np.array(
        [
            2.0 * (column + 0.5) / width - 1.0,
            1.0 - 2.0 * (row + 0.5) / height,
            2.0 * depth_buffer - 1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    camera_xyz = _divide_homogeneous(
        _invert(projection, "projection") @ clip,
        "camera",
    )
    world_xyz = _divide_homogeneous(
        _invert(projection @ view, "projection-view") @ clip,
        "world",
    )
    return BackprojectedPoint(
        camera_xyz=camera_xyz,
        world_xyz=world_xyz,
    )


def reproject_world_point(
    world_xyz: Sequence[float],
    width: int,
    height: int,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
) -> ReprojectedPoint:
    """Project one world point back to image and metric camera depth."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    world = np.asarray(world_xyz, dtype=np.float64)
    if world.shape != (3,) or not np.all(np.isfinite(world)):
        raise ValueError("world point must contain three finite values")
    view = _matrix4(view_matrix, "view")
    projection = _matrix4(projection_matrix, "projection")
    world_h = np.array([*world, 1.0], dtype=np.float64)
    camera_h = view @ world_h
    camera_xyz = _divide_homogeneous(camera_h, "camera")
    clip = projection @ camera_h
    ndc = _divide_homogeneous(clip, "clip")
    pixel_x = (ndc[0] + 1.0) * width / 2.0 - 0.5
    pixel_y = (1.0 - ndc[1]) * height / 2.0 - 0.5
    depth_m = -camera_xyz[2]
    if not math.isfinite(depth_m):
        raise ValueError("reprojected depth must be finite")
    return ReprojectedPoint(
        pixel_x=float(pixel_x),
        pixel_y=float(pixel_y),
        depth_m=float(depth_m),
    )
