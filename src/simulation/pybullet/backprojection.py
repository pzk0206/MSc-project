"""Depth backprojection for the PyBullet perception study.

Matrix and depth conventions follow the public PyBullet quickstart guide:
https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
The audit design and implementation are project-specific.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from src.simulation.pybullet.backend_comparison import (
    EXPECTED_TARGET_BACKENDS,
)


PIXEL_ERROR_THRESHOLD = 1.0
DEPTH_ERROR_THRESHOLD_M = 1e-4
RayTest = Callable[
    [
        tuple[float, float, float],
        tuple[float, float, float],
        int,
    ],
    tuple[int, tuple[float, float, float] | None],
]


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


@dataclass(frozen=True)
class BackprojectionAudit:
    """Post-hoc truth audit for one backprojected grasp centre."""

    target: str
    backend: str
    center_x: float
    center_y: float
    sampled_column: int | None
    sampled_row: int | None
    depth_m: float | None
    camera_x: float | None
    camera_y: float | None
    camera_z: float | None
    world_x: float | None
    world_y: float | None
    world_z: float | None
    reprojected_x: float | None
    reprojected_y: float | None
    reprojected_depth_m: float | None
    pixel_error: float | None
    depth_error_m: float | None
    coordinates_finite: bool
    valid_depth: bool
    reprojection_passed: bool
    segmentation_body_id: int | None
    expected_body_id: int
    segmentation_target_match: bool
    ray_body_id: int | None
    ray_target_match: bool
    ray_hit_position: tuple[float, float, float] | None
    ray_hit_distance_m: float | None
    gate_passed: bool
    failure_reason: str


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


def build_ray_segment(
    camera_eye: Sequence[float],
    world_xyz: Sequence[float],
    extension_m: float = 0.05,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build a ray ending slightly beyond a reconstructed surface point."""

    eye = np.asarray(camera_eye, dtype=np.float64)
    point = np.asarray(world_xyz, dtype=np.float64)
    if (
        eye.shape != (3,)
        or point.shape != (3,)
        or not np.all(np.isfinite(eye))
        or not np.all(np.isfinite(point))
    ):
        raise ValueError("camera eye and world point must contain finite xyz")
    if not math.isfinite(extension_m) or extension_m <= 0.0:
        raise ValueError("ray extension must be finite and positive")
    direction = point - eye
    distance = float(np.linalg.norm(direction))
    if not math.isfinite(distance) or distance <= 1e-12:
        raise ValueError("camera eye and world point must be distinct")
    ray_end = point + direction / distance * extension_m
    return (
        tuple(float(value) for value in eye),
        tuple(float(value) for value in ray_end),
    )


def _failed_audit(
    target: str,
    backend: str,
    center_x: float,
    center_y: float,
    expected_body_id: int,
    failure_reason: str,
) -> BackprojectionAudit:
    return BackprojectionAudit(
        target=target,
        backend=backend,
        center_x=center_x,
        center_y=center_y,
        sampled_column=None,
        sampled_row=None,
        depth_m=None,
        camera_x=None,
        camera_y=None,
        camera_z=None,
        world_x=None,
        world_y=None,
        world_z=None,
        reprojected_x=None,
        reprojected_y=None,
        reprojected_depth_m=None,
        pixel_error=None,
        depth_error_m=None,
        coordinates_finite=False,
        valid_depth=False,
        reprojection_passed=False,
        segmentation_body_id=None,
        expected_body_id=expected_body_id,
        segmentation_target_match=False,
        ray_body_id=None,
        ray_target_match=False,
        ray_hit_position=None,
        ray_hit_distance_m=None,
        gate_passed=False,
        failure_reason=failure_reason,
    )


def audit_backprojected_grasp(
    backend_row: Mapping[str, object],
    depth_m: np.ndarray,
    segmentation: np.ndarray,
    expected_body_id: int,
    camera_eye: Sequence[float],
    image_width: int,
    image_height: int,
    near: float,
    far: float,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
    client_id: int,
    ray_test: RayTest | None,
) -> BackprojectionAudit:
    """Backproject one saved 2-D centre, then run truth-only audits."""

    target = str(backend_row.get("target", ""))
    backend = str(backend_row.get("backend", ""))
    try:
        center_x = float(backend_row["center_x"])
        center_y = float(backend_row["center_y"])
    except (KeyError, TypeError, ValueError) as exc:
        return _failed_audit(
            target,
            backend,
            math.nan,
            math.nan,
            expected_body_id,
            f"invalid_grasp_center:{exc}",
        )

    try:
        sample = sample_nearest_depth(
            depth_m,
            center_x,
            center_y,
            near,
            far,
        )
        point = backproject_pixel(
            sample.column,
            sample.row,
            sample.depth_m,
            image_width,
            image_height,
            view_matrix,
            projection_matrix,
            near,
            far,
        )
        reprojection = reproject_world_point(
            point.world_xyz,
            image_width,
            image_height,
            view_matrix,
            projection_matrix,
        )
    except ValueError as exc:
        return _failed_audit(
            target,
            backend,
            center_x,
            center_y,
            expected_body_id,
            f"backprojection_error:{exc}",
        )

    numeric_coordinates = (
        *point.camera_xyz,
        *point.world_xyz,
        reprojection.pixel_x,
        reprojection.pixel_y,
        reprojection.depth_m,
    )
    coordinates_finite = all(
        math.isfinite(value) for value in numeric_coordinates
    )
    pixel_error = math.hypot(
        reprojection.pixel_x - sample.column,
        reprojection.pixel_y - sample.row,
    )
    depth_error_m = abs(reprojection.depth_m - sample.depth_m)
    reprojection_passed = (
        coordinates_finite
        and pixel_error <= PIXEL_ERROR_THRESHOLD
        and depth_error_m <= DEPTH_ERROR_THRESHOLD_M
    )

    segmentation_array = np.asarray(segmentation)
    if segmentation_array.shape != (image_height, image_width):
        return _failed_audit(
            target,
            backend,
            center_x,
            center_y,
            expected_body_id,
            "segmentation_shape_mismatch",
        )
    raw_segmentation = int(
        segmentation_array[sample.row, sample.column]
    )
    segmentation_body_id = (
        None
        if raw_segmentation < 0
        else raw_segmentation & ((1 << 24) - 1)
    )
    segmentation_target_match = segmentation_body_id == expected_body_id

    ray_body_id = None
    ray_hit_position = None
    ray_hit_distance_m = None
    ray_target_match = False
    if ray_test is not None:
        try:
            ray_from, ray_to = build_ray_segment(
                camera_eye,
                point.world_xyz,
            )
            ray_body_id, ray_hit_position = ray_test(
                ray_from,
                ray_to,
                client_id,
            )
            ray_body_id = int(ray_body_id)
            ray_target_match = ray_body_id == expected_body_id
            if ray_hit_position is not None:
                hit = np.asarray(ray_hit_position, dtype=np.float64)
                if hit.shape == (3,) and np.all(np.isfinite(hit)):
                    ray_hit_position = tuple(float(value) for value in hit)
                    ray_hit_distance_m = float(
                        np.linalg.norm(hit - np.asarray(point.world_xyz))
                    )
                else:
                    ray_hit_position = None
        except (TypeError, ValueError) as exc:
            ray_body_id = None
            ray_hit_position = None
            ray_hit_distance_m = None
            ray_target_match = False
            ray_failure = f"ray_test_error:{exc}"
        else:
            ray_failure = "" if ray_target_match else "ray_target_mismatch"
    else:
        ray_failure = "ray_test_unavailable"

    failures = []
    if not coordinates_finite:
        failures.append("non_finite_coordinates")
    if not reprojection_passed:
        failures.append("reprojection_threshold_failed")
    if not segmentation_target_match:
        failures.append("segmentation_target_mismatch")
    if ray_failure:
        failures.append(ray_failure)
    gate_passed = not failures

    return BackprojectionAudit(
        target=target,
        backend=backend,
        center_x=center_x,
        center_y=center_y,
        sampled_column=sample.column,
        sampled_row=sample.row,
        depth_m=sample.depth_m,
        camera_x=point.camera_xyz[0],
        camera_y=point.camera_xyz[1],
        camera_z=point.camera_xyz[2],
        world_x=point.world_xyz[0],
        world_y=point.world_xyz[1],
        world_z=point.world_xyz[2],
        reprojected_x=reprojection.pixel_x,
        reprojected_y=reprojection.pixel_y,
        reprojected_depth_m=reprojection.depth_m,
        pixel_error=pixel_error,
        depth_error_m=depth_error_m,
        coordinates_finite=coordinates_finite,
        valid_depth=True,
        reprojection_passed=reprojection_passed,
        segmentation_body_id=segmentation_body_id,
        expected_body_id=expected_body_id,
        segmentation_target_match=segmentation_target_match,
        ray_body_id=ray_body_id,
        ray_target_match=ray_target_match,
        ray_hit_position=ray_hit_position,
        ray_hit_distance_m=ray_hit_distance_m,
        gate_passed=gate_passed,
        failure_reason=";".join(failures),
    )


def summarize_backprojection_rows(
    rows: Sequence[BackprojectionAudit],
) -> dict[str, object]:
    """Summarize the exact nine target/backend rows as a pre-IK gate."""

    received_order = tuple((row.target, row.backend) for row in rows)
    if received_order != EXPECTED_TARGET_BACKENDS:
        raise ValueError(
            "backprojection rows must use exact target/backend order"
        )
    return {
        "protocol": "fixed_three_object_depth_backprojection_gate",
        "backprojection_result_count": 9,
        "coordinates_finite_count": sum(
            row.coordinates_finite for row in rows
        ),
        "valid_depth_count": sum(row.valid_depth for row in rows),
        "reprojection_passed_count": sum(
            row.reprojection_passed for row in rows
        ),
        "segmentation_target_match_count": sum(
            row.segmentation_target_match for row in rows
        ),
        "ray_target_match_count": sum(row.ray_target_match for row in rows),
        "backprojection_gate_passed": all(row.gate_passed for row in rows),
        "backprojection_complete": True,
        "pixel_error_threshold": PIXEL_ERROR_THRESHOLD,
        "depth_error_threshold_m": DEPTH_ERROR_THRESHOLD_M,
        "depth_used_after_2d_prediction": True,
        "segmentation_used_as_coordinate_input": False,
        "ray_test_used_as_coordinate_input": False,
        "ik_executed": False,
        "physical_grasp_executed": False,
    }


def summarize_available_backprojection_rows(
    rows: Sequence[BackprojectionAudit],
) -> dict[str, object]:
    """Summarize available rows without letting an incomplete gate pass."""

    received_order = tuple((row.target, row.backend) for row in rows)
    if received_order == EXPECTED_TARGET_BACKENDS:
        return summarize_backprojection_rows(rows)
    return {
        "protocol": "fixed_three_object_depth_backprojection_gate",
        "backprojection_result_count": len(rows),
        "coordinates_finite_count": sum(
            row.coordinates_finite for row in rows
        ),
        "valid_depth_count": sum(row.valid_depth for row in rows),
        "reprojection_passed_count": sum(
            row.reprojection_passed for row in rows
        ),
        "segmentation_target_match_count": sum(
            row.segmentation_target_match for row in rows
        ),
        "ray_target_match_count": sum(row.ray_target_match for row in rows),
        "backprojection_gate_passed": False,
        "backprojection_complete": False,
        "pixel_error_threshold": PIXEL_ERROR_THRESHOLD,
        "depth_error_threshold_m": DEPTH_ERROR_THRESHOLD_M,
        "depth_used_after_2d_prediction": True,
        "segmentation_used_as_coordinate_input": False,
        "ray_test_used_as_coordinate_input": False,
        "ik_executed": False,
        "physical_grasp_executed": False,
    }
