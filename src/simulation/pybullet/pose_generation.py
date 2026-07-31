"""Pure top-down hover-pose generation from verified image grasps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from src.simulation.pybullet.backprojection import (
    backproject_image_coordinate,
    backproject_pixel,
)


@dataclass(frozen=True)
class ToolPose:
    """One world-frame pose for the Panda grasp-target link."""

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PoseCandidate:
    """One symmetry-resolved top-down hover-pose candidate."""

    target: str
    backend: str
    symmetry_degrees: float
    finger_axis_world: tuple[float, float, float]
    closing_axis_world: tuple[float, float, float]
    approach_axis_world: tuple[float, float, float]
    surface_standoff_pose: ToolPose
    pregrasp_pose: ToolPose


def _rotation_matrix_to_quaternion(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a proper 3-D rotation matrix to an xyzw quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite and 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _candidate(
    target: str,
    backend: str,
    symmetry_degrees: float,
    finger_axis: np.ndarray,
    surface_point: np.ndarray,
    surface_standoff_m: float,
    pregrasp_offset_m: float,
) -> PoseCandidate:
    approach = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    closing = np.cross(approach, finger_axis)
    closing /= np.linalg.norm(closing)
    rotation = np.column_stack((finger_axis, closing, approach))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("tool rotation must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9
    ):
        raise ValueError("tool rotation must be right-handed")
    quaternion = _rotation_matrix_to_quaternion(rotation)
    standoff = surface_point - approach * surface_standoff_m
    pregrasp = standoff - approach * pregrasp_offset_m
    return PoseCandidate(
        target=target,
        backend=backend,
        symmetry_degrees=symmetry_degrees,
        finger_axis_world=tuple(float(value) for value in finger_axis),
        closing_axis_world=tuple(float(value) for value in closing),
        approach_axis_world=tuple(float(value) for value in approach),
        surface_standoff_pose=ToolPose(
            position=tuple(float(value) for value in standoff),
            quaternion_xyzw=quaternion,
        ),
        pregrasp_pose=ToolPose(
            position=tuple(float(value) for value in pregrasp),
            quaternion_xyzw=quaternion,
        ),
    )


def generate_top_down_pose_candidates(
    *,
    target: str,
    backend: str,
    column: int,
    row: int,
    depth_m: float,
    angle_degrees: float,
    width: int,
    height: int,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
    near: float,
    far: float,
    tangent_offset_px: float = 5.0,
    surface_standoff_m: float = 0.02,
    pregrasp_offset_m: float = 0.10,
) -> tuple[PoseCandidate, PoseCandidate]:
    """Generate the two 180-degree-symmetric top-down hover poses."""

    scalar_values = (
        angle_degrees,
        tangent_offset_px,
        surface_standoff_m,
        pregrasp_offset_m,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("pose-generation parameters must be finite")
    if tangent_offset_px <= 0.0:
        raise ValueError("tangent offset must be positive")
    if surface_standoff_m <= 0.0 or pregrasp_offset_m <= 0.0:
        raise ValueError("pose offsets must be positive")

    surface = backproject_pixel(
        column,
        row,
        depth_m,
        width,
        height,
        view_matrix,
        projection_matrix,
        near,
        far,
    )
    angle_radians = math.radians(angle_degrees)
    image_direction = np.array(
        (math.cos(angle_radians), math.sin(angle_radians)),
        dtype=np.float64,
    )
    centre = np.array((column, row), dtype=np.float64)
    negative = np.clip(
        centre - tangent_offset_px * image_direction,
        (0.0, 0.0),
        (width - 1.0, height - 1.0),
    )
    positive = np.clip(
        centre + tangent_offset_px * image_direction,
        (0.0, 0.0),
        (width - 1.0, height - 1.0),
    )
    first = backproject_image_coordinate(
        float(negative[0]),
        float(negative[1]),
        depth_m,
        width,
        height,
        view_matrix,
        projection_matrix,
        near,
        far,
    )
    second = backproject_image_coordinate(
        float(positive[0]),
        float(positive[1]),
        depth_m,
        width,
        height,
        view_matrix,
        projection_matrix,
        near,
        far,
    )
    tangent = np.asarray(second.world_xyz) - np.asarray(first.world_xyz)
    tangent[2] = 0.0
    norm = float(np.linalg.norm(tangent))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("projected image tangent must be non-zero")
    finger_axis = tangent / norm
    surface_point = np.asarray(surface.world_xyz, dtype=np.float64)
    return (
        _candidate(
            target,
            backend,
            0.0,
            finger_axis,
            surface_point,
            surface_standoff_m,
            pregrasp_offset_m,
        ),
        _candidate(
            target,
            backend,
            180.0,
            -finger_axis,
            surface_point,
            surface_standoff_m,
            pregrasp_offset_m,
        ),
    )
