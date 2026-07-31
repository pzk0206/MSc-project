"""Fixed virtual-camera utilities for the PyBullet perception pilot.

The depth-buffer conversion follows the public PyBullet camera model
documented in the official quickstart guide:
https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
The implementation here is project-specific and was not copied from a
third-party grasping repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the fixed pilot camera."""

    width: int = 640
    height: int = 480
    eye: tuple[float, float, float] = (1.0, 0.0, 1.15)
    target: tuple[float, float, float] = (0.5, 0.0, 0.62)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_degrees: float = 55.0
    near: float = 0.05
    far: float = 3.0

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 < self.near < self.far:
            raise ValueError("near must be positive and smaller than far")
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("fov must be between 0 and 180 degrees")


@dataclass(frozen=True)
class CameraFrame:
    """RGB, metric depth, segmentation, and matrices from one capture."""

    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]


def linearize_depth(
    depth_buffer: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    """Convert the OpenGL-style PyBullet depth buffer to metric depth."""

    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")

    buffer = np.asarray(depth_buffer, dtype=np.float32)
    if not np.all(np.isfinite(buffer)):
        raise ValueError("depth buffer contains non-finite values")

    depth = far * near / (far - (far - near) * buffer)
    if not np.all(np.isfinite(depth)):
        raise ValueError("linear depth contains non-finite values")
    return depth.astype(np.float32)


def capture_camera_frame(
    client_id: int,
    config: CameraConfig,
    renderer: int,
) -> CameraFrame:
    """Capture one validated RGB/depth/segmentation frame."""

    import pybullet as p

    config.validate()
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=config.eye,
        cameraTargetPosition=config.target,
        cameraUpVector=config.up,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=config.fov_degrees,
        aspect=config.width / config.height,
        nearVal=config.near,
        farVal=config.far,
    )
    returned_width, returned_height, rgba, depth_buffer, segmentation = (
        p.getCameraImage(
            width=config.width,
            height=config.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            physicsClientId=client_id,
        )
    )

    if (returned_width, returned_height) != (config.width, config.height):
        raise ValueError(
            "camera returned unexpected dimensions: "
            f"{returned_width}x{returned_height}"
        )

    rgba_array = np.asarray(rgba, dtype=np.uint8)
    expected_rgba_shape = (config.height, config.width, 4)
    if rgba_array.size != config.height * config.width * 4:
        raise ValueError(
            f"RGBA output must reshape to {expected_rgba_shape}, "
            f"received {rgba_array.shape}"
        )
    rgba_array = rgba_array.reshape(expected_rgba_shape)
    rgb = rgba_array[..., :3].copy()

    depth_buffer_array = np.asarray(depth_buffer, dtype=np.float32)
    segmentation_array = np.asarray(segmentation, dtype=np.int32)
    expected_plane_shape = (config.height, config.width)
    if depth_buffer_array.size != config.height * config.width:
        raise ValueError(
            f"depth output must reshape to {expected_plane_shape}, "
            f"received {depth_buffer_array.shape}"
        )
    if segmentation_array.size != config.height * config.width:
        raise ValueError(
            f"segmentation output must reshape to {expected_plane_shape}, "
            f"received {segmentation_array.shape}"
        )

    depth_m = linearize_depth(
        depth_buffer_array.reshape(expected_plane_shape),
        near=config.near,
        far=config.far,
    )
    segmentation_array = segmentation_array.reshape(expected_plane_shape)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("RGB output contains non-finite values")

    return CameraFrame(
        rgb=rgb,
        depth_m=depth_m,
        segmentation=segmentation_array,
        view_matrix=tuple(float(value) for value in view_matrix),
        projection_matrix=tuple(float(value) for value in projection_matrix),
    )
