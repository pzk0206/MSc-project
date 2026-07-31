import numpy as np
import pybullet as p
import pytest

from src.simulation.pybullet.backprojection import (
    backproject_pixel,
    metric_depth_to_buffer,
    reproject_world_point,
    sample_nearest_depth,
)


def test_nearest_depth_uses_half_up_pixel_rounding() -> None:
    depth = np.arange(20, dtype=np.float32).reshape(4, 5) / 10 + 0.1

    sample = sample_nearest_depth(
        depth,
        center_x=1.5,
        center_y=2.5,
        near=0.05,
        far=3.0,
    )

    assert (sample.column, sample.row) == (2, 3)
    assert sample.depth_m == pytest.approx(float(depth[3, 2]))


@pytest.mark.parametrize(
    ("center_x", "center_y"),
    [(-0.01, 0.0), (5.0, 0.0)],
)
def test_nearest_depth_rejects_centres_outside_image(
    center_x: float,
    center_y: float,
) -> None:
    with pytest.raises(ValueError, match="inside image"):
        sample_nearest_depth(
            np.ones((4, 5), dtype=np.float32),
            center_x,
            center_y,
            near=0.05,
            far=3.0,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.05, 3.0])
def test_nearest_depth_rejects_non_surface_depth(value: float) -> None:
    depth = np.full((2, 2), 0.8, dtype=np.float32)
    depth[1, 1] = value

    with pytest.raises(ValueError, match="depth"):
        sample_nearest_depth(
            depth,
            center_x=1.0,
            center_y=1.0,
            near=0.05,
            far=3.0,
        )


def test_metric_depth_inverse_maps_clip_planes() -> None:
    assert metric_depth_to_buffer(0.1, 0.1, 10.0) == pytest.approx(0.0)
    assert metric_depth_to_buffer(10.0, 0.1, 10.0) == pytest.approx(1.0)


def test_backprojection_recovers_hand_derived_world_point() -> None:
    view = p.computeViewMatrix(
        cameraEyePosition=(0.0, 0.0, 1.0),
        cameraTargetPosition=(0.0, 0.0, 0.0),
        cameraUpVector=(0.0, 1.0, 0.0),
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=10.0,
    )

    point = backproject_pixel(
        column=0,
        row=0,
        depth_m=0.5,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
        near=0.1,
        far=10.0,
    )

    assert point.camera_xyz == pytest.approx(
        (0.0, 0.0, -0.5), abs=1e-6
    )
    assert point.world_xyz == pytest.approx(
        (0.0, 0.0, 0.5), abs=1e-6
    )

    reprojection = reproject_world_point(
        point.world_xyz,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
    )
    assert (reprojection.pixel_x, reprojection.pixel_y) == pytest.approx(
        (0.0, 0.0), abs=1e-6
    )
    assert reprojection.depth_m == pytest.approx(0.5, abs=1e-6)
