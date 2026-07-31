import numpy as np
import pybullet as p
import pytest

from src.simulation.pybullet.backprojection import backproject_pixel
from src.simulation.pybullet.pose_generation import (
    generate_top_down_pose_candidates,
)


def _camera() -> tuple[tuple[float, ...], tuple[float, ...]]:
    view = p.computeViewMatrix(
        cameraEyePosition=(1.0, 0.0, 1.15),
        cameraTargetPosition=(0.5, 0.0, 0.62),
        cameraUpVector=(0.0, 0.0, 1.0),
    )
    projection = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=1.0,
        nearVal=0.05,
        farVal=3.0,
    )
    return tuple(view), tuple(projection)


@pytest.mark.parametrize("angle_degrees", [0.0, 90.0, -42.5])
def test_top_down_candidates_are_orthonormal_and_symmetric(
    angle_degrees: float,
) -> None:
    view, projection = _camera()
    surface = backproject_pixel(
        50,
        50,
        0.7,
        100,
        100,
        view,
        projection,
        0.05,
        3.0,
    )

    candidates = generate_top_down_pose_candidates(
        target="cube",
        backend="geometry",
        column=50,
        row=50,
        depth_m=0.7,
        angle_degrees=angle_degrees,
        width=100,
        height=100,
        view_matrix=view,
        projection_matrix=projection,
        near=0.05,
        far=3.0,
    )

    assert [item.symmetry_degrees for item in candidates] == [0.0, 180.0]
    first, second = candidates
    assert first.approach_axis_world == pytest.approx((0.0, 0.0, -1.0))
    assert second.finger_axis_world == pytest.approx(
        -np.asarray(first.finger_axis_world)
    )
    assert second.closing_axis_world == pytest.approx(
        -np.asarray(first.closing_axis_world)
    )
    for candidate in candidates:
        rotation = np.asarray(
            p.getMatrixFromQuaternion(
                candidate.surface_standoff_pose.quaternion_xyzw
            )
        ).reshape(3, 3)
        assert rotation.T @ rotation == pytest.approx(
            np.eye(3), abs=1e-9
        )
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        assert rotation[:, 0] == pytest.approx(
            candidate.finger_axis_world, abs=1e-7
        )
        assert rotation[:, 1] == pytest.approx(
            candidate.closing_axis_world, abs=1e-7
        )
        assert rotation[:, 2] == pytest.approx(
            candidate.approach_axis_world, abs=1e-7
        )
        assert candidate.surface_standoff_pose.position == pytest.approx(
            (*surface.world_xyz[:2], surface.world_xyz[2] + 0.02),
            abs=1e-7,
        )
        assert candidate.pregrasp_pose.position == pytest.approx(
            (*surface.world_xyz[:2], surface.world_xyz[2] + 0.12),
            abs=1e-7,
        )


def test_top_down_pose_supports_an_image_border_centre() -> None:
    view, projection = _camera()

    candidates = generate_top_down_pose_candidates(
        target="duck",
        backend="single",
        column=0,
        row=50,
        depth_m=0.7,
        angle_degrees=0.0,
        width=100,
        height=100,
        view_matrix=view,
        projection_matrix=projection,
        near=0.05,
        far=3.0,
    )

    assert len(candidates) == 2


@pytest.mark.parametrize(
    ("angle", "offset"),
    [(np.nan, 5.0), (0.0, 0.0), (0.0, np.inf)],
)
def test_top_down_pose_rejects_invalid_direction_inputs(
    angle: float,
    offset: float,
) -> None:
    view, projection = _camera()

    with pytest.raises(ValueError):
        generate_top_down_pose_candidates(
            target="duck",
            backend="single",
            column=50,
            row=50,
            depth_m=0.7,
            angle_degrees=angle,
            width=100,
            height=100,
            view_matrix=view,
            projection_matrix=projection,
            near=0.05,
            far=3.0,
            tangent_offset_px=offset,
        )
