import pytest

from src.simulation.pybullet.pose_generation import (
    generate_top_down_pose_from_world_point,
)


def test_truth_world_point_generates_top_down_cube_pregrasp() -> None:
    candidate = generate_top_down_pose_from_world_point(
        target="cube",
        backend="ground_truth",
        surface_point=(0.48, 0.0, 0.685),
        finger_axis_world=(0.8660254038, 0.5, 0.0),
    )

    assert candidate.surface_standoff_pose.position == pytest.approx(
        (0.48, 0.0, 0.705)
    )
    assert candidate.pregrasp_pose.position == pytest.approx(
        (0.48, 0.0, 0.805)
    )
    assert candidate.finger_axis_world == pytest.approx(
        (0.8660254038, 0.5, 0.0)
    )
    assert candidate.approach_axis_world == pytest.approx((0.0, 0.0, -1.0))
