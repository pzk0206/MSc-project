from dataclasses import replace
import math

import pybullet as p
import pytest

from src.simulation.pybullet.gripper_control import (
    GripperCloseConfig,
    execute_gripper_close,
)
from src.simulation.pybullet.kinematic_audit import (
    audit_pose_ik,
    resolve_panda_model,
)
from src.simulation.pybullet.pose_generation import (
    generate_top_down_pose_from_world_point,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"close_steps": 0}, "close_steps must be positive"),
        ({"hold_steps": -1}, "hold_steps must be non-negative"),
        (
            {"hold_steps": 10, "minimum_bilateral_hold_steps": 11},
            "minimum bilateral hold cannot exceed hold_steps",
        ),
        ({"closed_target_m": -0.001}, "closed_target_m must be within"),
        (
            {"arm_joint_tolerance_rad": 0.0},
            "arm_joint_tolerance_rad must be positive",
        ),
    ],
)
def test_gripper_close_config_rejects_invalid_protocol_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GripperCloseConfig(**kwargs)


def test_real_gripper_close_acquires_and_holds_bilateral_cube_contact() -> None:
    scene_config = replace(
        fixed_scene_config(MultiObjectStudyConfig()),
        robot_self_collision=True,
    )
    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        model = resolve_panda_model(scene.bodies.robot, scene.client_id)
        cube_id = scene.object_body_ids["cube"]
        cube_position, cube_quaternion = p.getBasePositionAndOrientation(
            cube_id,
            physicsClientId=scene.client_id,
        )
        cube_top_z = p.getAABB(
            cube_id,
            physicsClientId=scene.client_id,
        )[1][2]
        rotation = p.getMatrixFromQuaternion(cube_quaternion)
        candidate = generate_top_down_pose_from_world_point(
            target="cube",
            backend="ground_truth",
            surface_point=(cube_position[0], cube_position[1], cube_top_z),
            finger_axis_world=(rotation[0], rotation[3], rotation[6]),
            surface_standoff_m=0.005,
        )
        approach_ik = audit_pose_ik(
            scene.bodies.robot,
            scene.client_id,
            model,
            candidate.surface_standoff_pose,
        )
        assert approach_ik.gate_passed
        assert approach_ik.solution is not None
        for index, value in zip(
            model.arm_joint_indices,
            approach_ik.solution,
        ):
            p.resetJointState(
                scene.bodies.robot,
                index,
                value,
                physicsClientId=scene.client_id,
            )
        for index in model.finger_joint_indices:
            p.resetJointState(
                scene.bodies.robot,
                index,
                0.04,
                physicsClientId=scene.client_id,
            )
        p.performCollisionDetection(physicsClientId=scene.client_id)

        result = execute_gripper_close(
            robot_id=scene.bodies.robot,
            target_body_id=cube_id,
            client_id=scene.client_id,
            model=model,
            arm_hold_positions=approach_ik.solution,
            environment_body_ids=(
                scene.bodies.plane,
                scene.bodies.table,
                scene.object_body_ids["duck"],
                scene.object_body_ids["sphere"],
            ),
            allowed_environment_link_pairs=((-1, scene.bodies.table),),
        )

    assert result.bilateral_contact_acquired is True
    assert result.trailing_bilateral_contact_steps >= 60
    assert result.left_finger_contacted is True
    assert result.right_finger_contacted is True
    assert {event.robot_link for event in result.contact_events} == set(
        model.finger_joint_indices
    )
    assert all(
        math.isfinite(event.normal_force) and event.normal_force > 0.0
        for event in result.contact_events
    )
    assert result.prohibited_target_contact_count == 0
    assert result.environment_collision_count == 0
    assert result.self_collision_count == 0
    assert result.all_states_finite is True
    assert result.gate_passed is True
