from dataclasses import replace
import math

import pybullet as p
import pytest

from src.simulation.pybullet.gripper_control import execute_gripper_close
from src.simulation.pybullet.kinematic_audit import (
    audit_pose_ik,
    resolve_panda_model,
)
from src.simulation.pybullet.lift_control import (
    LiftConfig,
    execute_object_lift,
)
from src.simulation.pybullet.pose_generation import (
    ToolPose,
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
        ({"lift_steps": 0}, "lift_steps must be positive"),
        ({"hold_steps": 0}, "hold_steps must be positive"),
        ({"settle_steps": -1}, "settle_steps must be non-negative"),
        (
            {"tool_lift_command_m": 0.0},
            "tool_lift_command_m must be positive",
        ),
        (
            {
                "tool_lift_command_m": 0.10,
                "minimum_object_lift_m": 0.10,
            },
            "minimum object lift must be less than tool lift command",
        ),
        (
            {"maximum_hold_relative_drift_m": 0.0},
            "maximum_hold_relative_drift_m must be positive",
        ),
        (
            {
                "hold_steps": 10,
                "minimum_trailing_bilateral_contact_steps": 11,
            },
            "minimum trailing bilateral contact cannot exceed hold_steps",
        ),
        (
            {"arm_joint_tolerance_rad": 0.0},
            "arm_joint_tolerance_rad must be positive",
        ),
    ],
)
def test_lift_config_rejects_invalid_protocol_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LiftConfig(**kwargs)


def test_real_frozen_gripper_lifts_and_holds_cube_off_table() -> None:
    scene_config = replace(
        fixed_scene_config(MultiObjectStudyConfig()),
        robot_self_collision=True,
    )
    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        robot_id = scene.bodies.robot
        cube_id = scene.object_body_ids["cube"]
        model = resolve_panda_model(robot_id, scene.client_id)
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
        grasp_ik = audit_pose_ik(
            robot_id,
            scene.client_id,
            model,
            candidate.surface_standoff_pose,
        )
        assert grasp_ik.gate_passed
        assert grasp_ik.solution is not None
        for index, value in zip(model.arm_joint_indices, grasp_ik.solution):
            p.resetJointState(
                robot_id,
                index,
                value,
                physicsClientId=scene.client_id,
            )
        for index in model.finger_joint_indices:
            p.resetJointState(
                robot_id,
                index,
                0.04,
                physicsClientId=scene.client_id,
            )
        p.performCollisionDetection(physicsClientId=scene.client_id)
        gripper = execute_gripper_close(
            robot_id=robot_id,
            target_body_id=cube_id,
            client_id=scene.client_id,
            model=model,
            arm_hold_positions=grasp_ik.solution,
            environment_body_ids=(
                scene.bodies.plane,
                scene.bodies.table,
                scene.object_body_ids["duck"],
                scene.object_body_ids["sphere"],
            ),
            allowed_environment_link_pairs=((-1, scene.bodies.table),),
        )
        assert gripper.gate_passed
        close_row = gripper.trace[-1]
        lift_pose = ToolPose(
            position=(
                candidate.surface_standoff_pose.position[0],
                candidate.surface_standoff_pose.position[1],
                candidate.surface_standoff_pose.position[2] + 0.12,
            ),
            quaternion_xyzw=(
                candidate.surface_standoff_pose.quaternion_xyzw
            ),
        )
        lift_ik = audit_pose_ik(
            robot_id,
            scene.client_id,
            model,
            lift_pose,
        )
        assert lift_ik.gate_passed
        assert lift_ik.solution is not None
        reference_relative = tuple(
            tool - target
            for tool, target in zip(
                close_row.actual_tool_position,
                close_row.target_position,
            )
        )
        frozen_fingers = close_row.commanded_finger_positions

        result = execute_object_lift(
            robot_id=robot_id,
            target_body_id=cube_id,
            table_body_id=scene.bodies.table,
            client_id=scene.client_id,
            model=model,
            lift_arm_positions=lift_ik.solution,
            lift_target_pose=lift_pose,
            frozen_finger_positions=frozen_fingers,
            reference_target_position=close_row.target_position,
            reference_tool_relative_to_target=reference_relative,
            environment_body_ids=(
                scene.bodies.plane,
                scene.bodies.table,
                scene.object_body_ids["duck"],
                scene.object_body_ids["sphere"],
            ),
            allowed_environment_link_pairs=((-1, scene.bodies.table),),
        )

    assert result.lift_reached is True
    assert result.lift_settle_steps >= 1
    assert result.lift_endpoint_arm_error_rad <= 0.01
    assert result.object_lift_gate_passed is True
    assert result.table_release_gate_passed is True
    assert result.relative_stability_gate_passed is True
    assert result.lift_hold_gate_passed is True
    assert result.trailing_bilateral_contact_steps >= 120
    assert result.left_finger_contacted is True
    assert result.right_finger_contacted is True
    assert result.minimum_hold_object_lift_m >= 0.10
    assert result.maximum_hold_relative_drift_m <= 0.01
    assert result.hold_target_table_contact_count == 0
    assert result.prohibited_target_contact_count == 0
    assert result.environment_collision_count == 0
    assert result.self_collision_count == 0
    assert result.all_states_finite is True
    assert result.gate_passed is True
    assert len(result.trace) == 480 + result.lift_settle_steps
    assert {row.phase for row in result.trace} == {"lift", "lift_hold"}
    assert all(
        row.commanded_finger_positions == frozen_fingers
        for row in result.trace
    )
    assert all(
        math.isfinite(event.normal_force) and event.normal_force > 0.0
        for event in result.contact_events
    )
