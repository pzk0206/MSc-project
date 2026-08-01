from dataclasses import replace

import pybullet as p
import pytest

from src.simulation.pybullet.kinematic_audit import resolve_panda_model
from src.simulation.pybullet.motion_control import (
    MotionConfig,
    MotionSegment,
    execute_joint_motion,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps_per_segment": 0}, "steps_per_segment must be positive"),
        ({"settle_steps": -1}, "settle_steps must be non-negative"),
        ({"joint_tolerance_rad": 0.0}, "joint_tolerance_rad must be positive"),
        ({"clearance_m": 0.0}, "clearance_m must be positive"),
    ],
)
def test_motion_config_rejects_values_that_cannot_define_execution(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MotionConfig(**kwargs)


def test_motion_segment_rejects_a_target_without_seven_arm_values() -> None:
    with pytest.raises(
        ValueError,
        match="motion target must contain seven arm values",
    ):
        MotionSegment("outbound", (0.0,) * 6)


def test_real_motor_motion_reaches_a_safe_joint_target_and_returns() -> None:
    scene_config = replace(
        fixed_scene_config(MultiObjectStudyConfig()),
        robot_self_collision=True,
    )
    with PyBulletScene(scene_config) as scene:
        model = resolve_panda_model(
            scene.bodies.robot,
            scene.client_id,
        )
        neutral = tuple(model.rest_poses[:7])
        target = (*neutral[:6], neutral[6] + 0.1)
        for index, value in zip(model.arm_joint_indices, neutral):
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

        config = MotionConfig(steps_per_segment=120, settle_steps=240)
        result = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(
                MotionSegment("outbound", target),
                MotionSegment("return", neutral),
            ),
            environment_body_ids=(
                scene.bodies.plane,
                scene.bodies.table,
                *scene.object_body_ids.values(),
            ),
            allowed_environment_link_pairs=((-1, scene.bodies.table),),
            config=config,
        )

        final = tuple(
            float(
                p.getJointState(
                    scene.bodies.robot,
                    index,
                    physicsClientId=scene.client_id,
                )[0]
            )
            for index in model.arm_joint_indices
        )
        fingers = tuple(
            float(
                p.getJointState(
                    scene.bodies.robot,
                    index,
                    physicsClientId=scene.client_id,
                )[0]
            )
            for index in model.finger_joint_indices
        )

        assert result.gate_passed
        assert dict(result.segment_reached) == {
            "outbound": True,
            "return": True,
        }
        assert all(
            type(reached) is bool for _, reached in result.segment_reached
        )
        assert len(result.trace) >= 2 * config.steps_per_segment
        assert result.environment_collision_count == 0
        assert result.self_collision_count == 0
        assert result.all_states_finite
        assert max(abs(a - b) for a, b in zip(final, neutral)) <= 0.01
        assert fingers == pytest.approx((0.04, 0.04), abs=1e-3)
