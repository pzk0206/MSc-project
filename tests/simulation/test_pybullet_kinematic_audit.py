import pybullet as p
import pybullet_data
import pytest

from src.simulation.pybullet.kinematic_audit import (
    audit_pose_ik,
    resolve_panda_model,
)
from src.simulation.pybullet.pose_generation import ToolPose


def _panda_client() -> tuple[int, int]:
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
        physicsClientId=client_id,
    )
    return client_id, robot_id


def test_real_panda_is_resolved_by_joint_and_link_names() -> None:
    client_id, robot_id = _panda_client()
    try:
        model = resolve_panda_model(robot_id, client_id)

        assert model.arm_joint_indices == tuple(range(7))
        assert model.finger_joint_indices == (9, 10)
        assert model.movable_joint_indices == (*range(7), 9, 10)
        assert model.tool_link_index == 11
        assert len(model.lower_limits) == 9
        assert len(model.rest_poses) == 9
        assert model.rest_poses[-2:] == pytest.approx((0.04, 0.04))
        assert all(
            lower <= rest <= upper
            for lower, rest, upper in zip(
                model.lower_limits,
                model.rest_poses,
                model.upper_limits,
            )
        )
    finally:
        p.disconnect(client_id)


def test_real_panda_ik_passes_fk_and_restores_joint_states() -> None:
    client_id, robot_id = _panda_client()
    try:
        model = resolve_panda_model(robot_id, client_id)
        before = tuple(
            p.getJointState(
                robot_id, index, physicsClientId=client_id
            )[0]
            for index in model.movable_joint_indices
        )
        for index, value in zip(
            model.movable_joint_indices,
            model.rest_poses,
        ):
            p.resetJointState(
                robot_id,
                index,
                value,
                physicsClientId=client_id,
            )
        reachable_state = p.getLinkState(
            robot_id,
            model.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        for index, value in zip(model.movable_joint_indices, before):
            p.resetJointState(
                robot_id,
                index,
                value,
                physicsClientId=client_id,
            )
        pose = ToolPose(
            position=tuple(reachable_state[4]),
            quaternion_xyzw=tuple(reachable_state[5]),
        )

        audit = audit_pose_ik(
            robot_id,
            client_id,
            model,
            pose,
        )

        after = tuple(
            p.getJointState(
                robot_id, index, physicsClientId=client_id
            )[0]
            for index in model.movable_joint_indices
        )
        assert audit.solution is not None
        assert len(audit.solution) == 7
        assert audit.limits_passed
        assert audit.position_error_m <= 0.005
        assert audit.orientation_error_degrees <= 5.0
        assert audit.fk_passed
        assert audit.gate_passed
        assert audit.failure_reason == ""
        assert after == pytest.approx(before)
    finally:
        p.disconnect(client_id)


class _EmptyPhysics:
    JOINT_FIXED = p.JOINT_FIXED
    JOINT_REVOLUTE = p.JOINT_REVOLUTE
    JOINT_PRISMATIC = p.JOINT_PRISMATIC

    @staticmethod
    def getNumJoints(robot_id, physicsClientId):
        return 0


def test_resolver_rejects_a_model_without_named_panda_joints() -> None:
    with pytest.raises(ValueError, match="missing Panda arm joint"):
        resolve_panda_model(4, 7, physics=_EmptyPhysics())


def test_ik_rejects_non_unit_pose_quaternion_without_changing_state() -> None:
    client_id, robot_id = _panda_client()
    try:
        model = resolve_panda_model(robot_id, client_id)

        audit = audit_pose_ik(
            robot_id,
            client_id,
            model,
            ToolPose(
                position=(0.45, 0.0, 0.85),
                quaternion_xyzw=(0.0, 0.0, 0.0, 0.0),
            ),
        )

        assert not audit.gate_passed
        assert audit.failure_reason == "invalid_target_pose"
    finally:
        p.disconnect(client_id)
