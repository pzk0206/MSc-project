import pybullet as p
import pybullet_data
import pytest

from src.simulation.pybullet.kinematic_audit import (
    CandidateAudit,
    CollisionAudit,
    IKPoseAudit,
    audit_joint_path_clearance,
    audit_pose_ik,
    resolve_panda_model,
    select_candidate_pair,
)
from src.simulation.pybullet.pose_generation import (
    PoseCandidate,
    ToolPose,
)


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


def test_real_collision_audit_checks_41_states_and_restores() -> None:
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
        physicsClientId=client_id,
    )
    try:
        model = resolve_panda_model(robot_id, client_id)
        before = tuple(
            p.getJointState(
                robot_id, index, physicsClientId=client_id
            )[0]
            for index in model.movable_joint_indices
        )
        arm_rest = tuple(model.rest_poses[:7])

        audit = audit_joint_path_clearance(
            robot_id=robot_id,
            client_id=client_id,
            model=model,
            start_solution=arm_rest,
            pregrasp_solution=arm_rest,
            standoff_solution=arm_rest,
            environment_body_ids=(),
        )

        after = tuple(
            p.getJointState(
                robot_id, index, physicsClientId=client_id
            )[0]
            for index in model.movable_joint_indices
        )
        assert audit.checked_state_count == 41
        assert after == pytest.approx(before)
    finally:
        p.disconnect(client_id)


def _candidate_audit(
    symmetry: float,
    passed: bool,
    cost: float,
) -> CandidateAudit:
    pose = ToolPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    candidate = PoseCandidate(
        target="cube",
        backend="geometry",
        symmetry_degrees=symmetry,
        finger_axis_world=(1.0, 0.0, 0.0),
        closing_axis_world=(0.0, -1.0, 0.0),
        approach_axis_world=(0.0, 0.0, -1.0),
        surface_standoff_pose=pose,
        pregrasp_pose=pose,
    )
    ik = IKPoseAudit(
        solution=(0.0,) * 7,
        limits_passed=passed,
        position_error_m=0.0,
        orientation_error_degrees=0.0,
        fk_passed=passed,
        gate_passed=passed,
        failure_reason="" if passed else "ik_failed",
    )
    collision = CollisionAudit(
        clearance_passed=passed,
        checked_state_count=41,
        minimum_clearance_m=0.002,
        environment_collision_count=0,
        self_collision_count=0,
        failure_reason="" if passed else "collision_failed",
    )
    return CandidateAudit(
        candidate=candidate,
        pregrasp_ik=ik,
        standoff_ik=ik,
        collision=collision,
        total_normalized_joint_cost=cost,
        gate_passed=passed,
        selected=False,
        failure_reason="" if passed else "candidate_failed",
    )


def test_candidate_selection_prefers_lower_cost_and_zero_degree_ties() -> None:
    selected = select_candidate_pair(
        (_candidate_audit(0.0, True, 2.0), _candidate_audit(180.0, True, 1.0))
    )
    assert [row.selected for row in selected] == [False, True]

    tied = select_candidate_pair(
        (_candidate_audit(0.0, True, 1.0), _candidate_audit(180.0, True, 1.0))
    )
    assert [row.selected for row in tied] == [True, False]


def test_candidate_selection_selects_none_when_both_fail() -> None:
    selected = select_candidate_pair(
        (
            _candidate_audit(0.0, False, 1.0),
            _candidate_audit(180.0, False, 2.0),
        )
    )
    assert not any(row.selected for row in selected)
