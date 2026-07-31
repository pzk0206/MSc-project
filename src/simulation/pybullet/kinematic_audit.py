"""Offline Panda IK, FK, and static collision audit utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pybullet as p

from src.simulation.pybullet.pose_generation import ToolPose


POSITION_ERROR_THRESHOLD_M = 0.005
ORIENTATION_ERROR_THRESHOLD_DEGREES = 5.0
JOINT_LIMIT_TOLERANCE = 1e-6
_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
_TOOL_LINK_NAME = "panda_grasptarget"


@dataclass(frozen=True)
class PandaModelInfo:
    """Name-resolved Panda model indices and movable-joint limits."""

    arm_joint_indices: tuple[int, ...]
    finger_joint_indices: tuple[int, ...]
    movable_joint_indices: tuple[int, ...]
    tool_link_index: int
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    joint_ranges: tuple[float, ...]
    rest_poses: tuple[float, ...]


@dataclass(frozen=True)
class IKPoseAudit:
    """One inverse-kinematics result checked by forward kinematics."""

    solution: tuple[float, ...] | None
    limits_passed: bool
    position_error_m: float | None
    orientation_error_degrees: float | None
    fk_passed: bool
    gate_passed: bool
    failure_reason: str


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def resolve_panda_model(
    robot_id: int,
    client_id: int,
    physics: Any = p,
) -> PandaModelInfo:
    """Resolve required Panda joints and tool link by URDF names."""

    joints: dict[str, tuple[int, tuple[Any, ...]]] = {}
    links: dict[str, int] = {}
    movable_indices = []
    joint_count = int(
        physics.getNumJoints(robot_id, physicsClientId=client_id)
    )
    for index in range(joint_count):
        info = physics.getJointInfo(
            robot_id,
            index,
            physicsClientId=client_id,
        )
        name = _decode(info[1])
        link_name = _decode(info[12])
        if name in joints:
            raise ValueError(f"duplicate Panda joint name: {name}")
        if link_name in links:
            raise ValueError(f"duplicate Panda link name: {link_name}")
        joints[name] = (index, info)
        links[link_name] = index
        if int(info[2]) != physics.JOINT_FIXED:
            movable_indices.append(index)

    for name in _ARM_JOINT_NAMES:
        if name not in joints:
            raise ValueError(f"missing Panda arm joint: {name}")
        if int(joints[name][1][2]) != physics.JOINT_REVOLUTE:
            raise ValueError(f"Panda arm joint must be revolute: {name}")
    for name in _FINGER_JOINT_NAMES:
        if name not in joints:
            raise ValueError(f"missing Panda finger joint: {name}")
        if int(joints[name][1][2]) != physics.JOINT_PRISMATIC:
            raise ValueError(f"Panda finger joint must be prismatic: {name}")
    if _TOOL_LINK_NAME not in links:
        raise ValueError(f"missing Panda tool link: {_TOOL_LINK_NAME}")

    expected_movable_names = (*_ARM_JOINT_NAMES, *_FINGER_JOINT_NAMES)
    expected_indices = tuple(joints[name][0] for name in expected_movable_names)
    if tuple(movable_indices) != expected_indices:
        raise ValueError("Panda movable joints must match named arm and fingers")

    lower_limits = []
    upper_limits = []
    rest_poses = []
    for name in expected_movable_names:
        info = joints[name][1]
        lower = float(info[8])
        upper = float(info[9])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"invalid Panda joint limits: {name}")
        lower_limits.append(lower)
        upper_limits.append(upper)
        rest_poses.append(
            0.04 if name in _FINGER_JOINT_NAMES else (lower + upper) / 2.0
        )
    ranges = tuple(
        upper - lower
        for lower, upper in zip(lower_limits, upper_limits)
    )
    if any(value <= 0.0 for value in ranges):
        raise ValueError("Panda movable joint ranges must be positive")
    if any(
        not lower <= rest <= upper
        for lower, rest, upper in zip(
            lower_limits,
            rest_poses,
            upper_limits,
        )
    ):
        raise ValueError("Panda rest poses must remain inside joint limits")

    return PandaModelInfo(
        arm_joint_indices=tuple(joints[name][0] for name in _ARM_JOINT_NAMES),
        finger_joint_indices=tuple(
            joints[name][0] for name in _FINGER_JOINT_NAMES
        ),
        movable_joint_indices=expected_indices,
        tool_link_index=links[_TOOL_LINK_NAME],
        lower_limits=tuple(lower_limits),
        upper_limits=tuple(upper_limits),
        joint_ranges=ranges,
        rest_poses=tuple(rest_poses),
    )


def _failed_ik(reason: str) -> IKPoseAudit:
    return IKPoseAudit(
        solution=None,
        limits_passed=False,
        position_error_m=None,
        orientation_error_degrees=None,
        fk_passed=False,
        gate_passed=False,
        failure_reason=reason,
    )


def _validated_pose(
    pose: ToolPose,
) -> tuple[np.ndarray, np.ndarray] | None:
    position = np.asarray(pose.position, dtype=np.float64)
    quaternion = np.asarray(pose.quaternion_xyzw, dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
        or not math.isclose(
            float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6
        )
    ):
        return None
    return position, quaternion


def audit_pose_ik(
    robot_id: int,
    client_id: int,
    model: PandaModelInfo,
    pose: ToolPose,
    physics: Any = p,
) -> IKPoseAudit:
    """Solve one pose, verify FK, and restore all movable joint states."""

    validated = _validated_pose(pose)
    if validated is None:
        return _failed_ik("invalid_target_pose")
    target_position, target_quaternion = validated
    original_states = tuple(
        float(
            physics.getJointState(
                robot_id,
                index,
                physicsClientId=client_id,
            )[0]
        )
        for index in model.movable_joint_indices
    )
    arm_offsets = tuple(
        model.movable_joint_indices.index(index)
        for index in model.arm_joint_indices
    )
    try:
        for index, offset in zip(model.arm_joint_indices, arm_offsets):
            physics.resetJointState(
                robot_id,
                index,
                model.rest_poses[offset],
                physicsClientId=client_id,
            )
        for index in model.finger_joint_indices:
            physics.resetJointState(
                robot_id,
                index,
                0.04,
                physicsClientId=client_id,
            )
        raw_solution = tuple(
            float(value)
            for value in physics.calculateInverseKinematics(
                robot_id,
                model.tool_link_index,
                targetPosition=tuple(target_position),
                targetOrientation=tuple(target_quaternion),
                lowerLimits=tuple(
                    model.lower_limits[offset] for offset in arm_offsets
                ),
                upperLimits=tuple(
                    model.upper_limits[offset] for offset in arm_offsets
                ),
                jointRanges=tuple(
                    model.joint_ranges[offset] for offset in arm_offsets
                ),
                restPoses=tuple(
                    model.rest_poses[offset] for offset in arm_offsets
                ),
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=client_id,
            )
        )
        if len(raw_solution) != len(model.movable_joint_indices):
            return _failed_ik("unexpected_ik_solution_length")
        if not all(math.isfinite(value) for value in raw_solution):
            return _failed_ik("non_finite_ik_solution")
        arm_solution = tuple(raw_solution[offset] for offset in arm_offsets)
        limits_passed = all(
            model.lower_limits[offset] - JOINT_LIMIT_TOLERANCE
            <= value
            <= model.upper_limits[offset] + JOINT_LIMIT_TOLERANCE
            for value, offset in zip(
                arm_solution,
                arm_offsets,
            )
        )
        arm_positions = {
            index: value
            for index, value in zip(model.arm_joint_indices, arm_solution)
        }
        for index in model.arm_joint_indices:
            physics.resetJointState(
                robot_id,
                index,
                arm_positions[index],
                physicsClientId=client_id,
            )
        for index in model.finger_joint_indices:
            physics.resetJointState(
                robot_id,
                index,
                0.04,
                physicsClientId=client_id,
            )
        link_state = physics.getLinkState(
            robot_id,
            model.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        actual_position = np.asarray(link_state[4], dtype=np.float64)
        actual_quaternion = np.asarray(link_state[5], dtype=np.float64)
        position_error = float(
            np.linalg.norm(actual_position - target_position)
        )
        quaternion_dot = float(
            np.clip(
                abs(np.dot(actual_quaternion, target_quaternion)),
                0.0,
                1.0,
            )
        )
        orientation_error = math.degrees(2.0 * math.acos(quaternion_dot))
        fk_passed = (
            limits_passed
            and position_error <= POSITION_ERROR_THRESHOLD_M
            and orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
        )
        failures = []
        if not limits_passed:
            failures.append("joint_limit_failed")
        if position_error > POSITION_ERROR_THRESHOLD_M:
            failures.append("fk_position_threshold_failed")
        if orientation_error > ORIENTATION_ERROR_THRESHOLD_DEGREES:
            failures.append("fk_orientation_threshold_failed")
        return IKPoseAudit(
            solution=tuple(
                arm_positions[index] for index in model.arm_joint_indices
            ),
            limits_passed=limits_passed,
            position_error_m=position_error,
            orientation_error_degrees=orientation_error,
            fk_passed=fk_passed,
            gate_passed=not failures,
            failure_reason=";".join(failures),
        )
    except (TypeError, ValueError) as exc:
        return _failed_ik(f"ik_audit_error:{exc}")
    finally:
        for index, value in zip(model.movable_joint_indices, original_states):
            physics.resetJointState(
                robot_id,
                index,
                value,
                physicsClientId=client_id,
            )
