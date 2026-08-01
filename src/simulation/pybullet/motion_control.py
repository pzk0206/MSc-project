"""Deterministic motor-control primitives for staged Panda execution.

The implementation uses the public PyBullet API documented by the official
Bullet repository: https://github.com/bulletphysics/bullet3
No third-party grasp-execution code is copied or adapted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pybullet as p

from src.simulation.pybullet.kinematic_audit import PandaModelInfo


@dataclass(frozen=True)
class MotionConfig:
    """Fixed timing and gate thresholds for one joint-motion execution."""

    steps_per_segment: int = 240
    settle_steps: int = 240
    joint_tolerance_rad: float = 0.01
    clearance_m: float = 0.002

    def __post_init__(self) -> None:
        if self.steps_per_segment <= 0:
            raise ValueError("steps_per_segment must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if (
            not math.isfinite(self.joint_tolerance_rad)
            or self.joint_tolerance_rad <= 0.0
        ):
            raise ValueError("joint_tolerance_rad must be positive")
        if not math.isfinite(self.clearance_m) or self.clearance_m <= 0.0:
            raise ValueError("clearance_m must be positive")


@dataclass(frozen=True)
class MotionSegment:
    """One named seven-joint target."""

    name: str
    target_arm_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.target_arm_positions) != 7:
            raise ValueError("motion target must contain seven arm values")
        if not self.name.strip():
            raise ValueError("motion segment name must be non-empty")
        if not all(math.isfinite(value) for value in self.target_arm_positions):
            raise ValueError("motion target must contain finite arm values")


@dataclass(frozen=True)
class TrackedBodyPose:
    """Measured world pose of one optionally tracked rigid body."""

    body_id: int
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class MotionTraceRow:
    """Measured state after one real simulation step."""

    step: int
    phase: str
    commanded_arm_positions: tuple[float, ...]
    actual_arm_positions: tuple[float, ...]
    actual_finger_positions: tuple[float, float]
    actual_tool_position: tuple[float, float, float]
    actual_tool_quaternion_xyzw: tuple[float, float, float, float]
    maximum_joint_error_rad: float
    minimum_clearance_m: float
    environment_collision_count: int
    self_collision_count: int
    tracked_body_poses: tuple[TrackedBodyPose, ...] = ()


@dataclass(frozen=True)
class MotionExecutionResult:
    """Complete trace and gate result for a sequence of motor segments."""

    trace: tuple[MotionTraceRow, ...]
    segment_reached: tuple[tuple[str, bool], ...]
    minimum_clearance_m: float
    environment_collision_count: int
    self_collision_count: int
    all_states_finite: bool
    gate_passed: bool
    failure_reason: str


def _joint_positions(
    robot_id: int,
    client_id: int,
    indices: Sequence[int],
    physics: Any,
) -> tuple[float, ...]:
    return tuple(
        float(
            physics.getJointState(
                robot_id,
                index,
                physicsClientId=client_id,
            )[0]
        )
        for index in indices
    )


def _positive_joint_forces(
    robot_id: int,
    client_id: int,
    indices: Sequence[int],
    physics: Any,
) -> tuple[float, ...]:
    forces = tuple(
        float(
            physics.getJointInfo(
                robot_id,
                index,
                physicsClientId=client_id,
            )[10]
        )
        for index in indices
    )
    if any(not math.isfinite(force) or force <= 0.0 for force in forces):
        raise ValueError("Panda motor force limits must be finite and positive")
    return forces


def _clearance_sample(
    *,
    robot_id: int,
    client_id: int,
    model: PandaModelInfo,
    environment_body_ids: Sequence[int],
    allowed_environment_link_pairs: Sequence[tuple[int, int]],
    clearance_m: float,
    physics: Any,
) -> tuple[float, int, int]:
    minimum = clearance_m
    environment_collisions = 0
    self_collisions = 0
    allowed = {
        (int(link_index), int(body_id))
        for link_index, body_id in allowed_environment_link_pairs
    }
    adjacent = set(model.adjacent_link_pairs)
    physics.performCollisionDetection(physicsClientId=client_id)
    for body_id in environment_body_ids:
        contacts = physics.getClosestPoints(
            bodyA=robot_id,
            bodyB=int(body_id),
            distance=clearance_m,
            physicsClientId=client_id,
        )
        for contact in contacts:
            if (int(contact[3]), int(body_id)) in allowed:
                continue
            distance = float(contact[8])
            minimum = min(minimum, distance)
            if distance < clearance_m:
                environment_collisions += 1

    seen_pairs = set()
    contacts = physics.getClosestPoints(
        bodyA=robot_id,
        bodyB=robot_id,
        distance=clearance_m,
        physicsClientId=client_id,
    )
    for contact in contacts:
        pair = tuple(sorted((int(contact[3]), int(contact[4]))))
        if (
            pair[0] == pair[1]
            or pair in adjacent
            or pair in seen_pairs
        ):
            continue
        seen_pairs.add(pair)
        distance = float(contact[8])
        minimum = min(minimum, distance)
        if distance < clearance_m:
            self_collisions += 1
    return minimum, environment_collisions, self_collisions


def execute_joint_motion(
    *,
    robot_id: int,
    client_id: int,
    model: PandaModelInfo,
    segments: Sequence[MotionSegment],
    environment_body_ids: Sequence[int],
    allowed_environment_link_pairs: Sequence[tuple[int, int]] = (),
    tracked_body_ids: Sequence[int] = (),
    config: MotionConfig = MotionConfig(),
    physics: Any = p,
) -> MotionExecutionResult:
    """Command named arm targets while stepping and auditing real dynamics."""

    if len(model.arm_joint_indices) != 7:
        raise ValueError("Panda model must expose seven arm joints")
    if len(model.finger_joint_indices) != 2:
        raise ValueError("Panda model must expose two finger joints")
    if not segments:
        raise ValueError("motion execution requires at least one segment")
    arm_forces = _positive_joint_forces(
        robot_id, client_id, model.arm_joint_indices, physics
    )
    finger_forces = _positive_joint_forces(
        robot_id, client_id, model.finger_joint_indices, physics
    )
    trace = []
    reached_rows = []
    total_environment_collisions = 0
    total_self_collisions = 0
    global_minimum = config.clearance_m
    all_finite = True
    step_index = 0

    def command_and_sample(
        phase: str,
        command: tuple[float, ...],
    ) -> tuple[float, ...]:
        nonlocal step_index
        nonlocal total_environment_collisions
        nonlocal total_self_collisions
        nonlocal global_minimum
        nonlocal all_finite
        physics.setJointMotorControlArray(
            robot_id,
            model.arm_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=command,
            forces=arm_forces,
            physicsClientId=client_id,
        )
        physics.setJointMotorControlArray(
            robot_id,
            model.finger_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=(0.04, 0.04),
            forces=finger_forces,
            physicsClientId=client_id,
        )
        physics.stepSimulation(physicsClientId=client_id)
        step_index += 1
        actual = _joint_positions(
            robot_id, client_id, model.arm_joint_indices, physics
        )
        fingers = _joint_positions(
            robot_id, client_id, model.finger_joint_indices, physics
        )
        link_state = physics.getLinkState(
            robot_id,
            model.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        tool_position = tuple(float(value) for value in link_state[4])
        tool_quaternion = tuple(float(value) for value in link_state[5])
        tracked_poses = tuple(
            TrackedBodyPose(
                body_id=int(body_id),
                position=tuple(float(value) for value in pose[0]),
                quaternion_xyzw=tuple(float(value) for value in pose[1]),
            )
            for body_id in tracked_body_ids
            for pose in (
                physics.getBasePositionAndOrientation(
                    int(body_id),
                    physicsClientId=client_id,
                ),
            )
        )
        maximum_error = max(
            abs(actual_value - command_value)
            for actual_value, command_value in zip(actual, command)
        )
        minimum, environment_count, self_count = _clearance_sample(
            robot_id=robot_id,
            client_id=client_id,
            model=model,
            environment_body_ids=environment_body_ids,
            allowed_environment_link_pairs=allowed_environment_link_pairs,
            clearance_m=config.clearance_m,
            physics=physics,
        )
        global_minimum = min(global_minimum, minimum)
        total_environment_collisions += environment_count
        total_self_collisions += self_count
        values = (
            *command,
            *actual,
            *fingers,
            *tool_position,
            *tool_quaternion,
            *(
                value
                for tracked_pose in tracked_poses
                for value in (
                    *tracked_pose.position,
                    *tracked_pose.quaternion_xyzw,
                )
            ),
        )
        row_finite = all(math.isfinite(value) for value in values)
        all_finite = all_finite and row_finite
        trace.append(
            MotionTraceRow(
                step=step_index,
                phase=phase,
                commanded_arm_positions=command,
                actual_arm_positions=actual,
                actual_finger_positions=(fingers[0], fingers[1]),
                actual_tool_position=tool_position,
                actual_tool_quaternion_xyzw=tool_quaternion,
                maximum_joint_error_rad=float(maximum_error),
                minimum_clearance_m=float(minimum),
                environment_collision_count=environment_count,
                self_collision_count=self_count,
                tracked_body_poses=tracked_poses,
            )
        )
        return actual

    for segment in segments:
        start = np.asarray(
            _joint_positions(
                robot_id,
                client_id,
                model.arm_joint_indices,
                physics,
            ),
            dtype=np.float64,
        )
        target = np.asarray(segment.target_arm_positions, dtype=np.float64)
        actual = tuple(float(value) for value in start)
        for fraction in np.linspace(
            1.0 / config.steps_per_segment,
            1.0,
            config.steps_per_segment,
        ):
            command_values = start + fraction * (target - start)
            command = tuple(float(value) for value in command_values)
            actual = command_and_sample(segment.name, command)
        reached = bool(
            max(
                abs(actual_value - target_value)
                for actual_value, target_value in zip(actual, target)
            )
            <= config.joint_tolerance_rad
        )
        for _ in range(config.settle_steps):
            if reached:
                break
            actual = command_and_sample(
                segment.name,
                tuple(float(value) for value in target),
            )
            reached = bool(
                max(
                    abs(actual_value - target_value)
                    for actual_value, target_value in zip(actual, target)
                )
                <= config.joint_tolerance_rad
            )
        reached_rows.append((segment.name, reached))

    failures = []
    failures.extend(
        f"segment_not_reached:{name}"
        for name, reached in reached_rows
        if not reached
    )
    if total_environment_collisions:
        failures.append("environment_clearance_failed")
    if total_self_collisions:
        failures.append("self_clearance_failed")
    if not all_finite:
        failures.append("non_finite_state")
    return MotionExecutionResult(
        trace=tuple(trace),
        segment_reached=tuple(reached_rows),
        minimum_clearance_m=float(global_minimum),
        environment_collision_count=total_environment_collisions,
        self_collision_count=total_self_collisions,
        all_states_finite=all_finite,
        gate_passed=not failures,
        failure_reason=";".join(failures),
    )
