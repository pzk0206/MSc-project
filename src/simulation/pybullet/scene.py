"""Deterministic PyBullet scene lifecycle for the perception pilot.

This module uses the public PyBullet API and packaged ``pybullet_data``
resources documented by the official Bullet repository:
https://github.com/bulletphysics/bullet3
No third-party grasp-execution code is copied or adapted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class SceneObjectConfig:
    """Configuration for one additional named scene object."""

    name: str
    urdf: str
    position: tuple[float, float, float]
    yaw_degrees: float = 0.0
    rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SceneConfig:
    """Fixed scene resources and initial poses."""

    gui: bool = False
    seed: int = 42
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_self_collision: bool = False
    table_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    object_urdf: str = "duck_vhacd.urdf"
    object_position: tuple[float, float, float] = (0.55, 0.0, 0.66)
    object_yaw_degrees: float = 20.0
    object_name: str = "target_object"
    object_rgba: tuple[float, float, float, float] | None = None
    additional_objects: tuple[SceneObjectConfig, ...] = ()


@dataclass(frozen=True)
class SceneBodies:
    """Body IDs loaded into one physics client."""

    plane: int
    table: int
    robot: int
    target_object: int
    additional_objects: tuple[tuple[str, int], ...] = ()


class PyBulletScene:
    """Own one PyBullet connection and its deterministic pilot scene."""

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self.client_id = -1
        self.renderer = p.ER_TINY_RENDERER
        self._bodies: SceneBodies | None = None

    @property
    def bodies(self) -> SceneBodies:
        if self._bodies is None:
            raise RuntimeError("scene is not connected")
        return self._bodies

    @property
    def is_connected(self) -> bool:
        return self.client_id >= 0 and bool(p.isConnected(self.client_id))

    @property
    def object_body_ids(self) -> dict[str, int]:
        """Map configured object names to their loaded body IDs."""

        bodies = self.bodies
        return {
            self.config.object_name: bodies.target_object,
            **dict(bodies.additional_objects),
        }

    def object_poses(
        self,
    ) -> dict[
        str,
        dict[
            str,
            tuple[float, ...],
        ],
    ]:
        """Return current world pose for every named scene object."""

        poses = {}
        for name, body_id in self.object_body_ids.items():
            position, orientation = p.getBasePositionAndOrientation(
                body_id,
                physicsClientId=self.client_id,
            )
            poses[name] = {
                "position": tuple(float(value) for value in position),
                "orientation": tuple(float(value) for value in orientation),
            }
        return poses

    @staticmethod
    def _resolve_object_urdf(urdf: str) -> Path:
        object_path = Path(urdf)
        data_root = Path(pybullet_data.getDataPath()).resolve()
        if object_path.is_absolute():
            raise ValueError("object_urdf must resolve inside pybullet_data")
        candidate = (data_root / object_path).resolve()
        if not candidate.is_relative_to(data_root):
            raise ValueError("object_urdf must resolve inside pybullet_data")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"PyBullet object URDF does not exist: {candidate}"
            )
        return candidate

    @staticmethod
    def _validate_rgba(
        rgba: tuple[float, float, float, float] | None,
    ) -> None:
        if rgba is None:
            return
        if len(rgba) != 4:
            raise ValueError("object RGBA must contain four values")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rgba):
            raise ValueError("object RGBA values must be finite and within [0, 1]")

    def _object_configs(self) -> tuple[SceneObjectConfig, ...]:
        objects = (
            SceneObjectConfig(
                name=self.config.object_name,
                urdf=self.config.object_urdf,
                position=self.config.object_position,
                yaw_degrees=self.config.object_yaw_degrees,
                rgba=self.config.object_rgba,
            ),
            *self.config.additional_objects,
        )
        names = [obj.name for obj in objects]
        if any(not name.strip() for name in names):
            raise ValueError("object names must be non-empty")
        if len(names) != len(set(names)):
            duplicate = next(
                name for index, name in enumerate(names) if name in names[:index]
            )
            raise ValueError(f"duplicate scene object name: {duplicate}")
        for obj in objects:
            self._validate_rgba(obj.rgba)
            self._resolve_object_urdf(obj.urdf)
        return objects

    def _load_object(self, config: SceneObjectConfig) -> int:
        orientation = p.getQuaternionFromEuler(
            (0.0, 0.0, math.radians(config.yaw_degrees))
        )
        body_id = int(
            p.loadURDF(
                str(self._resolve_object_urdf(config.urdf)),
                basePosition=config.position,
                baseOrientation=orientation,
                physicsClientId=self.client_id,
            )
        )
        if config.rgba is not None:
            p.changeVisualShape(
                body_id,
                -1,
                rgbaColor=config.rgba,
                physicsClientId=self.client_id,
            )
        return body_id

    def connect(self) -> "PyBulletScene":
        if self.is_connected:
            return self

        object_configs = self._object_configs()
        connection_mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = int(p.connect(connection_mode))
        if self.client_id < 0:
            raise RuntimeError("failed to connect to PyBullet")

        try:
            data_root = pybullet_data.getDataPath()
            p.setAdditionalSearchPath(
                data_root,
                physicsClientId=self.client_id,
            )
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(
                *self.config.gravity,
                physicsClientId=self.client_id,
            )
            p.setTimeStep(
                self.config.time_step,
                physicsClientId=self.client_id,
            )
            np.random.seed(self.config.seed)

            plane = p.loadURDF(
                "plane.urdf",
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
            table = p.loadURDF(
                "table/table.urdf",
                basePosition=self.config.table_position,
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
            robot = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=self.config.robot_position,
                useFixedBase=True,
                flags=(
                    p.URDF_USE_SELF_COLLISION
                    if self.config.robot_self_collision
                    else 0
                ),
                physicsClientId=self.client_id,
            )
            target_object = self._load_object(object_configs[0])
            additional_objects = tuple(
                (config.name, self._load_object(config))
                for config in object_configs[1:]
            )
            self.renderer = (
                p.ER_BULLET_HARDWARE_OPENGL
                if self.config.gui
                else p.ER_TINY_RENDERER
            )
            self._bodies = SceneBodies(
                plane=int(plane),
                table=int(table),
                robot=int(robot),
                target_object=int(target_object),
                additional_objects=additional_objects,
            )
        except Exception:
            self.close()
            raise

        return self

    def step(self, count: int = 1) -> None:
        if not self.is_connected:
            raise RuntimeError("scene is not connected")
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client_id)

    def reset(self) -> None:
        if not self.is_connected:
            raise RuntimeError("scene is not connected")
        p.resetSimulation(physicsClientId=self.client_id)
        self._bodies = None

    def close(self) -> None:
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
        self.client_id = -1
        self._bodies = None

    def __enter__(self) -> "PyBulletScene":
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
