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
class SceneConfig:
    """Fixed scene resources and initial poses."""

    gui: bool = False
    seed: int = 42
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    object_urdf: str = "duck_vhacd.urdf"
    object_position: tuple[float, float, float] = (0.55, 0.0, 0.66)
    object_yaw_degrees: float = 20.0


@dataclass(frozen=True)
class SceneBodies:
    """Body IDs loaded into one physics client."""

    plane: int
    table: int
    robot: int
    target_object: int


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

    def _resolve_object_urdf(self) -> Path:
        object_path = Path(self.config.object_urdf)
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

    def connect(self) -> "PyBulletScene":
        if self.is_connected:
            return self

        connection_mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = int(p.connect(connection_mode))
        if self.client_id < 0:
            raise RuntimeError("failed to connect to PyBullet")

        try:
            data_root = pybullet_data.getDataPath()
            object_urdf = self._resolve_object_urdf()
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
                physicsClientId=self.client_id,
            )
            object_orientation = p.getQuaternionFromEuler(
                (0.0, 0.0, math.radians(self.config.object_yaw_degrees))
            )
            target_object = p.loadURDF(
                str(object_urdf),
                basePosition=self.config.object_position,
                baseOrientation=object_orientation,
                physicsClientId=self.client_id,
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
