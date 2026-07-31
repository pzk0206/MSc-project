import numpy as np
import pybullet as p
import pytest

from src.simulation.pybullet.camera import (
    CameraConfig,
    capture_camera_frame,
)
from src.simulation.pybullet.scene import (
    PyBulletScene,
    SceneConfig,
    SceneObjectConfig,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)


def test_direct_scene_contains_visible_object_and_closes() -> None:
    scene = PyBulletScene(SceneConfig(gui=False, seed=42)).connect()
    try:
        assert scene.client_id >= 0
        assert scene.bodies.robot >= 0
        assert scene.bodies.table >= 0
        assert scene.bodies.target_object >= 0
        scene.step(10)
        frame = capture_camera_frame(
            scene.client_id,
            CameraConfig(width=160, height=120),
            scene.renderer,
        )
        assert frame.rgb.shape == (120, 160, 3)
        assert frame.depth_m.shape == (120, 160)
        assert np.all(np.isfinite(frame.depth_m))
        object_mask = (
            frame.segmentation & ((1 << 24) - 1)
        ) == scene.bodies.target_object
        assert np.any(object_mask)
    finally:
        scene.close()

    assert not scene.is_connected


def test_scene_rejects_object_urdf_outside_pybullet_data() -> None:
    scene = PyBulletScene(
        SceneConfig(gui=False, object_urdf="../plane.urdf")
    )

    with pytest.raises(
        ValueError,
        match="object_urdf must resolve inside pybullet_data",
    ):
        scene.connect()

    assert not scene.is_connected


def test_scene_context_manager_closes_and_reset_clears_bodies() -> None:
    scene = PyBulletScene(SceneConfig(gui=False))

    with scene as connected:
        assert connected.is_connected
        connected.reset()
        assert connected.is_connected
        with pytest.raises(RuntimeError, match="not connected"):
            _ = connected.bodies

    assert not scene.is_connected


def test_fixed_multi_object_scene_exposes_visible_named_targets() -> None:
    config = SceneConfig(
        gui=False,
        object_name="duck",
        object_position=(0.52, -0.18, 0.67),
        object_yaw_degrees=0.0,
        object_rgba=(1.0, 0.8, 0.0, 1.0),
        additional_objects=(
            SceneObjectConfig(
                name="cube",
                urdf="cube_small.urdf",
                position=(0.48, 0.0, 0.66),
                rgba=(0.9, 0.1, 0.1, 1.0),
            ),
            SceneObjectConfig(
                name="sphere",
                urdf="sphere_small.urdf",
                position=(0.52, 0.18, 0.67),
                rgba=(0.1, 0.8, 0.1, 1.0),
            ),
        ),
    )

    with PyBulletScene(config) as scene:
        scene.step(10)
        assert set(scene.object_body_ids) == {"duck", "cube", "sphere"}
        assert len(set(scene.object_body_ids.values())) == 3
        poses = scene.object_poses()
        assert set(poses) == {"duck", "cube", "sphere"}
        assert set(poses["duck"]) == {"position", "orientation"}
        assert len(poses["duck"]["position"]) == 3
        assert len(poses["duck"]["orientation"]) == 4

        frame = capture_camera_frame(
            scene.client_id,
            CameraConfig(width=320, height=240),
            scene.renderer,
        )
        body_ids = frame.segmentation & ((1 << 24) - 1)
        for body_id in scene.object_body_ids.values():
            assert np.any(body_ids == body_id)


def test_scene_rejects_duplicate_object_names_and_closes() -> None:
    scene = PyBulletScene(
        SceneConfig(
            gui=False,
            object_name="duck",
            additional_objects=(
                SceneObjectConfig(
                    name="duck",
                    urdf="cube_small.urdf",
                    position=(0.48, 0.0, 0.66),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="duplicate scene object name: duck"):
        scene.connect()

    assert not scene.is_connected


def test_scene_can_opt_in_to_robot_self_collision() -> None:
    with PyBulletScene(
        SceneConfig(gui=False, robot_self_collision=True)
    ) as scene:
        assert scene.bodies.robot >= 0


def test_fixed_study_robot_does_not_penetrate_table_or_eject_targets() -> None:
    scene = PyBulletScene(
        fixed_scene_config(MultiObjectStudyConfig(device="cpu"))
    ).connect()
    try:
        p.performCollisionDetection(physicsClientId=scene.client_id)
        robot_table_contacts = p.getContactPoints(
            bodyA=scene.bodies.robot,
            bodyB=scene.bodies.table,
            physicsClientId=scene.client_id,
        )
        assert not robot_table_contacts

        scene.step(60)
        frame = capture_camera_frame(
            scene.client_id,
            CameraConfig(width=320, height=240),
            scene.renderer,
        )
        body_ids = frame.segmentation & ((1 << 24) - 1)
        for body_id in scene.object_body_ids.values():
            assert np.any(
                (frame.segmentation >= 0) & (body_ids == body_id)
            )
    finally:
        scene.close()
