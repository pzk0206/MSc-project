import numpy as np
import pytest

from src.simulation.pybullet.camera import (
    CameraConfig,
    capture_camera_frame,
)
from src.simulation.pybullet.scene import PyBulletScene, SceneConfig


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
