import numpy as np
import pytest
import pybullet

from src.simulation.pybullet.camera import (
    CameraConfig,
    capture_camera_frame,
    linearize_depth,
)


def test_camera_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="width and height"):
        CameraConfig(width=0).validate()
    with pytest.raises(ValueError, match="near.*far"):
        CameraConfig(near=1.0, far=0.5).validate()
    with pytest.raises(ValueError, match="fov"):
        CameraConfig(fov_degrees=180.0).validate()


def test_linearize_depth_maps_clip_planes_and_is_monotonic() -> None:
    buffer = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    depth = linearize_depth(buffer, near=0.1, far=10.0)

    assert depth[0] == pytest.approx(0.1)
    assert depth[-1] == pytest.approx(10.0, rel=1e-5)
    assert np.all(np.isfinite(depth))
    assert np.all(np.diff(depth) > 0)


def test_capture_preserves_rgb_channels_shapes_and_client_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    rgba = np.array(
        [[[10, 20, 30, 255], [40, 50, 60, 255]]],
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        pybullet,
        "computeViewMatrix",
        lambda **kwargs: [1.0] * 16,
    )
    monkeypatch.setattr(
        pybullet,
        "computeProjectionMatrixFOV",
        lambda **kwargs: [2.0] * 16,
    )

    def fake_get_camera_image(**kwargs):
        captured.update(kwargs)
        return (
            2,
            1,
            rgba,
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[3, -1]], dtype=np.int32),
        )

    monkeypatch.setattr(pybullet, "getCameraImage", fake_get_camera_image)

    frame = capture_camera_frame(
        client_id=17,
        config=CameraConfig(width=2, height=1, near=0.1, far=1.0),
        renderer=9,
    )

    assert frame.rgb.tolist() == [[[10, 20, 30], [40, 50, 60]]]
    assert frame.rgb.shape == (1, 2, 3)
    assert frame.depth_m.shape == (1, 2)
    assert frame.segmentation.shape == (1, 2)
    assert captured["physicsClientId"] == 17
    assert captured["renderer"] == 9
