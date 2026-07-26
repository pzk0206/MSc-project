import numpy as np
import pytest

from src.simulation.pybullet.visualization import (
    depth_to_uint8,
    draw_prediction,
    grasp_box_points,
    segmentation_to_bgr,
    validate_detection_box,
)


def test_grasp_box_points_for_axis_aligned_rectangle() -> None:
    points = grasp_box_points(
        center_x=50.0,
        center_y=40.0,
        width=20.0,
        height=10.0,
        angle_degrees=0.0,
    )

    assert points.shape == (4, 2)
    assert set(map(tuple, points.astype(int))) == {
        (40, 35),
        (60, 35),
        (60, 45),
        (40, 45),
    }


def test_visualization_rejects_invalid_grasp_and_detection_box() -> None:
    with pytest.raises(ValueError, match="finite"):
        grasp_box_points(np.nan, 1.0, 2.0, 3.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        grasp_box_points(1.0, 1.0, -2.0, 3.0, 0.0)
    with pytest.raises(ValueError, match="positive area"):
        validate_detection_box((5.0, 5.0, 5.0, 8.0), 20, 20)
    with pytest.raises(ValueError, match="image bounds"):
        validate_detection_box((-1.0, 2.0, 8.0, 9.0), 20, 20)


def test_depth_visualization_has_uint8_range() -> None:
    result = depth_to_uint8(
        np.array([[0.1, 0.5, 1.0]], dtype=np.float32),
        near=0.1,
        far=1.0,
    )

    assert result.dtype == np.uint8
    assert result.tolist() == [[0, 113, 255]]


def test_segmentation_visualization_keeps_background_black() -> None:
    result = segmentation_to_bgr(
        np.array([[-1, 0, 1]], dtype=np.int32)
    )

    assert result.shape == (1, 3, 3)
    assert result[0, 0].tolist() == [0, 0, 0]
    assert result[0, 1].tolist() != result[0, 2].tolist()


def test_draw_prediction_returns_bgr_without_mutating_rgb() -> None:
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    original = rgb.copy()

    drawn = draw_prediction(
        rgb,
        (20.0, 20.0, 100.0, 80.0),
        {
            "center_x": 60.0,
            "center_y": 50.0,
            "width": 30.0,
            "height": 12.0,
            "angle_degrees": 30.0,
        },
        prompt="small object",
        confidence=0.8,
        backend="geometry",
    )

    assert np.array_equal(rgb, original)
    assert drawn.shape == rgb.shape
    assert drawn[0, 0].tolist() == [0, 0, 255]
    assert np.any(drawn != np.array([0, 0, 255], dtype=np.uint8))
