"""Pure OpenCV visualizations for PyBullet perception outputs."""

from __future__ import annotations

from collections.abc import Mapping
import math

import cv2
import numpy as np


_ENTITY_COLORS_BGR = {
    "duck": (0, 255, 255),
    "cube": (0, 0, 255),
    "sphere": (0, 255, 0),
    "robot": (255, 0, 255),
}
_DETECTION_COLOR_BGR = (255, 0, 0)


def _validated_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb_array = np.asarray(rgb)
    if (
        rgb_array.ndim != 3
        or rgb_array.shape[2] != 3
        or rgb_array.dtype != np.uint8
    ):
        raise ValueError("RGB image must have shape (H, W, 3) and dtype uint8")
    return rgb_array


def validate_detection_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Validate one inclusive image-coordinate detection box."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("detection box coordinates must be finite")
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detection box must have positive area")
    if (
        x1 < 0.0
        or y1 < 0.0
        or x2 > image_width - 1
        or y2 > image_height - 1
    ):
        raise ValueError("detection box must remain inside image bounds")
    return x1, y1, x2, y2


def grasp_box_points(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> np.ndarray:
    """Return four OpenCV box points for a centre-format grasp."""

    values = np.asarray(
        [center_x, center_y, width, height, angle_degrees],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("grasp parameters must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("grasp width and height must be positive")
    return cv2.boxPoints(
        (
            (float(center_x), float(center_y)),
            (float(width), float(height)),
            float(angle_degrees),
        )
    )


def depth_to_uint8(
    depth_m: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    """Map metric depth between the camera clip planes to uint8."""

    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    depth = np.asarray(depth_m, dtype=np.float32)
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth contains non-finite values")
    normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def segmentation_to_bgr(segmentation: np.ndarray) -> np.ndarray:
    """Color body IDs deterministically while keeping background black."""

    segmentation_array = np.asarray(segmentation, dtype=np.int64)
    if segmentation_array.ndim != 2:
        raise ValueError("segmentation must be a two-dimensional array")
    output = np.zeros((*segmentation_array.shape, 3), dtype=np.uint8)
    foreground = segmentation_array >= 0
    body_ids = segmentation_array[foreground] & ((1 << 24) - 1)
    output[foreground, 0] = (body_ids * 67 + 53) % 256
    output[foreground, 1] = (body_ids * 97 + 101) % 256
    output[foreground, 2] = (body_ids * 131 + 193) % 256
    return output


def draw_ground_truth_boxes(
    rgb: np.ndarray,
    boxes: Mapping[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw fixed-color evaluation truth boxes on an RGB frame."""

    rgb_array = _validated_rgb(rgb)
    image = cv2.cvtColor(rgb_array.copy(), cv2.COLOR_RGB2BGR)
    height, width = rgb_array.shape[:2]
    for name, box in boxes.items():
        x1, y1, x2, y2 = validate_detection_box(box, width, height)
        color = _ENTITY_COLORS_BGR.get(name, (255, 255, 255))
        start = int(round(x1)), int(round(y1))
        end = int(round(x2)), int(round(y2))
        cv2.rectangle(image, start, end, color, 1)
        cv2.putText(
            image,
            name,
            (start[0], max(0, start[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def draw_target_evaluation(
    rgb: np.ndarray,
    requested_target: str,
    prompt: str,
    detection_box: tuple[int, int, int, int] | None,
    ground_truth_boxes: Mapping[str, tuple[int, int, int, int]],
    best_matching_target: str | None,
    score: float | None,
) -> np.ndarray:
    """Draw one prompt's requested truth, best match, and predicted box."""

    rgb_array = _validated_rgb(rgb)
    image = cv2.cvtColor(rgb_array.copy(), cv2.COLOR_RGB2BGR)
    height, width = rgb_array.shape[:2]

    highlighted = [
        ("requested", requested_target),
        ("best", best_matching_target),
    ]
    for role, name in highlighted:
        if name is None or name not in ground_truth_boxes:
            continue
        x1, y1, x2, y2 = validate_detection_box(
            ground_truth_boxes[name],
            width,
            height,
        )
        color = _ENTITY_COLORS_BGR.get(name, (255, 255, 255))
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            1 if role == "requested" else 2,
        )

    if detection_box is not None:
        x1, y1, x2, y2 = validate_detection_box(
            detection_box,
            width,
            height,
        )
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            _DETECTION_COLOR_BGR,
            1,
        )

    score_text = "none" if score is None else f"{score:.3f}"
    cv2.putText(
        image,
        (
            f"{prompt} | requested={requested_target} | "
            f"best={best_matching_target or 'none'} | score={score_text}"
        ),
        (5, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_prediction(
    rgb: np.ndarray,
    localization_box: tuple[float, float, float, float],
    grasp: Mapping[str, float],
    prompt: str,
    confidence: float,
    backend: str,
) -> np.ndarray:
    """Draw localization and centre-format grasp prediction in BGR."""

    rgb_array = _validated_rgb(rgb)
    image_height, image_width = rgb_array.shape[:2]
    x1, y1, x2, y2 = validate_detection_box(
        localization_box,
        image_width,
        image_height,
    )
    points = grasp_box_points(
        center_x=float(grasp["center_x"]),
        center_y=float(grasp["center_y"]),
        width=float(grasp["width"]),
        height=float(grasp["height"]),
        angle_degrees=float(grasp["angle_degrees"]),
    )

    image = cv2.cvtColor(rgb_array.copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        image,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (0, 255, 255),
        2,
    )
    cv2.polylines(
        image,
        [np.rint(points).astype(np.int32)],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    center = (
        int(round(float(grasp["center_x"]))),
        int(round(float(grasp["center_y"]))),
    )
    cv2.circle(image, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
    angle_radians = math.radians(float(grasp["angle_degrees"]))
    half_width = float(grasp["width"]) / 2.0
    direction_end = (
        int(round(center[0] + half_width * math.cos(angle_radians))),
        int(round(center[1] + half_width * math.sin(angle_radians))),
    )
    cv2.line(
        image,
        center,
        direction_end,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"{prompt} | score={confidence:.3f} | backend={backend}",
        (10, image_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image
