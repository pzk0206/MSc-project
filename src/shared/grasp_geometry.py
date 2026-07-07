"""
2D 抓取矩形的几何转换工具。

Cornell 原始标注格式是“四个角点”：

    (x1, y1), (x2, y2), (x3, y3), (x4, y4)

但是后面做 baseline、评估和画预测结果时，更常用的是中心格式：

    center_x, center_y, width, height, angle

其中：

- center_x, center_y
  抓取矩形中心点；

- width
  抓取矩形的长边长度，通常可以理解为夹爪开口方向上的跨度；

- height
  抓取矩形的短边长度，通常可以理解为抓取矩形的厚度；

- angle
  长边相对于图像 x 轴的角度。

注意：
这里仍然是 2D 图像坐标，不是机器人真实世界坐标。
"""

from __future__ import annotations

import math

import numpy as np


def normalize_angle_radians(angle: float) -> float:
    """
    把角度归一化到 [-pi/2, pi/2)。

    为什么要这样做？
    ----------------
    对平行夹爪抓取来说，角度 theta 和 theta + pi 通常表示同一个夹爪方向。

    例如：
        0 度 和 180 度

    在 2D 抓取矩形里可以看成等价方向。
    所以我们把角度压到一个固定范围，方便后面比较预测框和真实框。
    """

    while angle >= math.pi / 2:
        angle -= math.pi

    while angle < -math.pi / 2:
        angle += math.pi

    return angle


def rectangle_to_center_format(rectangle: np.ndarray) -> dict:
    """
    把一个 Cornell 四点抓取矩形转换成中心格式。

    参数
    ----------
    rectangle:
        shape 为 (4, 2) 的 numpy 数组。

        例如：

            [
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4],
            ]

    返回
    ----------
    dict:
        包含：

        - center_x
        - center_y
        - width
        - height
        - angle_radians
        - angle_degrees

    这里默认用矩形长边作为 angle 的方向。
    """

    rectangle = np.asarray(rectangle, dtype=np.float32)

    if rectangle.shape != (4, 2):
        raise ValueError(f"rectangle 必须是 shape=(4, 2)，实际是 {rectangle.shape}")

    # 中心点就是 4 个角点 x/y 坐标的平均值。
    center_x = float(np.mean(rectangle[:, 0]))
    center_y = float(np.mean(rectangle[:, 1]))

    # 取四个点，方便后面计算边长。
    point_0 = rectangle[0]
    point_1 = rectangle[1]
    point_2 = rectangle[2]
    point_3 = rectangle[3]

    # Cornell 标注一般按矩形顺序给出四个角点。
    # 因此相邻点之间的距离就是矩形边长。
    edge_01 = point_1 - point_0
    edge_12 = point_2 - point_1
    edge_23 = point_3 - point_2
    edge_30 = point_0 - point_3

    length_01 = float(np.linalg.norm(edge_01))
    length_12 = float(np.linalg.norm(edge_12))
    length_23 = float(np.linalg.norm(edge_23))
    length_30 = float(np.linalg.norm(edge_30))

    # 对边理论上长度应该接近：
    # length_01 ≈ length_23
    # length_12 ≈ length_30
    #
    # 为了对小数标注更稳，我们用对边平均值。
    side_a_length = (length_01 + length_23) / 2.0
    side_b_length = (length_12 + length_30) / 2.0

    # 选择长边作为 width 和 angle 的方向。
    # 如果 side_a 更长，方向使用 point_0 -> point_1。
    # 如果 side_b 更长，方向使用 point_1 -> point_2。
    if side_a_length >= side_b_length:
        width = side_a_length
        height = side_b_length
        direction = edge_01
    else:
        width = side_b_length
        height = side_a_length
        direction = edge_12

    # atan2(dy, dx) 会返回向量相对 x 轴的角度。
    angle_radians = math.atan2(float(direction[1]), float(direction[0]))
    angle_radians = normalize_angle_radians(angle_radians)
    angle_degrees = math.degrees(angle_radians)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": float(width),
        "height": float(height),
        "angle_radians": float(angle_radians),
        "angle_degrees": float(angle_degrees),
    }


def rectangles_to_center_format(rectangles: np.ndarray) -> list[dict]:
    """
    批量转换多个抓取矩形。

    参数
    ----------
    rectangles:
        shape 为 (N, 4, 2) 的数组。

    返回
    ----------
    list[dict]:
        每个 dict 对应一个抓取矩形的中心格式。
    """

    rectangles = np.asarray(rectangles, dtype=np.float32)

    if rectangles.ndim != 3 or rectangles.shape[1:] != (4, 2):
        raise ValueError(
            f"rectangles 必须是 shape=(N, 4, 2)，实际是 {rectangles.shape}"
        )

    return [rectangle_to_center_format(rectangle) for rectangle in rectangles]


def main() -> None:
    """
    一个很小的自检示例。

    直接运行：

        python src/shared/grasp_geometry.py

    可以确认几何转换函数是否能正常工作。
    """

    example_rectangle = np.array(
        [
            [253.0, 319.7],
            [309.0, 324.0],
            [306.0, 344.0],
            [251.0, 340.0],
        ],
        dtype=np.float32,
    )

    center_format = rectangle_to_center_format(example_rectangle)

    print("示例抓取矩形转换结果：")
    for key, value in center_format.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
