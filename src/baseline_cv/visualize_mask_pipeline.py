"""
可视化传统 CV baseline 的 mask 生成过程。

这个脚本用于回答一个很具体的问题：

    原图经过色彩过滤后变成什么样？
    再经过 border 过滤后变成什么样？
    border 过滤到底删掉了哪些区域？

输出是一张横向四联图：

    1. 原始 RGB 图
    2. 色彩/亮度过滤得到的初始 mask，以红色半透明叠加在原图上
    3. border 过滤 + morphology + 轮廓选择后的最终目标区域，以绿色半透明叠加在原图上
    4. 黑红差异图，用黑色显示最终保留目标，红色显示被过滤掉/未选中的区域

注意：
这里的 "border" 指的是图像边界过滤。
Cornell 图片边缘有时会出现桌边、背景、鞋子等干扰区域，
所以 baseline 会把图像最外圈的一小圈像素清掉。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RGB_PATH = Path("data/raw/cornell/01/pcd0100r.png")
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/baseline_cv/visualizations/bordereffect/pcd0100_mask_pipeline_overlay.png"
)


def create_color_mask(rgb_bgr: np.ndarray) -> np.ndarray:
    """
    只做色彩/亮度过滤，不做 border 过滤。

    baseline 的朴素假设是：
    Cornell 图像大多是浅色背景，物体通常比背景更暗或颜色更明显。

    因此这里使用 HSV：

    - saturation > 25：颜色比较明显；
    - value < 210：亮度比较低，不像白色背景。

    满足任意一个条件，就认为可能是物体区域。
    """

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 25) | (value < 210)).astype(np.uint8) * 255

    return mask


def apply_border_filter(mask: np.ndarray, border: int = 8) -> np.ndarray:
    """
    去掉图像最外圈 border 像素。

    这样做是为了减少图像边缘的背景、桌边、鞋子等干扰。
    """

    filtered_mask = mask.copy()

    filtered_mask[:border, :] = 0
    filtered_mask[-border:, :] = 0
    filtered_mask[:, :border] = 0
    filtered_mask[:, -border:] = 0

    return filtered_mask


def apply_morphology(mask: np.ndarray) -> np.ndarray:
    """
    对 mask 做形态学清理。

    MORPH_OPEN:
        先腐蚀再膨胀，用来去掉小噪点。

    MORPH_CLOSE:
        先膨胀再腐蚀，用来填补小空洞。
    """

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def choose_object_contour(mask: np.ndarray) -> np.ndarray | None:
    """
    从清理后的 mask 中选择最像目标物体的轮廓。

    这段逻辑和传统 CV baseline 里的思想一致：
    - 去掉太小的噪声；
    - 去掉贴着边界的大块背景；
    - 在候选里偏向面积较大、靠近图像中心的区域。
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    image_height, image_width = mask.shape
    image_center = np.array([image_width / 2.0, image_height / 2.0])

    best_contour = None
    best_score = -float("inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_border = (
            x <= 8
            or y <= 8
            or x + w >= image_width - 8
            or y + h >= image_height - 8
        )
        if touches_border:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        contour_center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        )
        distance_to_center = np.linalg.norm(contour_center - image_center)
        score = area / (1.0 + distance_to_center / 120.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def contour_to_mask(mask_shape: tuple[int, int], contour: np.ndarray | None) -> np.ndarray:
    """
    把选中的轮廓转换成单通道 mask。
    """

    selected_mask = np.zeros(mask_shape, dtype=np.uint8)

    if contour is not None:
        cv2.drawContours(selected_mask, [contour], contourIdx=-1, color=255, thickness=-1)

    return selected_mask


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    """
    把单通道 mask 以半透明颜色叠加到原图上。

    参数：
        image:
            原始 BGR 图片。

        mask:
            单通道 mask。非零区域会被上色。

        color:
            BGR 颜色。例如：
            红色是 (0, 0, 255)，绿色是 (0, 255, 0)。

        alpha:
            颜色叠加透明度，越大颜色越明显。
    """

    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:, :] = color

    mask_bool = mask > 0

    # 如果 mask 里没有任何被选中的像素，直接返回原图。
    # 批量处理时某些样本可能暂时选不出目标区域，不能因此让整个脚本中断。
    if not np.any(mask_bool):
        return overlay

    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool],
        1.0 - alpha,
        color_layer[mask_bool],
        alpha,
        0,
    )

    return overlay


def build_removed_region_panel(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
) -> np.ndarray:
    """
    构建黑红差异图。

    黑色：
        最终选中的目标区域。

    红色：
        没有被最终选中的区域。
    """

    panel = np.zeros((*before_mask.shape, 3), dtype=np.uint8)
    panel[:, :] = (0, 0, 255)
    panel[after_mask > 0] = (0, 0, 0)

    return panel


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """
    在图片左上角写标题。
    """

    canvas = image.copy()

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def build_four_panel_visualization(rgb_bgr: np.ndarray, border: int) -> tuple:
    """
    构建四宫格可视化结果。
    """

    color_mask = create_color_mask(rgb_bgr)
    border_filtered_mask = apply_border_filter(color_mask, border=border)
    morphology_mask = apply_morphology(border_filtered_mask)
    selected_contour = choose_object_contour(morphology_mask)
    selected_object_mask = contour_to_mask(morphology_mask.shape, selected_contour)

    original_panel = add_title(rgb_bgr, "1. Original RGB")
    color_mask_panel = add_title(
        overlay_mask(rgb_bgr, color_mask, color=(0, 0, 255), alpha=0.55),
        "2. Color/Brightness filter",
    )
    border_mask_panel = add_title(
        overlay_mask(rgb_bgr, selected_object_mask, color=(0, 255, 0), alpha=0.60),
        f"3. Final object after border",
    )
    difference_panel = add_title(
        build_removed_region_panel(color_mask, selected_object_mask),
        "4. Removed / kept difference",
    )

    # 横向拼接，得到类似论文/slide 中常见的流程展示。
    four_panel = np.hstack(
        [original_panel, color_mask_panel, border_mask_panel, difference_panel]
    )

    return four_panel, color_mask, border_filtered_mask, morphology_mask, selected_object_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize color filtering and border filtering for one Cornell RGB image."
    )
    parser.add_argument(
        "--rgb-path",
        type=Path,
        default=DEFAULT_RGB_PATH,
        help="Path to Cornell RGB image, e.g. data/raw/cornell/01/pcd0100r.png",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the four-panel visualization.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=8,
        help="Number of pixels removed from each image border.",
    )

    args = parser.parse_args()

    image = cv2.imread(str(args.rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图片：{args.rgb_path}")

    (
        four_panel,
        color_mask,
        border_mask,
        morphology_mask,
        selected_object_mask,
    ) = build_four_panel_visualization(
        image,
        border=args.border,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_success = cv2.imwrite(str(args.output_path), four_panel)
    if not save_success:
        raise RuntimeError(f"保存可视化图片失败：{args.output_path}")

    color_pixels = int(np.count_nonzero(color_mask))
    border_pixels = int(np.count_nonzero(border_mask))
    morphology_pixels = int(np.count_nonzero(morphology_mask))
    selected_object_pixels = int(np.count_nonzero(selected_object_mask))
    removed_pixels = color_pixels - border_pixels

    print("mask pipeline 可视化完成")
    print(f"输入图片：{args.rgb_path}")
    print(f"输出图片：{args.output_path}")
    print(f"border 大小：{args.border} px")
    print(f"色彩过滤后 mask 像素数：{color_pixels}")
    print(f"border 过滤后 mask 像素数：{border_pixels}")
    print(f"border 删除像素数：{removed_pixels}")
    print(f"形态学清理后 mask 像素数：{morphology_pixels}")
    print(f"最终选中目标区域像素数：{selected_object_pixels}")


if __name__ == "__main__":
    main()
