"""
运行 VLM 辅助的 2D 抓取矩形预测。

中文说明
--------
这个文件是 VLM 部分的第二阶段。

第一阶段 `run_grounding_dino_localization.py` 只回答：
    “VLM 能不能找到物体？”

这个文件回答：
    “VLM 找到物体以后，能不能帮助我们生成更准确的抓取矩形？”

所以它会读取第一阶段保存的 VLM 目标框，然后把传统 CV 的掩码/轮廓
限制在这个目标框附近，最后仍然用 Cornell 风格标准评估最终抓取框。

这个脚本把前面已经跑出来的 VLM / Grounding DINO 目标定位结果接到抓取框预测上：

    RGB 图像
        -> VLM 目标框
        -> 把传统 CV 掩码限制在 VLM 目标框内
        -> 轮廓几何后端
        -> Cornell 风格抓取矩形评估

它的意义不是“让 VLM 直接画抓取框”，而是测试一个更现实、也更容易解释的方案：

    VLM 负责开放词汇目标定位；
    传统几何后端负责把目标区域转换成抓取矩形。

输出格式尽量和传统 CV 基线保持一致，方便后面直接做论文里的对比表。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_cv.run_cv_baseline import (  # noqa: E402
    create_object_mask,
    predict_grasp_from_contour,
)
from src.shared.cornell_dataset import CornellGraspDataset  # noqa: E402
from src.shared.cornell_evaluation import (  # noqa: E402
    ANGLE_THRESHOLD_DEGREES,
    IOU_THRESHOLD,
    evaluate_prediction,
)
from src.shared.grasp_geometry import rectangles_to_center_format  # noqa: E402


# Cornell 原始数据目录。
DATASET_ROOT = Path("data/raw/cornell")

# 默认读取的 VLM 定位结果。
# 这个 CSV 来自 run_grounding_dino_localization.py。
DEFAULT_LOCALIZATION_CSV = Path(
    "data/processed/vlm/localization/grounding_dino_generic_small_object_predictions.csv"
)

# VLM 辅助抓取的所有结果放在 data/processed/vlm/grasp 下。
OUTPUT_DIR = Path("data/processed/vlm/grasp")

# 每张图的最终抓取预测和评估结果。
PREDICTIONS_CSV = OUTPUT_DIR / "vlm_assisted_grasp_predictions.csv"

# 全量实验的汇总结果，例如成功率、平均 IoU、平均角度误差。
SUMMARY_JSON = OUTPUT_DIR / "vlm_assisted_grasp_summary.json"

# 保存成功/失败案例图片。
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# 成功/失败案例拼图，方便快速肉眼检查。
OVERVIEW_PATH = VISUALIZATION_DIR / "vlm_assisted_success_failure_overview.png"

# 传统 CV 基线的结果，用来做同一批样本上的公平对比。
BASELINE_CV_PREDICTIONS_CSV = Path("data/processed/baseline_cv/cv_baseline_predictions.csv")


def read_localization_csv(csv_path: Path) -> dict[tuple[str, str], dict]:
    """
    读取 VLM 定位结果 CSV。

    返回值用 (object_directory, sample_id) 做 key。
    这样后面遍历 Cornell 数据集时，可以快速找到对应样本的 VLM 目标框。
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 VLM 定位结果：{csv_path}\n"
            "请先运行：python src/vlm/run_grounding_dino_localization.py --samples-per-directory 2"
        )

    localization_by_sample: dict[tuple[str, str], dict] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 一个 Cornell 样本由 object_directory + sample_id 唯一确定。
            # 例如：("01", "pcd0100")。
            key = (row["object_directory"], row["sample_id"])
            localization_by_sample[key] = row

    return localization_by_sample


def parse_vlm_box(row: dict, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """
    从 CSV 行里解析 VLM 目标框。

    Grounding DINO 输出的是 x1, y1, x2, y2。
    这里会把坐标裁剪到图片范围内，避免后续数组切片越界。
    """

    # 如果 VLM 阶段没有检测成功，这里直接返回 None。
    if int(row.get("detected", 0)) != 1:
        return None

    try:
        # CSV 里读出来都是字符串，所以要转成 float。
        x1 = float(row["box_x1"])
        y1 = float(row["box_y1"])
        x2 = float(row["box_x2"])
        y2 = float(row["box_y2"])
    except (TypeError, ValueError):
        return None

    image_height, image_width = image_shape[:2]

    # Grounding DINO 输出一般是左上/右下坐标，但为了安全还是用最小值/最大值。
    # 向下取整/向上取整可以让裁剪框稍微覆盖完整一点，不会漏掉边缘像素。
    left = int(np.clip(math.floor(min(x1, x2)), 0, image_width - 1))
    top = int(np.clip(math.floor(min(y1, y2)), 0, image_height - 1))
    right = int(np.clip(math.ceil(max(x1, x2)), 0, image_width - 1))
    bottom = int(np.clip(math.ceil(max(y1, y2)), 0, image_height - 1))

    # 如果目标框没有面积，说明定位结果无效。
    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    """
    适当放大 VLM 目标框。

    原因：
    - VLM 框可能刚好贴着物体边缘；
    - 抓取矩形有时需要覆盖物体边缘外一点点空间；
    - 放大一点可以减少因为目标框太紧导致的轮廓截断。
    """

    left, top, right, bottom = box
    image_height, image_width = image_shape[:2]

    width = right - left
    height = bottom - top

    # expand_ratio=0.10 表示左右各扩 10% 目标框宽度，上下各扩 10% 目标框高度。
    pad_x = int(round(width * expand_ratio))
    pad_y = int(round(height * expand_ratio))

    # 放大后仍然不能超过图像边界。
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width - 1, right + pad_x),
        min(image_height - 1, bottom + pad_y),
    )


def restrict_mask_to_box(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """
    只保留 VLM 目标框内的掩码。

    传统 CV 基线是在整张图里找物体；
    VLM 辅助版本是在 VLM 指出的区域里找物体。
    这就是两种方法最核心的区别。
    """

    left, top, right, bottom = box
    restricted = np.zeros_like(mask)

    # 这里是关键操作：
    # 原始掩码可能包含桌边、背景、鞋子等干扰；
    # 限制后的掩码只保留 VLM 目标框内的部分，其他地方全部清零。
    restricted[top : bottom + 1, left : right + 1] = mask[top : bottom + 1, left : right + 1]
    return restricted


def choose_contour_inside_vlm_box(
    restricted_mask: np.ndarray,
    vlm_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """
    在 VLM 目标框限制后的掩码里选择物体轮廓。

    和传统 CV 基线的 choose_object_contour 不同，这里不再强烈偏向图像中心；
    因为 VLM 已经告诉我们目标在哪里了。
    """

    contours, _ = cv2.findContours(
        restricted_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    left, top, right, bottom = vlm_box

    # 目标框面积用来判断一个轮廓相对于 VLM 目标框是否太小。
    box_area = max(1, (right - left) * (bottom - top))

    best_contour = None
    best_score = -float("inf")

    for contour in contours:
        # contourArea 是 OpenCV 的轮廓面积计算函数。面积太小的一般是噪声。
        area = cv2.contourArea(contour)
        if area < 40:
            continue

        # boundingRect 是 OpenCV 的普通水平外接框函数，只用于计算轮廓中心和尺寸。
        x, y, w, h = cv2.boundingRect(contour)
        contour_area_ratio = area / box_area

        # 排除特别小的噪声；但不要太严格，因为 Cornell 有些物体很细长。
        if contour_area_ratio < 0.005:
            continue

        contour_center_x = x + w / 2.0
        contour_center_y = y + h / 2.0
        box_center_x = (left + right) / 2.0
        box_center_y = (top + bottom) / 2.0

        # VLM 目标框已经提供了“目标大概在哪”的信息，
        # 所以离 VLM 目标框中心太远的轮廓可信度较低。
        distance_to_box_center = math.hypot(
            contour_center_x - box_center_x,
            contour_center_y - box_center_y,
        )

        # 在 VLM 目标框内，面积越大越像目标；离 VLM 目标框中心越近越可信。
        score = area / (1.0 + distance_to_box_center / 80.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def fallback_grasp_from_vlm_box(box: tuple[int, int, int, int]) -> dict:
    """
    当颜色/亮度掩码在 VLM 目标框内找不到轮廓时，用 VLM 目标框本身生成一个保底抓取框。

    这不是强方法，但它能让我们区分两种失败：
    - VLM 没有定位到物体；
    - VLM 定位到了，但传统掩码后端没分割出来。
    """

    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    if box_width >= box_height:
        # 目标框横向更长：物体长轴近似水平。
        long_side_angle_degrees = 0.0
        short_side = box_height
    else:
        # 目标框纵向更长：物体长轴近似竖直。
        long_side_angle_degrees = 90.0
        short_side = box_width

    # 和传统 CV 基线保持一致：抓取方向近似垂直于物体长轴。
    grasp_angle_degrees = long_side_angle_degrees + 90.0
    if grasp_angle_degrees > 90.0:
        grasp_angle_degrees -= 180.0

    # grasp_width / grasp_height 沿用传统 CV 基线的启发式规则：
    # width 大致跨过物体短边，height 表示局部抓取区域厚度。
    grasp_width = float(np.clip(short_side * 1.35, 20.0, 120.0))
    grasp_height = float(np.clip(short_side * 0.55, 15.0, 45.0))

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": grasp_width,
        "height": grasp_height,
        "angle_degrees": float(grasp_angle_degrees),
        "contour_area": 0.0,
        "used_fallback_box": True,
    }


def predict_grasp_with_vlm_box(
    image: np.ndarray,
    vlm_box: tuple[int, int, int, int],
    expand_ratio: float,
    use_box_fallback: bool,
) -> tuple[dict | None, np.ndarray, tuple[int, int, int, int], str]:
    """
    用 VLM 目标框辅助生成抓取矩形。

    返回：
    - prediction: 预测抓取框
    - restricted_mask: 只保留 VLM 区域后的掩码，方便调试
    - expanded_box: 放大后的 VLM 目标框
    - failure_reason: 如果失败，这里记录原因
    """

    # 先把 VLM 目标框稍微放大，避免目标框太紧导致物体轮廓被切断。
    expanded_box = expand_box(vlm_box, image.shape, expand_ratio)

    # 复用传统 CV 基线里的颜色/亮度分割规则。
    # 注意：这里不是在整张图里选目标，而是之后会用 VLM 目标框限制它。
    mask = create_object_mask(image)

    # 只保留 VLM 目标框附近的掩码，这是 VLM 辅助方法的核心。
    restricted_mask = restrict_mask_to_box(mask, expanded_box)

    # 在 VLM 限定区域里找最像目标的轮廓。
    contour = choose_contour_inside_vlm_box(restricted_mask, expanded_box)

    if contour is None:
        # 如果掩码仍然没找到轮廓，可以选择直接用 VLM 目标框生成一个保底抓取框。
        if use_box_fallback:
            return (
                fallback_grasp_from_vlm_box(expanded_box),
                restricted_mask,
                expanded_box,
                "mask_contour_missing_used_vlm_box_fallback",
            )
        return None, restricted_mask, expanded_box, "mask_contour_missing"

    # 找到轮廓后，复用传统 CV 基线的几何后端：
    # 最小面积外接矩形 -> 物体长轴/短轴 -> 抓取方向和尺寸。
    prediction = predict_grasp_from_contour(contour)
    prediction["used_fallback_box"] = False
    return prediction, restricted_mask, expanded_box, ""


def draw_vlm_grasp_visualization(
    image: np.ndarray,
    prediction: dict | None,
    ground_truths: list[dict],
    vlm_box: tuple[int, int, int, int] | None,
    output_path: Path,
    success: bool,
) -> None:
    """
    保存 VLM 辅助抓取可视化。

    颜色约定：
    - 黄色：VLM 定位框
    - 蓝色：预测抓取框
    - 绿色：Cornell 正抓取标注
    """

    canvas = image.copy()

    if vlm_box is not None:
        # 黄色框：VLM 给出的目标区域。
        left, top, right, bottom = vlm_box
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            "VLM box",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for gt in ground_truths:
        # 绿色框：Cornell 人工标注的正抓取矩形。
        rect = (
            (gt["center_x"], gt["center_y"]),
            (gt["width"], gt["height"]),
            gt["angle_degrees"],
        )
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(canvas, [points], True, (0, 255, 0), 1, cv2.LINE_AA)

    if prediction is not None:
        # 蓝色框：当前 VLM 辅助方法预测出的抓取矩形。
        pred_rect = (
            (prediction["center_x"], prediction["center_y"]),
            (prediction["width"], prediction["height"]),
            prediction["angle_degrees"],
        )
        pred_points = cv2.boxPoints(pred_rect).astype(np.int32)
        cv2.polylines(canvas, [pred_points], True, (255, 0, 0), 2, cv2.LINE_AA)

    status_text = "SUCCESS" if success else "FAIL"
    cv2.putText(
        canvas,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if success else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"保存可视化失败：{output_path}")


def compare_with_baseline_same_subset(rows: list[dict]) -> dict | None:
    """
    在同一批样本上对比传统 CV 基线。

    为什么要做这个？
    ----------------
    当前 VLM 定位只跑了小批量样本。
    所以 VLM 辅助方法的 20 张结果不能直接和传统 CV 基线的 885 张全量结果硬比。
    更公平的临时比较是：

        同样这 20 张图，传统 CV 基线是多少？
        同样这 20 张图，VLM 辅助方法是多少？
    """

    # 如果还没跑过传统 CV 基线，就无法做对比；此时直接返回 None。
    if not BASELINE_CV_PREDICTIONS_CSV.exists() or not rows:
        return None

    # 读取传统 CV 基线的每张图结果，并按样本 key 建索引。
    with BASELINE_CV_PREDICTIONS_CSV.open("r", newline="", encoding="utf-8") as file:
        baseline_rows = {
            (row["object_directory"], row["sample_id"]): row
            for row in csv.DictReader(file)
        }

    matched_rows = []
    for row in rows:
        key = (row["object_directory"], row["sample_id"])
        if key in baseline_rows:
            matched_rows.append(baseline_rows[key])

    if not matched_rows:
        return None

    sample_count = len(matched_rows)
    success_count = sum(int(row["success"]) for row in matched_rows)

    return {
        "baseline_csv": str(BASELINE_CV_PREDICTIONS_CSV),
        "same_subset_sample_count": sample_count,
        "baseline_success_count_on_same_subset": success_count,
        "baseline_success_rate_on_same_subset": success_count / sample_count,
        "baseline_mean_best_iou_on_same_subset": float(
            np.mean([float(row["best_iou"]) for row in matched_rows])
        ),
        "baseline_mean_angle_error_on_same_subset": float(
            np.mean([float(row["best_angle_error_degrees"]) for row in matched_rows])
        ),
    }


def save_visualization_overview(image_paths: list[Path], output_path: Path) -> None:
    """
    保存一张成功/失败总览图。

    单独打开很多图片很麻烦；总览图适合快速看这个方法到底失败在哪里。
    """

    if not image_paths:
        return

    from PIL import Image, ImageDraw

    thumb_width = 320
    thumb_height = 240
    label_height = 30
    columns = 2
    rows = (len(image_paths) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB").resize((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_height + 6), image_path.name, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    # 命令例子：
    #   conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py
    #
    # 如果想评估另一个提示词的 VLM 定位结果，可以传：
    #   --localization-csv path/to/other_predictions.csv
    parser = argparse.ArgumentParser(
        description="在 Cornell 样本上评估 VLM 辅助的抓取预测。"
    )
    parser.add_argument(
        "--localization-csv",
        type=Path,
        default=DEFAULT_LOCALIZATION_CSV,
        help="Grounding DINO 定位结果 CSV。",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.10,
        help="运行几何后端前，VLM 目标框需要向外扩大的比例。",
    )
    parser.add_argument(
        "--no-box-fallback",
        action="store_true",
        help="当掩码找不到轮廓时，不使用 VLM 目标框作为保底抓取框。",
    )
    parser.add_argument(
        "--max-visualizations-per-class",
        type=int,
        default=10,
        help="每类成功/失败最多保存多少张可视化图片。",
    )

    args = parser.parse_args()

    # 读取第一阶段 VLM 定位的 CSV。
    localization_by_sample = read_localization_csv(args.localization_csv)

    # 加载 Cornell 数据集，用于读取 RGB 图和真实标注抓取框。
    dataset = CornellGraspDataset(DATASET_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 每次运行时清空旧的成功/失败可视化，
    # 避免上一次实验的图片混在这一次结果里。
    for subdir_name in ["success", "failure"]:
        subdir = VISUALIZATION_DIR / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)

    rows: list[dict] = []
    success_count = 0
    detected_count = 0
    failed_to_predict_count = 0
    mask_contour_missing_count = 0
    fallback_count = 0
    saved_success_visualizations = 0
    saved_failure_visualizations = 0
    visualization_paths: list[Path] = []

    # 遍历 Cornell 全部样本；但只评估定位 CSV 里存在的样本。
    # 这样同一个脚本既能评估 20 张小样本，也能评估 885 张全量数据。
    for index in range(len(dataset)):
        sample = dataset[index]
        key = (sample["object_directory"], sample["sample_id"])

        # 这个脚本只评估定位 CSV 里已经跑过 VLM 的样本。
        if key not in localization_by_sample:
            continue

        localization_row = localization_by_sample[key]
        image = sample["rgb"]

        # 把 CSV 中的 VLM 目标框字符串解析成整数像素坐标。
        vlm_box = parse_vlm_box(localization_row, image.shape)

        prediction = None
        expanded_box = None
        failure_reason = ""

        if vlm_box is None:
            # VLM 没检测到，或者目标框无效：这张图无法进入后续抓取后端。
            failure_reason = "vlm_no_detection_or_invalid_box"
            failed_to_predict_count += 1
        else:
            detected_count += 1

            # 使用 VLM 目标框限制 OpenCV 几何后端，生成最终抓取矩形。
            prediction, _restricted_mask, expanded_box, failure_reason = (
                predict_grasp_with_vlm_box(
                    image=image,
                    vlm_box=vlm_box,
                    expand_ratio=args.expand_ratio,
                    use_box_fallback=not args.no_box_fallback,
                )
            )
            if prediction is None:
                failed_to_predict_count += 1

            # 统计掩码/轮廓后端失败次数。
            if "mask_contour_missing" in failure_reason:
                mask_contour_missing_count += 1

            # 统计有多少次不是用轮廓，而是用 VLM 目标框保底方案生成的预测。
            if prediction is not None and prediction.get("used_fallback_box", False):
                fallback_count += 1

        # Cornell cpos 是四点矩形，这里转成中心点/宽/高/角度格式，
        # 方便和预测框计算 IoU 与角度误差。
        positive_ground_truths = rectangles_to_center_format(sample["positive_rectangles"])

        # 使用和传统 CV 基线完全相同的 Cornell 风格评价标准：
        # IoU >= 0.25 且角度误差 <= 30°。
        evaluation = evaluate_prediction(prediction, positive_ground_truths)

        if evaluation["success"]:
            success_count += 1

        # 这一行会写入 predictions CSV，记录单张图的完整结果。
        row = {
            "sample_id": sample["sample_id"],
            "object_directory": sample["object_directory"],
            "success": int(evaluation["success"]),
            "best_iou": evaluation["best_iou"],
            "best_angle_error_degrees": evaluation["best_angle_error_degrees"],
            "matched_gt_index": evaluation["matched_gt_index"],
            "successful_match_iou": evaluation["successful_match_iou"],
            "successful_match_angle_error_degrees": evaluation[
                "successful_match_angle_error_degrees"
            ],
            "successful_matched_gt_index": evaluation[
                "successful_matched_gt_index"
            ],
            "vlm_detected": int(vlm_box is not None),
            "vlm_box_x1": "" if vlm_box is None else vlm_box[0],
            "vlm_box_y1": "" if vlm_box is None else vlm_box[1],
            "vlm_box_x2": "" if vlm_box is None else vlm_box[2],
            "vlm_box_y2": "" if vlm_box is None else vlm_box[3],
            "expanded_box_x1": "" if expanded_box is None else expanded_box[0],
            "expanded_box_y1": "" if expanded_box is None else expanded_box[1],
            "expanded_box_x2": "" if expanded_box is None else expanded_box[2],
            "expanded_box_y2": "" if expanded_box is None else expanded_box[3],
            "vlm_score": localization_row.get("score", ""),
            "vlm_label": localization_row.get("label", ""),
            "failed_to_predict": int(prediction is None),
            "failure_reason": failure_reason,
            "used_fallback_box": int(
                prediction is not None and prediction.get("used_fallback_box", False)
            ),
            "rgb_path": sample["rgb_path"],
        }

        if prediction is not None:
            # 有预测框时，把预测框的中心点/尺寸/角度写入 CSV。
            row.update(
                {
                    "pred_center_x": prediction["center_x"],
                    "pred_center_y": prediction["center_y"],
                    "pred_width": prediction["width"],
                    "pred_height": prediction["height"],
                    "pred_angle_degrees": prediction["angle_degrees"],
                    "contour_area": prediction["contour_area"],
                }
            )
        else:
            # 没有预测框时，相关字段留空，CSV 结构仍然保持一致。
            row.update(
                {
                    "pred_center_x": "",
                    "pred_center_y": "",
                    "pred_width": "",
                    "pred_height": "",
                    "pred_angle_degrees": "",
                    "contour_area": "",
                }
            )

        rows.append(row)

        # 为了避免保存 885 张成功图/失败图过多，这里只保存每类前 N 张。
        should_save_success = (
            evaluation["success"]
            and saved_success_visualizations < args.max_visualizations_per_class
        )
        should_save_failure = (
            not evaluation["success"]
            and saved_failure_visualizations < args.max_visualizations_per_class
        )

        if should_save_success or should_save_failure:
            subdir = "success" if evaluation["success"] else "failure"
            output_path = (
                VISUALIZATION_DIR
                / subdir
                / f"{sample['object_directory']}_{sample['sample_id']}.png"
            )
            draw_vlm_grasp_visualization(
                image=image,
                prediction=prediction,
                ground_truths=positive_ground_truths,
                vlm_box=expanded_box,
                output_path=output_path,
                success=evaluation["success"],
            )

            if evaluation["success"]:
                saved_success_visualizations += 1
            else:
                saved_failure_visualizations += 1
            visualization_paths.append(output_path)

    fieldnames = [
        # 样本身份。
        "sample_id",
        "object_directory",
        # Cornell 风格评价结果。
        "success",
        "best_iou",
        "best_angle_error_degrees",
        "matched_gt_index",
        "successful_match_iou",
        "successful_match_angle_error_degrees",
        "successful_matched_gt_index",
        # VLM 定位结果。
        "vlm_detected",
        "vlm_box_x1",
        "vlm_box_y1",
        "vlm_box_x2",
        "vlm_box_y2",
        "expanded_box_x1",
        "expanded_box_y1",
        "expanded_box_x2",
        "expanded_box_y2",
        "vlm_score",
        "vlm_label",
        # 失败原因和保底方案记录。
        "failed_to_predict",
        "failure_reason",
        "used_fallback_box",
        # 最终预测抓取框。
        "pred_center_x",
        "pred_center_y",
        "pred_width",
        "pred_height",
        "pred_angle_degrees",
        "contour_area",
        "rgb_path",
    ]

    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sample_count = len(rows)

    # 这些是论文表格最常用的统计量。
    success_rate = success_count / sample_count if sample_count else 0.0
    detection_rate = detected_count / sample_count if sample_count else 0.0
    mean_iou = float(np.mean([row["best_iou"] for row in rows])) if rows else 0.0
    mean_angle_error = (
        float(np.mean([row["best_angle_error_degrees"] for row in rows]))
        if rows
        else 0.0
    )

    # 和传统 CV 基线在同一批样本上做公平对比。
    baseline_same_subset = compare_with_baseline_same_subset(rows)

    # summary JSON 是整次实验的“总成绩单”。
    summary = {
        "method_name": "vlm_assisted_opencv_contour_min_area_rect_rgb",
        "dataset_root": str(DATASET_ROOT),
        "localization_csv": str(args.localization_csv),
        "sample_count": sample_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "vlm_detected_count": detected_count,
        "vlm_detection_rate_on_evaluated_samples": detection_rate,
        "failed_to_predict_count": failed_to_predict_count,
        "mask_contour_missing_count": mask_contour_missing_count,
        "fallback_count": fallback_count,
        "mean_best_iou": mean_iou,
        "mean_best_angle_error_degrees": mean_angle_error,
        "iou_threshold": IOU_THRESHOLD,
        "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
        "expand_ratio": args.expand_ratio,
        "use_box_fallback": not args.no_box_fallback,
        "predictions_csv": str(PREDICTIONS_CSV),
        "visualization_dir": str(VISUALIZATION_DIR),
        "overview_path": str(OVERVIEW_PATH),
        "baseline_cv_same_subset_comparison": baseline_same_subset,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # 保存成功/失败拼图，方便快速看预测框质量。
    save_visualization_overview(visualization_paths, OVERVIEW_PATH)

    print("VLM-assisted grasp 运行完成")
    print(f"定位结果 CSV：{args.localization_csv}")
    print(f"评估样本数量：{sample_count}")
    print(f"VLM 检测数量：{detected_count}")
    print(f"VLM 检测率：{detection_rate:.4f}")
    print(f"成功数量：{success_count}")
    print(f"最终抓取成功率：{success_rate:.4f}")
    print(f"无法生成预测数量：{failed_to_predict_count}")
    print(f"mask 轮廓缺失数量：{mask_contour_missing_count}")
    print(f"fallback 使用数量：{fallback_count}")
    print(f"平均 best IoU：{mean_iou:.4f}")
    print(f"平均角度误差：{mean_angle_error:.2f} 度")
    print(f"预测结果 CSV：{PREDICTIONS_CSV}")
    print(f"总结 JSON：{SUMMARY_JSON}")
    print(f"可视化目录：{VISUALIZATION_DIR}")
    print(f"总览图：{OVERVIEW_PATH}")

    if baseline_same_subset is not None:
        print()
        print("同一批样本上的 baseline_cv 对比")
        print(
            "baseline_cv 成功率："
            f"{baseline_same_subset['baseline_success_rate_on_same_subset']:.4f}"
        )
        print(
            "baseline_cv 平均 best IoU："
            f"{baseline_same_subset['baseline_mean_best_iou_on_same_subset']:.4f}"
        )
        print(
            "baseline_cv 平均角度误差："
            f"{baseline_same_subset['baseline_mean_angle_error_on_same_subset']:.2f} 度"
        )


if __name__ == "__main__":
    main()
