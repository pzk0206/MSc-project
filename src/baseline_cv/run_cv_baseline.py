"""
运行一个最小传统计算机视觉 baseline。

这个 baseline 对应项目计划书里的第一条主线：

    RGB 图像 -> OpenCV 目标区域提取 -> 旋转边界框 -> 2D 抓取矩形预测 -> Cornell-style 评估

它不是最终方法，也不是 VLM 方法。
它的意义是先建立一个“可复现、可评估、可对比”的传统 CV 下限。

后面 VLM-assisted pipeline 做出来以后，就可以和这个 baseline 比较：

    VLM 方法是否比传统 CV 几何方法更好？
"""

from __future__ import annotations

from pathlib import Path
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

from src.shared.cornell_dataset import CornellGraspDataset
from src.shared.cornell_evaluation import (
    ANGLE_THRESHOLD_DEGREES,
    IOU_THRESHOLD,
    angle_difference_degrees,
    evaluate_prediction,
    rotated_rect_iou,
)
from src.shared.grasp_geometry import rectangles_to_center_format, normalize_angle_radians


DATASET_ROOT = Path("data/raw/cornell")
OUTPUT_DIR = Path("data/processed/baseline_cv")
PREDICTIONS_CSV = OUTPUT_DIR / "cv_baseline_predictions.csv"
SUMMARY_JSON = OUTPUT_DIR / "cv_baseline_summary.json"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"


def create_object_mask(rgb_bgr: np.ndarray) -> np.ndarray:
    """
    从 RGB/BGR 图像中粗略分割物体区域。

    Cornell 图片大多是物体放在浅色背景上。
    所以这里用一个很朴素的规则：

    - 饱和度较高的区域，可能是有颜色的物体；
    - 亮度较低的区域，可能是深色物体；
    - 去掉贴着图像边界的大背景区域；
    - 用形态学操作清理噪声。

    这不是完美分割，但它正好适合作为 baseline：
    简单、可解释、容易被后续 VLM 方法超过。
    """

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # 非白色/非浅灰背景区域。
    mask = ((saturation > 25) | (value < 210)).astype(np.uint8) * 255

    # 去掉图像边缘区域，减少桌边、背景、鞋子等干扰。
    border = 8
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def choose_object_contour(mask: np.ndarray) -> np.ndarray | None:
    """
    从 mask 中选择最像物体的轮廓。

    选择策略：
    - 排除太小的噪声；
    - 排除贴着图像边界的大块区域；
    - 在剩下的候选里，偏向面积大且靠近图像中心的区域。
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

        # 面积越大越好，离图像中心越近越好。
        score = area / (1.0 + distance_to_center / 120.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def predict_grasp_from_contour(contour: np.ndarray) -> dict:
    """
    根据物体轮廓预测一个抓取矩形。

    baseline 思路：
    - 用 cv2.minAreaRect 找到物体的旋转外接矩形；
    - 取外接矩形中心作为抓取中心；
    - 取物体长轴方向作为抓取角度；
    - 用启发式规则生成一个抓取矩形大小。

    这很简单，但它给我们一个可运行的传统 CV 对照组。
    """

    (center_x, center_y), (rect_w, rect_h), raw_angle = cv2.minAreaRect(contour)

    # OpenCV 的 minAreaRect 角度定义比较别扭。
    # 这里统一转换成“长边方向角度”。
    if rect_w >= rect_h:
        long_side = rect_w
        short_side = rect_h
        angle_degrees = raw_angle
    else:
        long_side = rect_h
        short_side = rect_w
        angle_degrees = raw_angle + 90.0

    angle_radians = normalize_angle_radians(math.radians(angle_degrees))
    angle_degrees = math.degrees(angle_radians)

    # Cornell 的抓取矩形通常不是沿着物体长轴“顺着抓”，
    # 而是让夹爪横跨物体较窄的一侧。
    #
    # 例如遥控器是竖着放的，但合理抓取框往往是横着跨过遥控器。
    # 所以这里把物体长轴角度旋转 90 度，得到抓取矩形的主方向。
    angle_radians = normalize_angle_radians(angle_radians + math.pi / 2.0)
    angle_degrees = math.degrees(angle_radians)

    # 预测抓取框不要直接使用整个物体外接框。
    # 否则框会太大，和 Cornell 的局部抓取框不太匹配。
    #
    # 这里用物体短边估计抓取框宽度：
    # - width 大致横跨物体短边；
    # - height 保持较薄，表示局部抓取区域厚度。
    grasp_width = float(np.clip(short_side * 1.35, 20.0, 120.0))
    grasp_height = float(np.clip(short_side * 0.55, 15.0, 45.0))

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": grasp_width,
        "height": grasp_height,
        "angle_radians": float(angle_radians),
        "angle_degrees": float(angle_degrees),
        "contour_area": float(cv2.contourArea(contour)),
    }


def draw_prediction_visualization(
    image: np.ndarray,
    prediction: dict | None,
    ground_truths: list[dict],
    output_path: Path,
    success: bool,
    object_contour: np.ndarray | None = None,
) -> None:
    """
    保存少量 baseline 可视化结果。

    颜色约定和 VLM-assisted 可视化保持一致：

    - 黄色：传统 CV 从 mask/contour 找到的物体识别框；
    - 蓝色：baseline 预测抓取框；
    - 绿色：Cornell ground-truth 正抓取框。
    """

    canvas = image.copy()

    if object_contour is not None:
        # 黄色框表示传统 CV 前端识别出来的目标区域。
        # 这里用 minAreaRect 画旋转外接框，而不是普通水平框，
        # 这样可以更直观看到 OpenCV 认为物体的主方向是什么。
        object_rect = cv2.minAreaRect(object_contour)
        object_points = cv2.boxPoints(object_rect).astype(np.int32)
        cv2.polylines(canvas, [object_points], True, (0, 255, 255), 2, cv2.LINE_AA)

        x, y, _w, _h = cv2.boundingRect(object_contour)
        cv2.putText(
            canvas,
            "CV object box",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for gt in ground_truths:
        # 绿色框表示 Cornell 数据集给出的正抓取标注。
        rect = (
            (gt["center_x"], gt["center_y"]),
            (gt["width"], gt["height"]),
            gt["angle_degrees"],
        )
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(canvas, [points], True, (0, 255, 0), 1, cv2.LINE_AA)

    if prediction is not None:
        # 蓝色框表示传统 CV baseline 最终预测出的抓取矩形。
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
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    """
    运行全数据集 CV baseline。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # 每次运行 baseline 前只清理旧的 success/failure 可视化结果。
    # 不清理整个 visualizations 目录，因为 bordereffect 等 baseline 调试图也放在这里。
    for subdir_name in ["success", "failure"]:
        subdir = VISUALIZATION_DIR / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)

    dataset = CornellGraspDataset(DATASET_ROOT)

    rows = []
    success_count = 0
    failed_to_predict_count = 0

    # 保存少量可视化：前 10 个成功案例和前 10 个失败案例。
    saved_success_visualizations = 0
    saved_failure_visualizations = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["rgb"]

        mask = create_object_mask(image)
        contour = choose_object_contour(mask)

        prediction = None
        if contour is None:
            failed_to_predict_count += 1
        else:
            prediction = predict_grasp_from_contour(contour)

        positive_ground_truths = rectangles_to_center_format(sample["positive_rectangles"])
        evaluation = evaluate_prediction(prediction, positive_ground_truths)

        if evaluation["success"]:
            success_count += 1

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
            "failed_to_predict": int(prediction is None),
            "rgb_path": sample["rgb_path"],
        }

        if prediction is not None:
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

        should_save_success = evaluation["success"] and saved_success_visualizations < 10
        should_save_failure = (
            not evaluation["success"] and saved_failure_visualizations < 10
        )

        if should_save_success or should_save_failure:
            subdir = "success" if evaluation["success"] else "failure"
            output_path = (
                VISUALIZATION_DIR
                / subdir
                / f"{sample['object_directory']}_{sample['sample_id']}.png"
            )
            draw_prediction_visualization(
                image=image,
                prediction=prediction,
                ground_truths=positive_ground_truths,
                output_path=output_path,
                success=evaluation["success"],
                object_contour=contour,
            )

            if evaluation["success"]:
                saved_success_visualizations += 1
            else:
                saved_failure_visualizations += 1

    fieldnames = [
        "sample_id",
        "object_directory",
        "success",
        "best_iou",
        "best_angle_error_degrees",
        "matched_gt_index",
        "successful_match_iou",
        "successful_match_angle_error_degrees",
        "successful_matched_gt_index",
        "failed_to_predict",
        "pred_center_x",
        "pred_center_y",
        "pred_width",
        "pred_height",
        "pred_angle_degrees",
        "contour_area",
        "rgb_path",
    ]

    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success_rate = success_count / len(dataset)
    mean_iou = float(np.mean([row["best_iou"] for row in rows]))
    mean_angle_error = float(
        np.mean([row["best_angle_error_degrees"] for row in rows])
    )

    summary = {
        "baseline_name": "opencv_contour_min_area_rect_rgb",
        "dataset_root": str(DATASET_ROOT),
        "sample_count": len(dataset),
        "success_count": success_count,
        "success_rate": success_rate,
        "failed_to_predict_count": failed_to_predict_count,
        "mean_best_iou": mean_iou,
        "mean_best_angle_error_degrees": mean_angle_error,
        "iou_threshold": IOU_THRESHOLD,
        "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
        "predictions_csv": str(PREDICTIONS_CSV),
        "visualization_dir": str(VISUALIZATION_DIR),
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("传统 CV baseline 运行完成")
    print(f"样本数量：{len(dataset)}")
    print(f"成功数量：{success_count}")
    print(f"成功率：{success_rate:.4f}")
    print(f"无法生成预测数量：{failed_to_predict_count}")
    print(f"平均 best IoU：{mean_iou:.4f}")
    print(f"平均角度误差：{mean_angle_error:.2f} 度")
    print(f"预测结果 CSV：{PREDICTIONS_CSV}")
    print(f"总结 JSON：{SUMMARY_JSON}")
    print(f"可视化目录：{VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
