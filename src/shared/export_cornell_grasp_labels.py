"""
导出 Cornell 抓取标注为更适合后续实验使用的 CSV。

Cornell 原始标注是四点格式：

    x1,y1
    x2,y2
    x3,y3
    x4,y4

每 4 行组成一个抓取矩形。

这个脚本把它转换成中心格式：

    sample_id, label_type, grasp_index,
    center_x, center_y, width, height, angle_radians, angle_degrees

为什么要导出这个 CSV？
--------------------
因为后续做 baseline、VLM 几何方法、预测框评估时，
center/width/height/angle 比四个角点更容易计算和比较。

注意：
这个脚本不会修改 raw 原始数据。
它只是在 data/processed/shared/labels/ 下生成一个新的处理后标签表。
"""

from __future__ import annotations

from pathlib import Path
import csv
import sys


# 允许直接运行 python src/shared/export_cornell_grasp_labels.py。
# 因为直接运行脚本时，Python 默认不一定能找到同目录下的模块。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.cornell_dataset import CornellGraspDataset
from src.shared.grasp_geometry import rectangles_to_center_format


DATASET_ROOT = Path("data/raw/cornell")
OUTPUT_PATH = Path("data/processed/shared/labels/cornell_grasp_labels_center_format.csv")


def add_rectangles_to_rows(
    rows: list[dict],
    sample: dict,
    label_type: str,
) -> None:
    """
    把一个样本中的正/负抓取矩形追加到 rows 里。

    参数
    ----------
    rows:
        最终要写入 CSV 的行列表。

    sample:
        CornellGraspDataset 返回的单个样本字典。

    label_type:
        "positive" 或 "negative"。
    """

    if label_type == "positive":
        rectangles = sample["positive_rectangles"]
    elif label_type == "negative":
        rectangles = sample["negative_rectangles"]
    else:
        raise ValueError(f"未知 label_type：{label_type}")

    center_rectangles = rectangles_to_center_format(rectangles)

    for grasp_index, grasp in enumerate(center_rectangles):
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "object_directory": sample["object_directory"],
                "label_type": label_type,
                "grasp_index": grasp_index,
                "center_x": grasp["center_x"],
                "center_y": grasp["center_y"],
                "width": grasp["width"],
                "height": grasp["height"],
                "angle_radians": grasp["angle_radians"],
                "angle_degrees": grasp["angle_degrees"],
                "rgb_path": sample["rgb_path"],
                "cpos_path": sample["cpos_path"],
                "cneg_path": sample["cneg_path"],
            }
        )


def main() -> None:
    """
    主函数：读取 Cornell 数据集并导出中心格式抓取标签。
    """

    dataset = CornellGraspDataset(DATASET_ROOT)

    rows: list[dict] = []

    for index in range(len(dataset)):
        sample = dataset[index]

        # 主线最重要的是 positive，也就是 ground-truth 正抓取框。
        add_rectangles_to_rows(rows, sample, label_type="positive")

        # negative 暂时不是 baseline 主线，但保存下来以后可以分析或可视化。
        add_rectangles_to_rows(rows, sample, label_type="negative")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "object_directory",
        "label_type",
        "grasp_index",
        "center_x",
        "center_y",
        "width",
        "height",
        "angle_radians",
        "angle_degrees",
        "rgb_path",
        "cpos_path",
        "cneg_path",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    positive_count = sum(row["label_type"] == "positive" for row in rows)
    negative_count = sum(row["label_type"] == "negative" for row in rows)

    print("Cornell 抓取标签导出完成")
    print(f"数据集样本数：{len(dataset)}")
    print(f"正抓取标签数量：{positive_count}")
    print(f"负抓取标签数量：{negative_count}")
    print(f"总标签数量：{len(rows)}")
    print(f"输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
