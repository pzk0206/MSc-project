"""在固定 85 样本上逐样本比较几何后端与 CNN 后端。"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.cornell_dataset import (  # noqa: E402
    CornellGraspDataset,
    load_grasp_rectangles,
)

DATASET_ROOT = Path("data/raw/cornell")
GEOMETRIC_PREDICTIONS = Path(
    "data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv"
)
CNN_PREDICTIONS = Path(
    "data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv"
)
OUTPUT_DIR = Path("data/processed/vlm/backend_comparison")
TEST_DIRS = {"09", "10"}


def classify_pair(geometric_success: int, cnn_success: int) -> str:
    """按两个后端的成功/失败组合分类。"""
    outcomes = {
        (1, 1): "both_success",
        (0, 1): "cnn_only",
        (1, 0): "geometric_only",
        (0, 0): "both_failure",
    }
    try:
        return outcomes[(int(geometric_success), int(cnn_success))]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("success values must be 0 or 1") from exc


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"prediction CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _index_test_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["object_directory"], row["sample_id"]): row
        for row in rows
        if row["object_directory"] in TEST_DIRS
    }


def build_comparison_rows(
    geometric_rows: list[dict],
    cnn_rows: list[dict],
) -> list[dict]:
    """严格内连接两个后端在目录 09–10 的逐样本预测。"""
    geometric = _index_test_rows(geometric_rows)
    cnn = _index_test_rows(cnn_rows)
    keys = sorted(geometric.keys() & cnn.keys())
    if len(keys) != 85:
        raise RuntimeError(f"expected 85 common test samples, found {len(keys)}")

    comparisons = []
    for object_directory, sample_id in keys:
        geo = geometric[(object_directory, sample_id)]
        learned = cnn[(object_directory, sample_id)]
        comparisons.append(
            {
                "object_directory": object_directory,
                "sample_id": sample_id,
                "category": classify_pair(
                    int(geo["success"]),
                    int(learned["success"]),
                ),
                "geometric_success": int(geo["success"]),
                "geometric_best_iou": float(geo["best_iou"]),
                "geometric_angle_error_degrees": float(
                    geo["best_angle_error_degrees"]
                ),
                "geometric_center_x": float(geo["pred_center_x"]),
                "geometric_center_y": float(geo["pred_center_y"]),
                "geometric_width": float(geo["pred_width"]),
                "geometric_height": float(geo["pred_height"]),
                "geometric_angle_degrees": float(
                    geo["pred_angle_degrees"]
                ),
                "cnn_success": int(learned["success"]),
                "cnn_best_iou": float(learned["best_iou"]),
                "cnn_angle_error_degrees": float(
                    learned["best_angle_error_degrees"]
                ),
                "cnn_center_x": float(learned["pred_center_x"]),
                "cnn_center_y": float(learned["pred_center_y"]),
                "cnn_width": float(learned["pred_width"]),
                "cnn_height": float(learned["pred_height"]),
                "cnn_angle_degrees": float(learned["pred_angle_degrees"]),
                "vlm_box_x1": int(float(geo["expanded_box_x1"])),
                "vlm_box_y1": int(float(geo["expanded_box_y1"])),
                "vlm_box_x2": int(float(geo["expanded_box_x2"])),
                "vlm_box_y2": int(float(geo["expanded_box_y2"])),
                "rgb_path": geo["rgb_path"],
            }
        )
    return comparisons


def _method_metrics(rows: list[dict], prefix: str) -> dict:
    return {
        "count": len(rows),
        "success_count": sum(int(row[f"{prefix}_success"]) for row in rows),
        "success_rate": (
            sum(int(row[f"{prefix}_success"]) for row in rows) / len(rows)
        ),
        "mean_best_iou": float(
            np.mean([float(row[f"{prefix}_best_iou"]) for row in rows])
        ),
        "mean_angle_error_degrees": float(
            np.mean(
                [
                    float(row[f"{prefix}_angle_error_degrees"])
                    for row in rows
                ]
            )
        ),
    }


def build_summary(rows: list[dict]) -> dict:
    categories = Counter(row["category"] for row in rows)
    return {
        "test_count": len(rows),
        "categories": {
            name: int(categories.get(name, 0))
            for name in (
                "both_success",
                "cnn_only",
                "geometric_only",
                "both_failure",
            )
        },
        "vlm_geometric": _method_metrics(rows, "geometric"),
        "vlm_cnn_last_saved_run": _method_metrics(rows, "cnn"),
    }


def _draw_rotated_rectangle(
    canvas: np.ndarray,
    row: dict,
    prefix: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    rect = (
        (row[f"{prefix}_center_x"], row[f"{prefix}_center_y"]),
        (row[f"{prefix}_width"], row[f"{prefix}_height"]),
        row[f"{prefix}_angle_degrees"],
    )
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)


def _sample_paths() -> dict[tuple[str, str], object]:
    dataset = CornellGraspDataset(DATASET_ROOT)
    return {
        (sample.object_directory, sample.sample_id): sample
        for sample in dataset.samples
    }


def _render_cell(row: dict, sample_paths: object) -> np.ndarray:
    canvas = cv2.imread(str(sample_paths.rgb_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise FileNotFoundError(f"cannot read image: {sample_paths.rgb_path}")

    ground_truth = load_grasp_rectangles(
        sample_paths.cpos_path,
        allow_empty=False,
    )
    for rectangle in ground_truth:
        cv2.polylines(
            canvas,
            [rectangle.astype(np.int32)],
            True,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )

    _draw_rotated_rectangle(canvas, row, "cnn", (255, 0, 0), 3)
    _draw_rotated_rectangle(canvas, row, "geometric", (0, 0, 255), 3)
    cv2.rectangle(
        canvas,
        (row["vlm_box_x1"], row["vlm_box_y1"]),
        (row["vlm_box_x2"], row["vlm_box_y2"]),
        (0, 220, 220),
        2,
    )

    resized = cv2.resize(canvas, (320, 240), interpolation=cv2.INTER_AREA)
    title = np.full((42, 320, 3), 255, dtype=np.uint8)
    label = (
        f"{row['object_directory']}/{row['sample_id']}  "
        f"G:{row['geometric_best_iou']:.2f} "
        f"C:{row['cnn_best_iou']:.2f}"
    )
    cv2.putText(
        title,
        label,
        (7, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([title, resized])


def _select_examples(rows: list[dict], category: str) -> list[dict]:
    subset = [row for row in rows if row["category"] == category]
    if category == "cnn_only":
        key = lambda row: row["cnn_best_iou"] - row["geometric_best_iou"]
        subset.sort(key=key, reverse=True)
    elif category == "geometric_only":
        key = lambda row: row["geometric_best_iou"] - row["cnn_best_iou"]
        subset.sort(key=key, reverse=True)
    else:
        key = lambda row: row["cnn_best_iou"] + row["geometric_best_iou"]
        subset.sort(key=key)
    if len(subset) < 4:
        raise RuntimeError(f"category {category} has fewer than four samples")
    return subset[:4]


def save_failure_figure(rows: list[dict], output_path: Path) -> None:
    """保存 CNN-only、geometry-only 和共同失败的 3×4 对比图。"""
    paths = _sample_paths()
    image_rows = []
    for category in ("cnn_only", "geometric_only", "both_failure"):
        cells = [
            _render_cell(
                row,
                paths[(row["object_directory"], row["sample_id"])],
            )
            for row in _select_examples(rows, category)
        ]
        panel = cv2.hconcat(cells)
        banner = np.full((42, panel.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            banner,
            category.replace("_", " ").upper(),
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        image_rows.append(cv2.vconcat([banner, panel]))

    legend = np.full((52, image_rows[0].shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        legend,
        "GT: green   CNN: blue   Geometry: red   VLM crop: yellow",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = cv2.vconcat([legend, *image_rows])
    if not cv2.imwrite(str(output_path), figure):
        raise RuntimeError(f"failed to write figure: {output_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_comparison_rows(
        _load_csv(GEOMETRIC_PREDICTIONS),
        _load_csv(CNN_PREDICTIONS),
    )
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "sample_comparison.csv", rows)
    with (OUTPUT_DIR / "comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    save_failure_figure(rows, OUTPUT_DIR / "backend_failure_cases.png")

    print(f"共同测试样本: {summary['test_count']}")
    print(f"交叉分类: {summary['categories']}")
    print(
        "几何后端: "
        f"{summary['vlm_geometric']['success_count']}/"
        f"{summary['test_count']}"
    )
    print(
        "CNN 最后一轮: "
        f"{summary['vlm_cnn_last_saved_run']['success_count']}/"
        f"{summary['test_count']}"
    )
    print(f"比较产物: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
