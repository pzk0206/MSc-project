"""审计 Cornell 目录划分，并生成同样本后端比较所需的基础产物。"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.cornell_dataset import CornellGraspDataset  # noqa: E402

TRAIN_DIRS = {"01", "02", "03", "04", "05", "06"}
VAL_DIRS = {"07", "08"}
TEST_DIRS = {"09", "10"}

DATASET_ROOT = Path("data/raw/cornell")
BASELINE_PREDICTIONS = Path(
    "data/processed/baseline_cv/cv_baseline_predictions.csv"
)
GEOMETRIC_PREDICTIONS = Path(
    "data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv"
)
CNN_PREDICTIONS = Path(
    "data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv"
)
OUTPUT_DIR = Path("data/processed/shared/split_audit")


def assign_split(object_directory: str) -> str:
    """把 Cornell 子目录映射到固定的 train/val/test 划分。"""
    if object_directory in TRAIN_DIRS:
        return "train"
    if object_directory in VAL_DIRS:
        return "val"
    if object_directory in TEST_DIRS:
        return "test"
    raise ValueError(f"unknown Cornell directory: {object_directory}")


def summarize_predictions(rows: list[dict]) -> dict[str, dict]:
    """按固定划分汇总成功率、IoU 和角度误差。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[assign_split(row["object_directory"])].append(row)

    result: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        subset = grouped.get(split, [])
        result[split] = {
            "count": len(subset),
            "success_rate": (
                sum(int(row["success"]) for row in subset) / len(subset)
                if subset
                else 0.0
            ),
            "mean_best_iou": (
                float(np.mean([float(row["best_iou"]) for row in subset]))
                if subset
                else 0.0
            ),
            "mean_angle_error_degrees": (
                float(
                    np.mean(
                        [
                            float(row["best_angle_error_degrees"])
                            for row in subset
                        ]
                    )
                )
                if subset
                else 0.0
            ),
        }
    return result


def load_csv(path: Path) -> list[dict]:
    """读取预测 CSV，并在缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(f"prediction CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["object_directory"], row["sample_id"]): row
        for row in rows
    }


def _common_test_rows(
    geometric_rows: list[dict],
    cnn_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    geometric = _index_rows(geometric_rows)
    cnn = _index_rows(cnn_rows)
    common_keys = sorted(
        key
        for key in geometric.keys() & cnn.keys()
        if key[0] in TEST_DIRS
    )
    if len(common_keys) != 85:
        raise RuntimeError(
            f"expected 85 common test samples, found {len(common_keys)}"
        )
    return (
        [geometric[key] for key in common_keys],
        [cnn[key] for key in common_keys],
    )


def representative_quota(object_directory: str) -> int:
    """为三个 split 分配相同数量的 contact-sheet 样本。"""
    split = assign_split(object_directory)
    return 2 if split == "train" else 6


def _choose_representatives(
    dataset: CornellGraspDataset,
) -> list[dict]:
    grouped: dict[str, list] = defaultdict(list)
    for sample_paths in dataset.samples:
        grouped[sample_paths.object_directory].append(sample_paths)

    selected: list[dict] = []
    for object_directory in sorted(grouped):
        candidates = grouped[object_directory]
        count = min(representative_quota(object_directory), len(candidates))
        indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
        for index in indices:
            sample = candidates[int(index)]
            selected.append(
                {
                    "sample_id": sample.sample_id,
                    "object_directory": object_directory,
                    "split": assign_split(object_directory),
                    "rgb_path": str(sample.rgb_path),
                }
            )
    return selected


def _thumbnail(row: dict, width: int = 160, height: int = 120) -> np.ndarray:
    image = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read RGB image: {row['rgb_path']}")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 22), (255, 255, 255), -1)
    label = f"{row['object_directory']}/{row['sample_id']}"
    cv2.putText(
        resized,
        label,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return resized


def _split_panel(split: str, rows: list[dict]) -> np.ndarray:
    columns = 3
    thumbnails = [_thumbnail(row) for row in rows]
    while len(thumbnails) % columns:
        thumbnails.append(np.full_like(thumbnails[0], 245))
    image_rows = [
        cv2.hconcat(thumbnails[index : index + columns])
        for index in range(0, len(thumbnails), columns)
    ]
    grid = cv2.vconcat(image_rows)
    title = np.full((42, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title,
        f"{split.upper()} ({len(rows)} representatives)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return cv2.vconcat([title, grid])


def save_contact_sheet(rows: list[dict], output_path: Path) -> None:
    """生成 train/val/test 三栏代表样本图。"""
    panels = [
        _split_panel(split, [row for row in rows if row["split"] == split])
        for split in ("train", "val", "test")
    ]
    max_height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        bottom = max_height - panel.shape[0]
        padded.append(
            cv2.copyMakeBorder(
                panel,
                0,
                bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.hconcat(padded)):
        raise RuntimeError(f"failed to write contact sheet: {output_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset = CornellGraspDataset(DATASET_ROOT)
    baseline_rows = load_csv(BASELINE_PREDICTIONS)
    geometric_rows = load_csv(GEOMETRIC_PREDICTIONS)
    cnn_rows = load_csv(CNN_PREDICTIONS)

    dataset_counts = {"train": 0, "val": 0, "test": 0}
    for sample in dataset.samples:
        dataset_counts[assign_split(sample.object_directory)] += 1
    if sum(dataset_counts.values()) != 885:
        raise RuntimeError(
            f"expected 885 Cornell samples, found {sum(dataset_counts.values())}"
        )

    geometric_test, cnn_test = _common_test_rows(
        geometric_rows,
        cnn_rows,
    )
    geometric_test_metrics = summarize_predictions(geometric_test)["test"]
    cnn_test_metrics = summarize_predictions(cnn_test)["test"]

    metrics = {
        "dataset_sample_counts": dataset_counts,
        "methods": {
            "traditional_cv": summarize_predictions(baseline_rows),
            "vlm_geometric": summarize_predictions(geometric_rows),
            "vlm_cnn_last_saved_run": summarize_predictions(cnn_rows),
        },
        "common_test_sample_count": len(geometric_test),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "split_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    common_rows = []
    for method, summary in (
        ("vlm_geometric", geometric_test_metrics),
        ("vlm_cnn_last_saved_run", cnn_test_metrics),
    ):
        common_rows.append({"method": method, **summary})
    _write_csv(OUTPUT_DIR / "same_test_subset_metrics.csv", common_rows)

    representatives = _choose_representatives(dataset)
    _write_csv(OUTPUT_DIR / "representative_samples.csv", representatives)
    save_contact_sheet(
        representatives,
        OUTPUT_DIR / "cornell_split_contact_sheet.png",
    )

    print(f"数据集划分: {dataset_counts}")
    print("共同测试样本: 85")
    print(
        "几何后端: "
        f"{geometric_test_metrics['success_rate'] * 100:.2f}%"
    )
    print(
        "CNN 最后一轮: "
        f"{cnn_test_metrics['success_rate'] * 100:.2f}%"
    )
    print(f"审计产物: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
