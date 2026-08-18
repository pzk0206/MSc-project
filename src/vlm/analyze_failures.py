"""
VLM-assisted 抓取检测失败案例分析脚本。

分析 Cornell metric-v2 下 VLM-guided geometry 的失败样本，按与存在性
成功判据相容的模式分类并生成统计报告。

失败模式分类：
1. 没有任何 GT 达到 IoU 门槛；
2. 至少一个 GT 达到 IoU 门槛，但没有同一个 GT 同时通过角度门槛。

输出：
- docs/debugging/FAILURE_ANALYSIS_METRIC_V2.md：失败分析报告
- metric-v2 geometry 目录下的 failure_analysis.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VLM_PREDICTIONS = (
    PROJECT_ROOT
    / "data/processed/shared/cornell_metric_v2/vlm_geometry/predictions.csv"
)
BASELINE_PREDICTIONS = (
    PROJECT_ROOT
    / "data/processed/shared/cornell_metric_v2/baseline/predictions.csv"
)
OUTPUT_CSV = VLM_PREDICTIONS.parent / "failure_analysis.csv"
OUTPUT_MD = PROJECT_ROOT / "docs/debugging/FAILURE_ANALYSIS_METRIC_V2.md"

IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD = 30.0


def load_predictions() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    """加载 VLM 和 baseline 预测结果。"""
    with open(VLM_PREDICTIONS, newline="", encoding="utf-8") as f:
        vlm = list(csv.DictReader(f))

    baseline = {}
    if BASELINE_PREDICTIONS.exists():
        with open(BASELINE_PREDICTIONS, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                baseline[(r["object_directory"], r["sample_id"])] = r

    return vlm, baseline


def classify_failure(row: dict) -> str:
    """Classify one metric-v2 failure without mixing matches across GTs."""
    if int(row["success"]) != 0:
        raise ValueError("cannot classify a successful row as a failure")

    iou = float(row["best_iou"])
    angle = float(row["best_angle_error_degrees"])
    if iou < IOU_THRESHOLD:
        return "no_iou_qualified_match"
    if angle <= ANGLE_THRESHOLD:
        raise ValueError(
            "failure row is inconsistent with metric-v2 thresholds"
        )
    return "angle_after_iou_match"


def iou_severity(iou: float) -> str:
    """IoU 失败严重程度分级。"""
    if iou < 0.05:
        return "severe (<0.05)"
    elif iou < 0.10:
        return "major (0.05-0.10)"
    elif iou < 0.15:
        return "moderate (0.10-0.15)"
    elif iou < 0.20:
        return "minor (0.15-0.20)"
    else:
        return "borderline (0.20-0.25)"


def angle_severity(angle: float) -> str:
    """角度失败严重程度分级。"""
    if angle > 60:
        return "severe (>60°)"
    elif angle > 45:
        return "major (45°-60°)"
    else:
        return "borderline (30°-45°)"


def analyze_failures(
    vlm_rows: list[dict],
    baseline: dict[tuple[str, str], dict],
) -> tuple[list[dict], dict]:
    """主分析逻辑。"""
    failed = [r for r in vlm_rows if r["success"] == "0"]

    categories = Counter()
    iou_severities = Counter()
    angle_severities = Counter()
    vlm_fail_bl_success = []
    vlm_fail_bl_fail = []

    for r in failed:
        category = classify_failure(r)
        categories[category] += 1

        iou = float(r["best_iou"])
        angle = float(r["best_angle_error_degrees"])

        if iou < IOU_THRESHOLD:
            iou_severities[iou_severity(iou)] += 1
        if angle > ANGLE_THRESHOLD:
            angle_severities[angle_severity(angle)] += 1

        # 与 baseline 交叉对比
        key = (r["object_directory"], r["sample_id"])
        bl = baseline.get(key)
        if bl and bl["success"] == "1":
            vlm_fail_bl_success.append(r)
        else:
            vlm_fail_bl_fail.append(r)

    # 按 object_directory 统计失败分布
    dir_failure = Counter(r["object_directory"] for r in failed)

    # 分析 VLM box 大小与失败的关系
    box_areas = []
    for r in failed:
        try:
            w = float(r["vlm_box_x2"]) - float(r["vlm_box_x1"])
            h = float(r["vlm_box_y2"]) - float(r["vlm_box_y1"])
            box_areas.append(w * h)
        except (ValueError, TypeError):
            pass

    stats = {
        "total_samples": len(vlm_rows),
        "total_failed": len(failed),
        "total_success": len(vlm_rows) - len(failed),
        "categories": dict(categories),
        "iou_severities": dict(iou_severities),
        "angle_severities": dict(angle_severities),
        "vlm_fail_bl_success_count": len(vlm_fail_bl_success),
        "vlm_fail_bl_fail_count": len(vlm_fail_bl_fail),
        "dir_failure": dict(dir_failure),
        "mean_box_area": sum(box_areas) / len(box_areas) if box_areas else 0,
        "mean_iou_failed": sum(float(r["best_iou"]) for r in failed) / len(failed),
        "mean_angle_failed": sum(float(r["best_angle_error_degrees"]) for r in failed) / len(failed),
    }

    # 给每个失败样本打上分类标签
    annotated = []
    for r in failed:
        r_copy = dict(r)
        r_copy["failure_category"] = classify_failure(r)
        key = (r["object_directory"], r["sample_id"])
        bl = baseline.get(key)
        r_copy["baseline_success"] = bl["success"] if bl else "N/A"
        r_copy["baseline_iou"] = bl["best_iou"] if bl else "N/A"
        r_copy["baseline_angle"] = bl["best_angle_error_degrees"] if bl else "N/A"
        annotated.append(r_copy)

    return annotated, stats


def write_csv(annotated: list[dict]) -> None:
    """保存带分类标签的失败样本 CSV。"""
    if not annotated:
        return
    fieldnames = list(annotated[0].keys())
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)
    print(f"失败分析 CSV 已保存: {OUTPUT_CSV}")


def write_markdown_report(
    annotated: list[dict],
    stats: dict,
    baseline: dict,
) -> None:
    """Write a bounded metric-v2 diagnostic report."""

    del annotated, baseline
    total_failed = stats["total_failed"]
    no_iou = stats["categories"].get("no_iou_qualified_match", 0)
    angle_after_iou = stats["categories"].get(
        "angle_after_iou_match",
        0,
    )
    lines = [
        "# VLM-guided Geometry Metric-v2 失败分析",
        "",
        "日期：2026-08-17",
        "",
        "成功采用 `cornell_rectangle_any_gt_v2`：必须存在同一个 GT 同时",
        "通过 IoU 与角度门槛。下列类别是诊断，不是因果标签。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 总样本 | {stats['total_samples']} |",
        f"| 成功 | {stats['total_success']} |",
        f"| 失败 | {total_failed} |",
        f"| 成功率 | {stats['total_success']/stats['total_samples']*100:.2f}% |",
        "| 2D 预测覆盖率 | 100% |",
        "",
        "| 失败诊断 | 数量 | 占失败比例 |",
        "|---|---:|---:|",
        (
            "| 没有 GT 达到 IoU 0.25 | "
            f"{no_iou} | {no_iou/total_failed*100:.1f}% |"
        ),
        (
            "| 有 GT 达到 IoU，但无同一 GT 通过角度 | "
            f"{angle_after_iou} | {angle_after_iou/total_failed*100:.1f}% |"
        ),
        "",
        "与传统 CV 的逐样本交叉结果：",
        "",
        f"- Geometry 失败、CV 成功：{stats['vlm_fail_bl_success_count']}。",
        f"- 两者均失败：{stats['vlm_fail_bl_fail_count']}。",
        "",
        "注意：`best_*` 只描述最大-IoU GT；它不能与另一 GT 的角度混合",
        "来构造成功，也不能单独证明定位、尺度或方向是失败原因。",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"失败分析报告已保存: {OUTPUT_MD}")


def main() -> None:
    vlm_rows, baseline = load_predictions()
    annotated, stats = analyze_failures(vlm_rows, baseline)

    print("=== VLM-guided Geometric Pipeline 失败分析 ===")
    print(f"总样本: {stats['total_samples']}")
    print(f"成功: {stats['total_success']} ({stats['total_success']/stats['total_samples']*100:.1f}%)")
    print(f"失败: {stats['total_failed']} ({stats['total_failed']/stats['total_samples']*100:.1f}%)")
    print()
    print("失败模式分布:")
    for category, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count} ({count/stats['total_failed']*100:.1f}%)")
    print()
    print(f"IoU 严重程度: {stats['iou_severities']}")
    print(f"角度严重程度: {stats['angle_severities']}")
    print()
    print(f"VLM 失败但 Baseline 成功: {stats['vlm_fail_bl_success_count']} (退化案例)")
    print(f"两者都失败: {stats['vlm_fail_bl_fail_count']} (难例)")
    print()
    print(f"失败样本平均 IoU: {stats['mean_iou_failed']:.4f}")
    print(f"失败样本平均角度误差: {stats['mean_angle_failed']:.1f}°")

    write_csv(annotated)
    write_markdown_report(annotated, stats, baseline)


if __name__ == "__main__":
    main()
