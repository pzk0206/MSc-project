"""
批量生成 border effect / mask pipeline 可视化图片。

单张版本：
    src/baseline_cv/visualize_mask_pipeline.py

批量版本：
    本脚本会从 Cornell 的 01 到 10 子目录中抽样，
    为每个样本生成一张横向四联图：

    1. Original RGB
    2. Color/Brightness filter
    3. Final object after border
    4. Removed / kept difference

默认：
    每个子目录抽 2 张，总共约 20 张。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_cv.visualize_mask_pipeline import build_four_panel_visualization

import cv2
import numpy as np


DATASET_ROOT = Path("data/raw/cornell")
OUTPUT_DIRECTORY = Path("data/processed/baseline_cv/visualizations/bordereffect")


def sample_rgb_paths(
    dataset_root: Path,
    samples_per_directory: int,
    random_seed: int,
) -> list[Path]:
    """
    从 Cornell 01 到 10 子目录中抽样 RGB 图片路径。
    """

    random.seed(random_seed)

    selected_paths: list[Path] = []

    for directory_index in range(1, 11):
        directory_name = f"{directory_index:02d}"
        sample_directory = dataset_root / directory_name

        if not sample_directory.exists():
            print(f"跳过不存在的目录：{sample_directory}")
            continue

        rgb_paths = sorted(sample_directory.glob("pcd*r.png"))

        # 排除 backgrounds 之类非 pcd 样本已经由 glob 模式处理。
        if not rgb_paths:
            print(f"目录中没有 RGB 样本：{sample_directory}")
            continue

        number_to_sample = min(samples_per_directory, len(rgb_paths))
        selected = random.sample(rgb_paths, number_to_sample)
        selected_paths.extend(sorted(selected))

    return selected_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch visualize color filtering, border filtering and final selected object masks."
    )
    parser.add_argument(
        "--samples-per-directory",
        type=int,
        default=2,
        help="How many RGB images to sample from each Cornell subdirectory.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=8,
        help="Number of pixels removed from each image border.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory where generated visualizations will be saved.",
    )

    args = parser.parse_args()

    rgb_paths = sample_rgb_paths(
        dataset_root=DATASET_ROOT,
        samples_per_directory=args.samples_per_directory,
        random_seed=args.random_seed,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    for rgb_path in rgb_paths:
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"跳过无法读取的图片：{rgb_path}")
            continue

        (
            four_panel,
            color_mask,
            border_mask,
            morphology_mask,
            selected_object_mask,
        ) = build_four_panel_visualization(image, border=args.border)

        sample_directory = rgb_path.parent.name
        sample_id = rgb_path.stem.removesuffix("r")
        output_path = (
            args.output_directory
            / f"{sample_directory}_{sample_id}_mask_pipeline_overlay.png"
        )

        save_success = cv2.imwrite(str(output_path), four_panel)
        if not save_success:
            raise RuntimeError(f"保存图片失败：{output_path}")

        generated_paths.append(output_path)

        print(
            f"{sample_directory}/{sample_id}: "
            f"color={int(np.count_nonzero(color_mask))}, "
            f"border={int(np.count_nonzero(border_mask))}, "
            f"morphology={int(np.count_nonzero(morphology_mask))}, "
            f"selected={int(np.count_nonzero(selected_object_mask))}"
        )

    print()
    print("批量 mask pipeline 可视化完成")
    print(f"border 大小：{args.border} px")
    print(f"每个目录抽样数量：{args.samples_per_directory}")
    print(f"生成图片数量：{len(generated_paths)}")
    print(f"输出目录：{args.output_directory}")

    for output_path in generated_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
