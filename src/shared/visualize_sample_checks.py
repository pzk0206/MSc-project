"""
从 Cornell Grasping Dataset 中抽样生成标注可视化图片。

为什么要做这个脚本？
--------------------
我们已经确认 pcd0100 这个单样本能正确读取和画框。
但只看一个样本还不够，需要从不同子文件夹里多抽几张，
人工确认整个数据集的 RGB 图和抓取标注是否大体匹配。

这个脚本会：
1. 遍历 data/raw/cornell/01 到 data/raw/cornell/10；
2. 每个文件夹抽取固定数量的样本；
3. 读取 RGB 图像；
4. 读取 cpos 正抓取标注和 cneg 负抓取标注；
5. 把正抓取框画成绿色，把负抓取框画成红色；
6. 保存到 data/processed/shared/visualizations/sample_checks/。

注意：
----
生成出来的图片只是“人工检查用”，不是训练数据。
后面训练模型时仍然使用原始 RGB/depth 和原始/转换后的标签。
"""

from pathlib import Path
import random

import cv2
import numpy as np


# Cornell 原始数据所在目录。
DATASET_ROOT = Path("data/raw/cornell")

# 抽样可视化结果保存目录。
OUTPUT_DIRECTORY = Path("data/processed/shared/visualizations/sample_checks")

# 每个 Cornell 子目录抽几张图片。
SAMPLES_PER_DIRECTORY = 2

# 固定随机种子。
# 这样每次运行脚本，抽到的样本都是一样的，方便复现和写报告。
RANDOM_SEED = 42


def load_grasp_rectangles(label_path: Path, allow_empty: bool = False) -> np.ndarray:
    """
    读取 Cornell 抓取标注文件。

    参数：
        label_path:
            cpos.txt 或 cneg.txt 的文件路径。

        allow_empty:
            是否允许空文件。

            cpos 是正抓取标注，不应该为空；
            cneg 是负抓取标注，可以为空，空文件表示 0 个负抓取框。

    返回：
        shape 为 (N, 4, 2) 的数组。

        N 是抓取矩形数量；
        4 是每个矩形的四个角点；
        2 是每个角点的 x, y 坐标。
    """

    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件：{label_path}")

    # 如果文件大小是 0，说明它是空文件。
    if label_path.stat().st_size == 0:
        if allow_empty:
            return np.empty((0, 4, 2), dtype=np.float32)

        raise ValueError(f"标注文件为空：{label_path}")

    points = np.loadtxt(label_path, dtype=np.float32)
    points = np.atleast_2d(points)

    if points.shape[1] != 2:
        raise ValueError(
            f"标注格式错误：{label_path}，应该是两列 x y，实际 shape={points.shape}"
        )

    if points.shape[0] % 4 != 0:
        raise ValueError(
            f"标注格式错误：{label_path}，点数必须是 4 的倍数，实际点数={points.shape[0]}"
        )

    return points.reshape(-1, 4, 2)


def draw_grasp_rectangles(
    image: np.ndarray,
    rectangles: np.ndarray,
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    """
    在图片上画抓取矩形和文字标签。

    参数：
        image:
            OpenCV 读取的图片数组。

        rectangles:
            shape 为 (N, 4, 2) 的抓取矩形数组。

        color:
            画框颜色。OpenCV 使用 BGR 顺序：
            绿色是 (0, 255, 0)，红色是 (0, 0, 255)。

        label_prefix:
            写在框旁边的文字前缀，比如 POS 或 NEG。
    """

    for index, rectangle in enumerate(rectangles):
        integer_points = np.round(rectangle).astype(np.int32)

        cv2.polylines(
            image,
            [integer_points],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{label_prefix} {index}",
            tuple(integer_points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def visualize_one_sample(cpos_path: Path, output_directory: Path) -> Path:
    """
    可视化一个样本，并返回保存路径。

    参数：
        cpos_path:
            某个样本的 cpos 文件路径，例如：
            data/raw/cornell/01/pcd0100cpos.txt

        output_directory:
            可视化图片保存目录。
    """

    sample_directory = cpos_path.parent

    # cpos_path.stem:
    #   pcd0100cpos.txt -> pcd0100cpos
    #
    # removesuffix("cpos"):
    #   pcd0100cpos -> pcd0100
    sample_id = cpos_path.stem.removesuffix("cpos")

    rgb_path = sample_directory / f"{sample_id}r.png"
    cneg_path = sample_directory / f"{sample_id}cneg.txt"

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图片：{rgb_path}")

    positive_rectangles = load_grasp_rectangles(cpos_path, allow_empty=False)
    negative_rectangles = load_grasp_rectangles(cneg_path, allow_empty=True)

    draw_grasp_rectangles(
        image=image,
        rectangles=positive_rectangles,
        color=(0, 255, 0),
        label_prefix="POS",
    )

    draw_grasp_rectangles(
        image=image,
        rectangles=negative_rectangles,
        color=(0, 0, 255),
        label_prefix="NEG",
    )

    # 输出文件名加上子目录编号，避免不同目录里的样本重名时混淆。
    directory_name = sample_directory.name
    output_path = output_directory / f"{directory_name}_{sample_id}_annotations.png"

    save_success = cv2.imwrite(str(output_path), image)
    if not save_success:
        raise RuntimeError(f"保存可视化图片失败：{output_path}")

    return output_path


def main() -> None:
    """
    主函数：每个 Cornell 子目录抽样并生成可视化图片。
    """

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"找不到数据集目录：{DATASET_ROOT}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    generated_paths = []

    # Cornell 子目录是 01, 02, ..., 10。
    for directory_index in range(1, 11):
        directory_name = f"{directory_index:02d}"
        sample_directory = DATASET_ROOT / directory_name

        if not sample_directory.exists():
            print(f"跳过不存在的目录：{sample_directory}")
            continue

        cpos_paths = sorted(sample_directory.glob("pcd*cpos.txt"))

        if not cpos_paths:
            print(f"目录中没有 cpos 标注文件：{sample_directory}")
            continue

        # 如果某个目录样本数量少于 SAMPLES_PER_DIRECTORY，
        # 就全部使用；否则随机抽取指定数量。
        number_to_sample = min(SAMPLES_PER_DIRECTORY, len(cpos_paths))
        selected_cpos_paths = random.sample(cpos_paths, number_to_sample)
        selected_cpos_paths = sorted(selected_cpos_paths)

        for cpos_path in selected_cpos_paths:
            output_path = visualize_one_sample(cpos_path, OUTPUT_DIRECTORY)
            generated_paths.append(output_path)

    print("抽样可视化完成")
    print(f"输出目录：{OUTPUT_DIRECTORY}")
    print(f"生成图片数量：{len(generated_paths)}")
    print()
    print("生成的文件：")

    for output_path in generated_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
