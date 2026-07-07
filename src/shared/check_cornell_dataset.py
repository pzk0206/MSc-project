"""
检查 Cornell Grasping Dataset 是否完整、标注格式是否正常。

这个脚本做的事情：
1. 自动扫描 data/raw/cornell 下面所有 pcdXXXXcpos.txt 文件。
2. 根据 cpos 文件推断出每一个样本的 sample_id，例如 pcd0100。
3. 检查同一个样本是否同时拥有：
   - pcdXXXXr.png      RGB 彩色图
   - pcdXXXXd.tiff     深度图
   - pcdXXXXcpos.txt   正抓取标注
   - pcdXXXXcneg.txt   负抓取标注
   - pcdXXXX.txt       点云文件
4. 检查 cpos/cneg 标注文件格式是否合理：
   - 必须是两列：x y
   - 点的数量必须是 4 的倍数，因为 4 个角点组成 1 个抓取矩形
5. 检查抓取框坐标是否超出对应 RGB 图片范围。
6. 打印一个全数据集检查报告。
"""

from pathlib import Path

import cv2
import numpy as np


# Cornell 数据集根目录。
# 后面所有路径都会从这里开始寻找。
DATASET_ROOT = Path("data/raw/cornell")


def load_grasp_rectangles(label_path: Path, allow_empty: bool = False) -> np.ndarray:
    """
    读取一个 Cornell 抓取标注文件，并转换成矩形数组。

    参数：
        label_path:
            标注文件路径，可以是 cpos.txt，也可以是 cneg.txt。

        allow_empty:
            是否允许空标注文件。

            对 cpos 正抓取标注来说，通常不允许为空；
            因为没有正抓取框，模型就没有这个样本的正确答案。

            对 cneg 负抓取标注来说，可以允许为空；
            因为没有负抓取框不代表这个样本不能用于正抓取训练。

    返回：
        rectangles:
            形状为 (N, 4, 2) 的 numpy 数组。

            N 表示这个文件里有多少个抓取矩形。
            4 表示每个抓取矩形有 4 个角点。
            2 表示每个角点有 x 和 y 两个坐标。
    """

    # 如果标注文件不存在，直接报错。
    # 这里用 raise 是为了让问题尽早暴露，而不是让程序继续假装正常。
    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件：{label_path}")

    # 先检查文件内容是否为空。
    # Cornell 里可能出现空的 cneg.txt。
    # 如果 allow_empty=True，就把空文件理解为“0 个抓取矩形”。
    if label_path.stat().st_size == 0:
        if allow_empty:
            return np.empty((0, 4, 2), dtype=np.float32)

        raise ValueError(f"标注文件为空：{label_path}")

    # np.loadtxt 会把 txt 里的数字读取成 numpy 数组。
    # dtype=np.float32 表示用 32 位浮点数保存坐标，因为 Cornell 标注里有小数。
    points = np.loadtxt(label_path, dtype=np.float32)

    # 如果文件只有一行，np.loadtxt 可能读成一维数组，例如 shape=(2,)。
    # atleast_2d 可以保证它至少是二维数组，例如变成 shape=(1, 2)。
    points = np.atleast_2d(points)

    # Cornell 抓取标注每一行应该只有两个数字：x 和 y。
    # 所以第二个维度必须是 2。
    if points.shape[1] != 2:
        raise ValueError(
            f"标注文件格式错误，应该是两列 x y：{label_path}，实际 shape={points.shape}"
        )

    # 4 个点组成 1 个抓取矩形。
    # 所以总点数必须能被 4 整除。
    if points.shape[0] % 4 != 0:
        raise ValueError(
            f"标注点数量错误，点数必须是 4 的倍数：{label_path}，实际点数={points.shape[0]}"
        )

    # 把原来的点表：
    #   (总点数, 2)
    # 变成矩形表：
    #   (矩形数量, 4, 2)
    rectangles = points.reshape(-1, 4, 2)

    return rectangles


def check_coordinate_range(
    rectangles: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    """
    检查抓取矩形坐标是否都在图像范围内。

    参数：
        rectangles:
            形状为 (N, 4, 2) 的抓取矩形数组。

        image_width:
            RGB 图片宽度。Cornell 常见宽度是 640。

        image_height:
            RGB 图片高度。Cornell 常见高度是 480。

    返回：
        True:
            所有坐标都在图像范围内。

        False:
            至少有一个坐标越界。
    """

    # rectangles[:, :, 0] 取出所有角点的 x 坐标。
    x_coordinates = rectangles[:, :, 0]

    # rectangles[:, :, 1] 取出所有角点的 y 坐标。
    y_coordinates = rectangles[:, :, 1]

    # x 坐标合法范围是 [0, image_width)。
    # 例如 width=640，则合法 x 是 0 到 639。
    x_out_of_range = np.any((x_coordinates < 0) | (x_coordinates >= image_width))

    # y 坐标合法范围是 [0, image_height)。
    # 例如 height=480，则合法 y 是 0 到 479。
    y_out_of_range = np.any((y_coordinates < 0) | (y_coordinates >= image_height))

    return not (x_out_of_range or y_out_of_range)


def check_one_sample(cpos_path: Path) -> dict:
    """
    检查一个 Cornell 样本。

    Cornell 一个样本由多个同名前缀文件组成。
    例如 cpos_path 是：
        data/raw/cornell/01/pcd0100cpos.txt

    那么 sample_id 就是：
        pcd0100

    对应的其他文件就是：
        pcd0100r.png
        pcd0100d.tiff
        pcd0100cneg.txt
        pcd0100.txt
    """

    # cpos 文件名类似 pcd0100cpos.txt。
    # stem 会去掉最后的 .txt，得到 pcd0100cpos。
    cpos_stem = cpos_path.stem

    # 去掉末尾的 cpos，得到样本编号 pcd0100。
    sample_id = cpos_stem.removesuffix("cpos")

    # 当前样本所在文件夹，例如 data/raw/cornell/01。
    sample_directory = cpos_path.parent

    # 根据 sample_id 拼出同一个样本的其他文件路径。
    rgb_path = sample_directory / f"{sample_id}r.png"
    depth_path = sample_directory / f"{sample_id}d.tiff"
    cneg_path = sample_directory / f"{sample_id}cneg.txt"
    point_cloud_path = sample_directory / f"{sample_id}.txt"

    # result 用来记录这个样本的检查结果。
    result = {
        "sample_id": sample_id,
        "directory": str(sample_directory),
        "is_complete": True,
        "errors": [],
        "positive_grasp_count": 0,
        "negative_grasp_count": 0,
    }

    # 检查这个样本应该有的 5 个文件是否都存在。
    expected_files = {
        "rgb": rgb_path,
        "depth": depth_path,
        "cpos": cpos_path,
        "cneg": cneg_path,
        "point_cloud": point_cloud_path,
    }

    for file_type, file_path in expected_files.items():
        if not file_path.exists():
            result["is_complete"] = False
            result["errors"].append(f"缺少 {file_type} 文件：{file_path}")

    # 如果 RGB 图片缺失，后面就没法检查坐标范围。
    if not rgb_path.exists():
        return result

    # 用 OpenCV 读取 RGB 图片。
    # 注意 OpenCV 读取出来的颜色顺序是 BGR，但这里只检查尺寸，不影响结果。
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    # 如果 image 是 None，说明图片路径存在，但是 OpenCV 没有成功读取。
    if image is None:
        result["is_complete"] = False
        result["errors"].append(f"RGB 图片读取失败：{rgb_path}")
        return result

    # OpenCV 图片 shape 是：
    #   高度, 宽度, 通道数
    image_height, image_width = image.shape[:2]

    # 检查正抓取标注。
    try:
        positive_rectangles = load_grasp_rectangles(cpos_path, allow_empty=False)
        result["positive_grasp_count"] = len(positive_rectangles)

        if not check_coordinate_range(positive_rectangles, image_width, image_height):
            result["errors"].append(f"正抓取标注坐标越界：{cpos_path}")

    except Exception as error:
        result["errors"].append(f"正抓取标注读取失败：{cpos_path}，错误：{error}")

    # 检查负抓取标注。
    # 如果 cneg 文件本身缺失，前面已经记录过“缺少文件”，这里就不重复读取。
    if cneg_path.exists():
        try:
            negative_rectangles = load_grasp_rectangles(cneg_path, allow_empty=True)
            result["negative_grasp_count"] = len(negative_rectangles)

            if not check_coordinate_range(negative_rectangles, image_width, image_height):
                result["errors"].append(f"负抓取标注坐标越界：{cneg_path}")

        except Exception as error:
            result["errors"].append(f"负抓取标注读取失败：{cneg_path}，错误：{error}")

    # 只要 errors 非空，就说明这个样本有异常。
    if result["errors"]:
        result["is_complete"] = False

    return result


def main() -> None:
    """
    主函数：扫描整个 Cornell 数据集，并打印检查报告。
    """

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"找不到数据集目录：{DATASET_ROOT}")

    # rglob 会递归搜索所有子目录。
    # 这里我们用 cpos 文件作为“样本入口”，因为每个正式样本都应该有 cpos 标注。
    cpos_paths = sorted(DATASET_ROOT.rglob("pcd*cpos.txt"))

    if not cpos_paths:
        raise FileNotFoundError(f"没有找到任何 cpos 标注文件：{DATASET_ROOT}")

    sample_results = []

    # 逐个检查样本。
    for cpos_path in cpos_paths:
        sample_result = check_one_sample(cpos_path)
        sample_results.append(sample_result)

    # 统计总样本数。
    total_samples = len(sample_results)

    # 统计完整样本数。
    complete_samples = sum(result["is_complete"] for result in sample_results)

    # 统计异常样本。
    abnormal_samples = [
        result for result in sample_results if not result["is_complete"]
    ]

    # 统计正负抓取矩形数量。
    total_positive_grasps = sum(
        result["positive_grasp_count"] for result in sample_results
    )
    total_negative_grasps = sum(
        result["negative_grasp_count"] for result in sample_results
    )

    # 打印总报告。
    print("Cornell 数据集检查报告")
    print("=" * 40)
    print(f"数据集目录：{DATASET_ROOT}")
    print(f"样本总数：{total_samples}")
    print(f"完整样本数：{complete_samples}")
    print(f"异常样本数：{len(abnormal_samples)}")
    print(f"正抓取矩形总数：{total_positive_grasps}")
    print(f"负抓取矩形总数：{total_negative_grasps}")

    # 如果发现异常，打印前 20 个异常样本，避免输出太长。
    if abnormal_samples:
        print()
        print("异常样本示例，最多显示 20 个：")
        print("-" * 40)

        for result in abnormal_samples[:20]:
            print(f"{result['sample_id']}  位于 {result['directory']}")
            for error in result["errors"]:
                print(f"  - {error}")

        if len(abnormal_samples) > 20:
            print(f"... 还有 {len(abnormal_samples) - 20} 个异常样本未显示")
    else:
        print()
        print("没有发现缺失文件、标注格式错误或坐标越界问题。")


if __name__ == "__main__":
    main()
