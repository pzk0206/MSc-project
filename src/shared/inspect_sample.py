"""
观察并可视化一个 Cornell Grasping Dataset 单样本。

程序完成以下工作：

1. 读取一张 RGB 图片；
2. 读取正确抓取矩形 cpos；
3. 读取错误抓取矩形 cneg；
4. 将正确抓取框画成绿色；
5. 将错误抓取框画成红色；
6. 保存可视化结果。
"""

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. 定义输入和输出路径
# ============================================================

# Path 用来表示文件或文件夹路径。
# 相比直接使用字符串，Path 更容易进行路径拼接和检查。
sample_directory = Path("data/raw/cornell/01")

# “/”在这里不是数学除法。
# 对 Path 对象来说，它表示把目录和文件名拼接起来。
rgb_path = sample_directory / "pcd0100r.png"
positive_label_path = sample_directory / "pcd0100cpos.txt"
negative_label_path = sample_directory / "pcd0100cneg.txt"

# 程序生成的图片将保存在这里。
output_directory = Path("data/processed/shared/visualizations")

# parents=True：
# 如果上层目录不存在，也一起创建。
#
# exist_ok=True：
# 如果目录已经存在，不要报错。
output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "pcd0100_annotations.png"


# ============================================================
# 2. 定义读取抓取矩形标注的函数
# ============================================================

def load_grasp_rectangles(label_path: Path) -> np.ndarray:
    """
    读取 Cornell 抓取矩形标注。

    Cornell 标注文件中：
    - 每一行包含一个点的 x 和 y 坐标；
    - 每四行表示一个完整抓取矩形。

    参数
    ----------
    label_path:
        cpos.txt 或 cneg.txt 文件的路径。

    返回
    ----------
    rectangles:
        NumPy 数组，形状为：

            (矩形数量, 4, 2)

        例如：

            (5, 4, 2)

        表示文件中有 5 个矩形，
        每个矩形有 4 个点，
        每个点包含 x、y 两个坐标。
    """

    # 首先确认文件确实存在。
    # 如果路径错误，提前给出容易理解的错误。
    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件：{label_path}")

    # np.loadtxt 会读取纯文本中的数字。
    #
    # dtype=np.float32 表示使用 32 位浮点数。
    # 因为 Cornell 标注中包含 319.7 这样的非整数坐标，
    # 所以不能直接使用整数类型读取。
    points = np.loadtxt(label_path, dtype=np.float32)

    # 确保结果至少是二维数组。
    #
    # 正常情况下，points 的形状是：
    # (坐标点数量, 2)
    #
    # 例如 20 行标注会得到：
    # (20, 2)
    points = np.atleast_2d(points)

    # points.shape[1] 表示每一行有多少列。
    # Cornell 标注每行必须恰好包含 x 和 y 两列。
    if points.shape[1] != 2:
        raise ValueError(
            f"标注格式错误：{label_path} 每行应该包含两个数字，"
            f"实际数组形状为 {points.shape}"
        )

    # len(points) 表示坐标点的数量，也就是标注文件行数。
    number_of_points = len(points)

    # 每四个点组成一个矩形。
    # 因此总点数必须能够被 4 整除。
    #
    # % 是取余运算：
    # 8 % 4 = 0，表示可以组成两个矩形；
    # 10 % 4 = 2，表示有两个点多出来，标注格式有问题。
    if number_of_points % 4 != 0:
        raise ValueError(
            f"标注格式错误：{label_path} 包含 {number_of_points} 个点，"
            "坐标点数量不能被 4 整除。"
        )

    # reshape 用来改变数组形状。
    #
    # 原来的形状可能是：
    # (20, 2)
    #
    # 改变后是：
    # (5, 4, 2)
    #
    # -1 表示让 NumPy 自动计算矩形数量。
    rectangles = points.reshape(-1, 4, 2)

    return rectangles


# ============================================================
# 3. 定义绘制抓取矩形的函数
# ============================================================

def draw_grasp_rectangles(
    image: np.ndarray,
    rectangles: np.ndarray,
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    """
    在图片上绘制多个抓取矩形。

    参数
    ----------
    image:
        OpenCV 读取的图像数组。

    rectangles:
        形状为 (矩形数量, 4, 2) 的坐标数组。

    color:
        矩形线条颜色。

        注意 OpenCV 使用 BGR 顺序，不是 RGB：
        - 绿色：(0, 255, 0)
        - 红色：(0, 0, 255)
        - 蓝色：(255, 0, 0)

    label_prefix:
        标记文字，例如 POS 或 NEG。
    """

    # enumerate 会同时返回：
    # - index：当前矩形编号；
    # - rectangle：当前矩形的四个坐标点。
    for index, rectangle in enumerate(rectangles):
        # 标注坐标可能包含小数，例如 319.7。
        # OpenCV 绘图需要整数像素坐标。
        #
        # np.round：四舍五入；
        # astype(np.int32)：转换为 32 位整数。
        integer_points = np.round(rectangle).astype(np.int32)

        # cv2.polylines 用多个点绘制折线。
        #
        # [integer_points]：
        # OpenCV 要求轮廓列表，因此外面再加一层方括号。
        #
        # isClosed=True：
        # 把最后一个点和第一个点连接起来，形成闭合矩形。
        #
        # thickness=2：
        # 线条宽度为两个像素。
        cv2.polylines(
            image,
            [integer_points],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # 取矩形的第一个点，用于放置文字。
        text_position = tuple(integer_points[0])

        # 在矩形旁边写出 POS 0、POS 1 或 NEG 0。
        cv2.putText(
            image,
            f"{label_prefix} {index}",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


# ============================================================
# 4. 读取 RGB 图片
# ============================================================

# cv2.imread 读取图片。
#
# str(rgb_path)：
# OpenCV 接收字符串路径，因此把 Path 转换成字符串。
#
# cv2.IMREAD_COLOR：
# 以三通道彩色图片方式读取。
image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

# 如果路径不存在、文件损坏或读取失败，
# cv2.imread 不一定抛出错误，而是返回 None。
if image is None:
    raise FileNotFoundError(f"无法读取 RGB 图片：{rgb_path}")

# image.shape 返回：
# (高度, 宽度, 通道数量)
#
# Cornell RGB 图片通常是：
# (480, 640, 3)
image_height, image_width, number_of_channels = image.shape


# ============================================================
# 5. 读取正负抓取标注
# ============================================================

positive_rectangles = load_grasp_rectangles(positive_label_path)
negative_rectangles = load_grasp_rectangles(negative_label_path)


# ============================================================
# 6. 检查所有坐标是否位于图片范围内
# ============================================================

def check_coordinate_range(
    rectangles: np.ndarray,
    image_width: int,
    image_height: int,
    label_name: str,
) -> None:
    """检查抓取标注是否超出图像边界。"""

    # rectangles[:, :, 0] 取所有点的 x 坐标。
    x_coordinates = rectangles[:, :, 0]

    # rectangles[:, :, 1] 取所有点的 y 坐标。
    y_coordinates = rectangles[:, :, 1]

    # 图像合法坐标范围：
    # x：0 到 image_width - 1
    # y：0 到 image_height - 1
    has_invalid_x = np.any(
        (x_coordinates < 0) | (x_coordinates >= image_width)
    )

    has_invalid_y = np.any(
        (y_coordinates < 0) | (y_coordinates >= image_height)
    )

    if has_invalid_x or has_invalid_y:
        print(f"警告：{label_name} 中存在超出图像边界的坐标")
    else:
        print(f"{label_name} 坐标范围正常")


check_coordinate_range(
    positive_rectangles,
    image_width,
    image_height,
    "正抓取标注",
)

check_coordinate_range(
    negative_rectangles,
    image_width,
    image_height,
    "负抓取标注",
)


# ============================================================
# 7. 把抓取矩形画到图片上
# ============================================================

# 正确抓取使用绿色。
draw_grasp_rectangles(
    image=image,
    rectangles=positive_rectangles,
    color=(0, 255, 0),
    label_prefix="POS",
)

# 错误抓取使用红色。
draw_grasp_rectangles(
    image=image,
    rectangles=negative_rectangles,
    color=(0, 0, 255),
    label_prefix="NEG",
)


# ============================================================
# 8. 保存可视化结果
# ============================================================

# cv2.imwrite 将图片写入磁盘。
# 成功返回 True，失败返回 False。
save_success = cv2.imwrite(str(output_path), image)

if not save_success:
    raise RuntimeError(f"保存图片失败：{output_path}")


# ============================================================
# 9. 输出运行摘要
# ============================================================

print()
print("样本检查完成")
print(f"RGB 图片：{rgb_path}")
print(f"图像高度：{image_height}")
print(f"图像宽度：{image_width}")
print(f"图像通道数：{number_of_channels}")
print(f"正抓取矩形数量：{len(positive_rectangles)}")
print(f"负抓取矩形数量：{len(negative_rectangles)}")
print(f"可视化结果：{output_path}")
