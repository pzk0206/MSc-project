"""
Cornell Grasping Dataset 的正式数据读取器。

这个文件和前面的检查脚本不一样：

- check_cornell_dataset.py
  主要用于“检查数据集有没有问题”。

- visualize_sample_checks.py
  主要用于“抽样画图，人工确认标注是否合理”。

- cornell_dataset.py
  主要用于“后续训练/标签转换/实验时稳定读取数据”。

换句话说：
前两个脚本是体检工具；
这个文件是以后真正干活的数据入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import cv2
import numpy as np


@dataclass(frozen=True)
class CornellSamplePaths:
    """
    保存一个 Cornell 样本对应的所有文件路径。

    为什么要用 dataclass？
    --------------------
    因为一个样本有好几个文件：

    - RGB 图片
    - depth 深度图
    - cpos 正抓取标注
    - cneg 负抓取标注
    - point cloud 点云文件

    如果全部用零散变量传来传去，很容易混乱。
    dataclass 可以把这些路径打包成一个清楚的小对象。
    """

    sample_id: str
    object_directory: str
    rgb_path: Path
    depth_path: Path
    cpos_path: Path
    cneg_path: Path
    point_cloud_path: Path


def load_grasp_rectangles(label_path: Path, allow_empty: bool = False) -> np.ndarray:
    """
    读取 Cornell 的抓取矩形标注。

    Cornell 的 cpos/cneg 标注文件格式是：

        x1 y1
        x2 y2
        x3 y3
        x4 y4

    每 4 行表示一个抓取矩形。

    参数
    ----------
    label_path:
        标注文件路径，可以是 cpos.txt 或 cneg.txt。

    allow_empty:
        是否允许空标注文件。

        cpos 正抓取标注不应该为空，所以读取 cpos 时传 False。
        cneg 负抓取标注可以为空，所以读取 cneg 时传 True。

    返回
    ----------
    rectangles:
        shape 为 (N, 4, 2) 的 numpy 数组。

        N = 抓取矩形数量
        4 = 每个矩形的四个角点
        2 = 每个角点的 x, y 坐标
    """

    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件：{label_path}")

    # 空的 cneg 文件是 Cornell 里可能出现的可接受情况。
    # 这里返回一个形状正确的空数组，表示“0 个负抓取框”。
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


class CornellGraspDataset:
    """
    Cornell Grasping Dataset 读取器。

    这个类模仿 PyTorch Dataset 的常见形式：

    - len(dataset)
      返回数据集样本数量。

    - dataset[index]
      返回第 index 个样本。

    但它暂时不继承 torch.utils.data.Dataset。
    原因是我们现在还处在数据解析阶段，先保持依赖简单；
    等确认读取逻辑稳定后，再接 PyTorch 会更稳。
    """

    def __init__(self, dataset_root: str | Path):
        """
        初始化数据集。

        参数
        ----------
        dataset_root:
            Cornell 数据集根目录，例如：

                data/raw/cornell
        """

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"找不到数据集目录：{self.dataset_root}")

        # 扫描所有样本路径。
        self.samples = self._build_sample_index()

        if not self.samples:
            raise RuntimeError(f"没有找到任何 Cornell 样本：{self.dataset_root}")

    def _build_sample_index(self) -> list[CornellSamplePaths]:
        """
        扫描数据集目录，建立样本索引。

        为什么以 cpos 文件作为入口？
        --------------------------
        因为对 2D 抓取检测来说，正抓取标注是最核心的训练答案。
        一个样本如果没有 cpos，后面很难用于监督训练。

        所以这里用所有 pcd*cpos.txt 文件作为样本列表来源。
        """

        cpos_paths = sorted(self.dataset_root.rglob("pcd*cpos.txt"))
        samples: list[CornellSamplePaths] = []

        for cpos_path in cpos_paths:
            sample_directory = cpos_path.parent

            # pcd0100cpos.txt -> pcd0100cpos -> pcd0100
            sample_id = cpos_path.stem.removesuffix("cpos")

            rgb_path = sample_directory / f"{sample_id}r.png"
            depth_path = sample_directory / f"{sample_id}d.tiff"
            cneg_path = sample_directory / f"{sample_id}cneg.txt"
            point_cloud_path = sample_directory / f"{sample_id}.txt"

            sample = CornellSamplePaths(
                sample_id=sample_id,
                object_directory=sample_directory.name,
                rgb_path=rgb_path,
                depth_path=depth_path,
                cpos_path=cpos_path,
                cneg_path=cneg_path,
                point_cloud_path=point_cloud_path,
            )

            self._check_required_files(sample)
            samples.append(sample)

        return samples

    @staticmethod
    def _check_required_files(sample: CornellSamplePaths) -> None:
        """
        检查一个样本的必要文件是否存在。

        注意：
        这里检查的是文件是否存在，不检查文件内容。
        文件内容读取和格式检查会在 __getitem__ 里发生。
        """

        required_paths = {
            "rgb_path": sample.rgb_path,
            "depth_path": sample.depth_path,
            "cpos_path": sample.cpos_path,
            "cneg_path": sample.cneg_path,
            "point_cloud_path": sample.point_cloud_path,
        }

        missing = [
            f"{name}={path}" for name, path in required_paths.items() if not path.exists()
        ]

        if missing:
            missing_text = "\n".join(missing)
            raise FileNotFoundError(
                f"样本 {sample.sample_id} 缺少必要文件：\n{missing_text}"
            )

    def __len__(self) -> int:
        """
        返回数据集样本数量。

        这样就可以写：

            len(dataset)
        """

        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """
        读取第 index 个样本。

        返回的是一个字典，里面既有原始路径，也有已经读取好的数据。

        这样设计的好处：
        - 调试时能看到样本来自哪个文件；
        - 训练时能直接使用 rgb/depth/rectangles；
        - 后面保存错误案例时也容易追踪回原始样本。
        """

        sample_paths = self.samples[index]

        rgb_image = cv2.imread(str(sample_paths.rgb_path), cv2.IMREAD_COLOR)
        if rgb_image is None:
            raise FileNotFoundError(f"无法读取 RGB 图片：{sample_paths.rgb_path}")

        # IMREAD_UNCHANGED 表示保持 tiff 深度图原始数据类型。
        # 不要用 IMREAD_COLOR，否则深度图会被当成普通彩色图读坏。
        depth_image = cv2.imread(str(sample_paths.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"无法读取 depth 图片：{sample_paths.depth_path}")

        positive_rectangles = load_grasp_rectangles(
            sample_paths.cpos_path,
            allow_empty=False,
        )

        negative_rectangles = load_grasp_rectangles(
            sample_paths.cneg_path,
            allow_empty=True,
        )

        return {
            "sample_id": sample_paths.sample_id,
            "object_directory": sample_paths.object_directory,
            "rgb_path": sample_paths.rgb_path,
            "depth_path": sample_paths.depth_path,
            "cpos_path": sample_paths.cpos_path,
            "cneg_path": sample_paths.cneg_path,
            "point_cloud_path": sample_paths.point_cloud_path,
            "rgb": rgb_image,
            "depth": depth_image,
            "positive_rectangles": positive_rectangles,
            "negative_rectangles": negative_rectangles,
        }

    def save_index_csv(self, output_path: str | Path) -> None:
        """
        把数据集索引保存成 CSV 文件。

        这个 CSV 不保存图片本身，只保存每个样本对应的文件路径。
        它的意义是：

        - 以后写论文/报告时，可以说明数据集索引如何生成；
        - 以后训练脚本可以直接读取这个索引；
        - 如果路径或样本数量出问题，可以快速定位。
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_id",
                    "object_directory",
                    "rgb_path",
                    "depth_path",
                    "cpos_path",
                    "cneg_path",
                    "point_cloud_path",
                ],
            )
            writer.writeheader()

            for sample in self.samples:
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "object_directory": sample.object_directory,
                        "rgb_path": sample.rgb_path,
                        "depth_path": sample.depth_path,
                        "cpos_path": sample.cpos_path,
                        "cneg_path": sample.cneg_path,
                        "point_cloud_path": sample.point_cloud_path,
                    }
                )


def main() -> None:
    """
    简单测试这个 Dataset 是否能正常工作。

    直接运行：

        python src/shared/cornell_dataset.py

    它会：
    1. 建立 Cornell 数据集索引；
    2. 读取第一个样本；
    3. 打印关键 shape；
    4. 保存 cornell_dataset_index.csv。
    """

    dataset = CornellGraspDataset("data/raw/cornell")

    print("CornellGraspDataset 加载完成")
    print(f"样本数量：{len(dataset)}")

    sample = dataset[0]

    print()
    print("第一个样本信息：")
    print(f"sample_id：{sample['sample_id']}")
    print(f"object_directory：{sample['object_directory']}")
    print(f"rgb_path：{sample['rgb_path']}")
    print(f"depth_path：{sample['depth_path']}")
    print(f"RGB shape：{sample['rgb'].shape}")
    print(f"Depth shape：{sample['depth'].shape}")
    print(f"Depth dtype：{sample['depth'].dtype}")
    print(f"正抓取矩形 shape：{sample['positive_rectangles'].shape}")
    print(f"负抓取矩形 shape：{sample['negative_rectangles'].shape}")

    output_csv = Path("data/processed/shared/metadata/cornell_dataset_index.csv")
    dataset.save_index_csv(output_csv)

    print()
    print(f"数据集索引 CSV 已保存：{output_csv}")


if __name__ == "__main__":
    main()
