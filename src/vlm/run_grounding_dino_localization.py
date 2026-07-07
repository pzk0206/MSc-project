"""
在 Cornell RGB 图片上运行 Grounding DINO 目标定位。

中文说明
--------
这个文件是 VLM 部分的第一阶段：只负责“找物体”，不负责“画抓取框”。

为什么要把定位和抓取框生成分开？
因为项目需要回答两个不同问题：

1. VLM 能不能在 Cornell 图片里找到目标物体？
2. 找到物体以后，后面的算法能不能生成正确的抓取矩形？

如果两个步骤混在一起，失败时就不知道到底是 VLM 没找到物体，
还是抓取框生成逻辑有问题。

这个脚本是 VLM 流程的第一阶段：

    RGB 图片 + prompt
        -> Grounding DINO 目标框
        -> CSV 预测结果
        -> 定位可视化图片

注意：这个脚本还不生成抓取矩形。
它的目标是先检查 VLM 能不能定位目标物体。

预期输出：

    data/processed/vlm/localization/grounding_dino_<prompt>_predictions.csv
    data/processed/vlm/localization/grounding_dino_<prompt>_summary.json
    data/processed/vlm/visualizations/localization_checks/<prompt>/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import inspect
import json
import random
import re
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.cornell_dataset import CornellGraspDataset
from src.vlm.prompts import GENERIC_PROMPT, normalize_grounding_prompt


# Cornell 原始数据位置。
DATASET_ROOT = Path("data/raw/cornell")

# VLM 相关结果统一放在 data/processed/vlm 下，和 baseline_cv 分开。
OUTPUT_ROOT = Path("data/processed/vlm")

# 保存每张图的 VLM box 数值结果，例如 x1/y1/x2/y2、score、label。
LOCALIZATION_DIR = OUTPUT_ROOT / "localization"

# 保存画了 VLM box 的图片，方便肉眼检查定位是否合理。
VISUALIZATION_DIR = OUTPUT_ROOT / "visualizations" / "localization_checks"


def make_slug(text: str) -> str:
    """
    把 prompt 文本转换成安全的文件名片段。

    作用：
    把 prompt 转成可以放进文件名的字符串。

    例子：
        "small object" -> "small_object"

    这样不同 prompt 的实验结果不会互相覆盖。
    """

    # 全部转小写，去掉首尾空格，方便文件名统一。
    text = text.lower().strip()

    # 文件名里不要出现空格、斜杠、标点等特殊字符，统一替换成下划线。
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    # 如果 prompt 全是特殊字符，至少返回一个可用名字。
    return text or "empty"


def choose_sample_indices(
    dataset: CornellGraspDataset,
    samples_per_directory: int,
    random_seed: int,
    run_all: bool,
) -> list[int]:
    """
    选择本次要运行 VLM 定位的 Cornell 样本。

    默认不是全量运行，而是延续前面 sanity check 的风格：
    从每个 Cornell 子目录里抽少量样本。

    参数解释：
    - run_all=True：跑完整 885 张 Cornell 图片。
    - run_all=False：每个子文件夹抽 samples_per_directory 张，用于快速试验。
    - random_seed：保证每次抽样结果一致，方便复现实验。
    """

    if run_all:
        # 全量实验：直接返回 0 到 len(dataset)-1 的所有索引。
        return list(range(len(dataset)))

    # 固定随机种子，避免今天抽出来的样本和明天不一样。
    random.seed(random_seed)

    # Cornell 按物体目录分成 01、02、...、10。
    # 这里先把每个目录下有哪些样本 index 收集起来。
    directory_to_indices: dict[str, list[int]] = {}
    for index, sample_paths in enumerate(dataset.samples):
        directory_to_indices.setdefault(sample_paths.object_directory, []).append(index)

    selected_indices: list[int] = []

    for directory_name in sorted(directory_to_indices):
        indices = directory_to_indices[directory_name]
        number_to_sample = min(samples_per_directory, len(indices))

        # 每个目录抽同样数量的样本，避免只看某一类物体。
        selected_indices.extend(random.sample(indices, number_to_sample))

    # 排序不是算法必须，但输出顺序更稳定、更好读。
    return sorted(selected_indices)


def select_best_detection(result: dict) -> dict | None:
    """
    从 Grounding DINO 的候选结果中选择 score 最高的检测框。

    Grounding DINO 可能返回多个候选框。
    当前 Cornell 场景通常是单目标抓取，所以这里简单选择 score 最高的一个框。
    """

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    if len(boxes) == 0:
        return None

    # scores 可能是 torch Tensor，也可能已经是 numpy/list。
    # 为了后面用 np.argmax，这里统一转成 numpy。
    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    # score 最大的候选框就是当前脚本采用的 VLM 定位结果。
    best_index = int(np.argmax(scores_np))

    box = boxes[best_index]
    score = scores[best_index]
    label = labels[best_index] if len(labels) > best_index else ""

    # box 也可能是 torch Tensor；保存 CSV 前要变成普通 Python list。
    if hasattr(box, "detach"):
        box_values = box.detach().cpu().numpy().astype(float).tolist()
    else:
        box_values = np.asarray(box, dtype=float).tolist()

    # score 同理，转成普通 float，避免 json/csv 保存时类型不兼容。
    if hasattr(score, "detach"):
        score_value = float(score.detach().cpu().item())
    else:
        score_value = float(score)

    return {
        "box": box_values,
        "score": score_value,
        "label": str(label),
    }


def run_grounding_dino_on_image(
    image_path: Path,
    prompt: str,
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> dict | None:
    """
    对单张图片运行 Grounding DINO，并返回最佳检测结果。

    输入：
    - image_path：Cornell RGB 图片路径
    - prompt：文本提示，例如 "small object"
    - processor/model：Hugging Face 加载出的 Grounding DINO 组件
    - device：cpu 或 cuda

    输出：
    - None：没有检测到物体
    - dict：包含 box、score、label 的最佳检测结果
    """

    # PIL 读图是 Hugging Face processor 常用格式。
    pil_image = Image.open(image_path).convert("RGB")

    # 统一 prompt 格式，比如自动补句号。
    prompt = normalize_grounding_prompt(prompt)

    # processor 会把图片和文本变成模型需要的 tensor。
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")

    # 把所有输入 tensor 移到同一个设备上：CPU 或 GPU。
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 推理阶段不需要计算梯度，可以省显存、加速。
    with torch.no_grad():
        outputs = model(**inputs)

    # Hugging Face 后处理需要知道原图尺寸，才能把 box 缩放回原图坐标。
    # PIL size 是 (width, height)，这里需要的是 (height, width)。
    target_sizes = [pil_image.size[::-1]]

    # transformers 不同版本对这个 API 的参数名有过变化。
    # 这里先拿到当前环境中的后处理函数，再根据函数签名决定传什么参数。
    post_process = processor.post_process_grounded_object_detection
    signature = inspect.signature(post_process)

    # 后处理会把模型原始输出转成人类可读的检测框。
    post_process_kwargs = {
        "outputs": outputs,
        "input_ids": inputs.get("input_ids"),
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }

    # transformers 版本差异：
    # - 有些版本使用 threshold
    # - 有些示例/文档使用 box_threshold
    if "threshold" in signature.parameters:
        post_process_kwargs["threshold"] = box_threshold
    else:
        post_process_kwargs["box_threshold"] = box_threshold

    results = post_process(**post_process_kwargs)

    # results[0] 对应这一张图片；再从候选框里选 score 最高的。
    return select_best_detection(results[0])


def draw_localization_result(
    rgb_path: Path,
    output_path: Path,
    prompt: str,
    detection: dict | None,
) -> None:
    """
    把 Grounding DINO 定位结果画到 RGB 图片上。

    这一步不是模型必须的，而是为了人眼检查：
    如果 CSV 数值看起来很好，但图片上框错了，说明实验结论不能信。
    """

    # OpenCV 读取的是 BGR 图片；这里只是画框保存，不需要转 RGB。
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{rgb_path}")

    if detection is not None:
        # detection["box"] 是 [x1, y1, x2, y2]，表示左上角和右下角。
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
        score = detection["score"]
        label = detection["label"]

        # 蓝色矩形：VLM 找到的目标区域。
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            image,
            text,
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        # 如果没有检测结果，直接在图上写 NO DETECTION，方便筛失败案例。
        cv2.putText(
            image,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 把 prompt 也写在图上，避免以后忘了这批结果是用什么文本跑出来的。
    cv2.putText(
        image,
        f"prompt: {prompt}",
        (20, image.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"保存可视化失败：{output_path}")


def write_predictions_csv(rows: list[dict], output_path: Path) -> None:
    """
    把 VLM 定位结果写成和 baseline 风格接近的 CSV。

    CSV 的意义：
    - 后续 VLM-assisted grasp 脚本会读取这里的 box；
    - 论文实验表格可以从这里统计 detection rate；
    - 出错时能追踪每张图的路径、prompt、score 和 error。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # Cornell 样本身份信息。
        "sample_id",
        "object_directory",
        # prompt 信息，用于区分 generic prompt 和 user prompt 实验。
        "prompt_setting",
        "prompt",
        # VLM 是否检测到目标。
        "detected",
        # Grounding DINO 输出的 box 坐标。
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        # 检测置信度和文本 label。
        "score",
        "label",
        # 模型、原图路径和可视化路径，方便复现实验。
        "model_name",
        "rgb_path",
        "visualization_path",
        # 如果某张图推理报错，把错误信息写在这里。
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_localization_overview(image_paths: list[Path], output_path: Path) -> None:
    """
    保存一张定位结果总览图，方便快速人工检查。

    contact sheet 就是一张“大拼图”。
    用它可以快速扫一遍 VLM 框是否大体正确，不用一张一张打开。
    """

    if not image_paths:
        return

    from PIL import ImageDraw

    thumb_width = 320
    thumb_height = 240
    label_height = 30
    columns = 2
    rows = (len(image_paths) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB").resize((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_height + 6), image_path.name, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    # argparse 负责解析命令行参数。
    # 例如：
    #   python src/vlm/run_grounding_dino_localization.py --all --device cuda
    parser = argparse.ArgumentParser(
        description="在 Cornell RGB 图片上运行 Grounding DINO 定位。"
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face 模型 ID。",
    )
    parser.add_argument(
        "--prompt-setting",
        choices=["generic", "user_prompt"],
        default="generic",
        help="写入输出 CSV 的 prompt 设置名称。",
    )
    parser.add_argument(
        "--prompt",
        default=GENERIC_PROMPT,
        help="用于目标定位的 prompt。",
    )
    parser.add_argument(
        "--samples-per-directory",
        type=int,
        default=2,
        help="每个 Cornell 子目录抽取多少个样本运行。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行全部 Cornell 样本，而不是小批量 sanity-check 子集。",
    )
    parser.add_argument("--random-seed", type=int, default=42)

    # box_threshold 越高，模型越保守，低置信度框会被过滤掉。
    parser.add_argument("--box-threshold", type=float, default=0.25)

    # text_threshold 控制检测框和文本 prompt 的匹配强度。
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto 表示有 cuda 就用 cuda，否则使用 cpu。",
    )

    args = parser.parse_args()

    # auto 表示如果有 GPU 就用 cuda，否则退回 cpu。
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # CornellGraspDataset 负责扫描 raw/cornell 下面所有样本路径。
    dataset = CornellGraspDataset(DATASET_ROOT)

    # 决定本次实验跑哪些样本：小批量抽样或全量 885 张。
    selected_indices = choose_sample_indices(
        dataset=dataset,
        samples_per_directory=args.samples_per_directory,
        random_seed=args.random_seed,
        run_all=args.all,
    )

    print(f"加载模型：{args.model_id}")
    print(f"设备：{device}")
    print(f"prompt setting：{args.prompt_setting}")
    print(f"prompt：{args.prompt}")
    print(f"待处理样本数：{len(selected_indices)}")

    # processor 负责图片/文本预处理；model 是真正的 Grounding DINO 模型。
    # 第一次运行可能需要从 Hugging Face 下载模型文件。
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)

    # 把模型移动到 CPU/GPU，并切换到 eval 推理模式。
    model.to(device)
    model.eval()

    # run_name 决定输出文件名，避免不同 prompt 的实验互相覆盖。
    prompt_slug = make_slug(args.prompt)
    run_name = f"grounding_dino_{args.prompt_setting}_{prompt_slug}"

    predictions_path = LOCALIZATION_DIR / f"{run_name}_predictions.csv"
    summary_path = LOCALIZATION_DIR / f"{run_name}_summary.json"
    visualization_dir = VISUALIZATION_DIR / f"{args.prompt_setting}_{prompt_slug}"
    overview_path = OUTPUT_ROOT / "visualizations" / f"{run_name}_overview.png"

    rows: list[dict] = []
    detected_count = 0
    visualization_paths: list[Path] = []

    # 逐张图片运行 Grounding DINO。
    for index in selected_indices:
        sample = dataset[index]
        rgb_path = sample["rgb_path"]
        visualization_path = (
            visualization_dir
            / f"{sample['object_directory']}_{sample['sample_id']}_grounding_dino.png"
        )

        error_message = ""
        detection = None

        try:
            # 对单张 Cornell RGB 图运行 VLM 定位。
            detection = run_grounding_dino_on_image(
                image_path=rgb_path,
                prompt=args.prompt,
                processor=processor,
                model=model,
                device=device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        except Exception as error:
            # 单张图失败不让整个全量实验崩掉，而是把错误写进 CSV。
            error_message = str(error)

        # 不管是否检测成功，都保存一张可视化，方便之后检查失败案例。
        draw_localization_result(
            rgb_path=rgb_path,
            output_path=visualization_path,
            prompt=args.prompt,
            detection=detection,
        )

        if detection is not None:
            detected_count += 1
            x1, y1, x2, y2 = detection["box"]
            score = detection["score"]
            label = detection["label"]
        else:
            x1 = y1 = x2 = y2 = score = ""
            label = ""

        # 把这一张图的结果存成一行，最后统一写 CSV。
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "object_directory": sample["object_directory"],
                "prompt_setting": args.prompt_setting,
                "prompt": args.prompt,
                "detected": int(detection is not None),
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "score": score,
                "label": label,
                "model_name": args.model_id,
                "rgb_path": rgb_path,
                "visualization_path": visualization_path,
                "error": error_message,
            }
        )

        status = "detected" if detection is not None else "no detection"
        print(f"{sample['object_directory']}/{sample['sample_id']}: {status}")
        visualization_paths.append(visualization_path)

    write_predictions_csv(rows, predictions_path)

    # summary JSON 保存本次实验整体结果，后面写论文表格时直接读它即可。
    summary = {
        "model_name": args.model_id,
        "prompt_setting": args.prompt_setting,
        "prompt": args.prompt,
        "sample_count": len(selected_indices),
        "detected_count": detected_count,
        "detection_rate": detected_count / len(selected_indices)
        if selected_indices
        else 0.0,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "predictions_csv": str(predictions_path),
        "visualization_dir": str(visualization_dir),
        "overview_path": str(overview_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_localization_overview(visualization_paths, overview_path)

    print()
    print("Grounding DINO localization 完成")
    print(f"检测成功数量：{detected_count}/{len(selected_indices)}")
    print(f"检测率：{summary['detection_rate']:.4f}")
    print(f"预测 CSV：{predictions_path}")
    print(f"总结 JSON：{summary_path}")
    print(f"可视化目录：{visualization_dir}")
    print(f"总览图：{overview_path}")


if __name__ == "__main__":
    main()
