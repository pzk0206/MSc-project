"""
VLM-guided CNN Grasp Backend — 从 VLM crop 直接回归抓取矩形参数。

架构（对应论文第三条 pipeline）:

    RGB image + prompt
        -> VLM localization (Grounding DINO)
        -> VLM crop
        -> CNN grasp regressor  <-- 这个脚本
        -> grasp rectangle [center_x, center_y, width, height, angle]
        -> Cornell-style evaluation

与传统 CV baseline 和 VLM+geometric pipeline 的对比:
    1. Traditional CV baseline:  整图 OpenCV 分割 + 几何规则
    2. VLM + geometric backend:  VLM 定位 + OpenCV 几何规则
    3. VLM + CNN backend:        VLM 定位 + 学习式抓取回归  <-- 这个脚本

使用方式:
    # 训练 CNN backend
    conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode train

    # 评估全部样本
    conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode eval

    # 训练 + 评估
    conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Required by strict deterministic CUDA matrix operations. Set it before any
# model creates a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ——— isort / black 会把 torch 导入移到上面，这里保持注释结构清晰 ———
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 基线里定义的 evaluate_prediction 和 Cornell 标准常量。
from src.baseline_cv.run_cv_baseline import (  # noqa: E402
    ANGLE_THRESHOLD_DEGREES,
    IOU_THRESHOLD,
    evaluate_prediction,
)
from src.shared.cornell_dataset import CornellGraspDataset  # noqa: E402
from src.shared.grasp_geometry import normalize_angle_radians, rectangles_to_center_format  # noqa: E402
from src.vlm.cnn_grasp_models import (  # noqa: E402
    MultiHeadCNNGraspRegressor,
    SingleHeadCNNGraspRegressor,
    compute_multi_head_loss,
)

# ——— 路径常量 ———
DATASET_ROOT = Path("data/raw/cornell")
VLM_PREDICTIONS_CSV = Path("data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv")
BASELINE_PREDICTIONS_CSV = Path("data/processed/baseline_cv/cv_baseline_predictions.csv")

OUTPUT_DIR = Path("data/processed/vlm/cnn_grasp")
MODEL_WEIGHTS = OUTPUT_DIR / "cnn_grasp_model.pt"
PREDICTIONS_CSV = OUTPUT_DIR / "cnn_grasp_predictions.csv"
SUMMARY_JSON = OUTPUT_DIR / "cnn_grasp_summary.json"
TRAIN_HISTORY_JSON = OUTPUT_DIR / "training_history.json"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    model_weights: Path
    predictions_csv: Path
    summary_json: Path
    training_history_json: Path
    visualization_dir: Path


@dataclass(frozen=True)
class SeedOutputPaths:
    model_weights: Path
    training_history_json: Path
    predictions_csv: Path
    summary_json: Path


def build_output_paths(output_dir: Path) -> OutputPaths:
    """Build all generated artifact paths from one CLI-controlled directory."""
    return OutputPaths(
        output_dir=output_dir,
        model_weights=output_dir / "cnn_grasp_model.pt",
        predictions_csv=output_dir / "cnn_grasp_predictions.csv",
        summary_json=output_dir / "cnn_grasp_summary.json",
        training_history_json=output_dir / "training_history.json",
        visualization_dir=output_dir / "visualizations",
    )


def build_seed_output_paths(output_dir: Path, seed: int) -> SeedOutputPaths:
    """Build non-overlapping artifacts for one repeated training run."""
    return SeedOutputPaths(
        model_weights=output_dir / f"cnn_grasp_model_seed_{seed}.pt",
        training_history_json=output_dir / f"training_history_seed_{seed}.json",
        predictions_csv=output_dir / f"cnn_grasp_predictions_seed_{seed}.csv",
        summary_json=output_dir / f"cnn_grasp_summary_seed_{seed}.json",
    )


def resolve_output_dir(architecture: str, output_dir: Path | None) -> Path:
    """Choose an architecture-safe default without overriding an explicit path."""
    if output_dir is not None:
        return output_dir
    if architecture == "multi_head":
        return Path("data/processed/vlm/cnn_grasp_multi_head")
    return Path("data/processed/vlm/cnn_grasp")


def configure_output_paths(output_dir: Path) -> OutputPaths:
    """Configure legacy module globals while keeping path construction testable."""
    global OUTPUT_DIR, MODEL_WEIGHTS, PREDICTIONS_CSV
    global SUMMARY_JSON, TRAIN_HISTORY_JSON, VISUALIZATION_DIR

    paths = build_output_paths(output_dir)
    OUTPUT_DIR = paths.output_dir
    MODEL_WEIGHTS = paths.model_weights
    PREDICTIONS_CSV = paths.predictions_csv
    SUMMARY_JSON = paths.summary_json
    TRAIN_HISTORY_JSON = paths.training_history_json
    VISUALIZATION_DIR = paths.visualization_dir
    return paths


def configure_reproducibility(seed: int):
    """Seed all RNGs and require deterministic algorithms for this runtime."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

# ——— 训练超参数 ———
CROP_SIZE = 224          # VLM crop resize 到的正方形尺寸
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ——— 旧实验的数据集划分：按 Cornell 存储子目录划分 ———
TRAIN_DIRS = {"01", "02", "03", "04", "05", "06"}
VAL_DIRS = {"07", "08"}
TEST_DIRS = {"09", "10"}


# ======================================================================
# 数据集：VLM Crop 提取
# ======================================================================

def load_vlm_boxes() -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """读取 VLM 预测中的 expanded_box 坐标，按 (dir, sample_id) 索引。"""
    boxes = {}
    with open(VLM_PREDICTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["object_directory"], row["sample_id"])
            try:
                box = (
                    int(float(row["expanded_box_x1"])),
                    int(float(row["expanded_box_y1"])),
                    int(float(row["expanded_box_x2"])),
                    int(float(row["expanded_box_y2"])),
                )
                boxes[key] = box
            except (ValueError, TypeError):
                continue
    return boxes


def extract_crop_and_label(
    sample: dict,
    vlm_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[dict]] | None:
    """
    从 Cornell 样本中提取 VLM crop 和对应的抓取标签。

    返回:
        (crop_rgb, gt_rectangles_in_crop_coords) 或 None
    """
    x1, y1, x2, y2 = vlm_box
    image = sample["rgb"]  # HxWxC BGR
    h, w = image.shape[:2]

    # 边界裁剪防止越界
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # 将 Cornell 正抓取矩形转到 crop 坐标系
    positive_rects = sample["positive_rectangles"]  # (N, 4, 2)
    gt_list = []
    crop_w = x2 - x1
    crop_h = y2 - y1
    for rect in positive_rects:
        rect_crop = rect.copy()
        rect_crop[:, 0] -= x1
        rect_crop[:, 1] -= y1

        # 接受抓取中心在 crop 内的 GT（不要求四个角点全在 crop 内）
        center_x = float(np.mean(rect_crop[:, 0]))
        center_y = float(np.mean(rect_crop[:, 1]))
        if 0 <= center_x < crop_w and 0 <= center_y < crop_h:
            gt_list.append(rect_crop)

    if not gt_list:
        return None

    return crop, gt_list


def crop_to_tensor(crop: np.ndarray, target_size: int = CROP_SIZE) -> np.ndarray:
    """
    将 crop (HxWxC BGR) 转成归一化的 RGB tensor。
    返回 shape (3, target_size, target_size)，值域 [-1, 1] 附近。
    """
    import torch

    # BGR → RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # resize
    rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # HWC -> CHW, uint8 -> float32
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    # 简单归一化: 减均值除标准差 (ImageNet 统计量近似)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def gt_to_target(gt_rects: list[np.ndarray], crop_w: int, crop_h: int) -> dict:
    """
    将一个样本的多个 GT 抓取矩形转成训练目标。

    策略: 选择面积最大的 GT 矩形（通常是最主要的抓取）。
    输出归一化到 [0, 1] 的坐标和 sin/cos 角度。
    """
    if not gt_rects:
        raise ValueError("gt_rects 不能为空")

    # 选择面积最大的
    best_idx = 0
    best_area = -1.0
    for i, rect in enumerate(gt_rects):
        w_rect = np.linalg.norm(rect[1] - rect[0])
        h_rect = np.linalg.norm(rect[2] - rect[1])
        area = w_rect * h_rect
        if area > best_area:
            best_area = area
            best_idx = i

    best_rect = gt_rects[best_idx]
    center_fmt = rectangles_to_center_format(best_rect[np.newaxis, :, :])[0]

    # 归一化中心点 (相对于 crop 尺寸)
    cx_norm = center_fmt["center_x"] / crop_w
    cy_norm = center_fmt["center_y"] / crop_h

    # 归一化宽高 (除以 crop 对角线长度，保证数值范围合理)
    diag = math.sqrt(crop_w**2 + crop_h**2)
    w_norm = center_fmt["width"] / diag
    h_norm = center_fmt["height"] / diag

    # 角度用 sin(2θ) / cos(2θ) 处理 180° 等价性
    theta = center_fmt["angle_radians"]
    sin_2theta = math.sin(2 * theta)
    cos_2theta = math.cos(2 * theta)

    return {
        "cx": float(cx_norm),
        "cy": float(cy_norm),
        "width": float(w_norm),
        "height": float(h_norm),
        "sin_2theta": float(sin_2theta),
        "cos_2theta": float(cos_2theta),
    }


def build_all_samples() -> list[dict]:
    """
    遍历 Cornell 数据集并按 VLM box 构建全部有效 crop 样本。

    返回:
        每个 sample 是 dict:
        {"key": (dir, sample_id), "tensor": Tensor, "target": dict}
    """
    vlm_boxes = load_vlm_boxes()
    dataset = CornellGraspDataset(DATASET_ROOT)

    all_samples = []

    for i in range(len(dataset)):
        sample = dataset[i]
        key = (sample["object_directory"], sample["sample_id"])

        if key not in vlm_boxes:
            continue

        result = extract_crop_and_label(sample, vlm_boxes[key])
        if result is None:
            continue

        crop, gt_rects = result
        crop_h, crop_w = crop.shape[:2]

        try:
            target = gt_to_target(gt_rects, crop_w, crop_h)
        except (ValueError, IndexError):
            continue

        crop_tensor = crop_to_tensor(crop)

        item = {
            "key": key,
            "tensor": crop_tensor,
            "target": target,
        }
        all_samples.append(item)

    return all_samples


def partition_samples_by_role(
    samples: list[dict],
    roles: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition samples using explicit roles keyed by Cornell sample ID."""
    partitions = {"train": [], "validation": [], "test": []}
    for item in samples:
        sample_id = item["key"][1]
        if sample_id not in roles:
            raise ValueError(f"missing role for sample {sample_id}")
        role = roles[sample_id]
        if role not in partitions:
            raise ValueError(f"unknown role for sample {sample_id}: {role}")
        partitions[role].append(item)
    return (
        partitions["train"],
        partitions["validation"],
        partitions["test"],
    )


def build_datasets():
    """Build the legacy fixed-directory train/validation/test split."""
    all_samples = build_all_samples()
    roles = {
        item["key"][1]: (
            "train"
            if item["key"][0] in TRAIN_DIRS
            else "validation"
            if item["key"][0] in VAL_DIRS
            else "test"
        )
        for item in all_samples
    }
    train, val, test = partition_samples_by_role(all_samples, roles)

    print(f"数据集构建完成: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# Backward-compatible name used by existing analysis commands.
CNNGraspRegressor = SingleHeadCNNGraspRegressor


def create_model(architecture: str):
    if architecture == "single":
        return SingleHeadCNNGraspRegressor()
    if architecture == "multi_head":
        return MultiHeadCNNGraspRegressor()
    raise ValueError(f"unsupported architecture: {architecture}")


def flatten_model_output(output):
    """Convert either architecture output to the established six-value order."""
    import torch

    if isinstance(output, dict):
        return torch.cat(
            [output["centre"], output["size"], output["orientation"]],
            dim=1,
        )
    return output


def _state_dict_for_save(model, architecture: str) -> dict:
    if architecture == "single":
        return model.model.state_dict()
    return model.state_dict()


def _load_state_dict(model, architecture: str, state_dict: dict) -> None:
    if architecture == "single":
        model.model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


# ======================================================================
# 训练
# ======================================================================

def train_model(
    train_data: list[dict],
    val_data: list[dict],
    device: str = "cuda",
    seed: int = 42,
    architecture: str = "single",
    model_weights_path: Path | None = None,
    history_path: Path | None = None,
) -> tuple[object, list[dict]]:
    """训练 CNN 抓取回归网络。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    train_generator = configure_reproducibility(seed)

    class GraspCropDataset(Dataset):
        def __init__(self, samples: list[dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            t = s["target"]
            target_vec = torch.tensor([
                t["cx"], t["cy"], t["width"], t["height"],
                t["sin_2theta"], t["cos_2theta"],
            ], dtype=torch.float32)
            return s["tensor"], target_vec

    train_loader = DataLoader(
        GraspCropDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        GraspCropDataset(val_data), batch_size=BATCH_SIZE, shuffle=False,
    )

    model = create_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )

    # 损失: Smooth L1 — 对离群值不敏感
    criterion = nn.SmoothL1Loss()

    def calculate_losses(prediction, target):
        if architecture == "multi_head":
            return compute_multi_head_loss(prediction, target)
        total = criterion(prediction, target)
        return {"total": total}

    def empty_totals() -> dict[str, float]:
        names = (
            ("total", "centre", "size", "orientation", "unit_norm")
            if architecture == "multi_head"
            else ("total",)
        )
        return {name: 0.0 for name in names}

    def mean_totals(totals: dict[str, float], count: int) -> dict[str, float]:
        return {name: value / count for name, value in totals.items()}

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(1, NUM_EPOCHS + 1):
        # ——— 训练 ———
        model.train()
        train_totals = empty_totals()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            losses = calculate_losses(pred, batch_y)
            losses["total"].backward()
            optimizer.step()
            for name, loss in losses.items():
                train_totals[name] += loss.item() * batch_x.size(0)

        train_means = mean_totals(train_totals, len(train_data))
        train_loss = train_means["total"]

        # ——— 验证 ———
        model.eval()
        val_totals = empty_totals()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                losses = calculate_losses(pred, batch_y)
                for name, loss in losses.items():
                    val_totals[name] += loss.item() * batch_x.size(0)
        val_means = mean_totals(val_totals, len(val_data))
        val_loss = val_means["total"]

        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss),
        }
        if architecture == "multi_head":
            for name in ("centre", "size", "orientation", "unit_norm"):
                epoch_record[f"train_{name}_loss"] = float(train_means[name])
                epoch_record[f"val_{name}_loss"] = float(val_means[name])
        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.cpu().clone()
                for key, value in _state_dict_for_save(
                    model, architecture
                ).items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 恢复最佳权重
    if best_state is not None:
        _load_state_dict(model, architecture, best_state)

    # 保存模型
    selected_model_path = model_weights_path or MODEL_WEIGHTS
    selected_history_path = history_path or TRAIN_HISTORY_JSON
    selected_model_path.parent.mkdir(parents=True, exist_ok=True)
    selected_history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict_for_save(model, architecture), selected_model_path)
    print(f"模型已保存: {selected_model_path}")

    # 保存训练历史
    with open(selected_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


# ======================================================================
# 推理 + 评估
# ======================================================================

def predict_from_crop(
    model: object,
    crop_tensor: "torch.Tensor",
    crop_w: int,
    crop_h: int,
    device: str = "cuda",
) -> dict:
    """
    从 CNN 输出解码为抓取矩形（在原始图像坐标系中）。

    crop_w/crop_h: 原始 crop 的像素尺寸，用于反归一化。
    """
    import torch

    model.eval()
    with torch.no_grad():
        x = crop_tensor.unsqueeze(0).to(device)
        output = flatten_model_output(model(x))[0].cpu().numpy()

    cx_norm, cy_norm, w_norm, h_norm, sin_2t, cos_2t = output

    # 反归一化中心点
    center_x = float(np.clip(cx_norm, 0.0, 1.0) * crop_w)
    center_y = float(np.clip(cy_norm, 0.0, 1.0) * crop_h)

    # 反归一化尺寸
    diag = math.sqrt(crop_w**2 + crop_h**2)
    width = float(np.clip(w_norm, 0.01, 1.0) * diag)
    height = float(np.clip(h_norm, 0.01, 1.0) * diag)

    # 从 sin(2θ)/cos(2θ) 恢复角度
    theta = math.atan2(sin_2t, cos_2t) / 2.0
    theta = normalize_angle_radians(theta)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "angle_radians": float(theta),
        "angle_degrees": float(math.degrees(theta)),
    }


def _selected_sample_keys(
    samples: list[dict],
) -> set[tuple[str, str]]:
    """Return a unique set of sample keys selected for evaluation."""
    keys = [tuple(item["key"]) for item in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample key in evaluation selection")
    return set(keys)


def evaluate_model(
    model: object,
    all_samples: list[dict],
    dataset: CornellGraspDataset,
    vlm_boxes: dict[tuple[str, str], tuple[int, int, int, int]],
    device: str = "cuda",
) -> tuple[list[dict], dict]:
    """
    在显式选择的 Cornell 样本上评估 CNN backend。

    流程:
    1. 对每个样本，用 VLM box 裁剪
    2. CNN 预测抓取矩形
    3. 反算到原始图像坐标
    4. Cornell-style 评估
    5. 与 baseline 对照
    """
    # 加载 baseline 结果用于对照
    baseline_rows = {}
    if BASELINE_PREDICTIONS_CSV.exists():
        with open(BASELINE_PREDICTIONS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baseline_rows[(row["object_directory"], row["sample_id"])] = row

    rows = []
    success_count = 0
    eval_count = 0
    selected_keys = _selected_sample_keys(all_samples)

    for i in range(len(dataset)):
        sample = dataset[i]
        key = (sample["object_directory"], sample["sample_id"])

        if key not in selected_keys:
            continue
        if key not in vlm_boxes:
            continue

        vlm_box = vlm_boxes[key]
        x1, y1, x2, y2 = vlm_box
        h, w = sample["rgb"].shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = sample["rgb"][y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        crop_tensor = crop_to_tensor(crop)

        prediction = predict_from_crop(model, crop_tensor, crop_w, crop_h, device)

        # 反算到原始图像坐标
        prediction_img = {
            "center_x": prediction["center_x"] + x1,
            "center_y": prediction["center_y"] + y1,
            "width": prediction["width"],
            "height": prediction["height"],
            "angle_degrees": prediction["angle_degrees"],
        }

        positive_gt = rectangles_to_center_format(sample["positive_rectangles"])
        evaluation = evaluate_prediction(prediction_img, positive_gt)

        if evaluation["success"]:
            success_count += 1
        eval_count += 1

        # 获取 baseline 对照
        bl = baseline_rows.get(key, {})

        row = {
            "sample_id": sample["sample_id"],
            "object_directory": sample["object_directory"],
            "success": int(evaluation["success"]),
            "best_iou": evaluation["best_iou"],
            "best_angle_error_degrees": evaluation["best_angle_error_degrees"],
            "matched_gt_index": evaluation["matched_gt_index"],
            "pred_center_x": prediction_img["center_x"],
            "pred_center_y": prediction_img["center_y"],
            "pred_width": prediction_img["width"],
            "pred_height": prediction_img["height"],
            "pred_angle_degrees": prediction_img["angle_degrees"],
            "vlm_box_x1": x1, "vlm_box_y1": y1, "vlm_box_x2": x2, "vlm_box_y2": y2,
            "baseline_success": bl.get("success", "N/A"),
            "baseline_iou": bl.get("best_iou", "N/A"),
            "baseline_angle": bl.get("best_angle_error_degrees", "N/A"),
            "rgb_path": str(sample["rgb_path"]),
        }
        rows.append(row)

    # 统计指标
    mean_iou = float(np.mean([r["best_iou"] for r in rows])) if rows else 0.0
    mean_angle = float(np.mean([r["best_angle_error_degrees"] for r in rows])) if rows else 0.0
    success_rate = success_count / eval_count if eval_count else 0.0

    summary = {
        "method_name": "vlm_cnn_grasp_regressor_rgb",
        "sample_count": eval_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "mean_best_iou": mean_iou,
        "mean_best_angle_error_degrees": mean_angle,
        "iou_threshold": IOU_THRESHOLD,
        "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
        "predictions_csv": str(PREDICTIONS_CSV),
    }

    return rows, summary


def save_results(
    rows: list[dict],
    summary: dict,
    predictions_csv: Path | None = None,
    summary_json: Path | None = None,
) -> None:
    """保存预测结果和汇总。"""
    selected_predictions_csv = predictions_csv or PREDICTIONS_CSV
    selected_summary_json = summary_json or SUMMARY_JSON
    selected_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_summary_json.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(
            selected_predictions_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"预测结果 CSV: {selected_predictions_csv}")

    with open(selected_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"总结 JSON: {selected_summary_json}")


def save_visualizations(
    rows: list[dict],
    dataset: CornellGraspDataset,
    max_per_class: int = 10,
) -> None:
    """保存成功/失败可视化样本。"""
    for subdir_name in ["success", "failure"]:
        subdir = VISUALIZATION_DIR / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

    success_saved, fail_saved = 0, 0
    for row in rows:
        if row["success"] == 1 and success_saved >= max_per_class:
            continue
        if row["success"] == 0 and fail_saved >= max_per_class:
            continue

        # 找到对应样本
        for i in range(len(dataset)):
            s = dataset[i]
            if s["sample_id"] == row["sample_id"] and s["object_directory"] == row["object_directory"]:
                canvas = s["rgb"].copy()

                # 绿色: GT
                for gt in rectangles_to_center_format(s["positive_rectangles"]):
                    rect = ((gt["center_x"], gt["center_y"]), (gt["width"], gt["height"]), gt["angle_degrees"])
                    pts = cv2.boxPoints(rect).astype(np.int32)
                    cv2.polylines(canvas, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

                # 蓝色: CNN 预测
                pred_rect = ((row["pred_center_x"], row["pred_center_y"]),
                             (row["pred_width"], row["pred_height"]),
                             row["pred_angle_degrees"])
                pred_pts = cv2.boxPoints(pred_rect).astype(np.int32)
                cv2.polylines(canvas, [pred_pts], True, (255, 0, 0), 2, cv2.LINE_AA)

                # 黄色: VLM box (如果有)
                if "vlm_box_x1" in row and row["vlm_box_x1"] != "":
                    cv2.rectangle(canvas, (int(float(row["vlm_box_x1"])), int(float(row["vlm_box_y1"]))),
                                  (int(float(row["vlm_box_x2"])), int(float(row["vlm_box_y2"]))), (0, 255, 255), 1)

                status = "SUCCESS" if row["success"] == 1 else "FAIL"
                cv2.putText(canvas, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0) if row["success"] == 1 else (0, 0, 255), 2, cv2.LINE_AA)

                subdir = "success" if row["success"] == 1 else "failure"
                path = VISUALIZATION_DIR / subdir / f"{row['object_directory']}_{row['sample_id']}.png"
                cv2.imwrite(str(path), canvas)

                if row["success"] == 1:
                    success_saved += 1
                else:
                    fail_saved += 1
                break

    print(f"可视化已保存: {VISUALIZATION_DIR} (success={success_saved}, failure={fail_saved})")


# ======================================================================
# 主入口
# ======================================================================

def print_comparison_table(cnn_summary: dict) -> None:
    """打印三类方法对比表。"""
    # 读取 VLM+geometric 结果
    vlm_geo_path = Path("data/processed/vlm/grasp/vlm_assisted_grasp_summary.json")
    vlm_geo = {}
    if vlm_geo_path.exists():
        with open(vlm_geo_path) as f:
            vlm_geo = json.load(f)

    # 读取传统 CV baseline 结果
    cv_path = Path("data/processed/baseline_cv/cv_baseline_summary.json")
    cv_bl = {}
    if cv_path.exists():
        with open(cv_path) as f:
            cv_bl = json.load(f)

    print()
    print("=" * 80)
    print("三类方法对比")
    print("=" * 80)
    print(f"{'方法':<35} {'定位前端':<18} {'抓取后端':<18} {'成功率':<10} {'平均IoU':<10} {'平均角度':<10}")
    print("-" * 80)

    if cv_bl:
        print(f"{'Traditional CV baseline':<35} {'无 (全图阈值)':<18} {'OpenCV 几何':<18} "
              f"{cv_bl.get('success_rate', 0)*100:.1f}%{'':<4} "
              f"{cv_bl.get('mean_best_iou', 0):.4f}{'':<4} "
              f"{cv_bl.get('mean_best_angle_error_degrees', 0):.2f}°")

    if vlm_geo:
        print(f"{'VLM + Geometric backend':<35} {'Grounding DINO':<18} {'OpenCV 几何':<18} "
              f"{vlm_geo.get('success_rate', 0)*100:.1f}%{'':<4} "
              f"{vlm_geo.get('mean_best_iou', 0):.4f}{'':<4} "
              f"{vlm_geo.get('mean_best_angle_error_degrees', 0):.2f}°")

    print(f"{'VLM + CNN backend':<35} {'Grounding DINO':<18} {'CNN regressor':<18} "
          f"{cnn_summary.get('success_rate', 0)*100:.1f}%{'':<4} "
          f"{cnn_summary.get('mean_best_iou', 0):.4f}{'':<4} "
          f"{cnn_summary.get('mean_best_angle_error_degrees', 0):.2f}°")

    print("-" * 80)
    print()


def _train_one_run(
    train_data,
    val_data,
    device,
    seed,
    architecture="single",
    model_weights_path: Path | None = None,
    history_path: Path | None = None,
):
    """单次训练，返回模型和验证集 best loss。"""
    import torch
    model, history = train_model(
        train_data,
        val_data,
        device=device,
        seed=seed,
        architecture=architecture,
        model_weights_path=model_weights_path,
        history_path=history_path,
    )
    best_val_loss = min(h["val_loss"] for h in history)
    return model, best_val_loss


def _eval_on_splits(model, dataset, vlm_boxes, device):
    """在全部样本上评估，并按 train/val/test 分组统计。"""
    import torch

    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        key = (sample["object_directory"], sample["sample_id"])
        if key not in vlm_boxes:
            continue

        x1, y1, x2, y2 = vlm_boxes[key]
        h, w = sample["rgb"].shape[:2]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))

        crop = sample["rgb"][y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        crop_tensor = crop_to_tensor(crop)
        prediction = predict_from_crop(model, crop_tensor, crop_w, crop_h, device)

        prediction_img = {
            "center_x": prediction["center_x"] + x1,
            "center_y": prediction["center_y"] + y1,
            "width": prediction["width"],
            "height": prediction["height"],
            "angle_degrees": prediction["angle_degrees"],
        }

        positive_gt = rectangles_to_center_format(sample["positive_rectangles"])
        evaluation = evaluate_prediction(prediction_img, positive_gt)

        obj_dir = sample["object_directory"]
        if obj_dir in TRAIN_DIRS:
            split = "train"
        elif obj_dir in VAL_DIRS:
            split = "val"
        elif obj_dir in TEST_DIRS:
            split = "test"
        else:
            split = "unknown"

        rows.append({
            "sample_id": sample["sample_id"],
            "object_directory": obj_dir,
            "split": split,
            "success": int(evaluation["success"]),
            "best_iou": evaluation["best_iou"],
            "best_angle_error_degrees": evaluation["best_angle_error_degrees"],
            "pred_center_x": prediction_img["center_x"],
            "pred_center_y": prediction_img["center_y"],
            "pred_width": prediction_img["width"],
            "pred_height": prediction_img["height"],
            "pred_angle_degrees": prediction_img["angle_degrees"],
        })

    def stats(subset_rows):
        if not subset_rows:
            return {"count": 0, "success_rate": 0.0, "mean_iou": 0.0, "mean_angle": 0.0}
        return {
            "count": len(subset_rows),
            "success_rate": sum(r["success"] for r in subset_rows) / len(subset_rows),
            "mean_iou": float(np.mean([r["best_iou"] for r in subset_rows])),
            "mean_angle": float(np.mean([r["best_angle_error_degrees"] for r in subset_rows])),
        }

    return {
        "all": stats(rows),
        "train": stats([r for r in rows if r["split"] == "train"]),
        "val": stats([r for r in rows if r["split"] == "val"]),
        "test": stats([r for r in rows if r["split"] == "test"]),
        "rows": rows,
    }


def build_multi_run_summary(
    run_records: list[dict],
    architecture: str = "single",
) -> dict:
    """将逐轮指标汇总为稳定、可序列化的统计结构。"""

    def aggregate(split_name: str) -> dict:
        rates = [record[split_name]["success_rate"] for record in run_records]
        ious = [record[split_name]["mean_iou"] for record in run_records]
        angles = [record[split_name]["mean_angle"] for record in run_records]
        return {
            "success_rate_mean": float(np.mean(rates)),
            "success_rate_std": float(np.std(rates)),
            "mean_iou_mean": float(np.mean(ious)),
            "mean_iou_std": float(np.std(ious)),
            "mean_angle_mean": float(np.mean(angles)),
            "mean_angle_std": float(np.std(angles)),
        }

    return {
        "method": f"vlm_cnn_{architecture}_multi_run",
        "architecture": architecture,
        "num_runs": len(run_records),
        "seeds": [int(record["seed"]) for record in run_records],
        "all": aggregate("all"),
        "test": aggregate("test"),
        "per_run": [
            {
                "seed": int(record["seed"]),
                "best_val_loss": float(record["best_val_loss"]),
                "all_success_rate": float(record["all"]["success_rate"]),
                "test_success_rate": float(record["test"]["success_rate"]),
            }
            for record in run_records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-guided CNN grasp backend")
    parser.add_argument("--mode", choices=["train", "eval", "all", "multi"], default="all",
                        help="运行模式: train / eval / all / multi (多次训练评估)")
    parser.add_argument("--device", default="cuda",
                        help="设备: cuda / cpu")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="multi 模式下训练次数")
    parser.add_argument("--seed", type=int, default=42,
                        help="单次训练随机种子 (multi 模式下为起始种子)")
    parser.add_argument(
        "--architecture",
        choices=["single", "multi_head"],
        default="single",
        help="CNN 输出结构: single / multi_head",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="模型、训练历史、预测和汇总目录；默认按 architecture 隔离",
    )
    args = parser.parse_args()
    output_paths = configure_output_paths(
        resolve_output_dir(args.architecture, args.output_dir)
    )

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # === 多轮训练评估 ===
    if args.mode == "multi":
        print("\n>>> 构建数据集...")
        train_data, val_data, test_data = build_datasets()
        print(f"训练集: {len(train_data)}  验证集: {len(val_data)}  测试集: {len(test_data)}")

        vlm_boxes = load_vlm_boxes()
        dataset = CornellGraspDataset(DATASET_ROOT)

        run_records = []
        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx
            print(f"\n{'='*60}")
            print(f">>> Run {run_idx + 1}/{args.num_runs}  (seed={seed})")
            print(f"{'='*60}")

            seed_paths = build_seed_output_paths(output_paths.output_dir, seed)
            model, best_val_loss = _train_one_run(
                train_data,
                val_data,
                device,
                seed,
                architecture=args.architecture,
                model_weights_path=seed_paths.model_weights,
                history_path=seed_paths.training_history_json,
            )
            print(f"best val_loss = {best_val_loss:.6f}")

            result = _eval_on_splits(model, dataset, vlm_boxes, device)
            run_record = {
                "seed": seed,
                "best_val_loss": best_val_loss,
                "all": result["all"],
                "test": result["test"],
                "rows": result["rows"],
            }
            run_records.append(run_record)

            save_results(
                result["rows"],
                {
                    "method_name": (
                        f"vlm_cnn_{args.architecture}_grasp_regressor_rgb"
                    ),
                    "architecture": args.architecture,
                    "seed": seed,
                    "best_val_loss": best_val_loss,
                    "sample_count": result["all"]["count"],
                    "success_count": sum(
                        row["success"] for row in result["rows"]
                    ),
                    "success_rate": result["all"]["success_rate"],
                    "mean_best_iou": result["all"]["mean_iou"],
                    "mean_best_angle_error_degrees": result["all"]["mean_angle"],
                    "test_sample_count": result["test"]["count"],
                    "test_success_rate": result["test"]["success_rate"],
                    "test_mean_iou": result["test"]["mean_iou"],
                    "test_mean_angle_error_degrees": result["test"]["mean_angle"],
                    "iou_threshold": IOU_THRESHOLD,
                    "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
                    "predictions_csv": str(seed_paths.predictions_csv),
                },
                predictions_csv=seed_paths.predictions_csv,
                summary_json=seed_paths.summary_json,
            )

            print(f"  Full:      {result['all']['success_rate']*100:.1f}%  "
                  f"IoU={result['all']['mean_iou']:.4f}  angle={result['all']['mean_angle']:.2f}°")
            print(f"  Test only: {result['test']['success_rate']*100:.1f}%  "
                  f"IoU={result['test']['mean_iou']:.4f}  angle={result['test']['mean_angle']:.2f}°")

        # ——— 汇总统计 ———
        print(f"\n{'='*80}")
        print(f"多轮训练汇总 (n={args.num_runs})")
        print(f"{'='*80}")

        for subset_name in ["all", "test"]:
            rates = [
                record[subset_name]["success_rate"] * 100
                for record in run_records
            ]
            ious = [record[subset_name]["mean_iou"] for record in run_records]
            angles = [record[subset_name]["mean_angle"] for record in run_records]

            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            mean_iou = np.mean(ious)
            std_iou = np.std(ious)
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            print(f"\n  {subset_name.upper()} set:")
            print(f"    成功率:    {mean_rate:.2f}% ± {std_rate:.2f}%")
            print(f"    平均 IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
            print(f"    平均角度:  {mean_angle:.2f}° ± {std_angle:.2f}°")
            print(f"    各轮结果:  {[f'{r:.1f}%' for r in rates]}")

        # 保存汇总 JSON
        summary = build_multi_run_summary(
            run_records,
            architecture=args.architecture,
        )
        multi_json = OUTPUT_DIR / "multi_run_summary.json"
        with open(multi_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n多轮汇总 JSON: {multi_json}")

        # 保存最后一次的 predictions 和可视化
        last_record = run_records[-1]
        last_rows = last_record["rows"]
        save_results(last_rows, {
            "method_name": "vlm_cnn_multi_run_last",
            "architecture": args.architecture,
            "seed": last_record["seed"],
            "best_val_loss": last_record["best_val_loss"],
            "sample_count": last_record["all"]["count"],
            "success_count": int(
                last_record["all"]["success_rate"] * last_record["all"]["count"]
            ),
            "success_rate": last_record["all"]["success_rate"],
            "mean_best_iou": last_record["all"]["mean_iou"],
            "mean_best_angle_error_degrees": last_record["all"]["mean_angle"],
            "iou_threshold": IOU_THRESHOLD,
            "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
            "predictions_csv": str(PREDICTIONS_CSV),
        })
        save_visualizations(last_rows, dataset, max_per_class=10)

        return

    # === 单次训练/评估（原有逻辑） ===
    if args.mode in ("train", "all"):
        print("\n>>> 构建数据集...")
        train_data, val_data, test_data = build_datasets()
        print(f"训练集: {len(train_data)}  验证集: {len(val_data)}  测试集: {len(test_data)}")

        if not train_data:
            print("错误: 训练集为空，请检查 VLM 预测和数据集划分。")
            return

        print("\n>>> 训练 CNN 抓取回归网络...")
        model, history = train_model(
            train_data,
            val_data,
            device=device,
            seed=args.seed,
            architecture=args.architecture,
        )
        print(f"训练完成, best val_loss = {min(h['val_loss'] for h in history):.6f}")

    if args.mode in ("eval", "all"):
        print("\n>>> 全量评估...")
        import torch
        model = create_model(args.architecture).to(device)
        if MODEL_WEIGHTS.exists():
            state_dict = torch.load(
                MODEL_WEIGHTS,
                map_location=device,
                weights_only=True,
            )
            _load_state_dict(model, args.architecture, state_dict)
            print(f"加载模型权重: {MODEL_WEIGHTS}")
        else:
            print("警告: 未找到模型权重，使用随机初始化权重评估。")

        vlm_boxes = load_vlm_boxes()
        dataset = CornellGraspDataset(DATASET_ROOT)
        result = _eval_on_splits(model, dataset, vlm_boxes, device)
        rows = result["rows"]

        print(f"\nCNN Backend 评估结果:")
        for split_name in ["all", "train", "val", "test"]:
            s = result[split_name]
            if s["count"] > 0:
                print(f"  {split_name}: {s['count']} samples, "
                      f"success={s['success_rate']*100:.1f}%, "
                      f"IoU={s['mean_iou']:.4f}, angle={s['mean_angle']:.2f}°")

        summary = {
            "method_name": f"vlm_cnn_{args.architecture}_grasp_regressor_rgb",
            "architecture": args.architecture,
            "sample_count": result["all"]["count"],
            "success_count": int(result["all"]["success_rate"] * result["all"]["count"]),
            "success_rate": result["all"]["success_rate"],
            "mean_best_iou": result["all"]["mean_iou"],
            "mean_best_angle_error_degrees": result["all"]["mean_angle"],
            "test_success_rate": result["test"]["success_rate"],
            "test_mean_iou": result["test"]["mean_iou"],
            "test_mean_angle": result["test"]["mean_angle"],
            "iou_threshold": IOU_THRESHOLD,
            "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
            "predictions_csv": str(PREDICTIONS_CSV),
        }

        save_results(rows, summary)
        print_comparison_table(summary)
        save_visualizations(rows, dataset, max_per_class=10)


if __name__ == "__main__":
    main()
