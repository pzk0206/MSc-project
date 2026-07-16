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
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

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

# ——— 训练超参数 ———
CROP_SIZE = 224          # VLM crop resize 到的正方形尺寸
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ——— 数据集划分：按 Cornell 子目录划分，避免同物体泄露 ———
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


def build_datasets():
    """
    遍历 Cornell 数据集，按 VLM box 裁剪，按目录划分 train/val/test。

    返回:
        train_samples, val_samples, test_samples
        每个 sample 是 dict: {"key": (dir, sample_id), "tensor": Tensor, "target": dict}
    """
    import torch

    vlm_boxes = load_vlm_boxes()
    dataset = CornellGraspDataset(DATASET_ROOT)

    train, val, test = [], [], []

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

        obj_dir = sample["object_directory"]
        if obj_dir in TRAIN_DIRS:
            train.append(item)
        elif obj_dir in VAL_DIRS:
            val.append(item)
        elif obj_dir in TEST_DIRS:
            test.append(item)

    print(f"数据集构建完成: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ======================================================================
# CNN 模型定义
# ======================================================================

class CNNGraspRegressor:
    """
    轻量 CNN 抓取参数回归网络。

    输入:  (B, 3, 224, 224) 归一化 RGB
    输出:  (B, 6) = [cx, cy, width, height, sin(2θ), cos(2θ)]
    """

    def __init__(self):
        import torch
        import torch.nn as nn

        self.model = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            # Block 2: 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            # Block 3: 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            # Block 4: 14 -> 7
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7

            # 全局平均池化 -> (B, 256)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # 全连接
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 6),
        )

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x):
        return self.model(x)


# ======================================================================
# 训练
# ======================================================================

def train_model(
    train_data: list[dict],
    val_data: list[dict],
    device: str = "cuda",
) -> tuple[CNNGraspRegressor, list[dict]]:
    """训练 CNN 抓取回归网络。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

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
        GraspCropDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        GraspCropDataset(val_data), batch_size=BATCH_SIZE, shuffle=False,
    )

    model = CNNGraspRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, verbose=True,
    )

    # 损失: Smooth L1 — 对离群值不敏感
    criterion = nn.SmoothL1Loss()

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(1, NUM_EPOCHS + 1):
        # ——— 训练 ———
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_data)

        # ——— 验证 ———
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_data)

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}
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
        model.model.load_state_dict(best_state)

    # 保存模型
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), MODEL_WEIGHTS)
    print(f"模型已保存: {MODEL_WEIGHTS}")

    # 保存训练历史
    with open(TRAIN_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


# ======================================================================
# 推理 + 评估
# ======================================================================

def predict_from_crop(
    model: CNNGraspRegressor,
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
        output = model(x)[0].cpu().numpy()

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


def evaluate_model(
    model: CNNGraspRegressor,
    all_samples: list[dict],
    dataset: CornellGraspDataset,
    vlm_boxes: dict[tuple[str, str], tuple[int, int, int, int]],
    device: str = "cuda",
) -> tuple[list[dict], dict]:
    """
    在全量 Cornell 数据集上评估 CNN backend。

    流程:
    1. 对每个样本，用 VLM box 裁剪
    2. CNN 预测抓取矩形
    3. 反算到原始图像坐标
    4. Cornell-style 评估
    5. 与 baseline 对照
    """
    import torch

    # 加载 baseline 结果用于对照
    baseline_rows = {}
    if BASELINE_PREDICTIONS_CSV.exists():
        with open(BASELINE_PREDICTIONS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baseline_rows[(row["object_directory"], row["sample_id"])] = row

    rows = []
    success_count = 0
    eval_count = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        key = (sample["object_directory"], sample["sample_id"])

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


def save_results(rows: list[dict], summary: dict) -> None:
    """保存预测结果和汇总。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"预测结果 CSV: {PREDICTIONS_CSV}")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"总结 JSON: {SUMMARY_JSON}")


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

                # 黄色: VLM box
                cv2.rectangle(canvas, (int(row["vlm_box_x1"]), int(row["vlm_box_y1"])),
                              (int(row["vlm_box_x2"]), int(row["vlm_box_y2"])), (0, 255, 255), 1)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-guided CNN grasp backend")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all",
                        help="运行模式: train / eval / all")
    parser.add_argument("--device", default="cuda",
                        help="设备: cuda / cpu")
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    if args.mode in ("train", "all"):
        print("\n>>> 构建数据集...")
        train_data, val_data, test_data = build_datasets()
        print(f"训练集: {len(train_data)}  验证集: {len(val_data)}  测试集: {len(test_data)}")

        if not train_data:
            print("错误: 训练集为空，请检查 VLM 预测和数据集划分。")
            return

        print("\n>>> 训练 CNN 抓取回归网络...")
        model, history = train_model(train_data, val_data, device=device)
        print(f"训练完成, best val_loss = {min(h['val_loss'] for h in history):.6f}")

    if args.mode in ("eval", "all"):
        print("\n>>> 全量评估...")
        import torch
        model = CNNGraspRegressor().to(device)
        if MODEL_WEIGHTS.exists():
            model.model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
            print(f"加载模型权重: {MODEL_WEIGHTS}")
        else:
            print("警告: 未找到模型权重，使用随机初始化权重评估。")

        vlm_boxes = load_vlm_boxes()
        dataset = CornellGraspDataset(DATASET_ROOT)
        rows, summary = evaluate_model(model, [], dataset, vlm_boxes, device=device)

        print(f"\nCNN Backend 评估结果:")
        print(f"  评估样本数: {summary['sample_count']}")
        print(f"  成功数: {summary['success_count']}")
        print(f"  成功率: {summary['success_rate']*100:.2f}%")
        print(f"  平均 best IoU: {summary['mean_best_iou']:.4f}")
        print(f"  平均角度误差: {summary['mean_best_angle_error_degrees']:.2f}°")

        save_results(rows, summary)
        print_comparison_table(summary)

        # 保存可视化
        save_visualizations(rows, dataset, max_per_class=10)


if __name__ == "__main__":
    main()
