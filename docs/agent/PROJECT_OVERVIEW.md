# Project Overview

## Research topic

本项目研究 **VLM-guided 2D Robotic Grasp Rectangle Detection**：使用预训练开放词汇视觉语言模型作为目标定位前端，再由几何方法或学习式方法预测二维机器人抓取矩形。

## Research question

核心问题是：预训练 VLM 的开放词汇目标定位能力能否在不针对 Cornell 数据集微调定位模型的情况下，提高二维抓取检测性能；在相同 VLM 定位前端下，传统几何后端与 CNN 学习式后端各有什么优势和局限。

## Dataset

项目使用 Cornell Grasping Dataset。每个样本通常包括 RGB 图像、深度图、点云，以及正负抓取矩形标注。当前实验使用 885 个样本；本地数据放在：

```text
data/raw/cornell/
```

数据集和生成的实验产物不提交到 Git。

## Compared methods

### 1. Traditional CV baseline

直接在完整 RGB 图像上执行颜色和亮度阈值分割，提取主要物体轮廓，并用最小面积旋转外接矩形产生抓取预测。该方法不使用 VLM，作为低成本、可解释的基准。

### 2. VLM-guided geometric pipeline

Grounding DINO 根据文本 prompt 产生目标定位框；OpenCV 几何后端只在定位区域中执行分割和轮廓分析，并根据物体几何估计抓取矩形。

### 3. VLM-guided CNN pipeline

沿用相同的 Grounding DINO 定位前端，将目标 crop 输入轻量 CNN，回归抓取矩形的位置、尺寸和方向参数。该方法用于检验学习式抓取后端是否能改善位置精度和 unseen-object 泛化。

## Evaluation protocol

预测抓取矩形与 Cornell 正抓取标注逐一比较。只要存在一个标注同时满足以下条件，该预测即判定为成功：

```text
IoU >= 0.25
angular error <= 30°
```

主要报告成功率、mean best IoU 和 mean angle error。CNN 结果同时区分 single run 与 5-run mean ± std，避免将单次结果与重复实验统计混淆。

## Main findings

- VLM 定位将成功率从 Traditional CV 的 56.95% 提升至 VLM + geometric 的 73.33%，是当前系统的主要性能增益来源。
- CNN 后端在完整数据集上取得更高的 mean best IoU。
- CNN 后端在未见物体测试集上优于几何后端，体现出更好的泛化。
- 几何后端的平均角度误差更低，说明显式角度先验仍有价值。
- Grounding DINO 在当前 prompt 和数据设置下覆盖全部样本，因此后续改进重点应放在抓取后端。

## Scope and limitations

- 当前结论主要来自 Cornell Grasping Dataset，尚不能直接代表真实机器人系统或更复杂场景。
- 当前任务预测二维抓取矩形，没有执行真实机械臂控制或闭环抓取。
- Grounding DINO 的全覆盖结果不等同于定位框完美；定位误差仍可能影响后端。
- CNN 的重复实验统计降低了随机划分或初始化造成的偶然性，但仍需要更多数据集验证。

## Related documents

- [Current project status](CURRENT_STATUS.md)
- [Repository structure](PROJECT_STRUCTURE.md)
- [Code organization guidelines](CODE_ORGANIZATION.md)
- [Research plan](../planning/vlm_robotic_grasp_study_plan.md)
- [Failure analysis](../debugging/FAILURE_ANALYSIS.md)
- [Project worklog](../worklog/WORKLOG.md)
