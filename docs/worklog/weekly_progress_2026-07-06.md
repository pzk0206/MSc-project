# 周进展报告——VLM 引导的二维机器人抓取检测

日期：2026-07-06  
项目方向：VLM 引导的二维机器人抓取矩形检测

## 1. 本周一句话总结

本周完成了 Cornell Grasping Dataset 上的两个完整抓取检测基线，并完成全数据集量化对比：传统 OpenCV 基线成功率为 **56.95%**，VLM 引导的几何实验流程成功率提升到 **73.33%**。实验表明，VLM 能可靠完成目标定位，当前主要瓶颈已经转移到抓取框生成后端。

## 2. 本周完成的主要工作

### 2.1 数据集与评估流程

- 完成 Cornell Grasping Dataset 全量读取与标签解析。
- 将 Cornell 四点抓取矩形转换为 `center_x, center_y, width, height, angle` 格式。
- 实现 Cornell-style 评估标准：
  - IoU ≥ 0.25；
  - angle error ≤ 30°。
- 完成成功/失败案例可视化输出。

### 2.2 模型一：传统计算机视觉基线

方法流程：

```text
RGB image
→ HSV / brightness threshold
→ object mask
→ contour extraction
→ minAreaRect
→ grasp rectangle
```

作用：

- 作为没有 VLM 的传统方法对照组。
- 用于判断 VLM 定位是否能改善下游抓取检测。

### 2.3 模型二：VLM 引导的几何实验流程

方法流程：

```text
RGB image + prompt
→ Grounding DINO 目标定位
→ VLM bounding box
→ OpenCV geometric backend inside VLM box
→ grasp rectangle
```

使用 prompt：

```text
small object
```

作用：

- 使用 VLM 作为语言条件目标定位前端。
- 后端仍使用 OpenCV 几何方法生成抓取矩形。
- 用于验证 VLM 定位是否能提升最终抓取检测表现。

## 3. 两个模型全量结果对比

数据集：Cornell Grasping Dataset  
样本数：885

| 方法 | 是否使用 VLM | 后端 | 成功数 | 成功率 | 平均 best IoU | 平均角度误差 |
|---|---:|---|---:|---:|---:|---:|
| 传统计算机视觉基线 | 否 | OpenCV 几何后端 | 504 / 885 | 56.95% | 0.3360 | 29.62° |
| VLM 引导的几何实验流程 | 是 | VLM 边界框内的 OpenCV 几何后端 | 649 / 885 | 73.33% | 0.4182 | 14.81° |

VLM localization 单独结果：

| 模型 | Prompt | 检测数 | 检测率 |
|---|---|---:|---:|
| Grounding DINO | small object | 885 / 885 | 100% |

## 4. 关键发现

1. VLM 定位在 Cornell 数据集上非常稳定，Grounding DINO 使用通用提示词 `small object` 实现了 **100% 的目标定位检测率**。

2. 加入 VLM 定位后，最终抓取检测成功率从 **56.95%** 提升到 **73.33%**。

3. 平均角度误差从 **29.62°** 降低到 **14.81°**，说明 VLM 引导的方法不仅提升了位置匹配，也改善了抓取方向估计。

4. 当前主要瓶颈不再是目标定位，而是 VLM 边界框之后的**抓取矩形生成后端**。

5. 因此，现阶段不优先进行 VLM 微调。更合理的下一步是分析失败案例，并引入学习式 CNN 抓取框后端。

## 5. 本周生成的主要代码与结果文件

### 传统计算机视觉基线

代码：

```text
src/baseline_cv/run_cv_baseline.py
```

结果：

```text
data/processed/baseline_cv/cv_baseline_predictions.csv
data/processed/baseline_cv/cv_baseline_summary.json
data/processed/baseline_cv/visualizations/
```

### VLM 定位

代码：

```text
src/vlm/run_grounding_dino_localization.py
src/vlm/prompts.py
src/vlm/check_vlm_environment.py
```

结果：

```text
data/processed/vlm/localization/grounding_dino_generic_small_object_predictions.csv
data/processed/vlm/localization/grounding_dino_generic_small_object_summary.json
data/processed/vlm/visualizations/localization_checks/generic_small_object/
```

### VLM 引导的抓取检测

代码：

```text
src/vlm/run_vlm_assisted_grasp.py
```

结果：

```text
data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv
data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
data/processed/vlm/grasp/visualizations/
```

## 6. 更新后的研究方向

本项目目前可表述为：

```text
VLM 引导的二维机器人抓取矩形检测
```

更具体地说：

```text
使用开放词汇 VLM 作为语言条件目标定位前端，并比较不同抓取框生成后端对 2D 抓取矩形检测性能的影响。
```

当前实验路线：

```text
传统计算机视觉基线
→ VLM 引导的几何后端
→ VLM 引导的 CNN 后端
```

这可以理解为一个模块化 VLA-style grasp perception pipeline：

```text
视觉：RGB / RGB-D 图像
语言：提示词
动作表示：二维抓取矩形
```

## 7. 下周计划

### 7.1 失败案例分析

分析 VLM 引导的几何实验流程中失败的 236 个样本：

```text
总样本数：885
成功：649
失败：236
```

重点统计：

- IoU 不足导致失败；
- 角度误差过大导致失败；
- 抓取中心偏移；
- 抓取框宽高不合适；
- 物体形状不规则导致 OpenCV 几何后端失败。

### 7.2 开始 VLM 引导的 CNN 抓取后端

计划实现：

```text
RGB 图像 + 提示词
→ VLM 定位
→ VLM 裁剪区域
→ CNN 抓取回归器
→ 抓取矩形
```

第一版 CNN 输入：

```text
VLM 裁剪区域 RGB 图像
```

输出：

```text
center_x
center_y
width
height
sin(2θ)
cos(2θ)
```

后续如果 RGB 版本跑通，再扩展到 RGB-D。

## 8. 可以给导师汇报的简短版本

本周完成了 Cornell Grasping Dataset 上的两条完整基线。传统 OpenCV 抓取检测基线在 885 个样本上取得 56.95% 的成功率。随后，将 Grounding DINO 作为基于 VLM 的目标定位前端，并使用通用提示词“small object”。Grounding DINO 成功定位全部 885 个物体，最终的 VLM 引导几何抓取流程将抓取成功率提升至 73.33%。结果表明，VLM 在 Cornell 数据集上的定位非常可靠，剩余瓶颈在后续抓取矩形生成后端。下一步将分析失败案例，并实现轻量 VLM 引导的 CNN 抓取回归器，以替代手工几何后端。
