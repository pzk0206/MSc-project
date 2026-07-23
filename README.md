# MSc Project: VLM-guided 2D Robotic Grasp Detection

本仓库是 MSc Robotics & Artificial Intelligence 毕业项目代码库，研究方向为：

```text
VLM-guided 2D Robotic Grasp Rectangle Detection
```

项目目标是验证预训练开放词汇视觉语言模型是否可以作为机器人抓取任务中的目标定位前端，并比较传统几何后端与后续学习式后端在 2D 抓取矩形检测中的表现。

当前代码主要基于 Cornell Grasping Dataset，完成了数据解析、传统 OpenCV baseline、Grounding DINO 目标定位，以及 VLM-assisted 几何抓取矩形预测。

## 当前进展

当前已经完成三条可复现实验线：

1. **Traditional CV baseline**
   - RGB 图像
   - OpenCV 颜色/亮度阈值分割
   - 轮廓提取
   - 最小面积旋转外接矩形
   - Cornell-style 抓取矩形评估

2. **VLM-guided geometric pipeline**
   - RGB 图像 + 文本 prompt
   - Grounding DINO 开放词汇目标定位
   - 在 VLM box 内运行 OpenCV 几何后端
   - 生成并评估 2D 抓取矩形

3. **VLM-guided CNN pipeline**
   - RGB 图像 + 文本 prompt
   - Grounding DINO 开放词汇目标定位
   - VLM crop → 轻量 CNN 回归抓取矩形参数
   - 端到端评估

当前全量 Cornell 数据集结果（885 样本）：

| 方法 | 定位前端 | 抓取后端 | 成功数 | 成功率 | 平均 best IoU | 平均角度误差 |
|---|---:|---:|---:|---:|---:|---:|
| Traditional CV baseline | 无（全图阈值） | OpenCV 几何 | 504 | 56.95% | 0.3360 | 29.62° |
| VLM + Geometric backend | Grounding DINO | OpenCV 几何 | 649 | 73.33% | 0.4182 | **14.81°** |
| VLM + CNN backend | Grounding DINO | CNN regressor | 647 | 73.11% | **0.4476** | 15.97° |

**Test set 对比**（目录 09-10，85 个 CNN 未见过物体实例）：

| 方法 | Test 成功率 |
|---|---|
| VLM + Geometric backend | 75.3% (64/85) |
| VLM + CNN backend | **81.2% (69/85)** |

Grounding DINO 使用 prompt `small object` 在当前 Cornell 实验中实现了 885 / 885 的目标检测结果。

### 关键发现

- **VLM 定位带来最大提升**：从 56.95% → 73.33%，证明 VLM 作为定位前端价值显著
- **CNN 泛化更好**：在 unseen objects 上 81.2% vs 几何后端 75.3%
- **CNN IoU 更高**：0.4476 vs 0.4182，证明学习式方法的位置/尺寸预测更准确
- **几何方法角度更准**：14.81° vs 15.97°，"长轴垂直方向"启发式规则是有效的角度先验

## 仓库结构

```text
.
├── AGENTS.md
├── src/
│   ├── shared/
│   │   ├── cornell_dataset.py
│   │   ├── grasp_geometry.py
│   │   ├── check_cornell_dataset.py
│   │   ├── export_cornell_grasp_labels.py
│   │   ├── inspect_sample.py
│   │   └── visualize_sample_checks.py
│   ├── baseline_cv/
│   │   ├── run_cv_baseline.py
│   │   ├── visualize_mask_pipeline.py
│   │   └── visualize_mask_pipeline_batch.py
│   └── vlm/
│       ├── prompts.py
│       ├── run_grounding_dino_localization.py
│       ├── run_vlm_assisted_grasp.py
│       ├── run_cnn_grasp.py
│       ├── analyze_failures.py
│       ├── INSTALL.md
│       └── README.md
└── docs/
    ├── agent/
    │   ├── CURRENT_STATUS.md
    │   ├── PROJECT_OVERVIEW.md
    │   ├── PROJECT_STRUCTURE.md
    │   └── CODE_ORGANIZATION.md
    ├── debugging/
    │   ├── BUGLOG.md
    │   └── FAILURE_ANALYSIS.md
    ├── planning/
    │   └── vlm_robotic_grasp_study_plan.md
    └── worklog/
        ├── WORKLOG.md
        ├── weekly_progress_2026-07-06.md
        └── weekly_progress_2026-07-16.md
```

后续 AI 会话从 `docs/agent/CURRENT_STATUS.md` 开始；人工回顾从
`docs/worklog/WORKLOG.md` 开始。代码放置规则见
`docs/agent/CODE_ORGANIZATION.md`。

## 数据集

本项目使用 Cornell Grasping Dataset。数据集文件不上传到 GitHub，需要放在本地：

```text
data/raw/cornell/
```

一个 Cornell 样本通常包含：

```text
pcd0100r.png       RGB 图像
pcd0100d.tiff      depth 图像
pcd0100cpos.txt    正抓取矩形
pcd0100cneg.txt    负抓取矩形
pcd0100.txt        点云文件
```

`data/` 已经写入 `.gitignore`，避免把数据集、实验输出和可视化结果提交到仓库。

## 环境

主要 Python 依赖包括：

```text
opencv-python
numpy
Pillow
torch (2.5.1+cu121)
torchvision
transformers
```

当前 VLM 部分使用 Hugging Face Grounding DINO：

```text
IDEA-Research/grounding-dino-tiny
```

项目本地环境名：

```text
msc-grasp
```

## 运行方式

检查 Cornell 数据集：

```bash
conda run -n msc-grasp python src/shared/check_cornell_dataset.py
```

导出 Cornell 抓取标签中心格式：

```bash
conda run -n msc-grasp python src/shared/export_cornell_grasp_labels.py
```

运行传统 OpenCV baseline：

```bash
conda run -n msc-grasp python src/baseline_cv/run_cv_baseline.py
```

运行小批量 Grounding DINO 定位：

```bash
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py
```

运行全量 Grounding DINO 定位：

```bash
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda
```

运行 VLM-assisted 抓取矩形预测：

```bash
conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py
```

训练并评估 VLM-guided CNN grasp backend：

```bash
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all --device cuda
```

运行失败案例分析：

```bash
python3 src/vlm/analyze_failures.py
```

## 输出文件

实验输出默认保存到：

```text
data/processed/
```

主要结果包括：

```text
data/processed/baseline_cv/cv_baseline_predictions.csv
data/processed/baseline_cv/cv_baseline_summary.json

data/processed/vlm/localization/grounding_dino_generic_small_object_predictions.csv
data/processed/vlm/localization/grounding_dino_generic_small_object_summary.json

data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv
data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
data/processed/vlm/grasp/failure_analysis.csv

data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv
data/processed/vlm/cnn_grasp/cnn_grasp_summary.json
data/processed/vlm/cnn_grasp/cnn_grasp_model.pt
data/processed/vlm/cnn_grasp/training_history.json
```

这些结果文件不上传到 GitHub，避免仓库变大，也保证代码仓库只保存可复现逻辑。

## 评估标准

项目使用 Cornell grasp detection 常用的 rectangle metric：

```text
IoU >= 0.25
angle error <= 30 degrees
```

预测抓取矩形和 Cornell 正抓取标注进行匹配，只要满足 IoU 和角度误差两个条件，即认为该样本预测成功。

## 下一步计划

三条实验 pipeline 已经全部完成。下一阶段重点转向：

```text
1. 论文写作
   - 填充 dissertation LaTeX 各章节
   - 整理实验图表和三类方法对比
   - 完成 Introduction / Background / Methodology / Results / Discussion

2. 补充分析（如有时间）
   - CNN 预测的 per-sample 错误分析
   - 探索几何+CNN 混合后端（几何初始化角度 + CNN 回归位置）
   - 在更难数据集（Jacquard）上验证泛化性
```

当前核心结论已经形成：VLM 零样本定位显著提升抓取检测（+16.4%），瓶颈在抓取后端；
CNN 学习式后端在 unseen objects 上泛化优于几何后端（81.2% vs 75.3%），
但几何后端的角度先验仍然有价值。

## 不上传到 GitHub 的内容

以下内容已通过 `.gitignore` 排除：

```text
data/
*.zip
__pycache__/
.codex/
.agents/
.venv/
venv/
```

这样可以避免误提交 Cornell 数据集、实验输出、大压缩包、缓存文件和本地工具文件。
