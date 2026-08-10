# MSc Project: VLM-guided 2D Robotic Grasp Detection

本仓库是 MSc Robotics & Artificial Intelligence 毕业项目代码库，研究方向为：

```text
开放词汇视觉定位驱动的二维抓取检测与仿真抓取
VLM-guided 2D Robotic Grasp Rectangle Detection with PyBullet Simulation
```

项目验证了预训练开放词汇视觉语言模型（Grounding DINO）作为机器人抓取任务目标定位前端的有效性，比较了传统几何后端与轻量 CNN 回归后端在 2D 抓取矩形检测中的表现，并搭建 Franka Panda 7-DOF PyBullet 仿真系统完成了感知驱动的物理抓取验证。

## 当前进展

项目已完成四条可复现实验线：

1. **Traditional CV baseline** — 全图 OpenCV 颜色/亮度阈值分割 + 轮廓提取 + 最小面积旋转外接矩形
2. **VLM-guided geometric pipeline** — Grounding DINO 文本条件定位 + 定位框内 OpenCV 几何抓取后端
3. **VLM-guided CNN pipeline** — Grounding DINO 目标裁剪 + 单头/多头轻量回归网络
4. **PyBullet simulation pipeline** — 虚拟相机 RGB-D 感知 → 深度反投影 → IK/FK/碰撞审计 → POSITION_CONTROL 分阶段物理抓取

### Cornell 全量数据集结果（885 样本）

| 方法 | 成功数 | 成功率 | 平均最佳 IoU | 平均角度误差 |
|---|---:|---:|---:|---:|
| 传统 CV 基线 | 504 / 885 | 56.95% | 0.3360 | 29.62° |
| VLM + 几何后端 | 649 / 885 | 73.33% | 0.4182 | **14.81°** |
| 单头 CNN（五 seed 均值 ± 标准差） | — | 74.55% ± 1.77% | 0.4535 ± 0.0149 | 16.62° ± 0.83° |
| 多头 CNN（五 seed 均值 ± 标准差） | — | **75.59% ± 1.90%** | **0.4640 ± 0.0165** | **15.20° ± 0.29°** |

Grounding DINO 使用 prompt `small object`，在当前实验中成功定位全部 885 个 Cornell 样本。

VLM 定位前端带来的提升最大：成功率 +16.38 个百分点（56.95% → 73.33%）；这是当前受控比较中的最大测得增益。

### Image-wise 五折交叉验证（885 折外预测）

| 架构 | 成功数 | Pooled 成功率 | 平均 IoU | 平均角度误差 |
|---|---:|---:|---:|---:|
| 单头（432,454 参数） | 635 / 885 | 71.75% | 0.4390 | 17.74° |
| 多头（514,758 参数） | 647 / 885 | **73.11%** | **0.4580** | **17.40°** |

多头相对单头提高 12 个成功样本（+1.36 个百分点）和 0.0190 IoU，角度误差降低 0.34°，表明任务分头带来小幅总体优势。

### 固定测试子集（Cornell 目录 09–10，85 样本）

| 方法 | 测试集成功率 |
|---|---:|
| VLM + 几何后端 | 75.3%（64 / 85） |
| 单头 CNN（五 seed） | **80.47% ± 5.19%** |
| 多头 CNN（五 seed） | **80.47% ± 3.38%** |

### PyBullet 仿真物理抓取

搭建 Franka Panda 7-DOF PyBullet 仿真系统（`3.2.7`，API `202010061`）：

- **深度反投影门控**：3 目标 × 3 后端的九点验证全部通过（重投影、segmentation 与 ray-test 门控 9/9）
- **真值姿态抬升**：方块抬升约 120 mm 并稳定保持 240 步，`physical_grasp_success: true`
- **斜视相机失败定位**：感知抓取失败根因为斜视相机下 24.67–26.62 mm 的 XY 反投影偏差
- **头顶相机修正**：垂直向下相机将 XY 偏差降至 0.76 mm（改善 97.1%），完成感知驱动仿真抓取：抬升 119.94 mm、保持漂移 1.40 mm，`scientific_gate_passed: true`

### 关键发现

- **VLM 定位带来最大提升**：成功率从 56.95% → 73.33%（+16.38 pp）
- **CNN 后端 IoU 更高**：多头 0.4640 vs 几何 0.4182，位置和尺寸预测更准确
- **几何后端角度更准**：14.81° vs 多头 15.20°，物体长轴垂直方向仍是有效抓取角度先验
- **单目深度无法恢复三维中心**：斜视相机 XY 偏差 26.62 mm 导致夹爪仅触及物体边缘、抬升失败；头顶相机从根本上消除了这一偏差
- **感知驱动物理抓取成功**：头顶相机 + 抓取深度修正实现了从 RGB 图像到仿真抬升的完整链条

## 仓库结构

```text
.
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── src/
│   ├── shared/                          # 共享模块：Cornell 数据集、抓取几何、交叉验证
│   │   ├── cornell_dataset.py
│   │   ├── cornell_cross_validation.py
│   │   ├── grasp_geometry.py
│   │   ├── analyze_cornell_splits.py
│   │   └── ...
│   ├── baseline_cv/                     # 传统 CV 基线
│   │   ├── run_cv_baseline.py
│   │   └── visualize_mask_pipeline.py
│   ├── vlm/                             # VLM 定位 + 几何/CNN 抓取后端
│   │   ├── prompts.py
│   │   ├── run_grounding_dino_localization.py
│   │   ├── run_vlm_assisted_grasp.py
│   │   ├── cnn_grasp_models.py          # 单头/多头轻量回归网络定义
│   │   ├── run_cnn_grasp.py
│   │   ├── run_cnn_cross_validation.py
│   │   ├── analyze_failures.py
│   │   └── analyze_backend_comparison.py
│   └── simulation/
│       └── pybullet/                    # PyBullet 仿真系统
│           ├── scene.py                 # 确定性场景与 URDF 管理
│           ├── camera.py                # RGB、米制深度、segmentation 采集
│           ├── perception.py            # 仿真 RGB 接入 VLM + 抓取后端
│           ├── backprojection.py        # 深度反投影与九点门控
│           ├── pose_generation.py       # 俯视悬停姿态生成
│           ├── kinematic_audit.py       # IK/FK、关节限位、碰撞余量审计
│           ├── motion_control.py        # 位置电机轨迹执行
│           ├── gripper_control.py       # 双指闭合与目标接触评价
│           ├── lift_control.py          # 抬升与保持门控
│           ├── execution_plan.py        # 冻结感知执行计划
│           ├── run_stage6b_pipeline.py  # 感知驱动完整物理抓取链条
│           ├── run_overhead_preflight.py # 头顶相机预检
│           └── ...
├── tests/
│   └── simulation/                      # PyBullet 仿真测试套件（241 项）
│       ├── test_pybullet_camera.py
│       ├── test_pybullet_backprojection.py
│       ├── test_pybullet_kinematic_audit.py
│       ├── test_pybullet_truth_lift.py
│       ├── test_pybullet_stage6b_pipeline.py
│       └── ...
├── docs/
│   ├── agent/                           # AI 上下文文档（项目状态、结构、规范）
│   ├── debugging/                       # 调试记录与失败分析
│   ├── planning/                        # 架构设计与文献矩阵
│   ├── reporting/                       # 导师汇报 PDF 与图表生成
│   └── worklog/                         # 工作日志与周报
└── uog_dissertation_outline/            # 毕业论文 LaTeX 源码
    ├── l4proj.tex
    ├── l4proj.bib
    └── images/
```

## 数据集

本项目使用 Cornell Grasping Dataset。数据集文件不上传到 GitHub，需放在本地：

```text
data/raw/cornell/
```

每个 Cornell 样本包含：

```text
pcd0100r.png       RGB 图像
pcd0100d.tiff      深度图像
pcd0100cpos.txt    正抓取矩形
pcd0100cneg.txt    负抓取矩形
pcd0100.txt        点云文件
```

`data/` 已写入 `.gitignore`，数据集、实验输出和可视化结果不会提交到仓库。

## 环境

主要 Python 依赖：

```text
opencv-python
numpy
Pillow
torch (2.5.1+cu121)
torchvision
transformers
pybullet (3.2.7)
```

VLM 部分使用 Hugging Face Grounding DINO：

```text
IDEA-Research/grounding-dino-tiny
```

项目 Conda 环境名：

```text
msc-grasp
```

## 运行方式

### Cornell 实验

```bash
# 检查 Cornell 数据集
conda run -n msc-grasp python src/shared/check_cornell_dataset.py

# 导出 Cornell 抓取标签
conda run -n msc-grasp python src/shared/export_cornell_grasp_labels.py

# 运行传统 OpenCV 基线
conda run -n msc-grasp python src/baseline_cv/run_cv_baseline.py

# 全量 Grounding DINO 定位
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda

# VLM + 几何抓取后端
conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py

# 训练并评估 CNN 抓取后端
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all --device cuda

# 五次确定性重复实验
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode multi --num-runs 5 --device cuda

# Image-wise 五折交叉验证
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py --mode manifest
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py --mode run --architecture single --fold 0 --device cuda
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py --mode compare
```

### PyBullet 仿真

```bash
# 固定三物体目标选择与深度反投影
conda run -n msc-grasp python src/simulation/pybullet/run_multi_object_study.py --device cuda

# 姿态 IK/FK 与碰撞审计
conda run -n msc-grasp python src/simulation/pybullet/run_pose_ik_study.py

# 阶段 1–5：真值姿态逐步验证
conda run -n msc-grasp python src/simulation/pybullet/run_safe_motion_smoke.py
conda run -n msc-grasp python src/simulation/pybullet/run_truth_pregrasp.py
conda run -n msc-grasp python src/simulation/pybullet/run_truth_approach.py
conda run -n msc-grasp python src/simulation/pybullet/run_truth_contact.py
conda run -n msc-grasp python src/simulation/pybullet/run_truth_lift.py

# Stage 6A：几何感知执行计划预检（静态，不运动）
conda run -n msc-grasp python src/simulation/pybullet/run_geometry_execution_preflight.py --device cuda

# 头顶相机预检（消除斜视 XY 偏差）
conda run -n msc-grasp python src/simulation/pybullet/run_overhead_preflight.py --device cuda

# Stage 6B：感知驱动完整物理抓取链条
conda run -n msc-grasp python src/simulation/pybullet/run_stage6b_pipeline.py --device cuda
```

### 测试

```bash
# 运行全部回归测试（241 项）
conda run -n msc-grasp python -m pytest -q
```

## 评估标准

项目使用 Cornell grasp detection 标准 rectangle metric：

```text
IoU >= 0.25
角度误差 <= 30°
```

预测抓取矩形与 Cornell 正抓取标注进行匹配，满足两个条件即计为预测成功。

## 输出文件

实验输出保存到 `data/processed/`（Git 忽略）：

```text
data/processed/baseline_cv/
├── cv_baseline_predictions.csv
└── cv_baseline_summary.json

data/processed/vlm/
├── localization/                       # Grounding DINO 定位结果
├── grasp/                              # VLM + 几何后端结果
├── cnn_grasp/                          # 单头 CNN 结果
├── cnn_grasp_multi_head/               # 多头 CNN 结果
├── cnn_cross_validation/               # 五折交叉验证结果
│   ├── image_wise_folds_seed_42.json   # 共同 fold manifest（SHA-256 哈希）
│   ├── single/
│   ├── multi_head/
│   └── architecture_comparison.json
└── backend_comparison/                 # 后端逐样本比较

data/processed/pybullet/
├── multi_object_study/                 # 三目标九点反投影产物
└── grasp_execution/                    # 分阶段仿真抓取产物
    ├── stage_1_safe_motion/
    ├── stage_2_cube_pregrasp/
    ├── stage_3_open_approach/
    ├── stage_4_bilateral_contact/
    ├── stage_5_truth_cube_lift/
    ├── stage_6a_geometry_preflight/
    ├── stage_6b_perception_grasp/
    ├── stage_6b_multi_head_grasp/
    └── stage_overhead_grasp_v3/        # 头顶相机物理抓取成功
```

## 实验溯源

项目建立了可审计的实验溯源链：

- 确定性 seeds（42–46）用于五 seed 重复实验，每个 seed 独立保存权重、训练历史、逐样本预测和 summary
- Image-wise 五折使用共同 manifest（SHA-256: `b7d3e22a...`），单头和多头共用同一划分
- 每折独立保存 checkpoint、验证损失和 177 条测试预测
- PyBullet 物理执行产物包含完整状态轨迹 CSV、接触事件、metadata 和关键帧
- 完整回归测试：**241 项全部通过**

## 不上传到 GitHub 的内容

以下内容已通过 `.gitignore` 排除：

```text
data/                  # 数据集、实验输出、可视化
__pycache__/           # Python 缓存
.codex/ .agents/       # 本地工具文件
.venv/ venv/           # 虚拟环境
*.zip *.tar *.tar.gz   # 压缩包
uog_dissertation_outline/*.aux *.log *.out *.toc *.pdf  # LaTeX 编译产物
thesis_export/         # 论文导出目录
```

## 后续方向

论文已完成，核心实验结论已形成。后续研究方向包括：

- **RGB-D 融合**：将深度信息直接输入 CNN 后端，而非仅在反投影阶段使用
- **真实机器人验证**：将当前 PyBullet 仿真管线迁移至真实 Franka Panda 平台
- **多物体抓取规划**：在 clutter 场景中扩展目标选择与避碰抓取排序
- **其他数据集泛化**：在 Jacquard 等更大规模数据集上验证 CNN 后端泛化性
