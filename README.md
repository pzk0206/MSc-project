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

### Cornell metric-v2 主结果（885 样本）

成功判据为：存在同一个 Cornell 正抓取框同时满足 IoU ≥ 0.25 和角度误差
≤ 30°。CNN 行使用 5-fold 的 885 条折外预测；角度列只对实际产生预测的样本
求均值，覆盖率单独报告。

| 方法 | 成功数 | 成功率 | 平均最佳 IoU | 条件角度误差 | 预测覆盖率 |
|---|---:|---:|---:|---:|---:|
| 传统 CV 基线 | 522 / 885 | 58.98% | 0.3360 | 19.65° | 93.79% |
| VLM + 几何后端 | 659 / 885 | 74.46% | 0.4182 | **14.81°** | 100% |
| 单头 CNN（5-fold OOF） | 661 / 885 | 74.69% | 0.4390 | 17.74° | 100% |
| 多头 CNN（5-fold OOF） | **671 / 885** | **75.82%** | **0.4580** | 17.40° | 100% |

Grounding DINO 使用 prompt `small object`，对 885/885 个样本都返回了框；这只
表示 detection coverage，不等于定位精度。

VLM + geometry 相对整图 CV 增加 137 个成功样本（+15.48 个百分点），配对
精确 McNemar 检验 `p=5.24e-33`。重评分工具保留历史输入并记录 SHA-256，输出
位于 `data/processed/shared/cornell_metric_v2/`。

### Image-wise 五折交叉验证（885 折外预测）

| 架构 | 成功数 | Pooled 成功率 | 平均 IoU | 平均角度误差 |
|---|---:|---:|---:|---:|
| 单头（432,454 参数） | 661 / 885 | 74.69% | 0.4390 | 17.74° |
| 多头（514,758 参数） | 671 / 885 | **75.82%** | **0.4580** | **17.40°** |

多头相对单头增加 10 个成功样本（+1.13 个百分点）；但 46 个样本仅单头成功、
56 个仅多头成功，精确 McNemar `p=0.373`，不支持显著架构优势。两种网络的
参数量和损失也不同，因此这里只报告“当前架构包的描述性差异”。旧固定目录
五 seed 表包含训练/验证样本，不再作为主泛化结果。

### PyBullet 仿真物理抓取

搭建 Franka Panda 7-DOF PyBullet 仿真系统（`3.2.7`，API `202010061`）：

- **深度反投影门控**：3 目标 × 3 后端的九点验证全部通过（重投影、segmentation 与 ray-test 门控 9/9）
- **真值姿态抬升**：方块抬升约 120 mm 并稳定保持 240 步，`physical_grasp_success: true`
- **斜视 pilot 失败模式**：两次运行同时出现 24.67–26.62 mm 的 XY 反投影
  偏差与抬升失败，提示共享表面点/中心问题，但未做唯一根因消融
- **头顶 + 深抓取 pilot**：历史固定场景运行将 XY 偏差降至 0.76 mm，并在
  `-25 mm` 抓取深度下抬升 119.94 mm；这是联合干预的 N=1 可行性证据，不能
  解释为只改变相机的因果实验

### 关键发现

- **VLM 约束的几何流程带来最大描述性提升**：58.98% → 74.46%（+15.48 pp）
- **CNN 后端 IoU 更高**：多头 OOF 0.4580 vs 几何 0.4182
- **几何后端角度更低**：14.81° vs 多头 OOF 17.40°
- **单像素表面反投影不等于物体中心**：斜视 pilot 中 24.67--26.62 mm
  的 XY 偏差与两次抬升失败同时出现；历史顶视联合干预把偏差降到 0.76 mm
  并成功抬升，但尚不能把差异单独归因于相机
- **物理结果边界**：头顶相机 + 深抓取修订实现过一次完整链条；正式比较仍需
  同一控制器的配对重跑及计划中的 60 次试验

## 仓库结构

```text
.
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── src/
│   ├── shared/                          # 共享模块：Cornell 数据集、抓取几何、交叉验证
│   │   ├── cornell_dataset.py
│   │   ├── cornell_evaluation.py      # 共享 Cornell 判据与成功见证
│   │   ├── rescore_cornell_predictions.py # 不覆盖历史输入的重评分
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
│   └── simulation/                      # PyBullet 仿真测试套件（216 项）
│       ├── test_pybullet_camera.py
│       ├── test_pybullet_backprojection.py
│       ├── test_pybullet_kinematic_audit.py
│       ├── test_pybullet_truth_lift.py
│       ├── test_pybullet_stage6b_pipeline.py
│       └── ...
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

# 五次确定性重复实验（固定目录，仅作补充）
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode multi \
  --architecture single --num-runs 5 --device cuda

# Image-wise 五折交叉验证
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py --mode manifest
for architecture in single multi_head; do
  for fold in 0 1 2 3 4; do
    conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
      --mode run --architecture "$architecture" --fold "$fold" --device cuda
  done
  conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
    --mode aggregate --architecture "$architecture"
done
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py --mode compare

# 从保存的四组预测生成 metric-v2 审计产物（不覆盖输入）
conda run -n msc-grasp python -m src.shared.rescore_cornell_predictions \
  --source baseline=data/processed/baseline_cv/cv_baseline_predictions.csv \
  --source vlm_geometry=data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv \
  --source cnn_single_oof=data/processed/vlm/cnn_cross_validation/single/combined_predictions.csv \
  --source cnn_multi_head_oof=data/processed/vlm/cnn_cross_validation/multi_head/combined_predictions.csv \
  --paired-comparison baseline_vs_geometry=baseline,vlm_geometry \
  --paired-comparison single_vs_multi=cnn_single_oof,cnn_multi_head_oof \
  --paired-comparison geometry_vs_single=vlm_geometry,cnn_single_oof \
  --paired-comparison geometry_vs_multi=vlm_geometry,cnn_multi_head_oof
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

# 头顶 + -25 mm 深抓取修订协议预检
conda run -n msc-grasp python src/simulation/pybullet/run_overhead_preflight.py \
  --device cuda --output-dir data/processed/pybullet/grasp_execution/stage_overhead_preflight_strict

# 明确消费该新计划；不要省略 --plan-path 后误跑默认斜视计划
conda run -n msc-grasp python src/simulation/pybullet/run_stage6b_pipeline.py \
  --plan-path data/processed/pybullet/grasp_execution/stage_overhead_preflight_strict/execution_plan.json \
  --output-dir data/processed/pybullet/grasp_execution/stage_overhead_grasp_strict
```

### 测试

```bash
# 运行全部回归测试（当前 254 项）
conda run -n msc-grasp python -m pytest -q
```

## 评估标准

项目使用 Cornell grasp detection 标准 rectangle metric：

```text
IoU >= 0.25
角度误差 <= 30°
```

必须存在同一个正抓取标注同时满足两个条件才计为成功。`best_*` 字段继续记录
最大-IoU 匹配；`successful_match_*` 单独记录使成功成立的 GT，避免两种语义混用。

## 输出文件

实验输出保存到 `data/processed/`（Git 忽略）：

```text
data/processed/baseline_cv/
├── cv_baseline_predictions.csv
└── cv_baseline_summary.json

data/processed/shared/cornell_metric_v2/ # 独立重评分、输入/输出哈希与配对检验

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
    └── stage_overhead_grasp_v3/        # 历史顶视 + 深抓取联合 pilot
```

## 实验溯源

项目建立了可审计的实验溯源链：

- 确定性 seeds（42–46）用于五 seed 重复实验，每个 seed 独立保存权重、训练历史、逐样本预测和 summary
- Image-wise 五折使用共同 manifest（SHA-256: `b7d3e22a...`），单头和多头共用同一划分
- 每折独立保存 checkpoint、验证损失和 177 条测试预测
- Cornell metric-v2 总摘要 SHA-256：
  `6ed8d6cf3d621f2fa9171edfe1db0fc8af1d6d684c287f4b8565219128c6d02f`
  （源预测 SHA-256 与重评分输出 SHA-256 均记录在该 JSON 内）
- PyBullet 物理执行产物包含完整状态轨迹 CSV、接触事件、metadata 和关键帧
- 完整回归测试：**254 项全部通过**（2026-08-17，禁用 pytest cache）

## 后续方向

当前修正后的主结果和证据边界已形成；严格顶视协议仍需按上面的显式计划路径
重新运行。后续研究方向包括：

- **RGB-D 融合**：将深度信息直接输入 CNN 后端，而非仅在反投影阶段使用
- **真实机器人验证**：将当前 PyBullet 仿真管线迁移至真实 Franka Panda 平台
- **多物体抓取规划**：在 clutter 场景中扩展目标选择与避碰抓取排序
- **其他数据集泛化**：在 Jacquard 等更大规模数据集上验证 CNN 后端泛化性
