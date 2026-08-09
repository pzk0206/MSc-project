# 项目结构

## 仓库目录

```text
.
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── src/
│   ├── shared/
│   │   ├── cornell_dataset.py
│   │   ├── cornell_cross_validation.py
│   │   ├── grasp_geometry.py
│   │   ├── analyze_cornell_splits.py
│   │   ├── check_cornell_dataset.py
│   │   ├── export_cornell_grasp_labels.py
│   │   ├── inspect_sample.py
│   │   └── visualize_sample_checks.py
│   ├── baseline_cv/
│   │   ├── run_cv_baseline.py
│   │   ├── visualize_mask_pipeline.py
│   │   └── visualize_mask_pipeline_batch.py
│   ├── vlm/
│   │   ├── prompts.py
│   │   ├── run_grounding_dino_localization.py
│   │   ├── run_vlm_assisted_grasp.py
│   │   ├── cnn_grasp_models.py
│   │   ├── run_cnn_grasp.py
│   │   ├── run_cnn_cross_validation.py
│   │   ├── analyze_failures.py
│   │   ├── analyze_backend_comparison.py
│   │   ├── INSTALL.md
│   │   └── README.md
│   └── simulation/
│       └── pybullet/
│           ├── README.md
│           ├── scene.py
│           ├── camera.py
│           ├── perception.py
│           ├── backend_comparison.py
│           ├── backprojection.py
│           ├── pose_generation.py
│           ├── kinematic_audit.py
│           ├── motion_control.py
│           ├── gripper_control.py
│           ├── lift_control.py
│           ├── execution_plan.py
│           ├── center_bias_diagnostic.py
│           ├── grasp_execution.py
│           ├── run_pose_ik_study.py
│           ├── run_safe_motion_smoke.py
│           ├── run_truth_pregrasp.py
│           ├── run_truth_approach.py
│           ├── run_truth_contact.py
│           ├── run_truth_lift.py
│           ├── run_geometry_execution_preflight.py
│           ├── run_center_bias_diagnostic.py
│           ├── run_stage6a2_recovery_preflight.py
│           ├── center_recovery.py
│           ├── run_stage6b_pipeline.py
│           ├── run_overhead_preflight.py
│           ├── run_overhead_side_grasp.py
│           ├── run_multi_head_preflight.py
│           ├── target_selection.py
│           ├── visualization.py
│           ├── run_pilot.py
│           └── run_multi_object_study.py
├── tests/
│   └── simulation/
│       ├── test_pybullet_camera.py
│       ├── test_pybullet_backend_comparison.py
│       ├── test_pybullet_backprojection.py
│       ├── test_pybullet_pose_generation.py
│       ├── test_pybullet_kinematic_audit.py
│       ├── test_pybullet_pose_ik_runner.py
│       ├── test_pybullet_motion_control.py
│       ├── test_pybullet_safe_motion_runner.py
│       ├── test_pybullet_truth_pregrasp.py
│       ├── test_pybullet_truth_approach.py
│       ├── test_pybullet_gripper_control.py
│       ├── test_pybullet_truth_contact.py
│       ├── test_pybullet_lift_control.py
│       ├── test_pybullet_truth_lift.py
│       ├── test_pybullet_execution_plan.py
│       ├── test_pybullet_geometry_execution_preflight.py
│       ├── test_pybullet_center_bias_diagnostic.py
│       ├── test_pybullet_center_bias_runner.py
│       ├── test_pybullet_stage6a2_recovery_preflight.py
│       ├── test_pybullet_stage6b_pipeline.py
│       ├── test_pybullet_perception.py
│       ├── test_pybullet_smoke.py
│       ├── test_pybullet_target_selection.py
│       ├── test_pybullet_visualization.py
│       ├── test_pybullet_runner.py
│       └── test_pybullet_multi_object_runner.py
├── docs/
│   ├── agent/
│   │   ├── CURRENT_STATUS.md
│   │   ├── PROJECT_OVERVIEW.md
│   │   ├── PROJECT_STRUCTURE.md
│   │   ├── DISSERTATION_WRITING_GUIDE.md
│   │   ├── CODE_ORGANIZATION.md
│   │   └── vlm_robotic_grasp_study_plan.md
│   ├── debugging/
│   │   ├── BUGLOG.md
│   │   ├── FAILURE_ANALYSIS.md
│   │   └── WORKLOG.md
│   ├── planning/
│   │   ├── cnn_architecture_rationale.md
│   │   └── modern_2d_grasp_literature_matrix.md
│   ├── reporting/
│   │   ├── generate_cnn_architecture.py
│   │   ├── generate_concise_speaking_notes_pdf.py
│   │   ├── generate_supervisor_progress_report.py
│   │   ├── generate_supervisor_progress_report_en.py
│   │   ├── supervisor_progress_report_2026-07-23.pdf
│   │   ├── supervisor_progress_report_2026-07-23_en.pdf
│   │   ├── supervisor_progress_report_speaking_notes_bilingual.md
│   │   ├── supervisor_progress_report_speaking_notes_concise_bilingual.md
│   │   └── supervisor_progress_report_speaking_notes_concise_bilingual.pdf
│   ├── worklog/
│   │   ├── WORKLOG.md
│   │   ├── weekly_progress_2026-07-06.md
│   │   └── weekly_progress_2026-07-16.md
│   └── superpowers/
│       ├── plans/
│       └── specs/
└── uog_dissertation_outline/
    ├── l4proj.tex
    ├── l4proj.bib
    ├── l4proj.cls
    ├── images/
    └── fonts/
```

`__init__.py` 文件和论文模板的单个素材未全部展开；完整列表以仓库实际文件为准。

## 模块职责

| 路径 | 职责 |
|---|---|
| `src/shared/cornell_dataset.py` | 解析 Cornell 样本、图像和抓取标注 |
| `src/shared/cornell_cross_validation.py` | 生成、持久化并审计确定性的 Cornell image-wise fold manifest |
| `src/shared/grasp_geometry.py` | 抓取矩形几何转换、IoU 和角度评估 |
| `src/shared/analyze_cornell_splits.py` | 审计 Cornell 固定目录划分、共同测试样本和代表图 |
| `src/shared/check_cornell_dataset.py` | 检查数据集完整性 |
| `src/shared/export_cornell_grasp_labels.py` | 导出中心参数形式的抓取标签 |
| `src/shared/inspect_sample.py` | 检查单个 Cornell 样本 |
| `src/shared/visualize_sample_checks.py` | 批量可视化样本和标注检查 |
| `src/baseline_cv/run_cv_baseline.py` | 运行并评估传统计算机视觉基线 |
| `src/baseline_cv/visualize_mask_pipeline.py` | 可视化单样本分割与抓取框流程 |
| `src/baseline_cv/visualize_mask_pipeline_batch.py` | 批量生成基线流程可视化 |
| `src/vlm/prompts.py` | 保存 VLM 定位提示词配置 |
| `src/vlm/run_grounding_dino_localization.py` | 运行 Grounding DINO 开放词汇定位 |
| `src/vlm/run_vlm_assisted_grasp.py` | 运行 VLM 定位 + 几何抓取后端 |
| `src/vlm/cnn_grasp_models.py` | 定义旧权重兼容的单头 CNN、多头 CNN 及多头损失 |
| `src/vlm/run_cnn_grasp.py` | 训练、评估并重复运行单头或多头 CNN 抓取后端 |
| `src/vlm/run_cnn_cross_validation.py` | 运行可恢复的单头/多头 image-wise 五折训练、聚合与成对比较 |
| `src/vlm/analyze_failures.py` | 汇总 VLM 引导的几何实验流程失败案例 |
| `src/vlm/analyze_backend_comparison.py` | 在固定测试子集上逐样本比较几何和 CNN 后端 |
| `src/simulation/pybullet/scene.py` | 管理确定性 PyBullet 场景、URDF 和 client 生命周期 |
| `src/simulation/pybullet/camera.py` | 采集 RGB、米制深度、segmentation 和相机矩阵 |
| `src/simulation/pybullet/perception.py` | 将仿真 RGB 适配到现有 Grounding DINO、几何和 CNN 接口 |
| `src/simulation/pybullet/backend_comparison.py` | 对三后端中心格式抓取框做有限性、目标 mask 和图像边界诊断，不生成性能排名 |
| `src/simulation/pybullet/backprojection.py` | 用米制深度和 PyBullet 相机矩阵将二维抓取中心恢复为相机/世界坐标，并以重投影、segmentation 和射线命中执行九点事后门控 |
| `src/simulation/pybullet/pose_generation.py` | 将九点结果和二维方向转换为两个世界 -Z 俯视悬停姿态候选，并支持从真值世界表面点直接生成同约定姿态 |
| `src/simulation/pybullet/kinematic_audit.py` | 按名称解析 Panda，并执行可恢复的离线 IK/FK、关节限位和静态碰撞余量审计 |
| `src/simulation/pybullet/motion_control.py` | 用 Panda 位置电机执行命名关节轨迹，逐仿真步记录关节/FK、收敛、动态碰撞余量及可选目标刚体真值位姿 |
| `src/simulation/pybullet/gripper_control.py` | 保持 Panda 手臂并缓慢闭合双指，以真实 body/link/正法向力分类目标接触，取得双指接触后冻结命令并持续审计 |
| `src/simulation/pybullet/lift_control.py` | 冻结双指命令并执行手臂抬升/保持，逐步审计 cube 上升量、桌面接触、工具相对漂移、双指接触和禁止碰撞 |
| `src/simulation/pybullet/execution_plan.py` | 定义 Stage 6A 严格冻结的 geometry 感知执行计划；校验协议、相机、预测、三个位姿、七关节解、82 状态审计和唯一候选 |
| `src/simulation/pybullet/center_bias_diagnostic.py` | 纯计算 Stage 6A.1 表面点—cube 质心 XY 偏差及名义顶面 Z 偏差，冻结 0.025 m 半高和 0.005 m 参考门槛并严格序列化 |
| `src/simulation/pybullet/grasp_execution.py` | 复用真值 cube 的场景准备、姿态预检、分段电机执行、目标稳定性、抓取深度、双指接触与抬升保持门控及阶段证据写入 |
| `src/simulation/pybullet/run_pose_ik_study.py` | 独立读取九点产物，审计 18 个候选并保存 CSV/JSON；不执行电机、轨迹或夹爪 |
| `src/simulation/pybullet/run_safe_motion_smoke.py` | 阶段 1 安全空中往返 runner；先静态预检，再执行电机并保存 CSV/JSON/关键帧，不靠近或抓取目标 |
| `src/simulation/pybullet/run_truth_pregrasp.py` | 阶段 2 真值方块上方到达 runner；检查目标稳定性并执行张开夹爪 pregrasp，逐步保存目标相对位姿，不下降或闭合 |
| `src/simulation/pybullet/run_truth_approach.py` | 阶段 3 张开夹爪垂直接近 runner；先门控 pregrasp，再下降至 cube 顶面以上 0.02 m，不闭合或评价接触 |
| `src/simulation/pybullet/run_truth_contact.py` | 阶段 4 真值方块接触 runner；从 0.02 m 接触前高度张开下探至 0.005 m，再闭合并保持双指目标接触，不抬升 |
| `src/simulation/pybullet/run_truth_lift.py` | 阶段 5 真值方块抬升 runner；重放阶段 2--4 后冻结夹爪，垂直抬升 cube 并保持 240 步，保存完整物理抓取证据 |
| `src/simulation/pybullet/run_geometry_execution_preflight.py` | Stage 6A 同场景 VLM + geometry runner；实时定位、反投影并审计两个执行候选，只写冻结计划，不调用电机或夹爪 |
| `src/simulation/pybullet/run_overhead_preflight.py` | Stage 6A 头顶相机几何预检；相机垂直向下置于场景正上方，单像素反投影无中心恢复，消除斜视偏差 |
| `src/simulation/pybullet/run_overhead_side_grasp.py` | 头顶相机侧向抓取变体；与 run_overhead_preflight.py 共享同一 pipeline |
| `src/simulation/pybullet/run_center_bias_diagnostic.py` | Stage 6A.1 离线 runner；交叉校验并哈希现有 Stage 6A 证据，在独立目录保存后验中心偏差和全 false 执行元数据 |
| `src/simulation/pybullet/target_selection.py` | 使用仿真真值框事后评价多物体 prompt 目标选择，不向模型注入 segmentation |
| `src/simulation/pybullet/visualization.py` | 绘制定位框、目标选择真值、二维抓取框和深度/分割诊断图 |
| `src/simulation/pybullet/run_pilot.py` | 编排第一阶段仿真感知 pilot、CLI、产物和失败元数据 |
| `src/simulation/pybullet/run_multi_object_study.py` | 一次渲染和模型加载后运行三条主 prompt 与一条 generic 诊断，保存二维后端结果并执行三目标乘三后端的深度反投影门控 |
| `src/simulation/pybullet/README.md` | 说明仿真命令、固定协议、评价边界、官方 PyBullet 来源和输出 |
| `docs/planning/cnn_architecture_rationale.md` | 逐项区分当前 CNN 的文献依据与工程选择 |
| `docs/planning/modern_2d_grasp_literature_matrix.md` | 统一比较现代二维抓取方法的输入、划分、指标和可比性 |
| `docs/agent/vlm_robotic_grasp_study_plan.md` | 默认读取的研究总计划；记录 Cornell 范围及 PyBullet 分阶段运动、闭合、抬升和几何/多头比较门控 |
| `docs/agent/DISSERTATION_WRITING_GUIDE.md` | 记录论文各章节目标字数、核心内容、批判性分析要求和推荐写作顺序 |
| `docs/reporting/generate_cnn_architecture.py` | 生成论文使用的 CNN 矢量结构图 |
| `docs/reporting/generate_supervisor_progress_report.py` | 生成三页中文导师项目进展汇报 PDF |
| `docs/reporting/generate_supervisor_progress_report_en.py` | 生成三页英文导师项目进展汇报 PDF |
| `docs/reporting/generate_concise_speaking_notes_pdf.py` | 将简洁双语讲稿转换为适合 iPad 阅读的 PDF |
| `docs/reporting/supervisor_progress_report_2026-07-23.pdf` | 2026-07-23 导师项目进展汇报 |
| `docs/reporting/supervisor_progress_report_2026-07-23_en.pdf` | 2026-07-23 英文导师项目进展汇报 |
| `docs/reporting/supervisor_progress_report_speaking_notes_bilingual.md` | 英文汇报讲稿、中文提示和常见问答 |
| `docs/reporting/supervisor_progress_report_speaking_notes_concise_bilingual.md` | 对应三页英文汇报的简洁双语讲稿 |
| `docs/reporting/supervisor_progress_report_speaking_notes_concise_bilingual.pdf` | 适合 iPad 竖屏阅读的简洁双语讲稿 |
| `uog_dissertation_outline/l4proj.tex` | 毕业论文主 LaTeX 文档 |
| `uog_dissertation_outline/l4proj.bib` | 毕业论文 BibTeX 文献库 |
| `uog_dissertation_outline/l4proj.cls` | 毕业论文模板与当前 LaTeX 兼容设置 |
| `uog_dissertation_outline/images/cnn_architecture.pdf` | 轻量 CNN 的矢量结构图 |

## 数据流

```text
data/raw/cornell/
        │
        ├── 传统计算机视觉 ─────────────> data/processed/baseline_cv/
        │
        └── Grounding DINO 定位 ─────────> data/processed/vlm/localization/
                                             │
                                             ├── 几何抓取
                                             │      └── data/processed/vlm/grasp/
                                             └── CNN 抓取
                                                    ├── data/processed/vlm/cnn_grasp/
                                                    ├── data/processed/vlm/cnn_grasp_multi_head/
                                                    └── data/processed/vlm/cnn_cross_validation/

PyBullet 场景
        └── 固定虚拟相机 ───────────────> RGB / depth / segmentation
                                             │
                                             └── 现有 VLM + 抓取后端
                                                    │
                                                    └── 二维中心 + depth + 相机矩阵
                                                           └── 世界坐标与九点审计
                                                                  └── 两个俯视悬停候选
                                                                  ├── 头顶相机 preflight（消除斜视 XY 偏差 26.55→0.76mm）
                                                                  │      └── 抓取深度修正（TCP 从表面+5mm 下探至 cube 中心-25mm）
                                                                  │             └── Stage 6B 物理抓取首次成功
                                                                         └── Panda IK/FK + 41 状态碰撞审计
                                                                                ├── data/processed/pybullet/
                                                                                └── 阶段 1 电机安全空中往返
                                                                                       └── grasp_execution/stage_1_safe_motion/
                                                                                              └── 阶段 2 真值 cube 上方 pregrasp
                                                                                                     └── grasp_execution/stage_2_cube_pregrasp/
                                                                                                            └── 阶段 3 张开夹爪垂直接近
                                                                                                                   └── grasp_execution/stage_3_open_approach/
                                                                                                                          └── 阶段 4 真值 cube 双指接触
                                                                                                                                 └── grasp_execution/stage_4_bilateral_contact/
                                                                                                                                        └── 阶段 5 真值 cube 抬升与保持
                                                                                                                                               └── grasp_execution/stage_5_truth_cube_lift/
        └── Stage 6A VLM + geometry 冻结计划
               └── grasp_execution/stage_6a_geometry_preflight/
                      └── Stage 6A.1 只读中心偏差诊断
                             └── grasp_execution/stage_6a1_center_bias_diagnostic/
```

`data/` 被 `.gitignore` 排除。源码和文档不得依赖已提交的生成结果。

## 常用命令

项目默认 Conda 环境名为 `msc-grasp`。

```bash
# 检查 Cornell 数据集
conda run -n msc-grasp python src/shared/check_cornell_dataset.py

# 导出 Cornell 抓取标签
conda run -n msc-grasp python src/shared/export_cornell_grasp_labels.py

# 运行传统计算机视觉基线
conda run -n msc-grasp python src/baseline_cv/run_cv_baseline.py

# 小批量运行 Grounding DINO 定位
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py

# 全量运行 Grounding DINO 定位
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda

# 运行 VLM 引导的几何抓取
conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py

# 训练并评估 CNN 抓取后端
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all --device cuda

# 五次重复 CNN 实验
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode multi --num-runs 5 --device cuda

# 训练并评估多头 CNN
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all \
  --architecture multi_head \
  --output-dir data/processed/vlm/cnn_grasp_multi_head --device cuda

# 生成并审计共同的 image-wise 五折清单
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode manifest

# 运行一个可恢复的单头 fold
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode run --architecture single --fold 0 --device cuda

# 五折完成后聚合并比较架构
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode aggregate --architecture single
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode compare

# 失败案例分析
python3 src/vlm/analyze_failures.py

# Cornell 数据划分审计
conda run -n msc-grasp python src/shared/analyze_cornell_splits.py

# 固定测试子集后端比较
conda run -n msc-grasp python src/vlm/analyze_backend_comparison.py

# PyBullet DIRECT 感知 pilot（小鸭目标诊断）
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend geometry --device cuda --prompt "yellow rubber duck"

# 固定三物体目标选择研究
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py \
  --device cuda \
  --output-dir data/processed/pybullet/multi_object_study

# 对已保存九点结果运行静态姿态、IK/FK 和碰撞余量审计
conda run -n msc-grasp python \
  src/simulation/pybullet/run_pose_ik_study.py \
  --input-dir data/processed/pybullet/multi_object_study \
  --output-dir data/processed/pybullet/multi_object_study

# 阶段 1：安全空中电机往返
conda run -n msc-grasp python \
  src/simulation/pybullet/run_safe_motion_smoke.py

# 阶段 2：使用真值方块移动到目标上方，夹爪保持张开
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_pregrasp.py

# 阶段 3：张开夹爪从 pregrasp 垂直接近接触前高度
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_approach.py

# 阶段 4：张开下探到抓取深度，再闭合并保持双指目标接触
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_contact.py

# 阶段 5：冻结双指命令，垂直抬升真值方块并保持
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_lift.py

# Stage 6A：实时 VLM + geometry 生成静态执行计划，不运动
conda run -n msc-grasp python \
  src/simulation/pybullet/run_geometry_execution_preflight.py \
  --device cuda

# Stage 6A.1：只读冻结 Stage 6A 产物，保存中心偏差诊断
conda run -n msc-grasp python -m \
  src.simulation.pybullet.run_center_bias_diagnostic

# 运行全部测试；使用 python -m pytest 保证仓库根目录可导入
conda run -n msc-grasp python -m pytest -q

# 重新生成 CNN 矢量结构图
MPLCONFIGDIR=/tmp/msc-mplconfig conda run -n msc-grasp \
  python docs/reporting/generate_cnn_architecture.py

# 编译毕业论文（首次使用会下载标准 LaTeX 宏包缓存）
cd uog_dissertation_outline
XDG_CACHE_HOME=/tmp/msc-tectonic-cache conda run -n msc-grasp \
  tectonic --keep-logs l4proj.tex
cd ..
```

依赖安装和 VLM 环境说明见 [`../../src/vlm/INSTALL.md`](../../src/vlm/INSTALL.md)。

## 主要输出

```text
data/processed/baseline_cv/
├── cv_baseline_predictions.csv
└── cv_baseline_summary.json

data/processed/vlm/localization/
├── grounding_dino_generic_small_object_predictions.csv
└── grounding_dino_generic_small_object_summary.json

data/processed/vlm/grasp/
├── vlm_assisted_grasp_predictions.csv
├── vlm_assisted_grasp_summary.json
└── failure_analysis.csv

data/processed/vlm/cnn_grasp/
├── cnn_grasp_predictions.csv
├── cnn_grasp_summary.json
├── cnn_grasp_model.pt
└── training_history.json

data/processed/vlm/cnn_cross_validation/
├── image_wise_folds_seed_42.csv
├── image_wise_folds_seed_42.json
├── single/
│   ├── fold_0/ ... fold_4/
│   ├── combined_predictions.csv
│   └── cross_validation_summary.json
├── multi_head/
│   ├── fold_0/ ... fold_4/
│   ├── combined_predictions.csv
│   └── cross_validation_summary.json
└── architecture_comparison.json

data/processed/shared/split_audit/
├── representative_samples.csv
├── split_metrics.json
├── same_test_subset_metrics.csv
└── cornell_split_contact_sheet.png

data/processed/vlm/backend_comparison/
├── sample_comparison.csv
├── comparison_summary.json
└── backend_failure_cases.png

data/processed/pybullet/pilot/
├── rgb.png
├── depth.npy
├── depth_visualization.png
├── segmentation.png
├── localization.png
├── prediction.png
└── metadata.json

data/processed/pybullet/multi_object_study/
├── rgb.png
├── depth.npy
├── depth_visualization.png
├── segmentation.png
├── ground_truth_boxes.png
├── results.csv
├── backend_results.csv
├── backend_comparison.json
├── backprojection_results.csv
├── backprojection_summary.json
├── pose_ik_candidates.csv
├── pose_ik_summary.json
├── pose_ik_metadata.json
├── summary.json
├── metadata.json
└── targets/
    ├── duck/
    ├── cube/
    ├── sphere/
    └── generic/
```

## 文档阅读顺序

1. [项目当前状态](CURRENT_STATUS.md) — 最新进展、结果和下一步。
2. [项目概览](PROJECT_OVERVIEW.md) — 稳定的研究背景和方法。
3. [项目结构](PROJECT_STRUCTURE.md) — 当前目录、模块和命令。
4. [代码组织规范](CODE_ORGANIZATION.md) — 新代码放置和模块拆分规则。
5. [论文写作指南](DISSERTATION_WRITING_GUIDE.md) — 章节字数、内容重点和写作顺序。
6. [项目工作日志](../worklog/WORKLOG.md) — 按时间回顾已完成工作。

新增、移动或删除主要模块后必须更新本文件。
