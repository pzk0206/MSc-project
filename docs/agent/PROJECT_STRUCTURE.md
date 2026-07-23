# Project Structure

## Repository map

```text
.
├── AGENTS.md
├── README.md
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
├── docs/
│   ├── agent/
│   │   ├── CURRENT_STATUS.md
│   │   ├── PROJECT_OVERVIEW.md
│   │   ├── PROJECT_STRUCTURE.md
│   │   └── CODE_ORGANIZATION.md
│   ├── debugging/
│   │   ├── BUGLOG.md
│   │   └── FAILURE_ANALYSIS.md
│   ├── planning/
│   │   └── vlm_robotic_grasp_study_plan.md
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

## Module responsibilities

| Path | Responsibility |
|---|---|
| `src/shared/cornell_dataset.py` | 解析 Cornell 样本、图像和抓取标注 |
| `src/shared/grasp_geometry.py` | 抓取矩形几何转换、IoU 和角度评估 |
| `src/shared/check_cornell_dataset.py` | 检查数据集完整性 |
| `src/shared/export_cornell_grasp_labels.py` | 导出中心参数形式的抓取标签 |
| `src/shared/inspect_sample.py` | 检查单个 Cornell 样本 |
| `src/shared/visualize_sample_checks.py` | 批量可视化样本和标注检查 |
| `src/baseline_cv/run_cv_baseline.py` | 运行并评估 Traditional CV baseline |
| `src/baseline_cv/visualize_mask_pipeline.py` | 可视化单样本分割与抓取框流程 |
| `src/baseline_cv/visualize_mask_pipeline_batch.py` | 批量生成 baseline 流程可视化 |
| `src/vlm/prompts.py` | 保存 VLM 定位 prompt 配置 |
| `src/vlm/run_grounding_dino_localization.py` | 运行 Grounding DINO 开放词汇定位 |
| `src/vlm/run_vlm_assisted_grasp.py` | 运行 VLM 定位 + 几何抓取后端 |
| `src/vlm/run_cnn_grasp.py` | 训练、评估并重复运行 CNN 抓取后端 |
| `src/vlm/analyze_failures.py` | 汇总 VLM-guided geometric pipeline 失败案例 |
| `uog_dissertation_outline/l4proj.tex` | dissertation 主 LaTeX 文档 |
| `uog_dissertation_outline/l4proj.bib` | dissertation BibTeX 文献库 |

## Data flow

```text
data/raw/cornell/
        │
        ├── Traditional CV ──────────────> data/processed/baseline_cv/
        │
        └── Grounding DINO localization ─> data/processed/vlm/localization/
                                             │
                                             ├── geometric grasp
                                             │      └── data/processed/vlm/grasp/
                                             └── CNN grasp
                                                    └── data/processed/vlm/cnn_grasp/
```

`data/` 被 `.gitignore` 排除。源码和文档不得依赖已提交的生成结果。

## Main commands

项目默认 Conda 环境名为 `msc-grasp`。

```bash
# 检查 Cornell 数据集
conda run -n msc-grasp python src/shared/check_cornell_dataset.py

# 导出 Cornell 抓取标签
conda run -n msc-grasp python src/shared/export_cornell_grasp_labels.py

# 运行 Traditional CV baseline
conda run -n msc-grasp python src/baseline_cv/run_cv_baseline.py

# 小批量 Grounding DINO 定位
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py

# 全量 Grounding DINO 定位
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda

# VLM-guided geometric grasp
conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py

# 训练并评估 CNN grasp backend
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode all --device cuda

# 五次重复 CNN 实验
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --mode multi --num-runs 5 --device cuda

# 失败案例分析
python3 src/vlm/analyze_failures.py
```

依赖安装和 VLM 环境说明见 [`../../src/vlm/INSTALL.md`](../../src/vlm/INSTALL.md)。

## Main outputs

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
```

## Documentation reading order

1. [CURRENT_STATUS.md](CURRENT_STATUS.md) — 最新进展、结果和下一步。
2. [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — 稳定的研究背景和方法。
3. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — 当前目录、模块和命令。
4. [CODE_ORGANIZATION.md](CODE_ORGANIZATION.md) — 新代码放置和模块拆分规则。
5. [WORKLOG.md](../worklog/WORKLOG.md) — 按时间回顾已完成工作。

新增、移动或删除主要模块后必须更新本文件。
