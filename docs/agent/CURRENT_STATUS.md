# Current Project Status

> AI task entry point: read this file before starting project work.

Last updated: 2026-07-23

## Current phase

三条核心实验 pipeline 已完成并在 Cornell Grasping Dataset 的 885 个样本上完成评估。当前项目重心已从基础实验实现转向 dissertation 写作、结果整理和补充分析。

## Completed pipelines

1. **Traditional CV baseline**
   - 在整幅 RGB 图像上进行颜色/亮度阈值分割。
   - 提取轮廓并计算最小面积旋转外接矩形。
   - 使用 Cornell rectangle metric 评估预测抓取框。
2. **VLM-guided geometric pipeline**
   - 使用 Grounding DINO 和文本 prompt 完成目标定位。
   - 在定位框内运行 OpenCV 几何抓取后端。
3. **VLM-guided CNN pipeline**
   - 使用 Grounding DINO 产生目标 crop。
   - 使用轻量 CNN 回归抓取矩形参数。
   - 支持 single run 和 five-run 重复实验汇总。

## Latest verified results

### Full Cornell dataset（885 samples）

| Method | Success | Success rate | Mean best IoU | Mean angle error |
|---|---:|---:|---:|---:|
| Traditional CV baseline | 504 / 885 | 56.95% | 0.3360 | 29.62° |
| VLM + geometric backend | 649 / 885 | 73.33% | 0.4182 | **14.81°** |
| VLM + CNN backend（single run） | 647 / 885 | 73.11% | **0.4476** | 15.97° |
| VLM + CNN backend（5-run mean ± std） | — | 74.51% ± 1.38% | 0.4510 ± 0.0081 | 16.49° ± 0.72° |

### Unseen-object test set（Cornell directories 09–10, 85 samples）

| Method | Test success rate |
|---|---:|
| VLM + geometric backend | 75.3%（64 / 85） |
| VLM + CNN backend（single run） | 81.2%（69 / 85） |
| VLM + CNN backend（5-run mean ± std） | **82.35% ± 4.53%** |

Grounding DINO 使用 prompt `small object` 在当前实验中成功定位全部 885 个 Cornell 样本。

## Confirmed findings

- VLM 定位前端带来的提升最大：56.95% → 73.33%。
- CNN 后端在 unseen objects 上优于几何后端，说明学习式后端的泛化更好。
- CNN 后端的平均 IoU 更高，位置和尺寸预测更准确。
- 几何后端的角度误差更低，说明“物体长轴的垂直方向”仍是有效的抓取角度先验。
- 当前主要瓶颈已从目标定位转移到抓取框后端。

## Current repository state

- 核心实现位于 `src/shared/`、`src/baseline_cv/` 和 `src/vlm/`。
- 实验数据和产物位于被 Git 忽略的 `data/` 目录。
- dissertation LaTeX 框架位于 `uog_dissertation_outline/`。
- 项目文档已按 AI 上下文、调试、计划和工作日志分类。
- 文档整理没有改变任何实验代码、模型行为或结果文件。

## Next priorities

1. 填充 dissertation 的 Introduction、Background、Methodology、Results 和 Discussion。
2. 整理三种方法的实验表格、图像和可复现命令。
3. 如时间允许，补充 CNN per-sample 错误分析。
4. 可选探索几何角度先验与 CNN 位置回归结合的混合后端。
5. 可选在 Jacquard 等更困难的数据集上验证泛化。

## Read next

- 项目研究背景与方法：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 当前代码和数据结构：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 新代码放置与拆分规则：[CODE_ORGANIZATION.md](CODE_ORGANIZATION.md)
- 完整工作时间线：[../worklog/WORKLOG.md](../worklog/WORKLOG.md)
- 调试历史：[../debugging/BUGLOG.md](../debugging/BUGLOG.md)
- 失败案例分析：[../debugging/FAILURE_ANALYSIS.md](../debugging/FAILURE_ANALYSIS.md)
- 研究计划：[../planning/vlm_robotic_grasp_study_plan.md](../planning/vlm_robotic_grasp_study_plan.md)

## Maintenance rule

仅在项目阶段、已验证指标、主要结论或下一步优先级发生变化时更新本文件。历史过程写入 `docs/worklog/`，详细调试记录写入 `docs/debugging/`；不要在这里记录未经验证的实验结果。
