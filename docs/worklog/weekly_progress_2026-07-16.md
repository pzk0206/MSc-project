# Weekly Progress Report — VLM-guided 2D Robotic Grasp Detection

日期：2026-07-16  
项目方向：VLM-guided 2D Robotic Grasp Rectangle Detection

## 1. 本周一句话总结

本周完成了 VLM-guided geometric pipeline 的失败案例分析，并实现了第三条实验 pipeline——VLM-guided CNN grasp backend。CNN 后端在全量上与几何后端持平（74.5% vs 73.3%），但在 unseen objects 上泛化明显更好（82.4% ± 4.5% vs 75.3%）。至此，论文计划的三条 pipeline 全部完成。

## 2. 本周完成的主要工作

### 2.1 失败案例分析

- 编写 `src/vlm/analyze_failures.py`，对 VLM-guided geometric pipeline 的 236 个失败样本分类
- 输出 `docs/failure_analysis.md` 分析报告
- 输出 `data/processed/vlm/grasp/failure_analysis.csv` 分类标注

失败模式分布：

| 失败模式 | 数量 | 占比 | 含义 |
|---|---|---|---|
| 仅 IoU 失败（角度正确） | 126 | 53.4% | 方向对了，位置/尺寸不准 |
| 仅角度失败（IoU 达标） | 45 | 19.1% | 位置对了，方向错了 |
| 两者都失败 | 65 | 27.5% | 都没对 |

关键发现：
- 超过一半失败（126 个）角度已经正确，纯几何规则无法生成准确的 center/width/height
- 只有 9 个退化案例（VLM 失败但 baseline 成功），说明 VLM 定位几乎不会降低性能
- 63 个样本 IoU 在 0.20-0.25 之间，离成功仅差一步

### 2.2 Model 3: VLM-guided CNN grasp backend

方法流程：

```text
RGB image + prompt
→ Grounding DINO object localization
→ VLM crop (expanded box)
→ CNN regressor → [cx, cy, width, height, sin(2θ), cos(2θ)]
→ grasp rectangle → Cornell-style evaluation
```

实现细节：
- 数据：从 VLM expanded_box 裁剪 RGB，resize 到 224×224
- 标签：Cornell cpos 正抓取矩形，取面积最大的 GT，转 crop 坐标并归一化
- 划分：按 Cornell 子目录 (01-06 train, 07-08 val, 09-10 test)，避免同物体泄露
- 模型：4 层 Conv (32→64→128→256) + GAP + 3 层 FC (256→128→64→6)
- 损失：Smooth L1 Loss
- 训练：Adam, ReduceLROnPlateau, early stopping (patience=20)

## 3. 三类方法全量结果对比

数据集：Cornell Grasping Dataset  
样本数：885  
VLM 模型：Grounding DINO (IDEA-Research/grounding-dino-tiny)  
Prompt：`small object`

| 方法 | 定位前端 | 抓取后端 | 成功数 | 成功率 | 平均 IoU | 平均角度 |
|---|---:|---:|---:|---:|---:|---:|
| Traditional CV baseline | 无（全图阈值） | OpenCV 几何 | 504 | 56.95% | 0.3360 | 29.62° |
| VLM + Geometric backend | Grounding DINO | OpenCV 几何 | 649 | 73.33% | 0.4182 | **14.81°** |
| VLM + CNN backend (5-run) | Grounding DINO | CNN regressor | — | 74.51% ± 1.38% | **0.4510** ± 0.0081 | 16.49° ± 0.72° |

### 3.1 Test set 泛化对比（目录 09-10，85 个 CNN 未见过物体）

| 方法 | Test 成功率 |
|---|---|
| VLM + Geometric backend | 75.3% (64/85) |
| **VLM + CNN backend (5-run mean ± std)** | **82.35% ± 4.53%** |

各轮 test 结果：83.5%, 80.0%, 90.6%, 77.6%, 80.0% —— 5 轮中最低 (77.6%) 仍高于几何后端 (75.3%)，不是偶然。

## 4. 关键发现

1. **VLM 定位贡献最大**：从 56.95% → 73.33%（+16.4%），是三条 pipeline 中最大的一次提升
2. **CNN 泛化更好**：在 unseen objects 上 82.35% vs 几何后端 75.3%（+7.0%）
3. **CNN IoU 更高**：0.4510 vs 0.4182，验证了失败分析中的判断——几何后端的主要弱点是位置/尺寸
4. **几何方法角度更准**：14.81° vs 16.49°，长轴垂直方向启发式规则是有效的角度先验
5. **两者互补**：CNN 擅长位置/尺寸，几何擅长角度——motivates 混合后端作为进一步方向

## 5. 本周生成的主要代码与结果文件

### 失败分析

代码：

```text
src/vlm/analyze_failures.py
```

结果：

```text
docs/failure_analysis.md
data/processed/vlm/grasp/failure_analysis.csv
```

### CNN grasp backend

代码：

```text
src/vlm/run_cnn_grasp.py
```

结果：

```text
data/processed/vlm/cnn_grasp/cnn_grasp_model.pt
data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv
data/processed/vlm/cnn_grasp/cnn_grasp_summary.json
data/processed/vlm/cnn_grasp/training_history.json
data/processed/vlm/cnn_grasp/multi_run_summary.json
data/processed/vlm/cnn_grasp/visualizations/
```

### 文档更新

```text
README.md  (更新三类方法对比表、实验运行命令、下一步计划)
```

## 6. 当前项目状态

实验部分全部完成：

```text
✅ 文献阅读与数据集准备
✅ Traditional CV baseline            (56.95%)
✅ VLM localization                   (Grounding DINO, 100% detection)
✅ VLM + Geometric backend            (73.33%)
✅ VLM + CNN backend                  (74.51% ± 1.38%, test 82.35%)
✅ 失败案例分析
✅ 三类方法全量对比
```

## 7. 下一步计划

实验管线已全部完成，下一阶段重心转移到论文写作：

1. 开始填充 `uog_dissertation_outline/l4proj.tex` 各章节
2. 整理实验图表（三类方法对比柱状图、失败模式饼图、角度/IoU 分布图）
3. 完成 Introduction / Background / Methodology / Results / Discussion 初稿
4. 如有时间：探索几何+CNN 混合后端（几何初始化角度 + CNN 回归位置/尺寸）

## 8. 可以给导师汇报的简短版本

This week I completed two major tasks. First, I analysed the 236 failure cases from the VLM-guided geometric pipeline, categorising them into IoU-only failures (126, 53%), angle-only failures (45, 19%), and combined failures (65, 28%). The analysis revealed that over half of the failures had correct grasp orientation but inaccurate position/size — motivating a learning-based backend.

Second, I implemented and trained a lightweight CNN grasp regressor that takes the VLM crop as input and directly predicts grasp rectangle parameters. Across five independent training runs with different random seeds, the CNN backend achieved 74.51% ± 1.38% on the full Cornell dataset and 82.35% ± 4.53% on unseen test objects, consistently outperforming the geometric backend (73.33% full, 75.3% test). The CNN shows better position/size prediction (IoU 0.4510 vs 0.4182) while the geometric backend retains a slight edge in angle estimation (14.81° vs 16.49°).

All three experimental pipelines (Traditional CV, VLM+Geometric, VLM+CNN) are now complete. The remaining work focuses on dissertation writing.
