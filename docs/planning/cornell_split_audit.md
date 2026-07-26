# Cornell 数据划分审计

日期：2026-07-26

## 审计目的

本项目按 Cornell 子目录划分数据：

- train：01–06，共 600 个样本；
- validation：07–08，共 200 个样本；
- test：09–10，共 85 个样本。

审计目标不是事后证明该划分公平，而是检查三个集合的物体构成和测量难度
是否足以支持“未见物体泛化”表述。

## 检查方法

1. 从每个训练目录等间距选择 2 张图，从每个验证/测试目录等间距选择 6 张
   图，使三个 split 各有 12 张代表图；
2. 人工观察代表图中的物体形状、对称性、细长程度和潜在抓取歧义；
3. 分别计算传统 CV、VLM + geometry 和 CNN 最后一轮在三个 split 上的
   成功率、平均 best IoU 和平均角度误差；
4. 在完全相同的 85 个测试样本上比较几何后端和 CNN 最后一轮。

代表图由 `src/shared/analyze_cornell_splits.py` 生成。它用于检查分布，不代表
完整的物体类别标注。

## 代表物体人工检查

形状标签：

```text
regular_boxlike / elongated / round / thin_flat / irregular_branched
```

难度标签：

```text
low / medium / high
```

主要因素：

```text
symmetry / multiple_valid_grasps / weak_boundary / complex_shape / narrow_part
```

| 目录 | split | 代表物体观察 | 主要形状 | 难度 | 判断依据 |
|---|---|---|---|---|---|
| 01 | train | 小型长条工具、线缆、盒状物和带分支零件 | elongated / irregular_branched | medium | narrow_part / complex_shape |
| 02 | train | 小盒、塞子、刷状物和不规则小件 | regular_boxlike / irregular_branched | medium | weak_boundary / complex_shape |
| 03 | train | 盒状物、圆形件、杯状物和勺形物 | regular_boxlike / round / elongated | medium | symmetry / multiple_valid_grasps |
| 04 | train | 勺形物、圆盘和细长杆状物 | elongated / round | medium | symmetry / narrow_part |
| 05 | train | 小瓶、线缆、环形物和剪刀 | elongated / irregular_branched | high | complex_shape / narrow_part |
| 06 | train | 包装件、眼镜、瓶状物和细长工具 | elongated / irregular_branched | high | complex_shape / weak_boundary |
| 07 | val | 笔状物、勺形物、钳状物和弯曲零件 | elongated / irregular_branched | high | narrow_part / complex_shape |
| 08 | val | 半透明物、圆形物、水果和弯曲零件 | round / irregular_branched | medium | symmetry / weak_boundary |
| 09 | test | 小盒、小瓶、软包装和卷曲小件 | regular_boxlike / round / irregular_branched | medium | weak_boundary / symmetry |
| 10 | test | 环形物、细杆、勺形物和小型零件 | elongated / round / thin_flat | medium | narrow_part / symmetry |

人工观察说明三个集合都包含规则、细长和不规则物体，但样例不足以建立严格
的类别匹配关系。目录编号也不是公开的难度分层标签，因此不能仅凭目录划分
推断测试集与训练集具有相同难度。

## 客观分组统计

| 方法 | split | 样本数 | 成功率 | 平均 best IoU | 平均角度误差 |
|---|---|---:|---:|---:|---:|
| Traditional CV | train | 600 | 53.17% | 0.3226 | 30.58° |
| Traditional CV | val | 200 | 61.00% | 0.3478 | 33.72° |
| Traditional CV | test | 85 | 74.12% | 0.4022 | 13.20° |
| VLM + geometry | train | 600 | 71.00% | 0.4091 | 15.37° |
| VLM + geometry | val | 200 | 79.50% | 0.4567 | 14.33° |
| VLM + geometry | test | 85 | 75.29% | 0.3921 | 12.01° |
| VLM + CNN（最后一轮，seed 46） | train | 600 | 73.83% | 0.4420 | 15.46° |
| VLM + CNN（最后一轮，seed 46） | val | 200 | 73.50% | 0.4784 | 18.13° |
| VLM + CNN（最后一轮，seed 46） | test | 85 | 80.00% | 0.4562 | 21.01° |

传统 CV 在测试集上的成功率比训练集高 20.95 个百分点，平均角度误差低
17.38°。这说明即使不考虑 CNN 学习，目录 09–10 对现有图像处理和几何规则
也呈现出不同的测量难度。VLM + geometry 在验证集上的成功率反而高于测试
集，进一步说明目录之间不是单调的难度分层。

## 同一 85 样本的后端比较

| 后端 | 成功数 | 成功率 | 平均 best IoU | 平均角度误差 |
|---|---:|---:|---:|---:|
| VLM + geometry | 64 / 85 | 75.29% | 0.3921 | 12.01° |
| VLM + CNN（最后一轮，seed 46） | 68 / 85 | 80.00% | 0.4562 | 21.01° |

该比较支持“最后一轮 CNN 在固定 85 样本上有更高成功率和 IoU、几何后端
有更低角度误差”，但不支持把差异推广到任意未见物体。

五次 CNN 实验的测试成功率均值为 82.35% ± 4.53%。这个聚合数字用于描述
随机种子波动，不用于逐样本交叉分类，因为当前只保存了最后一轮的逐样本
预测。

## Image-wise 五折协议审计

Cornell image-wise 五折只要求每张图具有稳定且唯一的 sample ID，因此当前
885 张图可以可靠生成五折清单。协议定义依据 Lenz、Lee 和 Saxena 的
“Deep Learning for Detecting Robotic Grasps”（IJRR 2015）：
<https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf>。
本项目的清单生成与校验代码为独立实现，没有复制外部交叉验证代码。

正式清单使用 seed 42。每个 fold 包含 566 个训练样本、142 个验证样本和
177 个测试样本；五个测试集合两两互斥，合并后恰好覆盖全部 885 张图。
单头和多头 CNN 将共同读取同一个 JSON 文件：

```text
data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.json
SHA-256: b7d3e22a145f50add6d57a70bf0abb87b4b12ee674541deab0d7fee9a286bc2d
```

Object-wise 五折需要“图像到物体实例”的权威映射。原始数据、论文和已检查
的公开实现均未提供本项目可审计使用的完整映射；`01`–`10` 只是存储目录，
不能当作 object ID。因此当前不生成或声称 object-wise 结果。

Image-wise 协议允许同一物体的不同视角落入训练和测试集合，所以它能支持
标准的逐图性能比较，但不能证明对未见物体的泛化。

## 审计结论

**样例和统计不足以确认可比性，因此不作一般化泛化结论。**

论文采用以下表述：

> The CNN achieved a higher mean success rate on the fixed 85-sample test
> subset from directories 09–10. However, the split composition and the
> markedly stronger traditional-CV performance on this subset indicate that
> it should not be interpreted as evidence of general unseen-object
> generalisation.

## 输出与复现

```bash
conda run -n msc-grasp python src/shared/analyze_cornell_splits.py
```

输出：

```text
data/processed/shared/split_audit/representative_samples.csv
data/processed/shared/split_audit/split_metrics.json
data/processed/shared/split_audit/same_test_subset_metrics.csv
data/processed/shared/split_audit/cornell_split_contact_sheet.png
```
