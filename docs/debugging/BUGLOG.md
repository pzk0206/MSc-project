# Cornell 数据集解析与传统 CV Baseline 调试记录

> 本文件保留完整调试历史。当前状态见
> [`../agent/CURRENT_STATUS.md`](../agent/CURRENT_STATUS.md)，失败模式汇总见
> [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)。

日期：2026-06-26  
项目：基于开放词汇视觉语言模型的 2D 机器人抓取矩形检测与少样本适应性评估

## 1. 本阶段目标

本阶段的目标不是直接进入 VLM，也不是训练深度学习模型，而是先完成一个可复现的数据处理和传统计算机视觉 baseline。

具体目标包括：

1. 理解 Cornell Grasping Dataset 的文件结构；
2. 验证 RGB 图像与抓取标注是否能够正确对应；
3. 抽样可视化 ground-truth 抓取矩形；
4. 建立正式的数据读取器；
5. 将 Cornell 四点抓取框转换为中心格式；
6. 跑出一个传统 OpenCV baseline，作为后续 VLM-assisted pipeline 的对照组。

这一阶段遵循的原则是：

```text
先确认数据能被稳定读取，再做 baseline；
先有传统 CV 对照组，再引入 VLM。
```

## 2. Cornell 数据集文件结构理解

Cornell 数据集中，一个样本由多个同名前缀文件组成。以 `pcd0100` 为例：

```text
pcd0100r.png       RGB 彩色图像
pcd0100d.tiff      depth 深度图
pcd0100cpos.txt    正抓取矩形标注
pcd0100cneg.txt    负抓取矩形标注
pcd0100.txt        3D 点云文件
```

本项目计划书的主线是 2D 抓取矩形检测，因此当前重点是：

```text
RGB 图像
cpos 正抓取标注
2D 抓取矩形几何转换
Cornell-style 评估
```

depth 图像和点云文件目前保留在数据读取器中，但不是当前 baseline 的主线输入。

## 3. 单样本可视化检查

首先编写并运行了单样本检查脚本：

```text
src/shared/inspect_sample.py
```

该脚本读取：

```text
data/raw/cornell/01/pcd0100r.png
data/raw/cornell/01/pcd0100cpos.txt
data/raw/cornell/01/pcd0100cneg.txt
```

并将：

```text
绿色框：正抓取标注 POS
红色框：负抓取标注 NEG
```

画在 RGB 图像上。

运行结果显示：

```text
正抓取标注 坐标范围正常
负抓取标注 坐标范围正常

样本检查完成
图像高度：480
图像宽度：640
图像通道数：3
正抓取矩形数量：4
负抓取矩形数量：3
```

人工观察结果：

```text
抓取框与物体基本对齐；
没有出现整体偏移；
x/y 坐标顺序正确；
标注解析逻辑初步可信。
```

## 4. 全数据集完整性检查

随后编写了全数据集检查脚本：

```text
src/shared/check_cornell_dataset.py
```

该脚本检查：

1. 样本文件是否齐全；
2. RGB 图像是否可读取；
3. cpos/cneg 是否为两列坐标；
4. 标注点数量是否为 4 的倍数；
5. 抓取框坐标是否超出图像范围。

初次运行时发现一个异常样本：

```text
pcd0154
data/raw/cornell/01/pcd0154cneg.txt
```

异常原因：

```text
cneg 负抓取标注文件为空
```

工程判断：

```text
cpos 为空：严重问题，因为没有正抓取监督信号；
cneg 为空：可接受，可以视为 0 个负抓取框。
```

因此修改解析逻辑：

```text
空 cneg 文件 -> 返回 shape=(0, 4, 2) 的空数组
空 cpos 文件 -> 仍然报错
```

修正后全数据集检查结果：

```text
样本总数：885
完整样本数：885
异常样本数：0
正抓取矩形总数：5111
负抓取矩形总数：2909
```

结论：

```text
Cornell 数据集当前可用于后续实验。
```

## 5. 抽样可视化检查

为了避免只验证单个样本，编写了抽样可视化脚本：

```text
src/shared/visualize_sample_checks.py
```

抽样策略：

```text
Cornell 子目录 01 到 10
每个目录抽取 2 张
总共生成 20 张可视化图
```

输出目录：

```text
data/processed/shared/visualizations/sample_checks/
```

人工检查结论：

```text
20 张抽样图中，抓取框整体与物体对齐；
没有发现系统性坐标错误；
没有发现宽高反转或 x/y 反转；
POS/NEG 标注都集中在物体附近；
解析逻辑可以继续用于全数据集处理。
```

## 6. 正式 Dataset Parser

接着编写正式数据读取器：

```text
src/shared/cornell_dataset.py
```

该模块定义了：

```text
CornellGraspDataset
```

支持：

```python
dataset = CornellGraspDataset("data/raw/cornell")
sample = dataset[0]
```

返回内容包括：

```text
sample_id
object_directory
rgb_path
depth_path
cpos_path
cneg_path
point_cloud_path
rgb
depth
positive_rectangles
negative_rectangles
```

运行验证结果：

```text
样本数量：885
第一个样本：pcd0100
RGB shape：(480, 640, 3)
Depth shape：(480, 640)
Depth dtype：float32
正抓取矩形 shape：(4, 4, 2)
负抓取矩形 shape：(3, 4, 2)
```

同时生成数据集索引文件：

```text
data/processed/shared/metadata/cornell_dataset_index.csv
```

意义：

```text
后续训练、评估、错误分析都可以基于统一索引进行；
不需要在不同脚本中重复手写路径扫描逻辑。
```

## 7. 抓取框几何格式转换

Cornell 原始标注是四点格式：

```text
(x1, y1), (x2, y2), (x3, y3), (x4, y4)
```

但后续 baseline 和评估更适合使用中心格式：

```text
center_x
center_y
width
height
angle_radians
angle_degrees
```

因此编写了：

```text
src/shared/grasp_geometry.py
```

并导出全数据集标签：

```text
src/shared/export_cornell_grasp_labels.py
```

输出文件：

```text
data/processed/shared/labels/cornell_grasp_labels_center_format.csv
```

导出结果：

```text
数据集样本数：885
正抓取标签数量：5111
负抓取标签数量：2909
总标签数量：8020
```

意义：

```text
四点格式适合原始数据存储；
中心格式更适合模型预测、IoU 计算、角度误差计算和实验统计。
```

### 7.1 这样转换的论文依据

把四点矩形转换成 `center_x, center_y, width, height, angle` 不是随意设计的，而是为了对齐 Cornell grasp detection 文献中常用的 2D grasp rectangle 表示。

Redmon and Angelova (2015) 使用五维抓取表示：

```text
g = {x, y, θ, h, w}
```

其中：

```text
(x, y) 表示抓取矩形中心；
θ 表示相对水平轴的方向；
h 和 w 表示矩形尺寸。
```

Lenz et al. (2015) 也说明他们 follow rectangle-based grasp detection method，并使用 Cornell 数据集中的 positive/negative grasping rectangles。更早的 Jiang et al. (2011) 则提出/使用了 RGB-D grasping 的 rectangle representation。

因此，本项目把 Cornell 原始四点格式转换为中心格式，目的是让后续预测、IoU 计算和角度误差计算与经典 Cornell grasp detection 文献保持一致。

## 8. 传统 OpenCV Baseline

根据项目计划书，传统 CV baseline 包括：

```text
OpenCV
轮廓分析
旋转边界框
2D 抓取矩形生成
Cornell-style evaluation
```

因此编写了：

```text
src/baseline_cv/run_cv_baseline.py
```

baseline pipeline：

```text
RGB 图像
    ↓
HSV/亮度阈值生成粗略物体 mask
    ↓
轮廓提取
    ↓
选择最可能的物体轮廓
    ↓
minAreaRect 生成旋转外接矩形
    ↓
根据物体方向生成一个预测抓取矩形
    ↓
与 cpos ground truth 进行 Cornell-style 评估
```

评估标准：

```text
IoU >= 0.25
角度误差 <= 30 度
```

### 8.1 为什么先做传统 CV baseline？

项目计划书的核心问题之一是比较：

```text
VLM-assisted grasp detection pipeline
vs.
traditional computer vision baseline
```

因此，在进入 VLM 之前，需要先构建一个可复现的传统 CV 对照组。这个 baseline 的目的不是追求 SOTA，而是提供一个简单、透明、可解释的下限。

文献上，Lenz et al. (2015) 在 Cornell 数据集实验中也将自己的深度学习方法与 Jiang et al. (2011) 的 hand-engineered feature baseline 进行对比，并报告 chance baseline。这说明在 grasp detection 研究中，先建立传统/手工特征 baseline 再比较新方法，是合理的实验结构。

需要注意的是，本项目当前 OpenCV baseline 并不是完整复现 Jiang et al. (2011)。更准确地说，它是一个轻量工程 baseline：

```text
浅色背景假设
颜色/亮度阈值分割
轮廓提取
旋转最小外接矩形
启发式生成抓取矩形
```

其中 `findContours`、`contourArea`、`minAreaRect` 和 `rotatedRectangleIntersection` 来自 OpenCV 官方的 structural analysis and shape descriptors 工具。也就是说，这个 baseline 的定位是：

```text
不是复现某一篇传统方法；
而是基于 Cornell 矩形抓取表示和 OpenCV 几何工具构建的可解释传统 CV baseline。
```

### 8.2 为什么用 IoU 和角度误差判断准确率？

这是 Cornell grasp detection 文献中的标准 rectangle metric。

Lenz et al. (2015) 对 rectangle metric 的定义是：

```text
将算法预测的 top-ranked grasping rectangle 与 ground-truth rectangles 比较；
如果方向误差超过 30°，该预测直接被拒绝；
剩余预测使用 intersection over union；
如果 IoU 至少达到 25%，则认为预测正确。
```

Redmon and Angelova (2015) 也采用同样的 rectangle metric，并明确写出两个条件：

```text
1. 抓取角度必须在 ground truth 的 30° 以内；
2. 预测框与 ground truth 的 Jaccard index 必须大于 25%。
```

其中 Jaccard index 就是 IoU：

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Redmon and Angelova (2015) 还解释了为什么 IoU 阈值不是普通目标检测中常见的 50%，而是 25%：Cornell 的 ground-truth grasps 并不是穷尽所有可能抓取；一个方向正确、但只与某个标注抓取框重合 25% 的矩形，仍然可能是一个可行抓取。

因此，本项目 baseline 使用：

```text
IoU >= 0.25
angle error <= 30 degrees
```

不是主观设定，而是沿用 Cornell grasp detection 领域常用的 rectangle metric。

## 9. Baseline 调试过程

初始 baseline 中，预测抓取方向采用物体长轴方向。

初始结果：

```text
成功率约 12.1%
平均 best IoU：0.2260
平均角度误差：67.80 度
```

通过可视化观察发现：

```text
预测框基本能落在物体上；
但角度经常与 Cornell 标注相差约 90 度；
问题主要不是目标定位失败，而是抓取方向定义不合适。
```

分析后发现：

```text
Cornell 抓取框通常表示夹爪横跨物体较窄的一侧；
因此抓取矩形方向往往更接近物体长轴的垂直方向，
而不是直接沿着物体长轴。
```

因此修正 baseline：

```text
物体长轴方向 + 90 度 -> 预测抓取方向
```

修正后结果：

```text
样本数量：885
成功数量：504
成功率：0.5695
无法生成预测数量：55
平均 best IoU：0.3360
平均角度误差：29.62 度
```

输出文件：

```text
data/processed/baseline_cv/cv_baseline_predictions.csv
data/processed/baseline_cv/cv_baseline_summary.json
data/processed/baseline_cv/visualizations/
```

### 9.1 为什么把物体长轴方向旋转 90°？

初始 baseline 直接使用物体长轴作为抓取矩形方向时，成功率只有约 12.1%，平均角度误差达到 67.80°。可视化显示，预测框通常落在物体上，但方向与 Cornell ground truth 经常接近垂直。

这和抓取矩形的物理含义有关：

```text
抓取矩形不是普通目标检测框；
它表达的是平行夹爪闭合前的位置、方向和开口。
```

对于细长物体，例如遥控器、笔、工具等，合理抓取通常是让夹爪横跨物体较窄的一侧，而不是沿着物体长轴夹过去。因此，使用物体长轴的垂直方向作为抓取矩形方向，在很多 Cornell 样本上更接近 ground truth。

这个调整与 Redmon and Angelova (2015) 对抓取矩形的解释一致：抓取矩形的尺寸和方向对应 gripper plates 的位置与方向，而不是简单的物体外接框方向。

## 10. 当前 Baseline 结论

当前传统 CV baseline 的成功率为：

```text
56.95%
```

该结果说明：

```text
传统 CV 方法在 Cornell 上可以形成一个可用但有限的 baseline；
它依赖背景简单、物体和背景颜色/亮度差异明显；
当分割失败、物体颜色接近背景、轮廓不完整或姿态复杂时，性能会下降。
```

这正好为后续 VLM-assisted pipeline 提供了对照：

```text
如果 VLM 能更稳定地定位目标物体区域，
再接同样的几何抓取生成后端，
理论上应该减少目标区域提取错误，
并提高整体 grasp detection 成功率。
```

## 11. 当前项目状态

已完成：

```text
数据集解压与结构理解
单样本标注可视化
全数据集完整性检查
空 cneg 文件异常处理
抽样可视化 sanity check
正式 Dataset Parser
数据集索引 CSV
抓取框中心格式转换
中心格式标签 CSV
传统 CV baseline
baseline 评估与可视化
```

下一步：

```text
进入 VLM / open-vocabulary object localization 阶段。
```

建议下一步不是直接让 VLM 预测抓取框，而是先做：

```text
RGB image + text prompt
    ↓
VLM / open-vocabulary detector
    ↓
目标 bounding box 或 mask
    ↓
OpenCV 几何抓取后端
    ↓
预测 2D grasp rectangle
    ↓
与传统 CV baseline 对比
```
## 11.1. 今日调试工作

2026-06-29：

- 修复并恢复 `src/visualize_current_params.py` 脚本，解决原脚本内容丢失的问题。
- 将可视化输出改为与用户要求一致的 4 条并排面板：
  1. 原图
  2. 通过色彩/HSV 分割提取的 mask（border=0）
  3. 经过边界裁剪后的 mask（border=150）
  4. border 处理前后差异图
- 删除旧的可视化图片并重新生成，覆盖 `data/processed/visualizations/border_effect_current/` 中的结果。
- 调整可视化逻辑，使第二张和第三张为纯 mask 颜色展示，第四张展示 `border=0` 与 `border=150` 的差异区域。

该工作目的是让调试可视化更直观，方便比较 border 裁剪前后的目标区域提取效果。
## 12. 可以写入 dissertation 的简短表述

英文草稿：

```text
Before integrating open-vocabulary vision-language models, a reproducible traditional computer vision baseline was implemented. The Cornell Grasping Dataset was first parsed and validated through single-sample and multi-sample visual sanity checks. Grasp annotations were converted from four-corner rectangles into a centre-based representation containing position, size and orientation. A simple OpenCV baseline was then constructed using colour/brightness thresholding, contour extraction and rotated bounding boxes. Using the Cornell-style criterion of IoU greater than 0.25 and angle error below 30 degrees, the baseline achieved a success rate of 56.95% over 885 samples. This baseline provides a reference point for evaluating whether VLM-based object localization improves downstream 2D grasp rectangle detection.
```

中文解释：

```text
在引入开放词汇视觉语言模型之前，本项目首先实现了一个可复现的传统计算机视觉 baseline。项目先对 Cornell 数据集进行了单样本和多样本可视化检查，并将抓取标注从四点矩形格式转换为中心点、尺寸和角度格式。随后，使用 OpenCV 的阈值分割、轮廓提取和旋转边界框方法生成 2D 抓取矩形。在 Cornell-style 评估标准下，该传统 CV baseline 在 885 个样本上取得了 56.95% 的成功率。该结果将作为后续 VLM-assisted pipeline 的对照基准。
```

## 13. 本阶段操作与论文依据对应表

| 本项目操作 | 为什么这样做 | 主要依据 |
|---|---|---|
| 使用 Cornell Grasping Dataset | 该数据集是 2D grasp rectangle detection 常用基准，包含 RGB-D 图像和人工抓取矩形标注 | Jiang et al. (2011), Lenz et al. (2015), Redmon and Angelova (2015) |
| 解析 cpos/cneg 四点矩形 | Cornell ground truth 使用 oriented rectangles 表示抓取 | Jiang et al. (2011), Lenz et al. (2015) |
| 转换为 center/width/height/angle | 领域中常用五维 2D grasp representation `g={x,y,θ,h,w}` | Redmon and Angelova (2015) |
| 使用 IoU/Jaccard + angle error 评估 | Cornell rectangle metric 同时考虑位置、尺寸、重合率和方向 | Lenz et al. (2015), Redmon and Angelova (2015) |
| 设置 angle error <= 30° | Cornell rectangle metric 中方向误差超过 30° 会被拒绝 | Lenz et al. (2015), Redmon and Angelova (2015) |
| 设置 IoU >= 0.25 | Cornell rectangle metric 中 IoU/Jaccard 至少 25% 认为预测正确 | Lenz et al. (2015), Redmon and Angelova (2015) |
| 建立传统 CV baseline | 为后续 VLM-assisted pipeline 提供可解释对照组 | 项目计划书目标；Lenz et al. (2015) 中也比较了手工特征/传统 baseline |
| 使用 OpenCV 轮廓和旋转矩形 | 用简单几何方法从目标区域生成 oriented rectangle，作为轻量工程 baseline | OpenCV 官方文档；项目计划书中的 OpenCV/轮廓/PCA/旋转边界框方向 |

## 14. 参考论文与资料

1. Jiang, Y., Moseson, S., and Saxena, A. (2011). *Efficient Grasping from RGB-D Images: Learning Using a New Rectangle Representation*. ICRA 2011.  
   作用：提出/使用 RGB-D grasping 的 rectangle representation，是 Cornell grasp detection 传统表示的重要来源。Lenz et al. (2015) 的参考文献中将其列为 `[28]`。

2. Lenz, I., Lee, H., and Saxena, A. (2015). *Deep Learning for Detecting Robotic Grasps*. The International Journal of Robotics Research.  
   链接：https://arxiv.org/abs/1301.3592  
   作用：使用 Cornell grasping dataset；说明每张图有多个 positive/negative grasping rectangles；采用 rectangle metric：方向误差超过 30° 拒绝，IoU 至少 25% 认为正确。

3. Redmon, J., and Angelova, A. (2015). *Real-Time Grasp Detection Using Convolutional Neural Networks*. ICRA 2015.  
   链接：https://arxiv.org/abs/1412.3128  
   作用：明确使用五维抓取表示 `g={x,y,θ,h,w}`；说明 Cornell grasp labels 是 2D oriented rectangles；采用 rectangle metric：角度 30° 内且 Jaccard/IoU 大于 25%。

4. OpenCV Documentation. *Structural Analysis and Shape Descriptors*.  
   链接：https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html  
   作用：说明 `findContours`、`contourArea`、`minAreaRect`、`rotatedRectangleIntersection` 等几何函数的定义。本项目传统 CV baseline 使用这些函数实现轮廓提取、旋转矩形生成和旋转矩形 IoU 计算。

## 15. VLM-assisted grasp 全量实验记录

2026-06-30：

本阶段完成了从 VLM 目标定位到最终抓取矩形检测的完整 pipeline。

### 15.1. 实验流程

实验流程为：

```text
Cornell RGB image
    ↓
Grounding DINO / open-vocabulary VLM localization
    ↓
VLM object bounding box
    ↓
在 VLM box 附近限制 OpenCV mask 和 contour
    ↓
minAreaRect / 几何后端生成 grasp rectangle
    ↓
Cornell-style metric 评估
```

该流程对应项目计划书中的 VLM-assisted grasp detection 思路：

- VLM 负责开放词汇目标定位；
- OpenCV 几何方法负责从目标区域生成 2D 抓取矩形；
- 最终仍然使用 Cornell grasp rectangle metric 评价，而不是只评价目标检测框。

### 15.2. 运行命令

全量 VLM localization：

```bash
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda
```

全量 VLM-assisted grasp evaluation：

```bash
conda run -n msc-grasp python src/vlm/run_vlm_assisted_grasp.py
```

### 15.3. 输出文件

VLM localization 输出：

```text
data/processed/vlm/localization/grounding_dino_generic_small_object_predictions.csv
data/processed/vlm/localization/grounding_dino_generic_small_object_summary.json
data/processed/vlm/visualizations/localization_checks/generic_small_object/
data/processed/vlm/visualizations/grounding_dino_generic_small_object_overview.png
```

VLM-assisted grasp 输出：

```text
data/processed/vlm/grasp/vlm_assisted_grasp_predictions.csv
data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
data/processed/vlm/grasp/visualizations/success/
data/processed/vlm/grasp/visualizations/failure/
data/processed/vlm/grasp/visualizations/vlm_assisted_success_failure_overview.png
```

### 15.4. 全量结果

| 方法 | 样本数 | 成功数 | 成功率 | 平均 best IoU | 平均角度误差 |
|---|---:|---:|---:|---:|---:|
| Traditional CV baseline | 885 | 504 | 56.95% | 0.3360 | 29.62° |
| VLM-assisted CV | 885 | 649 | 73.33% | 0.4182 | 14.81° |

VLM localization 本身结果：

```text
检测成功数量：885 / 885
检测率：100%
```

VLM-assisted grasp 结果：

```text
最终抓取成功数量：649 / 885
最终抓取成功率：73.33%
无法生成预测数量：0
mask 轮廓缺失数量：0
fallback 使用数量：0
平均 best IoU：0.4182
平均角度误差：14.81 度
```

### 15.5. 当前结论

该结果说明：

1. 在 Cornell 数据集上，Grounding DINO 使用通用 prompt `small object` 已经能够稳定定位目标物体。
2. 仅将 VLM 作为目标定位前端，也能明显提升最终 grasp rectangle detection 的表现。
3. 相比传统 CV baseline，VLM-assisted 方法不仅成功率更高，而且角度误差显著降低。
4. 当前瓶颈已经从“是否能找到物体”转移到“如何从物体区域生成更符合 Cornell 标注的抓取矩形”。

因此，短期内不需要优先做 VLM 微调。
更合理的下一步是分析 VLM-assisted 的失败案例，并改进抓取矩形生成后端，例如：

- 调整抓取框 width / height 的启发式比例；
- 使用 depth 图辅助判断物体边界和可抓取区域；
- 使用 PCA / skeleton / 多候选 grasp sampling 生成多个候选框；
- 将 VLM 定位结果与 RGB-D 几何特征结合。

### 15.6. 关于是否继续微调 VLM 的阶段性判断

项目初步计划书中包含 VLM few-shot adaptation / fine-tuning 的设想。
但是根据当前全量实验结果，现阶段不应优先微调 VLM。

原因如下：

```text
VLM localization detection rate: 885 / 885 = 100%
VLM-assisted grasp success rate: 649 / 885 = 73.33%
```

这说明：

- VLM 负责的目标定位阶段已经非常稳定；
- 当前失败样本并不是因为物体没有被找到；
- 主要误差来自 VLM box 之后的 grasp rectangle generation 阶段；
- 也就是从目标区域生成 Cornell-style 抓取矩形的几何逻辑仍然不够强。

因此，当前研究重点应从：

```text
如何微调 VLM 让它更会找物体
```

转移为：

```text
在 VLM 已经可靠定位物体的前提下，如何生成更准确的抓取矩形
```

换句话说，当前系统的主要瓶颈已经不是 object localization，而是 grasp generation backend。

更严谨的论文表述可以是：

```text
Although the initial project plan considered few-shot adaptation or fine-tuning of the VLM, the zero-shot Grounding DINO localization achieved a 100% detection rate on the Cornell dataset using a generic prompt. Therefore, the main limitation was not object localization, but the downstream grasp rectangle generation stage. As a result, this project focuses on analysing and improving the grasp generation backend rather than fine-tuning the VLM at this stage.
```

中文解释：

```text
虽然初步计划中考虑了对 VLM 进行少样本适应或微调，但实验发现，零样本 Grounding DINO 在 Cornell 数据集上使用通用 prompt 已经取得了 100% 的目标检测率。因此，当前系统的主要限制并不在目标定位，而在后续抓取矩形生成阶段。因此，本项目现阶段将重点从 VLM 微调转向抓取生成后端的分析与改进。
```

这并不意味着 VLM 微调完全没有价值。
它可以保留为后续扩展或 future work，例如：

- 当数据集变成更复杂真实场景时，VLM 可能不再 100% 定位成功；
- 当 prompt 从 generic object 变成多物体语言指令时，VLM adaptation 可能重新变得重要；
- 当研究目标从 Cornell 单目标抓取扩展到开放世界多目标抓取时，VLM 微调或 prompt adaptation 仍然值得研究。

但在当前 Cornell 单目标实验阶段，优先微调 VLM 的收益不如优先改进抓取框生成逻辑。

### 15.7. 可以写入 dissertation 的简短表述

英文草稿：

```text
After establishing the traditional computer vision baseline, an open-vocabulary VLM-assisted pipeline was implemented. Grounding DINO was used as a localization front-end with the generic prompt "small object", and the resulting bounding box was used to constrain the OpenCV contour-based grasp rectangle generator. On the full Cornell Grasping Dataset of 885 samples, the VLM-assisted method achieved a grasp detection success rate of 73.33%, compared with 56.95% for the traditional CV baseline. The mean angular error was also reduced from 29.62 degrees to 14.81 degrees. These results suggest that open-vocabulary localization can substantially improve the downstream geometric grasp rectangle pipeline, although the remaining failures indicate that grasp rectangle generation, rather than object localization alone, remains a key bottleneck.
```

中文解释：

```text
在完成传统计算机视觉 baseline 后，本项目实现了一个 VLM-assisted 抓取检测流程。该流程使用 Grounding DINO 和通用 prompt “small object” 对目标物体进行定位，然后将 VLM 输出的 bounding box 用于限制 OpenCV 轮廓提取和几何抓取框生成。在 Cornell Grasping Dataset 的 885 个样本上，VLM-assisted 方法取得了 73.33% 的抓取检测成功率，高于传统 CV baseline 的 56.95%。同时，平均角度误差也从 29.62 度下降到 14.81 度。该结果说明，开放词汇 VLM 定位能够有效改善后续几何抓取矩形检测，但剩余失败案例也表明，真正的瓶颈已经逐渐转向抓取矩形生成策略本身。
```
