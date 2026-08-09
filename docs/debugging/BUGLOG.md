# 项目调试记录

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
3. 抽样可视化真实标注抓取矩形；
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

## 6. 正式数据集解析器

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

Lenz 等人（2015）也说明其采用基于矩形的抓取检测方法，并使用 Cornell 数据集中的正负抓取矩形。更早的 Jiang 等人（2011）则提出并使用了 RGB-D 抓取的矩形表示。

因此，本项目把 Cornell 原始四点格式转换为中心格式，目的是让后续预测、IoU 计算和角度误差计算与经典 Cornell grasp detection 文献保持一致。

## 8. 传统 OpenCV 基线

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

### 8.1 为什么先做传统计算机视觉基线？

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

这是 Cornell 抓取检测文献中的标准矩形指标。

Lenz 等人（2015）对矩形指标的定义是：

```text
将算法预测的最高排名抓取矩形与真实标注矩形比较；
如果方向误差超过 30°，该预测直接被拒绝；
剩余预测使用 intersection over union；
如果 IoU 至少达到 25%，则认为预测正确。
```

Redmon 和 Angelova（2015）也采用同样的矩形指标，并明确写出两个条件：

```text
1. 抓取角度必须在 ground truth 的 30° 以内；
2. 预测框与 ground truth 的 Jaccard index 必须大于 25%。
```

其中 Jaccard index 就是 IoU：

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Redmon 和 Angelova（2015）还解释了为什么 IoU 阈值不是普通目标检测中常见的 50%，而是 25%：Cornell 的真实抓取标注并未穷尽所有可能抓取；一个方向正确、但只与某个标注抓取框重合 25% 的矩形，仍然可能是一个可行抓取。

因此，本项目 baseline 使用：

```text
IoU >= 0.25
angle error <= 30 degrees
```

不是主观设定，而是沿用 Cornell 抓取检测领域常用的矩形指标。

## 9. 基线调试过程

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

## 10. 当前基线结论

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
进入 VLM / 开放词汇目标定位阶段。
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
## 12. 可以写入毕业论文的简短表述

论文表述草稿：

```text
在引入开放词汇视觉语言模型之前，本项目首先实现了一个可复现的传统计算机视觉基线。项目通过单样本和多样本可视化检查解析并验证 Cornell Grasping Dataset，将四点抓取矩形标注转换为包含位置、尺寸和方向的中心参数表示。随后，使用颜色与亮度阈值分割、轮廓提取和旋转边界框构建简单的 OpenCV 基线。在 IoU 大于 0.25 且角度误差小于 30 度的 Cornell 评估标准下，该基线在 885 个样本上取得 56.95% 的成功率。这一结果为评估 VLM 目标定位能否改善后续二维抓取矩形检测提供了参照。
```

简要说明：

```text
在引入开放词汇视觉语言模型之前，本项目首先实现了一个可复现的传统计算机视觉 baseline。项目先对 Cornell 数据集进行了单样本和多样本可视化检查，并将抓取标注从四点矩形格式转换为中心点、尺寸和角度格式。随后，使用 OpenCV 的阈值分割、轮廓提取和旋转边界框方法生成 2D 抓取矩形。在 Cornell-style 评估标准下，该传统 CV baseline 在 885 个样本上取得了 56.95% 的成功率。该结果将作为后续 VLM-assisted pipeline 的对照基准。
```

## 13. 本阶段操作与论文依据对应表

| 本项目操作 | 为什么这样做 | 主要依据 |
|---|---|---|
| 使用 Cornell Grasping Dataset | 该数据集是 2D grasp rectangle detection 常用基准，包含 RGB-D 图像和人工抓取矩形标注 | Jiang et al. (2011), Lenz et al. (2015), Redmon and Angelova (2015) |
| 解析 cpos/cneg 四点矩形 | Cornell ground truth 使用 oriented rectangles 表示抓取 | Jiang et al. (2011), Lenz et al. (2015) |
| 转换为 center/width/height/angle | 领域中常用五维 2D grasp representation `g={x,y,θ,h,w}` | Redmon and Angelova (2015) |
| 使用 IoU/Jaccard + 角度误差评估 | Cornell 矩形指标同时考虑位置、尺寸、重合率和方向 | Lenz 等（2015），Redmon 和 Angelova（2015） |
| 设置角度误差 <= 30° | Cornell 矩形指标中方向误差超过 30° 会被拒绝 | Lenz 等（2015），Redmon 和 Angelova（2015） |
| 设置 IoU >= 0.25 | Cornell 矩形指标中 IoU/Jaccard 至少 25% 认为预测正确 | Lenz 等（2015），Redmon 和 Angelova（2015） |
| 建立传统 CV baseline | 为后续 VLM-assisted pipeline 提供可解释对照组 | 项目计划书目标；Lenz et al. (2015) 中也比较了手工特征/传统 baseline |
| 使用 OpenCV 轮廓和旋转矩形 | 用简单几何方法从目标区域生成 oriented rectangle，作为轻量工程 baseline | OpenCV 官方文档；项目计划书中的 OpenCV/轮廓/PCA/旋转边界框方向 |

## 14. 参考论文与资料

1. Jiang, Y., Moseson, S., and Saxena, A. (2011). *Efficient Grasping from RGB-D Images: Learning Using a New Rectangle Representation*. ICRA 2011.  
   作用：提出/使用 RGB-D grasping 的 rectangle representation，是 Cornell grasp detection 传统表示的重要来源。Lenz et al. (2015) 的参考文献中将其列为 `[28]`。

2. Lenz, I., Lee, H., and Saxena, A. (2015). *Deep Learning for Detecting Robotic Grasps*. The International Journal of Robotics Research.  
   链接：https://arxiv.org/abs/1301.3592  
   作用：使用 Cornell 抓取数据集；说明每张图有多个正负抓取矩形；采用矩形指标：方向误差超过 30° 拒绝，IoU 至少 25% 认为正确。

3. Redmon, J., and Angelova, A. (2015). *Real-Time Grasp Detection Using Convolutional Neural Networks*. ICRA 2015.  
   链接：https://arxiv.org/abs/1412.3128  
   作用：明确使用五维抓取表示 `g={x,y,θ,h,w}`；说明 Cornell 抓取标签是二维旋转矩形；采用矩形指标：角度误差在 30° 内且 Jaccard/IoU 大于 25%。

4. OpenCV Documentation. *Structural Analysis and Shape Descriptors*.  
   链接：https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html  
   作用：说明 `findContours`、`contourArea`、`minAreaRect`、`rotatedRectangleIntersection` 等几何函数的定义。本项目传统 CV baseline 使用这些函数实现轮廓提取、旋转矩形生成和旋转矩形 IoU 计算。

## 15. VLM 辅助抓取全量实验记录

2026-06-30：

本阶段完成了从 VLM 目标定位到最终抓取矩形检测的完整 pipeline。

### 15.1. 实验流程

实验流程为：

```text
Cornell RGB 图像
    ↓
Grounding DINO / 开放词汇 VLM 定位
    ↓
VLM 目标边界框
    ↓
在 VLM 边界框附近限制 OpenCV 掩膜和轮廓
    ↓
minAreaRect / 几何后端生成抓取矩形
    ↓
Cornell 风格指标评估
```

该流程对应项目计划书中的 VLM-assisted grasp detection 思路：

- VLM 负责开放词汇目标定位；
- OpenCV 几何方法负责从目标区域生成 2D 抓取矩形；
- 最终仍然使用 Cornell 抓取矩形指标评价，而不是只评价目标检测框。

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
| 传统计算机视觉基线 | 885 | 504 | 56.95% | 0.3360 | 29.62° |
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
VLM 定位检测率：885 / 885 = 100%
VLM 辅助抓取成功率：649 / 885 = 73.33%
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

换句话说，当前系统的主要瓶颈已经不是目标定位，而是抓取生成后端。

更严谨的论文表述可以是：

```text
虽然初步项目计划考虑了 VLM 的少样本适应或微调，但零样本 Grounding DINO 使用通用提示词在 Cornell 数据集上取得了 100% 的检测率。因此，当前主要限制不在目标定位，而在后续抓取矩形生成阶段。基于这一结果，本项目现阶段重点分析和改进抓取生成后端，而不是微调 VLM。
```

中文解释：

```text
虽然初步计划中考虑了对 VLM 进行少样本适应或微调，但实验发现，零样本 Grounding DINO 在 Cornell 数据集上使用通用 prompt 已经取得了 100% 的目标检测率。因此，当前系统的主要限制并不在目标定位，而在后续抓取矩形生成阶段。因此，本项目现阶段将重点从 VLM 微调转向抓取生成后端的分析与改进。
```

这并不意味着 VLM 微调完全没有价值。
它可以保留为后续扩展或未来工作，例如：

- 当数据集变成更复杂真实场景时，VLM 可能不再 100% 定位成功；
- 当 prompt 从 generic object 变成多物体语言指令时，VLM adaptation 可能重新变得重要；
- 当研究目标从 Cornell 单目标抓取扩展到开放世界多目标抓取时，VLM 微调或 prompt adaptation 仍然值得研究。

但在当前 Cornell 单目标实验阶段，优先微调 VLM 的收益不如优先改进抓取框生成逻辑。

### 15.7. 可以写入毕业论文的简短表述

论文表述草稿：

```text
在建立传统计算机视觉基线后，本项目实现了一条开放词汇 VLM 辅助实验流程。Grounding DINO 使用通用提示词“small object”作为定位前端，其输出的边界框用于限制基于 OpenCV 轮廓的抓取矩形生成器。在包含 885 个样本的完整 Cornell Grasping Dataset 上，VLM 辅助方法取得了 73.33% 的抓取检测成功率，高于传统计算机视觉基线的 56.95%。平均角度误差也从 29.62 度降至 14.81 度。结果表明，开放词汇定位可以显著改善后续几何抓取矩形流程；剩余失败案例则说明，抓取矩形生成而非目标定位本身，已成为主要瓶颈。
```

简要说明：

```text
在完成传统计算机视觉 baseline 后，本项目实现了一个 VLM-assisted 抓取检测流程。该流程使用 Grounding DINO 和通用 prompt “small object” 对目标物体进行定位，然后将 VLM 输出的 bounding box 用于限制 OpenCV 轮廓提取和几何抓取框生成。在 Cornell Grasping Dataset 的 885 个样本上，VLM-assisted 方法取得了 73.33% 的抓取检测成功率，高于传统 CV baseline 的 56.95%。同时，平均角度误差也从 29.62 度下降到 14.81 度。该结果说明，开放词汇 VLM 定位能够有效改善后续几何抓取矩形检测，但剩余失败案例也表明，真正的瓶颈已经逐渐转向抓取矩形生成策略本身。
```

## 16. CNN 多轮汇总记录缺陷

日期：2026-07-24

### 16.1 已确认问题

1. `multi_run_summary.json` 的 `per_run.best_val_loss` 错误地写入了
   `all_success_rate`，因此旧文件中的逐轮验证损失不可用。
2. `run_cnn_grasp.py` 末尾存在两个相同的 `__main__` 入口，直接执行脚本会
   重复运行完整流程。
3. 当前保存的 `cnn_grasp_summary.json` 和
   `cnn_grasp_predictions.csv` 来自五次实验的最后一轮，不是项目记录中的
   73.11% 独立单次实验。

### 16.2 影响范围

- 五次实验的聚合成功率、IoU 和角度均值/标准差由每轮评估指标计算，不受
  `best_val_loss` 字段错误影响。
- 旧 `per_run.best_val_loss` 不得作为论文训练损失证据。
- 由于原始独立单次 JSON/CSV 已被覆盖，73.11% 单次结果不进入论文主表。

### 16.3 修复要求

- 从每轮训练历史中保存真实最佳验证损失；
- 将多轮汇总提取为可单元测试的纯函数；
- 保证脚本只有一个 `__main__` 入口；
- 不覆盖现有实验产物来验证修复。

## 17. 论文模板数学符号包冲突

日期：2026-07-24

### 17.1 现象

使用当前 Tectonic/LaTeX 发行版编译 `l4proj.tex` 时，正文处理前在
`amssymb.sty` 报错：

```text
LaTeX Error: Command `\Bbbk' already defined.
```

### 17.2 根因与验证

`l4proj.cls` 先加载 `newtxmath`，随后再次加载 `amssymb`。两个包都提供
`\Bbbk`，新版发行版将重复定义视为错误。最小样例稳定复现：

- `newtxmath + amssymb`：相同错误；
- 仅 `amssymb`：编译成功；
- `newtxmath + amsmath/amsfonts/amsbsy`：编译成功且常用数学符号正常。

### 17.3 修复

从模板的 AMS 包列表移除重复的 `amssymb`，保留 `newtxmath` 和其他 AMS
基础包。完整论文随后成功编译为 22 页 PDF，无 LaTeX fatal error。

## 18. CUDA 同 seed 训练不可重复

日期：2026-07-26

### 18.1 现象

多头 CNN 使用相同 seed 42、相同固定划分和同一 GPU 再次运行时，第一个
epoch 的损失只相差约 `2e-5`，但早停后的固定测试成功率从 82.35% 漂移至
76.47%。这说明原先的 `manual_seed` 设置只控制随机数，没有消除 CUDA 算法
非确定性。

### 18.2 根因

- cuDNN benchmark 和卷积算法没有显式确定性约束；
- DataLoader 没有使用独立的 seeded generator；
- `AdaptiveAvgPool2d` 的 CUDA 反向传播不支持 PyTorch 严格确定性模式。

### 18.3 修复与验证

- 固定 Python、NumPy、PyTorch 和所有 CUDA RNG；
- 设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- 关闭 `torch.backends.cudnn.benchmark`，启用 cuDNN deterministic 和
  `torch.use_deterministic_algorithms(True)`；
- 为训练 DataLoader 使用按实验 seed 初始化的 `torch.Generator`；
- 将固定 `224×224` 输入下的 `AdaptiveAvgPool2d(1)` 替换为数学等价的
  `AvgPool2d(7)`，保持参数量和可训练权重键不变。

在 GTX 1650 Ti 上用相同 seed 连续运行两次三轮小型训练，训练历史完全相同，
所有模型权重逐位一致。修复前生成的单头重跑和中断多头批次保留为诊断证据，
不进入最终单头/多头公平对照。

## 19. PyBullet 环境、测试入口与 prompt 歧义

日期：2026-07-27

### 19.1 `conda run pytest` 无法导入 `src`

直接运行 `conda run -n msc-grasp pytest -q` 时，pytest 可执行入口的
`sys.path` 不包含仓库根目录，测试收集阶段报
`ModuleNotFoundError: No module named 'src'`。同一环境使用：

```bash
conda run -n msc-grasp python -m pytest -q
```

可正确发现仓库并通过原有 32 项测试。该问题属于命令入口差异，不是源码回归；
项目文档统一使用 `python -m pytest`。

### 19.2 沙箱内外 CUDA 结果不同

受限沙箱内同一 `msc-grasp` 环境返回
`torch.cuda.is_available() == False`。在沙箱外运行相同 Conda 命令返回
`True`，并识别 `NVIDIA GeForce GTX 1650 Ti`。因此沙箱内的 `False` 不能
解释为驱动、CUDA 或 PyTorch 安装失败；真实 GPU 实验必须在允许访问
`/dev/dxg` 的环境运行。

### 19.3 generic prompt 选错机器人部件

PyBullet 默认场景同时包含 Panda 和黄色小鸭。使用 Cornell 主实验的 generic
prompt `small object` 时，Grounding DINO 返回 box
`[323, 68, 450, 150]`、score `0.5985`。人工检查定位图确认该框覆盖 Panda
末端，而非小鸭；runner 的 `status=success` 只表示管线产生合法输出，不代表
语义目标正确。

保持场景、模型、阈值和后端不变，仅将 prompt 改为
`yellow rubber duck` 后，box 为 `[318, 208, 374, 273]`、score `0.6985`，
人工检查确认覆盖小鸭。根因是多物体机器人场景中的 prompt 歧义，而不是颜色
通道、相机坐标或绘图错误。后续默认配置应使 prompt 与默认 URDF 配对，同时
保留其他物体和多物体目标的显式 prompt 选择能力。

## 20. Stage 6A 斜视相机中心偏差发现

日期：2026-07-28

### 20.1 现象

Stage 6A geometry preflight 通过全部静态门控（target selection、重投影、
IK/FK、碰撞审计），`scientific_gate_passed: true`。但使用 metadata 中
保存的 cube 真值质心 `(0.48, 0.0, 0.65)` m 和冻结半高 `0.025` m 进行
事后审计后发现：预测世界表面点相对 cube 质心的 X 偏差为 `0.0265` m，
Y 偏差为 `0.0022` m，XY 合成偏差为 `0.0266` m（`26.55` mm）。

### 20.2 根因分析

- 斜视相机位于 `X=1.0` m，cube 质心位于 `X=0.48` m。相机以倾斜角度
  观察 cube，可见表面点（面向相机的侧面）在 X 方向超前于质心。
- 单像素反投影恢复的是 **可见表面点**，而非物体质心。对于斜视相机，
  表面点 XY ≠ 质心 XY。
- Cube 半宽为 `0.025` m，偏差 `0.0266` m 已超过半宽，意味着夹爪中心
  会落到 cube 边缘而非中心。
- Z 方向偏差仅 `0.0005` m，高度恢复准确——问题完全在 XY 平面。

### 20.3 影响

- Stage 6A 的静态门控不检查夹爪中心相对于物体质心的偏差，因此
  `scientific_gate_passed: true` 不能解释为计划具有物理抓取成功条件。
- 该偏差是**斜视相机的固有特征**：单目深度只能恢复可见表面点，无法
  穿透物体得知质心位置。
- 该发现不影响 Stage 6A 静态安全门控的既有结论，但说明当前冻结计划
  在物理执行中可能遇到抓取中心不准确的问题。

### 20.4 验证

- Stage 6A.1 在独立目录完成只读诊断：读取 Stage 6A 正式产物的 RGB、
  metadata、summary 和冻结计划，交叉检查哈希与协议。
- 隔离 CUDA 重跑确认 RGB SHA-256、定位框、世界点和偏差数值完全一致。
- 正式 Stage 6A 目录在诊断前后 SHA-256 清单无差异。

## 21. Stage 6A.2 窗口化最小深度恢复失败

日期：2026-07-28

### 21.1 尝试方案

对 geometry 和 multi-head 共同应用窗口化（$5\times5$）目标 mask 最小
深度恢复：在抓取中心周围 $5\times5$ 窗口内取 segmentation mask 属于
目标且深度最小的像素，以期选到更接近物体顶面中心的点。

### 21.2 结果

| 后端 | 原始偏差 | 窗口化恢复后 | 变化 |
|---|---:|---:|---:|
| geometry | $26.55$ mm | $27.60$ mm | $+1.05$ mm（恶化）|
| multi-head | $24.60$ mm | $26.43$ mm | $+1.83$ mm（恶化）|

### 21.3 根因

窗口内最小深度自然选中朝向相机的物体边缘——与原始单像素的问题相同但
被 $5\times5$ 窗口放大。对于斜视相机，最小深度点更靠近相机，因此
选中的是 cube 的最近边缘而非顶面中心。

### 21.4 决策

该方案被排除。选择冻结为：回退至原始单像素反投影，原样消费 Stage 6A
冻结计划进入 Stage 6B 物理执行；偏差作为斜视相机固有特征在论文中诚实
披露。

## 22. Stage 6B 斜视相机物理抓取失败

日期：2026-08-01 至 2026-08-04

### 22.1 Geometry 后端

- 加载 Stage 6A 冻结计划执行完整七阶段物理抓取。
- Pregrasp/approach/grasp_depth 到达误差均 ≤ 1.04mm，静态和到达门控
  全部通过。
- 夹爪闭合并建立双指接触，但首次接触在闭合第 226/240 步（真值为 93 步），
  说明夹爪在闭合接近终点时才触及物体。
- **抬升失败**：cube 仅上升 0.38mm（门槛 100mm），末端-cube 相对漂移
  120mm（等于抬升命令量，说明 cube 完全未动），桌面接触 481 次。
- 根因：XY 偏差 26.62mm 导致夹爪仅触及 cube 边缘，抬升时立即滑脱。

### 22.2 Multi-head CNN 后端

- 新增多头 CNN preflight runner，生成无中心恢复的执行计划。
- 静态门控全部通过，Stage 6B 执行结果与 geometry 一致：
  cube 仅上升 0.34mm，漂移 120mm，抬升失败。
- XY 偏差 24.67mm，同样超过 cube 半宽。

### 22.3 配对结论

两个后端在同一阶段以相同模式失败，排除了后端选择作为根因。
**根因在感知管线前端：斜视相机下的单像素反投影无法恢复真实三维中心。**
该结论受限于固定场景 $N=1$ 和当前夹爪/物体几何。

## 23. 头顶相机 preflight 碰撞迭代

日期：2026-08-05 至 2026-08-06

### 23.1 方案设计

新增 `run_overhead_preflight.py`：相机置于场景正上方
(`eye=(0.5,0.0,1.3)`, `target=(0.5,0.0,0.62)`, `up=(0,1,0)`），
光轴垂直向下，从几何上消除斜视反投影的 XY 偏差。

### 23.2 首次尝试 `-0.06` m——碰撞失败

- 抓取深度 standoff 设为 `-0.06`，使 TCP 降至桌面高度 `Z=0.615` m。
- Preflight 碰撞审计失败：TCP 穿透桌子。
- 结论：TCP 不能低于桌面，抓取深度必须 > 桌面高度。

### 23.3 第二次尝试 `-0.025` m——preflight 通过

- 抓取深度 standoff 调整为 `-0.025`，TCP 在 cube 中心高度 `Z=0.650` m。
- Pregrasp offset 同步从 `0.115` 调整为 `0.145`，保持安全垂直距离。
- 三处工程修改：
  1. `pose_generation.py`：放开 `surface_standoff_m > 0` 约束，允许负值
  2. `run_overhead_preflight.py`：分别传入 contact 和 grasp_depth 的差异化参数
  3. `execution_plan.py`：将 approach/grasp 高度差从硬编码 `0.015` 放宽为
     `approach > grasp_depth`
- Preflight 通过：2/2 候选 gate_passed，minimum_clearance_m=0.002。

### 23.4 Stage 6B 物理抓取成功

- XY 中心偏差 0.76mm（斜视 26.55mm，改善 97.1%）。
- Pregrasp/approach/grasp_depth 到达误差均 ≤ 1.0mm。
- 双指首次接触在闭合第 41 步（斜视 226 步），夹爪扎实包裹 cube 中心。
- 抬升阶段 cube 上升 119.94mm，离桌干净（桌面接触仅 2 次），
  漂移 1.4mm，保持 240 步无异常，科学门控通过。
- **控制变量对照**：唯一改变的变量是相机位姿（连带抓取深度修正），
  其余条件不变。斜视失败→头顶成功，根因确在相机几何。

## 24. PyBullet 调试经验总结

### 24.1 分层诊断原则

PyBullet 调试中最容易犯的错误是把多层问题混在一起。本项目的分层顺序：

```text
坐标数据流 → 真值控制链 → 感知静态预检 → 感知物理执行 → 根因对照实验
```

每一层只在前一层通过后才进入。如果跳过真值控制链直接做感知物理执行，
就无法区分"控制链本身不可行"与"感知输入错误"。

### 24.2 静态门控 ≠ 物理成功

Stage 6A 通过全部静态门控（重投影、IK/FK、碰撞审计），但 Stage 6B
物理执行失败。静态门控检查的是几何和碰撞的**必要条件**，不能替代
动态接触质量和抓取稳定性。论文中必须区分"scientific_gate_passed"
的实际含义：通过当前阶段设计的门控检查，不等于物理抓取成功。

### 24.3 偏差诊断优于猜测修复

发现 26.55mm 偏差后，优先用只读诊断确认数值（Stage 6A.1）而非立即
修改算法。窗口化恢复尝试（Stage 6A.2）在了解斜视相机几何原理后可以
预判其会失败（最小深度 = 最近边缘），但仍用真实实验排除该方案，
提供了负向证据写入论文。

### 24.4 碰撞审计的参考价值

头顶相机首次 `-0.06` 尝试因穿透桌子被碰撞审计拦截。这说明碰撞审计
在抓取深度选择中起到了安全门控作用，验证了系统设计的合理性。
