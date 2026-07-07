# MSc 毕业设计初步项目计划书

基于视觉语言模型引导的 2D 机器人抓取检测


**学生姓名**

庞镇坤


**专业**

MSc Robotics & Artificial Intelligence


**指导老师**

Dr Jan Paul Siebert


**日期**

2026 年 6 月 18 日


## 1. 项目课题

基于开放词汇视觉语言模型引导的 2D 机器人抓取矩形检测

英文题目：VLM-guided 2D Robotic Grasp Rectangle Detection with Geometric and Learning-based Grasp Backends

本项目旨在研究预训练开放词汇视觉语言模型是否可以作为机器人抓取任务中的语言条件目标定位前端，并比较不同抓取框生成后端在 2D 抓取矩形检测中的表现。具体而言，项目将 VLM 输出的目标区域分别接入传统几何后端和轻量 CNN 学习式后端，用于生成 2D 抓取矩形。项目重点是模块化 VLA 风格的抓取感知，而不是完整机器人控制或真实机械臂执行。


## 2. 项目兴趣方向

我的主要项目兴趣方向是深度学习、计算机视觉、机器人感知和视觉语言模型。我尤其感兴趣的是预训练多模态模型如何帮助机器人理解物体和场景，特别是在目标 grounding、目标检测和机器人抓取检测任务中的应用。

从更宽泛的角度看，本项目与具身智能相关，但 MSc 阶段的研究范围将聚焦在机器人系统的视觉感知部分，而不是完整的机器人控制闭环。


## 3. 适合的数据集

本项目计划主要使用 Cornell Grasping Dataset。该数据集是 2D 机器人抓取检测领域常用的学术基准数据集，包含日常物体的 RGB-D 图像以及人工标注的旋转抓取矩形。

数据集公开可获取，适合 MSc 项目快速开展。

该数据集在机器人抓取检测研究中使用广泛，便于与已有研究对照。

数据集中包含 2D 抓取矩形标注，适合进行量化评估。

项目不依赖真实机械臂、传感器或硬件控制，风险较低。

数据集支持项目聚焦于视觉感知、目标定位和抓取矩形预测。


## 4. 研究问题与核心目标


### 核心研究问题

预训练开放词汇视觉语言模型作为目标定位前端时，能够在多大程度上提升 2D 机器人抓取矩形检测？

预训练 VLM 或开放词汇目标检测模型能否在不进行任务特定端到端训练的情况下，通过文本 prompt 在 Cornell 数据集中定位目标物体？

VLM 输出的目标区域能否通过不同抓取框生成后端，例如 OpenCV 几何规则或轻量 CNN 回归网络，转换为可靠的 2D 抓取矩形？

与传统计算机视觉 baseline 相比，VLM-guided pipeline 在抓取矩形准确率、角度误差和鲁棒性方面表现如何？当 VLM 定位可靠时，系统瓶颈是否会转移到 grasp rectangle generation 后端？


### 拓展研究问题

如果 VLM-guided 几何管线顺利完成，项目将进一步探索 VLM-guided CNN grasp backend 是否可以改善手工几何后端在抓取矩形生成上的限制。VLM 少样本适应或微调保留为复杂多目标场景下的 future work，不作为当前项目成败的核心要求。


## 5. 核心目标

整理 Cornell 数据集，解析图像和抓取标签，并可视化 ground-truth 抓取矩形。

构建可复现的传统计算机视觉 baseline，包括 OpenCV、轮廓分析、PCA 和旋转边界框方法。

部署预训练开放词汇视觉定位模型，例如 Grounding DINO、OWL-ViT 或 Florence-2。

将 VLM 目标定位结果与几何抓取估计后端结合，形成完整的 VLM-guided geometric grasp detection pipeline。

使用定位 IoU、抓取矩形 IoU、角度误差、Cornell-style 抓取成功标准和失败案例分析进行评估。

在 VLM-guided 几何管线稳定后，分析失败案例，并实现轻量 VLM-guided CNN grasp regressor，用于在 VLM crop 内学习抓取矩形生成。若时间允许，再探索 GG-CNN style dense prediction。


## 6. 初步方法设计

输入：Cornell 数据集中的 RGB 或 RGB-D 图像，以及描述目标物体或类别的文本 prompt。

视觉语言定位：使用预训练 VLM 或开放词汇目标检测模型根据 prompt 识别目标物体区域。

抓取框生成后端：首先利用 OpenCV 几何规则对 VLM 目标区域进行处理并估计 2D 抓取矩形；随后引入轻量 CNN，在 VLM crop 内直接回归抓取矩形参数。

量化评估：将预测抓取矩形与 Cornell 人工标注进行比较，并结合定性可视化、失败案例分析以及 Traditional CV / VLM-guided geometric / VLM-guided CNN 三类方法对比。

范围控制：本项目研究的是机器人抓取感知，不是完整机器人控制系统。这样既保留技术含量，也降低硬件和控制系统带来的风险。


## 7. 初步时间表与里程碑


**时间**


**阶段**


**目标与里程碑**

第 1-2 周

文献阅读与数据集准备

阅读机器人抓取检测、VLM grounding 和开放词汇检测相关论文；整理 Cornell 数据集；读取图片和标签；可视化人工标注抓取框；配置 Overleaf 模板。

第 3-4 周

传统计算机视觉 baseline

使用 OpenCV 实现分割、轮廓提取、PCA 或旋转边界框方法；生成初始抓取矩形；实现 IoU 与角度误差等基础评估。

第 5-6 周

VLM 辅助抓取检测管线

部署预训练开放词汇视觉定位模型；使用文本 prompt 定位目标物体；将定位结果接入几何抓取估计后端。

第 7-8 周

VLM-guided CNN 抓取后端与系统对比

在 VLM-guided geometric 结果基础上分析失败案例；实现轻量 VLM-guided CNN 抓取框回归网络；比较 Traditional CV、VLM-guided geometric 与 VLM-guided CNN 三类方法。

第 9-10 周

鲁棒性测试与失败案例分析

比较传统 baseline 与 VLM 辅助管线；分析定位失败、角度错误、不规则物体等失败模式；生成图表与定性案例。

第 11-12 周

论文整合与最终提交

完善论文各章节、参考文献、实验图表与格式；完成 Overleaf 最终版本。


## 8. 风险管理与备用方案


### 主要风险

项目中风险较高的部分包括 VLM 微调、PEFT 扩展以及端到端抓取网络训练。当前实验显示 VLM 零样本定位已经较稳定，因此 VLM 微调不作为优先核心交付；端到端网络将从轻量 CNN baseline 开始，降低训练和工程风险。


### 备用方案一

如果 VLM-guided CNN 训练效果有限，项目将聚焦于传统 CV baseline 与零样本 VLM-guided geometric pipeline 的量化对比，并围绕失败案例分析完成论文主体。


### 备用方案二

如果 VLM 部署不稳定，项目将聚焦于可复现的传统 CV baseline，并在条件允许时加入轻量目标检测方法作为替代对照。

安全核心交付：在 Cornell 数据集上完成一个可复现的 2D 机器人抓取检测管线，包括数据处理、抓取框可视化、传统 CV baseline、评估指标和失败模式分析。


## 9. 初始文献阅读矩阵


**文献**


**主题**


**与项目关系**

Redmon & Angelova (2015)

Real-Time Grasp Detection Using CNNs

2D 抓取检测的经典 CNN baseline。

Lenz, Lee & Saxena (2015)

Deep Learning for Detecting Robotic Grasps

机器人抓取预测领域的基础深度学习工作。

Liu et al. (2023)

Grounding DINO

开放词汇视觉 grounding 模块的重要参考。

Radford et al. (2021)

CLIP

视觉语言预训练和零样本识别的基础背景。

Hu et al. (2021)

LoRA

可选轻量微调/参数高效适应的理论参考。

Dettmers et al. (2023)

QLoRA

若尝试低成本微调，可作为扩展参考。


## 10. 预期贡献

本项目的预期贡献不是构建完整机器人操作系统，而是评估现代开放词汇视觉语言模型能否支持机器人抓取检测中的视觉感知环节。

构建可复现的 Cornell 数据集处理与评估管线。

实现传统计算机视觉 2D 抓取矩形预测 baseline。

实现 VLM-guided 机器人抓取检测管线，包括几何后端和学习式 CNN 后端。

使用标准指标进行量化对比。

分析 VLM-based grounding 以及不同抓取框生成后端在机器人抓取感知中有效或失败的条件。

如果时间和资源允许，探索 GG-CNN style dense prediction、复杂数据集泛化测试或 VLM 适应性方法作为扩展实验。


## 11. 实验后计划调整：VLM-guided 抓取感知与抓取框后端改进

根据当前阶段实验结果，项目计划需要进行一次合理调整。Grounding DINO 在 Cornell 数据集上使用通用 prompt “small object” 已经实现 885/885 的目标定位成功率，说明 VLM 负责的 object localization 阶段已经足够稳定。

然而，VLM-assisted pipeline 的最终抓取检测成功率为 649/885，即 73.33%。这说明剩余错误主要不来自目标定位，而来自 VLM box 之后的 grasp rectangle generation 阶段。也就是说，当前瓶颈已经从“能否找到物体”转移为“找到物体后如何生成更准确的抓取矩形”。

因此，项目后续不再优先进行 VLM 微调。VLM few-shot adaptation / fine-tuning 将保留为 future work，尤其适用于更复杂的多目标、开放场景或语言指令场景。当前阶段的主要改进方向是分析 VLM-guided geometric 方法的失败案例，并引入 VLM-guided CNN grasp backend 作为新的对照和改进方案。

新的实验路线为：Traditional CV baseline → VLM-guided geometric backend → VLM-guided CNN backend。前两者用于证明 VLM 定位前端对几何抓取框生成有帮助；第三者用于检验学习式 CNN 后端是否能够缓解手工几何规则在抓取角度、中心位置和尺寸估计上的限制。

学习式后端将优先从轻量 VLM-guided CNN rectangle regression 开始，即输入 VLM crop 的 RGB 或 RGB-D 图像，输出 center_x、center_y、width、height 和 angle。其中 angle 可用 sin(2θ) 和 cos(2θ) 表示，以处理平行夹爪的 180° 等价性。若时间允许，再进一步探索 GG-CNN style 方法，输出 grasp quality map、angle map 和 width map。

该调整并不偏离原计划，而是基于实验结果对研究重点进行收敛：当 VLM 零样本定位已经达到较高可靠性时，继续微调 VLM 的边际收益有限；相比之下，改进抓取矩形生成后端更能直接回应当前系统的主要误差来源。

本项目因此可表述为一个模块化 VLA 风格抓取感知研究：Vision 来自 RGB/RGB-D 图像，Language 来自目标 prompt，Action representation 则是 2D grasp rectangle。项目不训练完整 VLA 控制模型，而是聚焦于 VLM-guided perception 到 grasp action representation 的转换。
