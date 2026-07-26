# CNN 架构设计依据

## 定位

当前网络是本研究为小规模 Cornell 实验构建的轻量回归基线，而不是新颖的抓取
架构。它用于在同一个 Grounding DINO 定位前端下，与确定性几何后端进行受控
比较。模型共 `432,454` 个可训练参数；输入是定位后的 RGB crop，输出一个二维
抓取矩形。

## 设计来源对照

| 设计项 | 当前选择 | 文献依据 | 工程依据 | 是否为本项目原创 |
|---|---|---|---|---|
| 输入 | `224×224` RGB VLM crop | Redmon and Angelova (2015) 将图像缩放到 `224×224` 并直接回归抓取；但其输入包含深度 | 固定 crop 尺寸使几何和 CNN 后端共享相同定位结果，并控制显存与训练时间 | 否；将 VLM crop 接入此基线是本项目的系统组合 |
| 卷积深度 | 四个卷积块 | 深层 CNN 抓取回归由 Lenz et al. (2015)、Redmon and Angelova (2015) 和 Kumra and Kanan (2017) 提供总体先例 | Cornell 训练集仅 600 张；四次降采样后保留 `7×7` 特征，避免采用大型骨干 | **是，自主工程选择；不是文献中的既有架构** |
| 通道数 | `32/64/128/256` | 常见 CNN 随空间分辨率降低而增加通道；没有论文使用与本项目完全相同的序列 | 在表示能力、参数量和单 GPU/CPU 可运行性之间折中 | **是，自主工程选择** |
| 卷积核与下采样 | 首层 `5×5, s=2`，其余 `3×3`；每块后最大池化 | 早期抓取 CNN 使用卷积特征提取；本项目的具体核、步幅和池化组合没有直接复现某篇论文 | 首层扩大感受野并快速降采样；其余 `3×3` 控制参数量 | **是，自主工程选择** |
| Global Average Pooling | 固定 `7×7` 平均池化，`256×7×7 → 256` | 全局池化是轻量分类/回归网络的常见压缩方式；现代密集抓取网络则通常保留空间特征 | 固定输入下与自适应全局平均池化数学等价，同时支持严格确定性 CUDA 反向传播 | **是，自主工程选择** |
| 全连接回归头 | `256→128→64→6`，Dropout `0.3/0.2` | Redmon and Angelova (2015) 证明单次前向直接回归抓取参数可行 | 小型 MLP 足以连接全局特征与六维输出；Dropout 限制小数据集过拟合 | **宽度和 Dropout 为自主工程选择** |
| 输出形式 | 单个抓取矩形 `(cx, cy, w, h, θ)` | Jiang et al. (2011) 提出二维抓取矩形表示；Lenz et al. (2015) 和 Redmon and Angelova (2015) 延续该任务形式 | 与 Cornell 标注和现有几何后端输出一致，便于同样本比较 | 否 |
| 方向编码 | `sin(2θ), cos(2θ)` | GG-CNN（Morrison et al., 2018）以双角正弦/余弦处理平行夹爪的 `π` 周期对称性 | 避免 `θ` 与 `θ+π` 等价时的角度边界跳变 | 否；应用到本回归头 |
| 损失 | 六维输出上的 Smooth L1 | 直接回归沿用 Redmon and Angelova (2015) 的问题设定；Smooth L1 是稳健回归损失 | 相比 MSE 降低少量异常标注或困难样本对梯度的影响 | **具体损失组合为自主工程选择** |
| 优化 | Adam，学习率 `1e-3`，weight decay `1e-4` | Adam 和权重衰减是标准优化方法，并非抓取领域贡献 | 快速获得稳定基线；权重衰减抑制小样本过拟合 | 否；超参数为自主工程选择 |
| 训练停止 | 最大 80 epoch，验证损失 early stopping，patience 20 | 标准模型选择流程 | 避免在 600 张训练图像上继续拟合；只用验证集选择 checkpoint | 否；阈值为自主工程选择 |
| 数据划分 | 目录 `01–06 / 07–08 / 09–10` 对应 train/val/test | Cornell 文献常报告 image-wise 或 object-wise 五折结果，但本项目未复现这两个标准协议 | 目录隔离用于构建固定的未见目录测试，但目录难度并未被证明相等 | **本项目实验协议；不可称为标准 Cornell object-wise split** |
| 成功指标 | 最大矩形 IoU `>0.25` 且角度误差 `<30°` | Jiang et al. (2011) 的 rectangle metric；后续 Cornell 工作沿用 | 与已有二维抓取检测结果保持有限可比性 | 否 |

## 结构与张量尺寸

| 阶段 | 操作 | 输出尺寸 |
|---|---|---|
| 输入 | RGB crop | `3×224×224` |
| Block 1 | Conv `5×5, s=2, 32` + BN + ReLU + MaxPool | `32×56×56` |
| Block 2 | Conv `3×3, 64` + BN + ReLU + MaxPool | `64×28×28` |
| Block 3 | Conv `3×3, 128` + BN + ReLU + MaxPool | `128×14×14` |
| Block 4 | Conv `3×3, 256` + BN + ReLU + MaxPool | `256×7×7` |
| 汇聚 | 固定 `7×7` Global Average Pooling | `256` |
| 回归头 | FC 128 + FC 64 | `64` |
| 输出 | Linear | `[cx, cy, w, h, sin(2θ), cos(2θ)]` |

## 论文中的安全表述

可写成：

> We use a 432k-parameter convolutional regressor as a controlled lightweight
> baseline. The rectangle output follows established Cornell grasp
> representations, while the double-angle orientation encoding follows the
> symmetry-aware representation used by GG-CNN. The exact four-block channel
> schedule and MLP widths are engineering choices for this study and are not
> claimed as a novel architecture.

不要写成：

- “提出了一种新 CNN 架构”；
- “在 Cornell 上达到 82.35% 的标准 object-wise 性能”；
- “测试集更高说明网络具有更强的未见物体泛化能力”。

后两项分别忽略了非标准目录划分和集合难度未确认的问题。

## 主要局限

1. RGB crop 不包含深度，而多数高性能 Cornell 方法使用 RGB-D 或深度。
2. 全局池化后直接回归一个矩形，无法表达多个可行抓取。
3. 单一回归头没有显式分离中心、尺寸和方向目标。
4. 目录划分不是标准 image-wise/object-wise 五折协议。
5. 模型依据是可解释的工程组合，但其有效性仍需消融或多头对照实验验证。
