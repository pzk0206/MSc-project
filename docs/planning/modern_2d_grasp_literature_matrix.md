# 现代二维抓取检测文献矩阵

## 比较口径

本项目采用 Cornell rectangle metric：预测矩形与任一真值矩形的 IoU 大于
`0.25`，且夹爪方向误差小于 `30°` 时视为成功。本项目现已同时报告固定目录
划分和 image-wise 五折，但输入仅为 Grounding DINO 裁剪后的 RGB，且具体
fold 成员不是来自其他论文未公开的清单。因此，下表中的百分比主要用于定位
技术发展，而不能组成严格的排行榜。

可比性标签含义：

- **直接可比**：相同数据、划分、输入和 Cornell rectangle metric；
- **有限可比**：指标相同，但输入或划分不同；
- **不可直接比较**：数据集、任务或物理执行指标不同。

本矩阵没有将任何文献标为“直接可比”，因为没有一项同时复现本项目的 RGB
VLM crop、单矩形输出和完全相同的 fold 成员。

## 统一比较矩阵

| 文献 | 任务 | 输入模态 | 数据集 | 划分方式 | 输出形式 | 评价指标 | 报告结果 | 与本项目可比性 |
|---|---|---|---|---|---|---|---|---|
| Jiang et al. (2011) | 二维抓取检测 | RGB-D | Cornell | image-wise / object-wise | 单个抓取矩形 | Cornell rectangle metric | `60.5% / 58.3%`（Lenz et al., 2015, Table III 对原方法的统一复现） | **有限可比**：指标相同，但输入、划分和定位方式不同 |
| Lenz et al. (2015) | 两阶段候选评分 | RGB-D | Cornell | image-wise / object-wise | 候选矩形评分 | Cornell rectangle metric | `73.9% / 75.6%`（Table III） | **有限可比**：指标相同，但 RGB-D 和标准划分不同 |
| Redmon & Angelova (2015) | 实时直接回归 / 多抓取检测 | RGB-D | Cornell | image-wise / object-wise 五折 | 单矩形回归或网格多矩形 | Cornell rectangle metric | Direct `84.4% / 84.9%`；MultiGrasp `88.0% / 87.1%`（Table I） | **有限可比**：同为矩形回归，但使用深度和标准五折 |
| Kumra & Kanan (2017) | 基于 ResNet 的抓取分类/回归 | RGB-D 双流 | Cornell | image-wise / object-wise 五折 | 候选抓取分类与回归 | Cornell rectangle metric | `89.21% / 88.96%`（Table I） | **有限可比**：指标相同，但模型、深度输入和划分不同 |
| Morrison et al. (2018), GG-CNN | 密集生成抓取 | 深度 | Cornell；另有物理实验 | 论文的增强训练与 image/object split | 像素级质量、角度和宽度图 | Cornell rectangle metric；物理抓取成功率 | Cornell `73% / 69%`（Table I）；动态抓取 `83%`、移动家庭物体 `88%`（Abstract） | Cornell 为**有限可比**；物理结果**不可直接比较** |
| Park et al. (2018) | 高分辨率全卷积抓取检测 | RGB-D | Cornell | 论文内部协议 | 密集多抓取预测 | Cornell rectangle metric | 最高 `96.6%`、`6–20 ms`（Abstract）；该预印本已撤回并由 REM 版本取代 | **有限可比**：同一指标但输入和密集输出不同；撤回稿数字不用于正式主表排名 |
| Kumra et al. (2020), GR-ConvNet | 生成式残差抓取网络 | RGB-D（n-channel） | Cornell、Jacquard；物理实验 | Cornell image-wise / object-wise | 像素级质量、角度和宽度图 | Cornell rectangle metric；Jacquard；物理成功率 | Cornell `97.7% / 96.6%`；Jacquard `94.6%`（Results/Table I） | Cornell 为**有限可比**；Jacquard和物理结果**不可直接比较** |
| Li et al. (2022) | Gaussian-guided 密集抓取 | RGB-D | Cornell、Jacquard | Cornell image-wise / object-wise | 像素级生成式抓取图 | Cornell rectangle metric | Cornell `99.0% / 98.3%`；Jacquard `95.9%`（Table I） | Cornell 为**有限可比**；Jacquard**不可直接比较** |
| Vuong et al. (2024) | 语言驱动抓取检测 | RGB + 自然语言 | Grasp-Anything++ | 大规模合成/开放词汇协议 | 条件扩散式语言目标抓取 | 该论文的语言驱动数据集指标 | 数据集包含超过 `1M` 图像、`3M` 物体和 `10M` 指令（Abstract/Section 3）；不报告本项目 Cornell 目录指标 | **不可直接比较**：任务、数据和评价协议均不同 |
| 本项目单头 CNN | VLM crop 单矩形回归 | RGB | Cornell | image-wise 五折，seed 42 manifest | 单个抓取矩形 | Cornell rectangle metric | pooled `71.75%`，IoU `0.4390`，角度 `17.74°` | 项目基线 |
| 本项目多头 CNN | 共享主干、中心/尺寸/方向分头回归 | RGB | Cornell | 与单头字节相同的 image-wise manifest | 单个抓取矩形 | Cornell rectangle metric | pooled `73.11%`，IoU `0.4580`，角度 `17.40°` | 与本项目单头**直接成对可比**；与 RGB-D 文献仅有限可比 |

## 对本项目最有用的技术脉络

1. **矩形表示与直接回归。** Jiang et al. 建立二维矩形表示；Redmon and
   Angelova 证明单次 CNN 前向回归可实时运行。这支撑当前“单 crop → 单矩形”
   基线，但不支撑其具体层数或通道数。
2. **对称方向表示。** GG-CNN 使用
   `sin(2θ), cos(2θ)` 表达平行夹爪的 `π` 周期性，直接支撑当前方向输出。
3. **现代方法趋向密集预测。** GG-CNN、GR-ConvNet 和 Gaussian-guided CNN
   都预测像素级质量、方向和宽度图，能表达一个场景中的多个抓取；当前全局
   单矩形回归在表达力上更弱。
4. **高数字不等于可公平比较。** 现代结果常使用深度、数据增强、标准五折或
   不同数据集。本项目的贡献应定位为共享 VLM 前端下的后端比较，而不是 SOTA
   竞争。
5. **语言条件是相关但不同的前沿。** Vuong et al. 展示语言指定目标与抓取的
   联合任务，适合作为 Grounding DINO 前端的研究背景，而不能作为 Cornell
   成功率基线。

## 建议写入论文的总结

> Reported Cornell accuracy has increased substantially as the field moved from
> candidate scoring and global regression toward RGB-D dense generative
> prediction. These figures are only contextual for our system: our RGB-only
> VLM crops, single-rectangle output, and project-generated fold membership
> differ from prior protocols.
> Accordingly, we treat the CNN as a lightweight controlled backend rather than
> a state-of-the-art grasp detector.

## 原始来源

- Jiang et al. (2011), DOI: <https://doi.org/10.1109/ICRA.2011.5980145>
- Lenz et al. (2015), DOI: <https://doi.org/10.1177/0278364914549607>
- Redmon & Angelova (2015): <https://arxiv.org/abs/1412.3128>
- Kumra & Kanan (2017): <https://arxiv.org/abs/1611.08036>
- Morrison et al. (2018): <https://arxiv.org/abs/1804.05172>
- Park et al. (2018): <https://arxiv.org/abs/1809.05828>
  （该 arXiv 条目已撤回，并注明由 <https://arxiv.org/abs/1812.07762> 取代）
- Kumra et al. (2020): <https://arxiv.org/abs/1909.04810>
- Li et al. (2022): <https://arxiv.org/abs/2205.04003>
- Vuong et al. (2024), CVPR open access:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-Driven_Grasp_Detection_CVPR_2024_paper.html>
