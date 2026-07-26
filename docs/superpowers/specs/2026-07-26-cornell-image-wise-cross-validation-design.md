# Cornell Image-wise 五折交叉验证设计

日期：2026-07-26

## 1. 背景与决策

Lenz、Lee 和 Saxena 对 Cornell 抓取检测实验采用五折交叉验证，并区分：

- image-wise：按图像划分；
- object-wise：同一物体的全部图像必须留在同一 fold。

当前本地 Cornell 数据包包含 885 个样本，但没有提供 885 张图像到约 240 个
物体实例的映射。原始论文说明了 object-wise 协议，却没有公开当前数据包可直接
使用的实例清单。相邻图像虽然呈现连续物体视角，但自动视觉分组会把大幅旋转
误判为物体边界，不能作为可靠元数据。

因此本阶段作出以下决定：

1. 实现并运行可严格复现的 image-wise 五折；
2. object-wise 保持阻塞，不使用目录编号或自动聚类结果伪造物体 ID；
3. 如果以后获得可信物体实例映射，则复用同一套 fold 和训练接口补充
   object-wise 实验；
4. 论文必须说明当前 image-wise 结果不衡量未见物体泛化。

协议依据：

- Ian Lenz, Honglak Lee, and Ashutosh Saxena, *Deep Learning for Detecting
  Robotic Grasps*, IJRR 2015.
- 原文：
  <https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf>

本设计只依据论文定义协议，不复制论文或第三方仓库的实现代码。

## 2. 目标

为当前 VLM 引导的轻量 CNN 后端建立一套可审计的 image-wise 五折实验，使：

- 885 个样本各自恰好作为一次外层测试样本；
- 每折的训练、验证和测试集合互不重叠；
- 测试 fold 不参与早停、模型选择或超参数选择；
- 单头和多头 CNN 使用完全相同的 fold 清单；
- 每折保存独立模型、历史、预测和汇总；
- 最终报告五个测试 fold 合并后的逐样本指标，以及五折指标的均值和标准差；
- 五折结果和五随机种子重复实验保持分开，不混写统计含义。

## 3. 不在本阶段完成的内容

- 不把 Cornell 的 `01`–`10` 目录当作物体实例 ID；
- 不发布由视觉相似度自动推断的 object-wise 结果；
- 不增加 RGB-D、Jacquard、预训练 CNN 骨干或大规模超参数搜索；
- 不在 image-wise 五折中重新运行 Grounding DINO；继续复用已经保存并审计的
  885 个定位框；
- 不复制外部交叉验证实现。

## 4. 划分协议

### 4.1 外层五折

按稳定排序后的 885 个 `sample_id` 建立索引，使用固定随机种子 `42` 打乱一次，
然后尽可能均匀分成五个测试 fold。五个测试 fold 的大小应为：

```text
177, 177, 177, 177, 177
```

每个样本恰好出现在一个外层测试 fold 中。fold 清单写入 CSV 和 JSON，并记录：

- 协议名；
- 随机种子；
- fold 编号；
- `sample_id`；
- 原始 Cornell 子目录；
- 当前角色：train、validation 或 test。

### 4.2 内层验证集

对每个外层 fold 剩余的 708 个样本进行独立、确定性的训练/验证划分。使用由
主种子和 fold 编号派生的固定种子，在不接触外层测试样本的前提下划出约 20%
作为验证集：

```text
train = 566
validation = 142
test = 177
```

验证集只用于验证损失、学习率调度、早停和选择最佳 checkpoint。最终指标只在
外层测试 fold 上计算。

image-wise 协议允许同一物体的不同视角分散在训练、验证和测试中；这正是它与
object-wise 协议的区别，不能据此声称未见物体泛化。

### 4.3 随机性

- fold 清单生成种子固定为 `42`；
- 每个 fold 的模型训练种子固定为 `42`；
- 单头和多头使用相同 fold、相同训练种子和相同确定性 CUDA 设置；
- fold 变化和随机种子变化是不同因素，本实验不把五折称为“五次随机种子实验”。

固定相同训练种子有助于减少架构比较中的额外随机差异。五折间的指标变化主要
反映测试图像集合变化，但仍不能被解释为独立总体抽样。

## 5. 代码结构

### 5.1 共享 fold 模块

新增 `src/shared/cornell_cross_validation.py`，只负责：

- 从稳定的样本 ID 列表生成 image-wise 五折；
- 为每个外层 fold 生成 train/validation/test 角色；
- 校验覆盖、互斥、大小和确定性；
- 保存、加载并校验 fold manifest。

该模块不导入 PyTorch，也不训练模型，使划分逻辑可以独立测试。

### 5.2 CNN 五折入口

新增 `src/vlm/run_cnn_cross_validation.py`，负责：

- 解析 `--architecture single|multi_head`；
- 读取或生成同一个 fold manifest；
- 复用 `run_cnn_grasp.py` 中的数据准备、训练、评估和保存逻辑；
- 对五个 fold 逐一训练和测试；
- 保存每折产物和最终汇总。

如果现有训练函数无法安全接收显式索引，先进行最小范围重构，使数据索引成为
函数参数；保持现有固定目录实验命令和输出格式兼容。

### 5.3 输出目录

```text
data/processed/vlm/cnn_cross_validation/
├── image_wise_folds_seed_42.csv
├── image_wise_folds_seed_42.json
├── single_head/
│   ├── fold_0/
│   │   ├── model.pt
│   │   ├── training_history.json
│   │   ├── predictions.csv
│   │   └── summary.json
│   ├── fold_1/ ... fold_4/
│   ├── combined_predictions.csv
│   └── cross_validation_summary.json
└── multi_head/
    └── 与 single_head 相同
```

所有生成实验产物继续由 Git 忽略。源码、测试、设计、结果溯源和论文文字进入
版本控制。

## 6. 汇总口径

每个架构输出两类统计：

1. **合并测试预测**：连接五个外层测试 fold 的 885 行预测，每个样本一行；
   由此计算总体成功率、平均最佳 IoU 和平均角度误差。
2. **五折统计**：计算五个 fold 的成功率、IoU 和角度误差的均值与总体标准差，
   同时保留每折样本数。

单头与多头比较必须同时报告：

- 合并 885 样本的指标差；
- 每折成对指标差；
- 五折均值和标准差；
- 不通过只挑选表现最好的 fold 得出结论。

由于本项目使用 Grounding DINO RGB crop、单矩形回归和轻量自定义 CNN，即使
采用相同 image-wise 划分与 Cornell rectangle metric，也只能与 RGB-D、深度或
密集多抓取论文作有限比较。

## 7. 错误处理与审计

在任何训练开始前，manifest 校验必须确认：

- 样本总数为 885；
- sample ID 唯一；
- 正好有五个 fold；
- 每个 fold 中 train、validation、test 两两不相交；
- 每折三个集合的并集等于全部 885 个样本；
- 五个 test 集两两不相交且并集等于全部样本；
- 单头与多头读取的 manifest 文件哈希一致。

在汇总前还必须确认：

- 每折预测行数等于 177；
- 五折合并后恰好 885 行且 sample ID 唯一；
- 所有指标有限；
- 每折记录的最佳验证损失与训练历史一致；
- 架构名、fold 编号、种子和 manifest 哈希均写入汇总。

任何检查失败时立即停止，不生成可用于论文的最终汇总。

## 8. 测试策略

按照测试驱动开发实施：

1. 先写 fold 生成测试并确认因功能不存在而失败；
2. 实现最小 fold 生成逻辑并通过；
3. 先写泄漏、覆盖、错误 manifest 测试并确认失败；
4. 实现校验和持久化；
5. 先写训练入口的显式索引和产物路径测试并确认失败；
6. 实现训练编排；
7. 使用小型合成数据或少量真实样本进行 CPU/GPU 冒烟测试；
8. 正式训练前运行全部测试、Python 编译检查和 manifest 审计；
9. 正式结果生成后再次运行产物审计。

测试不通过 monkeypatch 隐藏文件系统或数据选择副作用；fold 核心逻辑使用真实
列表和临时目录验证。

## 9. 外部代码和出处要求

- 实现前检查是否确实需要参考外部源代码；
- 如果复制或实质性改编任何代码，必须在相关文件附近写明原作者、项目或论文、
  许可证和可访问链接；
- 同时在项目结果溯源文档中记录改编范围；
- 普通调用 Python、NumPy、PyTorch 或 OpenCV 公共 API 不视为复制外部实现；
- 如果只依据论文中的协议定义自行实现，则引用论文并明确说明实现为本项目
  自行编写。

## 10. 完成标准

- fold manifest 通过全部覆盖和无泄漏测试；
- 相同种子重复生成的 manifest 字节一致；
- 单头和多头完成五折，并各有五套独立产物；
- 每个架构的合并预测为 885 个唯一样本；
- 正式汇总通过自动审计；
- Results 和 Discussion 准确区分固定目录实验、五随机种子实验和 image-wise
  五折实验；
- object-wise 未完成原因和结论限制被明确记录；
- 所有实际引用或改编的外部代码具有符合 `AGENTS.md` 的出处标注。
