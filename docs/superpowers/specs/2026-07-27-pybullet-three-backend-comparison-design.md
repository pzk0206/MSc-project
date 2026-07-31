# PyBullet 三后端抓取框并列检查设计

## 目标

在已经通过目标选择门控的固定鸭、方块、球体场景中，让几何、正式单头 CNN
seed 42 和正式多头 CNN seed 42 接收完全相同的 Grounding DINO 定位框，
生成可审计的二维抓取矩形和并列图。该研究只检查后端接口与输出几何，不在
缺少仿真抓取真值时构造成功率或性能排名。

## 冻结输入

- 场景、相机、seed、三条 main prompt 和一条 generic prompt 保持现状。
- Grounding DINO 每次研究只加载一次，每条 prompt 只定位一次。
- 目标选择仍比较鸭、方块、球体和 Panda 的 segmentation 真值框。
- 只有 `correct_target=true` 的 main prompt 才进入抓取后端。
- generic `small object` 只保留目标选择诊断，不进入抓取后端。
- segmentation、body ID 和真值框不得输入 Grounding DINO 或任何抓取后端。

## 固定后端

按稳定顺序运行：

1. `geometry`，不加载权重；
2. `single`，固定使用
   `data/processed/vlm/cnn_grasp_single_head_deterministic/cnn_grasp_model_seed_42.pt`；
3. `multi_head`，固定使用
   `data/processed/vlm/cnn_grasp_multi_head_deterministic/cnn_grasp_model_seed_42.pt`。

两个 CNN 模型各加载一次，并复用于三个正确目标。权重缺失、损坏或架构不匹配
均视为基础设施失败；不得自动选择其他 seed、其他目录或 geometry 回退。

## 数据流

```text
固定场景 → 一次 RGB/depth/segmentation
                 │
                 ├─ segmentation → 事后目标真值与 mask
                 │
                 └─ RGB → 一次 Grounding DINO 模型加载
                              │
                              └─ 每条 prompt 一个 Localization
                                      │
                                      ├─ geometry
                                      ├─ single seed 42
                                      └─ multi_head seed 42
```

三个后端必须接收同一幅 BGR 图像和同一个 `Localization` 对象；不得重新定位、
重新裁剪真值区域或按后端修改检测框。

## 输出

保留现有场景、目标选择和 metadata 产物。每个正确 main 目标增加：

```text
targets/<target>/
├── evaluation.png
├── prediction.png
├── geometry_prediction.png
├── single_prediction.png
├── multi_head_prediction.png
└── backend_comparison.png
```

其中 `prediction.png` 与 `geometry_prediction.png` 保存同一 geometry 图像；
前者用于兼容现有固定输出，后者让并列后端命名明确。

CSV 使用一条目标选择行加嵌套 JSON 会降低审计性，因此新增独立
`backend_results.csv`，每个正确目标与后端一行，稳定顺序为
duck/cube/sphere × geometry/single/multi_head。字段包括：

- target、prompt、backend、weights_path；
- 共用 detection box 和 score；
- center、width、height、angle；
- 参数是否全部有限；
- 中心是否位于目标 mask；
- 旋转抓取矩形四个顶点是否全部位于图像内；
- 后端 failure reason。

新增 `backend_comparison.json`，汇总行数、每后端有限输出数、中心落入 mask
数量、框在图像内数量和权重路径。它是诊断汇总，不输出“最佳后端”。

metadata 保持：

```json
{
  "segmentation_used_as_model_input": false,
  "physical_grasp_executed": false
}
```

## 可视化

三张单后端图继续使用：

- 黄色框：Grounding DINO 定位框；
- 蓝色旋转框：二维抓取矩形；
- 绿点：抓取中心；
- 绿色中心轴：图像平面内的抓取矩形角度，不表示机械臂下降方向。

并列图按 geometry、single、multi_head 从左到右排列，并明确标注后端名称。
不改变模型数值，只改善人工比较。

## 错误处理

- 场景、相机、真值实体、检测模型或 CNN 权重失败：写入 failure metadata，
  CLI 返回非零。
- 单条 main prompt 无检测或目标选择错误：保留目标选择结果，不运行任何后端。
- 后端返回非有限或无效抓取参数：记录该后端失败，不把它改写为其他后端结果；
  其余目标与后端继续运行。
- 已知生成文件运行前按固定路径清理，避免旧 prediction 被误认成本次结果。

## 测试与验收

测试驱动覆盖：

- 三个后端对正确目标均被调用，错误目标和 generic 不调用；
- single 和 multi_head 模型各只加载一次；
- 三个后端收到相同的 RGB 与 Localization；
- 后端行和输出路径顺序稳定；
- 有限性、中心 mask 和旋转框图像边界检查；
- 缺失或不匹配权重写出明确 failure metadata；
- 并列图不修改输入且包含三个标注面板；
- 原单物体 runner 和现有目标选择输出继续通过回归。

真实 GPU 验收使用冻结场景与正式 seed 42 权重，逐张检查三目标共九张预测图
和三张并列图。只记录可见事实以及几何检查；不使用 Cornell 指标，也不声称
二维框可以直接驱动机械臂或代表物理抓取成功。
