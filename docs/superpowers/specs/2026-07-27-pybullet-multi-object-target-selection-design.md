# PyBullet 多物体目标选择研究设计

日期：2026-07-27

## 研究目的

在一个包含 Franka Panda 和三个候选物体的固定 PyBullet 场景中，验证
Grounding DINO 能否根据明确的自然语言 prompt 选择指定目标，并把正确定位
结果接入现有二维几何抓取后端。

该研究补充 Cornell 单物体图像实验，回答：

> 当机器人部件和多个候选物体同时出现在虚拟相机画面中时，语言条件定位能否
> 选择指定的抓取目标？

它是小型可行性与失败模式研究，不是新的 Cornell 性能实验，不评价物理抓取
成功率，也不替代 Cornell 抓取框 IoU 与角度指标。

## 已比较的方案

### 方案 A：一个固定三物体场景

- 一次渲染固定场景；
- 对三个目标分别使用明确 prompt；
- 用 segmentation 真值评价目标选择；
- 正确定位后生成二维几何抓取框。

优点是范围小、变量少、结果容易解释。缺点是不能支持随机布局泛化结论。

### 方案 B：五个手工布局

在方案 A 基础上改变五次物体位置。证据更充分，但增加模型运行、失败诊断和
论文分析工作，并开始接近独立定量实验。

### 方案 C：大量随机场景

随机化物体、位置、相机和光照，形成仿真 benchmark。它可以研究统计泛化，
但需要遮挡控制、有效场景筛选、批量恢复和更完整的实验设计，超出当前论文
剩余时间。

### 选择

本阶段采用方案 A。方案 B 只有在固定场景完整通过且论文时间充足时才重新
设计；方案 C 作为未来工作。

## 范围

### 本次实现

- 保留现有单物体 `run_pilot.py` 的行为和输出兼容性。
- 新增一个固定三物体场景配置。
- 使用 PyBullet 自带的鸭、方块和球体资源。
- 为三个物体设置固定位置和明显不同的颜色。
- 固定相机、renderer、模型、阈值和随机种子。
- 场景只渲染一次，Grounding DINO 只加载一次。
- 对三个明确 prompt 分别运行定位。
- 使用 segmentation 为三个目标和 Panda body ID 生成评价真值 mask 与水平
  真值框；Panda 只作为 distractor，不是主目标。
- 计算目标框 IoU、最佳匹配物体和目标选择是否正确。
- 对正确定位结果运行现有几何抓取后端。
- 检查二维抓取中心是否位于指定目标 mask 内。
- 额外运行一次 `small object`，只作为 prompt 歧义诊断，不计入三目标主结果。
- 保存逐目标图像、CSV、JSON、场景配置和失败原因。

### 本次不实现

- 不随机化物体位置、相机、纹理或光照。
- 不训练或微调 Grounding DINO。
- 不把 segmentation、body ID 或真值框输入定位或抓取算法。
- 不将真值框替代 Grounding DINO 框。
- 不重新训练 CNN，不把 CNN 仿真质量作为本研究主结果。
- 不做深度反投影、相机到世界坐标转换、逆运动学或夹爪执行。
- 不把三个案例解释为多物体泛化率或真实机器人性能。

## 固定场景

保留平面、桌面和 Franka Panda。Panda 继续出现在相机画面中，作为机器人场景
中的真实干扰项，不通过裁剪移除。

三个目标物体为：

| target name | URDF | RGBA | 初始位置 | prompt |
|---|---|---|---|---|
| `duck` | `duck_vhacd.urdf` | `(1.0, 0.8, 0.0, 1.0)` | `(0.52, -0.18, 0.67)` | `yellow rubber duck` |
| `cube` | `cube_small.urdf` | `(0.9, 0.1, 0.1, 1.0)` | `(0.48, 0.00, 0.66)` | `red cube` |
| `sphere` | `sphere_small.urdf` | `(0.1, 0.8, 0.1, 1.0)` | `(0.52, 0.18, 0.67)` | `green sphere` |

每个物体使用固定 yaw；方块可使用非零 yaw，球体 yaw 不影响外观。加载后运行
固定 60 个仿真 step 使物体稳定，再采集一帧。若物体相互碰撞、离开桌面或在
segmentation 中不可见，场景验证失败，不能进入模型推理。

颜色通过 PyBullet 官方 `changeVisualShape` API 设置。颜色是场景实验变量，
必须写入 metadata，不能只存在于代码常量。

## 架构与文件

### 修改 `scene.py`

增加不可变的 `SceneObjectConfig`：

```python
@dataclass(frozen=True)
class SceneObjectConfig:
    name: str
    urdf: str
    position: tuple[float, float, float]
    yaw_degrees: float
    rgba: tuple[float, float, float, float]
```

`SceneConfig` 增加可选的 `additional_objects`，默认空 tuple。现有
`object_urdf`、`object_position`、`object_yaw_degrees` 和
`SceneBodies.target_object` 保留，保证单物体 pilot 和已有测试继续工作。

`PyBulletScene.object_body_ids` 返回包含默认目标和附加目标的
`dict[str, int]`。多物体研究为默认鸭指定名称 `duck`，附加 `cube` 和
`sphere`。评价实体另外包含已有 `SceneBodies.robot`，名称固定为 `robot`。
所有资源仍必须解析在 `pybullet_data` 内，所有 PyBullet 调用显式传入
client ID。

### 新增 `target_selection.py`

只包含与模型无关的纯评价函数：

```python
segmentation_mask_for_body(segmentation, body_id) -> np.ndarray
mask_to_box(mask) -> tuple[int, int, int, int]
box_iou(box_a, box_b) -> float
evaluate_target_selection(
    predicted_box,
    requested_target,
    ground_truth_boxes,
    iou_threshold=0.25,
) -> TargetSelectionEvaluation
grasp_center_inside_mask(grasp, target_mask) -> bool
```

`TargetSelectionEvaluation` 至少包含：

- `requested_target`；
- `requested_target_iou`；
- `best_matching_target`；
- `best_iou`；
- `correct_target`；
- `iou_threshold`。

### 新增 `run_multi_object_study.py`

负责固定实验编排，不复制单图模型实现：

```text
连接固定场景
→ 稳定 60 step
→ 采集并保存一帧
→ 从 segmentation 生成三个目标和 robot distractor 的评价真值
→ 加载一次 Grounding DINO
→ 依次运行三个明确 prompt
→ 评价目标选择
→ 正确时调用现有 geometry 后端
→ 检查抓取中心
→ 运行一次 generic prompt 诊断
→ 保存逐目标与汇总产物
→ finally 关闭连接
```

Grounding DINO 必须复用
`src.simulation.pybullet.perception.localize_object`，抓取必须复用
`predict_grasp`。不得将三个目标拆成三个独立场景运行，也不得重复加载模型。

## 数据与评价

### segmentation 解码

PyBullet segmentation 整数的低 24 位保存 body unique ID。背景为 `-1`。
评价函数使用：

```python
decoded_body_id = segmentation & ((1 << 24) - 1)
```

但只在原值非负的位置比较 body ID，避免把背景解码为伪目标。

### 真值框

对指定 body 的所有可见像素取：

```text
x1 = min(x)
y1 = min(y)
x2 = max(x)
y2 = max(y)
```

真值框使用与预测框相同的左上/右下图像坐标约定。三个主目标 mask 中任一个
为空时停止实验并记录 `target_not_visible`，不能把该目标计为模型未检测。
Panda mask 为空时同样停止，因为该固定研究明确要求机器人部件作为干扰项。
桌面、平面和背景不进入最佳匹配计算，避免大面积背景框主导评价。

### 目标选择判定

对每个明确 prompt：

```text
correct_target =
    requested_target_iou >= 0.25
    AND best_matching_target == requested_target
```

`0.25` 是本 pilot 的工程门控阈值，不是 Cornell 抓取矩形指标，也不引用为
标准目标检测阈值。论文同时报告原始 IoU，避免只呈现二值结果。

`best_matching_target` 在 `duck`、`cube`、`sphere` 和 `robot` 四个评价实体
中选择，因此再次框住 Panda 时能够明确记录为 `robot`，而不是含糊地归为三个
物体 IoU 都很低。

若多个真值框得到完全相同的最高 IoU，结果为 `ambiguous_match`，不算正确。
若 Grounding DINO 无检测，记录 `no_detection`。模型失败不阻止保存其他目标
结果。

### 抓取框辅助检查

只有 `correct_target=True` 时才运行几何后端。检查：

- 抓取参数全部有限；
- width 和 height 为正；
- 抓取中心位于图像范围；
- 四舍五入后的中心像素属于指定目标 mask。

`grasp_center_inside_mask` 只是二维几何合理性代理，不是抓取成功。抓取框质量
的主要定量证据仍是 Cornell 标注上的 IoU 与角度误差。

### generic prompt 诊断

`small object` 在同一 RGB 上运行一次，记录它最佳匹配的对象。该行设置：

```text
result_role = diagnostic
```

三条明确 prompt 设置：

```text
result_role = main
```

汇总的 `main_target_count`、`correct_target_count` 和选择率只使用 main 行，
不把 generic 行混入分母。

## 输出

默认输出目录：

```text
data/processed/pybullet/multi_object_study/
├── rgb.png
├── depth.npy
├── depth_visualization.png
├── segmentation.png
├── ground_truth_boxes.png
├── results.csv
├── summary.json
├── metadata.json
└── targets/
    ├── duck_localization.png
    ├── duck_prediction.png
    ├── cube_localization.png
    ├── cube_prediction.png
    ├── sphere_localization.png
    ├── sphere_prediction.png
    └── generic_small_object_localization.png
```

若某目标无检测或目标选择错误，不生成对应 prediction 图；旧的固定输出必须在
本次运行前清除，避免历史图片冒充新结果。不得删除输出目录内名称不属于本实验
清单的其他文件。

`results.csv` 每行至少记录：

- `result_role`；
- `requested_target`；
- `prompt`；
- `detected`；
- 预测框和 score；
- 三个真值目标和 robot distractor 的 IoU；
- `best_matching_target`；
- `correct_target`；
- `failure_reason`；
- 抓取中心、尺寸、角度；
- `grasp_center_inside_target`。

`summary.json` 至少记录：

- `protocol: fixed_three_object_prompt_selection_pilot`；
- 三个 main 目标数量；
- 正确目标数量；
- main 选择率；
- main 平均 requested-target IoU；
- generic prompt 的最佳匹配结果；
- 不包含物理抓取成功率。

`metadata.json` 记录 PyBullet、模型、设备、renderer、seed、三个 URDF、颜色、
初始和稳定后姿态、相机矩阵、阈值、所有输出路径，以及：

```json
{
  "segmentation_used_as_model_input": false,
  "physical_grasp_executed": false
}
```

## 可视化

- `ground_truth_boxes.png`：三种不同颜色的 segmentation 真值框和 target name；
- 每个 localization 图：预测框、requested target、prompt、score；
- 每个 prediction 图：定位框、几何抓取框、中心、方向和
  `grasp_center_inside_target`；
- 错误目标选择图同时标出请求目标真值框和实际最佳匹配目标，避免只看一个预测框
  无法判断错误。

可视化只用于审计和论文案例图，数值结果以 CSV/JSON 为准。

## 错误处理

- 任一 URDF 不存在或越出 `pybullet_data`：场景阶段失败；
- target name 重复：配置阶段失败；
- RGBA 非法或物体不可见：相机/场景验证失败；
- CUDA 请求不可用：直接失败，不回退 CPU；
- Grounding DINO 无检测：记录该 target 失败并继续其他 prompt；
- 预测框非法：记录 `invalid_detection_box`；
- 几何后端失败：保留定位评价并记录 `grasp_backend_failed`；
- 输出文件写入失败：实验返回非零退出码；
- 所有失败 metadata 均保持 `physical_grasp_executed: false`。

## 测试

### 纯单元测试

- body ID 解码不把背景当作物体；
- 已知 mask 正确转换为 inclusive 真值框；
- 已知重叠框的 IoU 使用手算值；
- 请求目标 IoU 达标且为唯一最大值时选择正确；
- IoU 低于阈值、最佳对象错误和最高 IoU 并列分别失败；
- 抓取中心在 mask 内、外和越界时结果正确；
- main 与 diagnostic 行在汇总时分开。

### DIRECT 集成测试

- 三个物体和 Panda 均成功加载；
- 三个 target name 和 body ID 唯一；
- 稳定后每个目标在 segmentation 中具有非空 mask；
- 三个目标和 Panda 真值框位于 640×480 图像内；三个目标框互不完全覆盖；
- 关闭后无残留连接。

### Runner 测试

使用假的模型边界、真实文件落盘，验证：

- 场景和模型各初始化一次；
- 三个 main prompt 和一个 diagnostic prompt 顺序稳定；
- 错误目标不调用抓取后端；
- 单目标失败不阻止其他结果；
- CSV、summary、metadata 和预期图像完整；
- segmentation 不出现在 detector 或 grasp 函数参数中。

真实 Grounding DINO 运行不进入普通快速测试；使用沙箱外 CUDA 显式执行并
人工检查所有图像。

## 论文报告

论文最多使用：

- 一张固定三物体 RGB 场景图；
- 一张明确 prompt 正确选择目标的结果图；
- 一张 `small object` 错误选择 Panda 或其他对象的失败图；
- 一张包含三个 main prompt 的定位 IoU 与选择结果表。

Methodology 说明 segmentation 只用于仿真评价。Findings 报告保存的原始结果，
Discussion 分析语言明确性和从 Cornell 单物体图像到机器人场景的域差异。
不得使用“多物体泛化”“机器人抓取成功”或“RGB-D 模型”等表述。

## 进入机械臂抓取阶段的门控

只有满足以下条件后，才单独设计深度反投影与 Panda 抓取：

1. 三个明确 prompt 均完成运行并保存结果；
2. 三个目标均为 `correct_target=True`；
3. 三个几何抓取输出均有限且中心位于各自 target mask；
4. 所有普通测试通过；
5. 人工检查 RGB、真值框、定位框和抓取框没有颜色或坐标错误；
6. 论文记录明确这是二维感知门控。

如果模型没有达到 3/3，不修改或隐藏结果，也不通过 segmentation 替代模型。
先记录失败并决定是调整 prompt、场景可见性，还是只把机械控制作为独立 oracle
验证；不得直接声称完整视觉抓取管线可用。

## 依赖与代码来源

- 仿真仅使用 Erwin Coumans、Yunfei Bai 等维护的 Bullet Physics /
  PyBullet 官方 API 和 `pybullet_data`：
  https://github.com/bulletphysics/bullet3
- Grounding DINO 继续复用项目现有 Hugging Face/Transformers 接口和已有
  定位模块，不复制模型实现。
- 几何抓取继续复用项目现有 `predict_grasp_with_vlm_box`。
- 如果实现参考、复制或改编外部示例，源文件必须写明原作者、项目或论文名称、
  可访问链接以及本项目的改编范围；不得表述为本项目原创。
