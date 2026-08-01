# PyBullet 几何感知执行计划预检设计

## 目标

在已经通过的真值方块阶段 1--5 与几何感知后端之间增加一个可独立验收的
Stage 6A。程序必须在同一个固定 PyBullet 场景中实时完成：相机采集、
Grounding DINO `red cube` 定位、geometry 抓取框预测、中心深度反投影、
两个 `0°/180°` 对称俯视姿态生成以及静态 IK/FK/碰撞预检。通过后输出唯一的
冻结执行计划，供下一阶段物理执行消费；本阶段不发送任何电机或夹爪命令。

## 为什么先做预检桥接

直接把感知代码塞进阶段 5 状态机会同时改变感知、姿态生成和物理执行，失败时
难以区分是定位、几何后端、深度、IK、碰撞还是控制问题。读取 7 月 31 日保存的
九点结果虽然简单，但感知和执行不处于同一次场景实例，不能形成最强的端到端
证据。Stage 6A 因此采用同一场景内的实时感知和静态预检，并把物理执行留给
下一小步。

## 冻结边界

- 场景固定使用现有 `fixed_scene_config`、seed 42、640×480 相机和 cube。
- prompt 固定为 `red cube`；模型固定为
  `IDEA-Research/grounding-dino-tiny`；box/text threshold 均为 `0.25`。
- 抓取后端只允许 `geometry`，不加载单头或多头权重，不允许失败后回退真值。
- 深度只在二维中心产生后使用；segmentation 和 `rayTest` 不得参与定位、
  抓取框、深度或三维点生成。
- 姿态偏移冻结为接触前 `0.02 m`、抓取深度 `0.005 m`、pregrasp 额外
  `0.10 m`；物理阶段的闭合、抬升和保持参数本阶段不修改。
- 不根据 cube 真值中心移动、修正或重排预测中心和角度。
- 本阶段所有 metadata 中电机、闭合、接触、抬升和物理抓取标志必须为假。

## 架构与组件

### `execution_plan.py`

定义独立、可序列化的感知执行计划类型，避免让下一阶段依赖 runner 内部字典：

- `PerceptionEvidence`：prompt、定位框/分数、geometry 中心/尺寸/角度、采样
  像素、深度、世界表面点以及事后真值审计结果。
- `PlannedPoseCandidate`：对称角、pregrasp/approach/grasp-depth 三个位姿、
  七关节 IK 解、IK/FK 门控、41 状态静态碰撞结果、代价和选择标志。
- `GeometryExecutionPlan`：协议版本、场景 seed、相机参数、RGB SHA-256、
  后端名、两个候选、唯一选中候选和冻结控制协议摘要。

计划加载时重新校验版本、backend、候选数量、唯一选择、数值有限性、三个高度
关系和全部门控，防止手工编辑或过期文件被下一阶段静默执行。

### `run_geometry_execution_preflight.py`

runner 负责同一场景编排和证据落盘：

1. 连接固定场景并稳定 60 步，采集一次 RGB、米制深度、segmentation 和矩阵。
2. 保存 RGB 后加载一次 Grounding DINO，仅运行 `red cube`。
3. 在定位框内调用现有 geometry 后端；禁止加载 CNN 或使用真值回退。
4. 对预测中心执行现有最近像素半向上采样和深度反投影。
5. 以预测角度生成两个对称世界 `-Z` 候选，并为每个候选另外生成
   `0.005 m` 抓取深度姿态。
6. 对两个候选分别执行现有 IK/FK、关节限位和中立位→pregrasp→抓取深度
   41 状态碰撞审计。静态安全审计允许 Panda 底座—桌面安装接触，但不豁免
   cube、duck 或 sphere。
7. 使用既有确定性候选选择规则：只在通过全部预检的候选中按关节路程代价
   选择，代价相同优先 `0°`。
8. segmentation、目标框 IoU 和射线命中仅在预测产生后作为审计字段；不得
   修改预测。如果目标、坐标或安全审计失败，不写可执行计划，只保存失败证据。

依赖边界沿用现有 runner 的 dataclass 注入方式，使测试能替换模型加载与推理，
但真实正式命令必须绑定现有 Grounding DINO 和 geometry 实现。

## 数据流

```text
固定 PyBullet 场景（同一 client）
  → RGB/depth/segmentation/matrices
  → RGB + "red cube" → Grounding DINO box
  → RGB + box → geometry rectangle
  → rectangle centre + depth + matrices → world surface point
  → rectangle angle → 0°/180° top-down poses
  → IK/FK + 41-state clearance
  → unique selected candidate
  → execution_plan.json
```

segmentation、cube body ID 和 `rayTest` 只从三维点之后流向审计字段，不存在
返回定位、geometry、深度采样或候选生成的反向路径。

## 门控与失败保真

总 `scientific_gate_passed` 需要同时满足：

- Grounding DINO 返回有限、正面积定位框；
- 定位框事后唯一匹配 cube 且 IoU ≥ `0.25`；
- geometry 参数有限、宽高为正、中心在图像内；
- 中心深度有效，坐标有限，重投影门控通过；
- segmentation 与射线事后命中 cube；
- 恰好两个对称候选完成审计，至少一个通过 IK/FK、关节限位与 `2 mm` 余量；
- 恰好一个候选被确定性选择；
- 没有执行电机、闭合、接触、抬升或物理抓取。

任何异常均写入 `metadata.json` 的 `failure_stage`、`failure_reason` 和真实已完成
字段。失败时不得生成或保留旧的 `execution_plan.json`。

## 输出

默认目录：
`data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight/`

- `rgb.png`、`depth.npy`、`segmentation.png`；
- `localization.png`、`geometry_prediction.png`；
- `candidates.csv`：两个候选的位姿、IK/FK、碰撞、代价和选择结果；
- `summary.json`、`metadata.json`；
- `execution_plan.json`：仅总门控通过时存在。

## 测试与正式验证

- 纯单元测试覆盖配置拒绝非法 target/backend、计划序列化与加载时的完整性
  校验、旧计划清理和候选唯一选择。
- 真实 PyBullet 测试注入固定 localization/geometry 输出，但使用真实相机
  深度、反投影、IK/FK 和碰撞审计，证明执行计划来自同一场景且没有电机步进。
- runner 测试检查所有非执行 metadata 标志、候选顺序、输入哈希和失败保真。
- 正式证据使用真实 CUDA Grounding DINO；若运行环境没有 CUDA则明确失败，
  不静默转 CPU，也不把注入测试称为 VLM 正式结果。

## 结论边界与下一步

Stage 6A 通过只能说明“VLM + geometry 已产生一个在固定场景中通过静态安全
预检的方块执行计划”，不能称为感知抓取成功或仿真抓取成功。下一小步 Stage
6B 才会加载并复核该计划，然后调用冻结的阶段 1--5 物理控制链。多头 CNN 在
几何 Stage 6B 完成前不接入，避免同时调试两个后端。
