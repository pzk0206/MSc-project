# PyBullet Stage 6A.1 中心偏差诊断设计

## 目标

Stage 6A 已在固定 PyBullet 场景中实时运行 Grounding DINO 与 geometry
后端，并生成通过静态安全预检的冻结 `execution_plan.json`。事后检查发现，
单像素深度反投影恢复的是斜视相机可见的 cube 表面点，而不是 cube 平面中心；
该点相对 cube 真值质心的 XY 偏差约为 `26.55 mm`。

Stage 6A.1 的目标是把这一现象转换为独立、机器可读、可重复验证的后验证据，
同时严格保护 Stage 6A 的原始感知输出和冻结执行计划。该阶段不修正预测、不
生成新执行计划，也不移动机械臂。

## 研究边界

- 正式 Stage 6A.1 只读取现有 Stage 6A 正式产物，不重新运行 VLM 或 PyBullet。
- cube 真值仅用于后验评价，不得成为感知、反投影、候选生成或计划修正的输入。
- `5 mm` XY 门槛只作为与真值控制阶段一致的诊断参考，不追溯改变 Stage 6A
  的静态安全门控结果。
- Stage 6A.1 不调用 `stepSimulation`、电机、轨迹、夹爪、接触或抬升逻辑。
- 可重复性复核必须写入独立目录，不得覆盖正式 Stage 6A 或 Stage 6A.1 产物。
- 本阶段不得表述为物理抓取、感知抓取成功或后端成功率。

## 方案

采用“正式离线审计 + 隔离重复性复核”的组合方案。

正式离线审计以已有 Stage 6A `summary.json`、`metadata.json` 和
`execution_plan.json` 为只读输入。诊断器先验证三份输入属于冻结协议下的
同一运行，再计算中心偏差并写入独立输出目录。原始 Stage 6A 目录中的任何
文件均不得修改。

正式离线审计通过后，可在新的重复性目录重新运行一次 Stage 6A，并对该目录
应用完全相同的诊断器。重复性结果只回答端到端输出及偏差是否可复现，不能
替代正式原始结果，也不能用于静默更新冻结执行计划。

## 组件与职责

### `center_bias_diagnostic.py`

提供纯计算和严格序列化能力：

- 定义冻结的 Stage 6A.1 诊断协议版本。
- 接收预测世界点、cube 真值质心、冻结的 `cube_small.urdf` 半高
  `0.025 m` 和 XY 参考门槛。
- 计算带符号的 X/Y/Z 偏差、XY 欧氏偏差及各参考门槛判断。
- 拒绝非有限数值、非正门槛和结构不完整的数据。
- 使用 `allow_nan=False` 写入稳定 JSON，并可选写入一行 CSV。

该模块不依赖 PyBullet、Torch、Transformers 或模型权重，便于独立测试和复用。

### `run_center_bias_diagnostic.py`

负责 Stage 6A.1 的文件编排：

- 默认读取正式 Stage 6A 目录。
- 将诊断输出写入独立的
  `data/processed/pybullet/grasp_execution/stage_6a1_center_bias_diagnostic/`。
- 严格读取并交叉检查 Stage 6A summary、metadata 和 execution plan。
- 验证 backend、target、prompt、协议版本、RGB SHA-256、世界预测点和非执行
  边界的一致性。
- 从 metadata 中读取拍摄时保存的 cube 真值质心，并以冻结的
  `cube_half_extent_m = 0.025` 计算名义顶面参考高度；Stage 6A 没有保存 AABB，
  因此不得启动场景重新查询或把该派生值表述为重新测得的 AABB。
- 成功时写入 `center_bias_diagnostic.json`、
  `center_bias_diagnostic.csv` 和 `metadata.json`。
- 失败时不保留陈旧成功文件，并写入含 `failure_stage`、`error` 和全部非执行
  标志的失败 metadata。

### 重复性复核

重复性复核不需要新的计算实现。它在独立 Stage 6A rerun 目录运行现有
`run_geometry_execution_preflight`，再把该目录交给同一个 Stage 6A.1 runner。
正式结果与 rerun 结果分别保存，不自动合并，也不自动修改执行决策。

## 数据流

```text
正式 Stage 6A summary / metadata / execution plan（只读）
                         |
                         v
              输入结构与跨文件一致性检查
                         |
                         v
 prediction world point + cube truth snapshot + frozen half extent + thresholds
                         |
                         v
              纯计算中心偏差与诊断判断
                         |
                         v
Stage 6A.1 JSON + CSV + metadata（独立目录、全部非执行）
```

重复性复核沿用同一数据流，但输入和输出均位于明确标记的 rerun 目录。

## 输出契约

`center_bias_diagnostic.json` 至少记录：

- Stage 6A.1 协议版本和来源 Stage 6A 协议版本；
- 来源文件路径及 SHA-256；
- backend、target、prompt、seed 和 RGB SHA-256；
- 预测世界表面点；
- cube 真值质心、冻结半高及派生的名义顶面参考高度；
- X/Y 带符号偏差、XY 合成偏差和顶面 Z 带符号偏差；
- `xy_reference_threshold_m = 0.005`；
- 是否满足 XY 参考门槛；
- 明确的 `diagnostic_only`、`plan_modified = false`、
  `scientific_gate_reinterpreted = false`；
- 所有电机、轨迹、夹爪、接触、抬升和物理抓取标志为 `false`。

CSV 只提供同一组核心数值的一行扁平表示，便于论文表格和独立审计，不引入
JSON 之外的新结论。

## 失败处理

下列情况必须拒绝成功诊断：

- 任一输入文件缺失、JSON 无法解析或不符合 Stage 6A 冻结协议；
- RGB 哈希、backend、target、prompt、seed 或世界预测点跨文件不一致；
- cube 真值快照缺失，或冻结半高不是严格的 `0.025 m`；
- 输入或计算结果包含 NaN/Inf；
- Stage 6A metadata 显示拍摄后继续步进、执行电机、轨迹、夹爪、接触或抬升；
- 输出目录与正式 Stage 6A 输入目录相同。

失败不得覆盖或删除输入目录文件。runner 只清理自己输出目录中的陈旧
Stage 6A.1 成功产物，并保留失败 metadata 供审计。

## 测试与验证

实现遵循测试驱动开发：

1. 先为纯计算协议写失败测试，覆盖已知数值、有限性、门槛和严格序列化。
2. 再为 runner 写失败测试，使用临时复制的 Stage 6A 风格产物验证跨文件一致性、
   输入只读、输出隔离和失败保真。
3. 验证 runner 不导入或调用模型、仿真步进及执行控制逻辑。
4. 运行 simulation 相关回归和完整项目测试。
5. 对正式 Stage 6A 产物运行离线诊断，独立核对约 `0.026550 m` XY 偏差和
   约 `0.000509 m` 名义顶面 Z 偏差。
6. 正式离线证据通过后，在新目录做一次 CUDA Stage 6A 重跑和同协议诊断，
   将其明确标记为重复性复核。

## 验收条件

- 正式 Stage 6A 原始目录的文件哈希在诊断前后完全不变。
- 正式 Stage 6A.1 输出完整、数值有限、跨文件溯源一致。
- 正式结果明确显示 XY 偏差是否超过 `5 mm` 参考门槛，同时不改变 Stage 6A
  的 `scientific_gate_passed` 解释。
- 所有运动与抓取标志保持 `false`。
- 重复性复核使用独立目录，且其结果与正式结果被清楚区分。
- 自动测试、编译检查和 `git diff --check` 通过后，才更新项目状态和工作日志。
