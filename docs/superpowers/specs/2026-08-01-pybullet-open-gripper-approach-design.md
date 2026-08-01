# PyBullet 张开夹爪垂直接近设计

## 目标与范围

阶段 3 从已经通过的真值方块 pregrasp 开始，让 Panda 在夹爪保持张开的条件下
沿世界 Z 方向下降到接触前抓取高度。本阶段只证明“能够安全到达抓取前姿态”，
不发送闭合命令、不把接触作为成功条件、不抬升物体，也不计为仿真抓取成功。

固定 cube 的 AABB 顶面是高度基准。继续复用现有姿态生成约定：pregrasp 位于
顶面以上 `0.12 m`，surface-standoff 位于顶面以上 `0.02 m`，因此垂直下降
距离固定为 `0.10 m`。工具朝向和 XY 坐标在两点之间保持不变。

## 方案选择

考虑过三种实现：

1. 复制阶段 2 runner 并增加下降段。实现直接，但会复制场景、预检、门控和
   CSV/JSON 逻辑，后续闭合与抬升会继续放大重复。
2. 在 `run_truth_pregrasp.py` 中增加阶段模式。改动较少，但会让命名为阶段 2
   的 CLI 文件承担多阶段状态机职责。
3. 抽取共同真值执行核心，阶段 2/3 使用各自薄 runner。需要一次受控重构，
   但职责清晰，并符合研究总计划已预留的 `grasp_execution.py` 边界。

采用方案 3。现有 `run_truth_pregrasp.py` 的公共配置、函数和 CLI 保持兼容；
新增 `run_truth_approach.py` 作为阶段 3 入口。共同核心只放在
`src/simulation/pybullet/grasp_execution.py`，不修改静态
`kinematic_audit.py` 的职责。

## 组件与数据流

`grasp_execution.py` 接收统一的 `TruthExecutionConfig` 和阶段枚举。它负责：

1. 创建固定场景并等待 60 步；将 Panda 初始化到中立位且双指为 `0.04 m`。
2. 对 cube 再执行 60 步稳定性检查，最大质心位移门槛保持 `0.001 m`。
3. 从 cube 真值中心、姿态和 AABB 顶面生成 pregrasp 与 surface-standoff。
4. 对两个姿态分别执行 `5 mm/5°` IK/FK 审计，并对
   中立→pregrasp→standoff 全路径执行 `2 mm` 静态碰撞余量审计。
5. 阶段 2 只执行 pregrasp；阶段 3 先执行并门控 pregrasp，再单独执行
   approach。若 pregrasp 动态门控失败，不发送 approach 段命令。
6. 每个真实仿真步记录关节、双指、末端、cube 真值位姿、末端相对 cube 位姿
   和碰撞余量；分别保存起点、pregrasp 与 approach 图像。

阶段 3 的 summary 使用 `cube_truth_open_approach`，轨迹 phase 必须恰好为
`pregrasp` 和 `approach`。阶段 2 的既有字段名、文件名和语义保持不变。

## 阶段 3 成功门控

阶段 3 总科学门控必须同时满足：

- cube 执行前稳定性通过；
- pregrasp 和 surface-standoff 的 IK/FK 均通过 `5 mm/5°`；
- 中立→pregrasp→standoff 静态路径通过 `2 mm` 余量；
- pregrasp 与 approach 两个动态段均到达，且无环境或自碰撞；
- approach 终点全姿态误差不超过 `5 mm/5°`；
- 终点相对 cube 中心 XY 偏差不超过 `5 mm`；
- 终点工具高度相对本次 cube 顶面为 `0.02 m ± 0.005 m`；
- cube 在稳定检查和两个动态段中的最大位移不超过 `0.001 m`；
- 双指相对 `0.04 m` 张开目标的最大误差不超过 `0.001 m`；
- 夹爪闭合命令次数严格为零，所有记录状态有限。

动态碰撞检查仍把 cube 作为环境刚体，因此任何小于 `2 mm` 的意外接近都会使
阶段失败。该检查不是目标接触评价；metadata 中 `contact_evaluated` 保持
`false`。

## 输出与失败处理

默认输出目录为
`data/processed/pybullet/grasp_execution/stage_3_open_approach/`，包含：

- `state_trace.csv`；
- 空表头 `contact_events.csv`，明确本阶段没有评价接触；
- `summary.json` 和 `metadata.json`；
- `start.png`、`pregrasp.png`、`approach.png`。

预稳定、IK/FK、静态余量或 pregrasp 动态门控任一失败时立即停止，不执行后续
段，并保存已经产生的证据与明确 `failure_stage`。metadata 设置
`vertical_approach_executed=true` 仅表示 approach 段实际发送并步进；
`descent_to_contact_executed`、`gripper_close_commanded`、
`contact_evaluated`、`object_lifted` 和 `physical_grasp_executed` 全部保持
`false`。

## 测试与验收

先用现有阶段 2 真实集成测试保护重构兼容性，再添加阶段 3 真实 PyBullet 测试。
阶段 3 测试核验文件、phase、两段到达、终点高度、目标稳定、夹爪张开、零碰撞
和全部未执行标志。正式 evidence run 通过后人工检查三张关键帧，再更新中文
研究计划、当前状态、项目结构和工作日志。完整项目测试必须保持全绿。

## 自检

- 没有把阶段 4 的闭合或接触评价带入阶段 3。
- `0.10 m` 下降量与既有 `0.12/0.02 m` 两个高度定义一致。
- 阶段 2 的入口和语义保持兼容，公共逻辑只抽取一次。
- 所有门槛均沿用已批准研究计划，没有为获得通过而放宽。
