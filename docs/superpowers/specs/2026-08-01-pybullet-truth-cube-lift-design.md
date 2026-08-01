# PyBullet 真值方块抬升与保持设计

## 目标与边界

阶段 5 从阶段 4 已验证的双指闭合接触状态继续运行，在同一个 PyBullet scene
和 client 中保持阶段 4 冻结的双指位置命令，只让 Panda 末端沿世界 Z 方向
上移。到达后保持固定时长，并以方块质心高度、桌面接触、夹爪相对漂移和真实
手指接触共同判断方块是否被夹爪持续持有。

本阶段仍使用固定 cube 的 PyBullet 真值姿态，不运行 Grounding DINO、几何
后端或多头 CNN。只有阶段 1--5 的全部门控通过，才可把本次试验表述为“一次
使用真值姿态的仿真抓取成功”；这不等于感知后端端到端抓取成功率。

## 方案选择

考虑三种抬升控制：

1. 冻结阶段 4 首次双指接触时的夹爪位置命令，只插值手臂关节到上方目标。
   该方案只增加抬升变量，可直接检验阶段 4 的闭合接触能否承载方块。
2. 抬升期间继续把双指目标减小到 `0.0 m`。可能增加夹紧力，但会把额外挤压
   与阶段 4 的既有接触能力混合，并增加滑移或弹飞风险。
3. 改用夹爪力控制。能显式控制握持力，但会新增目标力、增益和稳定性调参，
   超出当前最小真值抓取链范围。

采用方案 1。抬升期间每步继续命令阶段 4 最终
`commanded_finger_positions`，不得自动收紧、重抓或修改接触阈值。

## 抬升目标与预检

阶段 4 的抓取深度工具姿态作为抬升起点。抬升目标保持完全相同的四元数和
世界 XY，只把工具世界 Z 增加 `0.12 m`。研究成功门槛仍按既有计划要求方块
质心相对闭合结束时至少上升 `0.10 m`；额外 `0.02 m` 是预先固定的位置控制
误差余量，不改变评价阈值。

发送电机命令前必须完成：

- 阶段 4 的稳定性、IK/FK、路径、抓取深度和双指接触门控全部通过；
- 抬升目标通过七关节 IK、关节限位及 `5 mm/5°` FK 门槛；
- 对 Panda 与 plane、table、duck、sphere 的抬升路径做静态余量审计；cube
  因为是预期被手指接触并随夹爪运动的目标，不进入普通环境余量集合；
- 阶段 4 最终双指命令、实测关节、工具和 cube 状态全部有限。

任何预检或阶段 4 门控失败都保存失败阶段，但不发送抬升命令。

## 独立抬升控制器

新增 `src/simulation/pybullet/lift_control.py`，负责：

- 从当前七关节状态到抬升 IK 解线性插值 240 个真实仿真步；插值结束后沿用
  现有运动控制模式，最多用 240 个 `lift` settle 步等待臂误差进入门槛；
- 确认到达后继续保持抬升目标和冻结双指命令 240 个完整 `lift_hold` 步；
- 每步采样命令/实测关节、双指、工具位姿、cube 位姿和相对位移；
- 分类左右手指对 cube 的有限正法向力接触；
- 分类 cube--table 接触、Panda 非手指--cube 接触、其他环境接触和自碰撞；
- 计算工具终点误差、方块上升量、末段稳定保持和数值有限性。

`grasp_execution.py` 只负责编排阶段 1--4、生成抬升 IK、调用控制器和合并
证据。新增 `run_truth_lift.py` 作为阶段 5 薄 CLI。现有阶段 2--4 runner 与
输出边界保持兼容。

## 成功判据

`LiftConfig` 默认值固定为：

- `lift_steps=240`；
- `settle_steps=240`；
- `hold_steps=240`；
- `tool_lift_command_m=0.12`；
- `minimum_object_lift_m=0.10`；
- `maximum_hold_relative_drift_m=0.01`；
- `minimum_trailing_bilateral_contact_steps=120`；
- `arm_joint_tolerance_rad=0.01`。

阶段 5 总门控必须同时满足：

- 抬升目标 IK/FK 与静态路径预检通过；
- 手臂抬升段达到目标，工具终点位置/方向误差不超过 `5 mm/5°`；
- 最后连续 240 个 `lift_hold` 步中，cube 相对闭合结束质心上升量每步均不少于
  `0.10 m`；
- 同一 240 步中 cube--table 接触每步均为零；
- 同一 240 步中 cube 相对工具的平移偏移相对抬升起点漂移不超过 `0.01 m`；
- 运行结束时连续双指同时正法向力接触不少于 120 步，左右手指都记录过有效
  接触；
- Panda 非手指--cube 接触、其他环境碰撞和去除相邻刚体后的自碰撞均为零；
- 进入保持前的手臂终点误差不超过 `0.01 rad`，全部状态有限；运动中的最大
  命令跟踪滞后另作诊断，不替代终点到达判据。

高度、桌面解除和相对漂移共同证明方块不是留在桌面、弹飞或从夹爪滑落；双指
接触作为独立物理证据。任何一项失败都不得设置 `physical_grasp_executed`。

## 数据与失败处理

阶段 5 的 `state_trace.csv` 包含
`pregrasp/approach/grasp_depth/close/contact_hold/lift/lift_hold` 七种 phase。
接触事件继续使用 step、phase、robot link、target body 和法向力字段，并合并
闭合与抬升期间的真实事件。

逐步额外记录 cube 相对闭合结束的上升量、cube--table 接触、cube 相对工具
向量及其相对起点漂移。若抬升期间失去接触或方块跌落，控制器仍完成预定轨迹
与保持并保存真实失败证据；不得自动下降、重抓、增大夹紧或再次试验。

## 输出与元数据

默认输出目录为
`data/processed/pybullet/grasp_execution/stage_5_truth_cube_lift/`，至少包含：

- `state_trace.csv`、`contact_events.csv`；
- `summary.json`、`metadata.json`；
- `start.png`、`pregrasp.png`、`approach.png`、`grasp_depth.png`、
  `closed.png`、`lifted.png`、`lift_hold.png`。

成功时 metadata 中 `gripper_close_commanded`、`gripper_closed`、
`contact_evaluated`、`target_contacted`、`object_lifted` 和
`physical_grasp_executed` 均为 `true`，同时保留 `perception_executed=false`。
失败时只把实际执行且通过相应门控的边界字段置真。

## 测试与验收

先为配置校验和已建立接触的真实 PyBullet 抬升写失败测试，再实现独立控制器；
随后为完整阶段 5 runner 写真实集成失败测试。定向回归必须同时覆盖阶段 2--5，
完整项目回归不得破坏 Cornell 或感知流程。

正式 evidence run 通过后核对 JSON、七种 phase、双指事件、最后 240 步高度与
桌面接触、七张 640×480 图。人工检查 `closed/lifted/lift_hold`，确认方块从
桌面升起并持续位于双指之间，再更新中文状态和工作日志。

## 自检

- 没有接入感知后端、sphere、duck 或批量成功率实验。
- 方块成功阈值保持既有 `0.10 m`，工具命令余量预先固定为 `0.12 m`。
- 抬升不改变阶段 4 冻结的夹爪命令，不用重抓掩盖失败。
- 抬升成功同时依赖高度、离桌、相对稳定和真实手指接触，不以单一位置值替代
  物理保持。
- 阶段 5 通过只支持真值姿态单次仿真抓取成功，不支持两个感知后端的排名。
