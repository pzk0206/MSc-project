# PyBullet 方块双指闭合接触设计

## 目标与边界

阶段 4 从已经通过的 cube 接触前姿态开始，保持 Panda 手臂在 approach 关节
目标，只闭合两个手指并验证左右手指是否都与目标 cube 建立真实 PyBullet
接触。检测到双指接触后保持夹爪，不执行任何抬升轨迹。

本阶段通过只能表述为“闭合与双指目标接触通过”，不能表述为抓取成功。只有
后续阶段 5 将物体抬升并保持，才可以把一次试验计入仿真抓取成功率。

## 方案选择

考虑三种闭合方式：

1. 在固定时长内始终命令双指到 `0.0 m`。实现最简单，但方块接触后仍持续
   收紧，可能产生不必要的挤压、滑移或弹飞。
2. 以位置控制缓慢闭合，首次检测到双指同时有效接触后冻结当时的闭合命令，
   再保持固定步数。该方案能限制过度收紧，并产生清晰的接触获取时刻。
3. 新增基于接触力的闭环力控制。控制更细，但会同时引入力目标、增益和稳定性
   调参，超出当前“先证明能闭合接触”的最小阶段范围。

采用方案 2。闭合最多使用 240 个仿真步，从每指 `0.04 m` 线性减小目标位置；
双指同时接触后停止减小目标，并保持该命令 120 步。阶段通过要求运行结束时
至少连续 60 步保持双指同时接触。

## 组件职责

新增 `src/simulation/pybullet/gripper_control.py`，只负责夹爪闭合、手臂保持、
逐步状态采样、目标接触分类和禁止碰撞分类。它不创建场景、不求 IK，也不决定
阶段输出目录。

现有 `grasp_execution.py` 新增 `CLOSE_CONTACT` 阶段：复用阶段 2/3 的目标
稳定、双姿态 IK/FK、静态余量、pregrasp 和 approach 动态门控；只有上述门控
都通过才调用夹爪控制。新增 `run_truth_contact.py` 作为阶段 4 薄 CLI。

## 接触定义与安全分类

Panda 两个手指 joint/link 由现有名称解析得到，固定为
`model.finger_joint_indices`，不通过硬编码猜测。

一条有效目标接触必须同时满足：

- `bodyA` 为 Panda，`bodyB` 为本次 cube；
- Panda link 是左手指或右手指之一；
- PyBullet `contactNormalForce` 为有限数且严格大于 `0 N`；
- 接触记录中的 body/link 与本次场景真实 ID 一致。

每个仿真步分别计算左、右有效接触；只有同一步左右均为真才算双指接触。接触
事件 CSV 记录 step、phase、robot link、target body 和法向力，不把桌面、其他
物体或机器人其他 link 的接触误记为目标接触。

以下情况属于禁止碰撞并使阶段失败：

- Panda 非手指 link 接触 cube；
- Panda 接触 plane、duck 或 sphere；
- 除既有 Panda 底座—桌面安装豁免外，Panda 接触桌面；
- 去除相邻/固定刚体内部接触后仍存在 Panda 自碰撞。

cube 与桌面接触是本阶段的预期状态，不计为禁止碰撞。cube 在闭合期间的位移、
高度变化和姿态变化全部记录，但不设置新的成功阈值；阶段 4 不用位移门槛掩盖
或替代双指接触判据。

## 控制与状态记录

`GripperCloseConfig` 默认值为：`close_steps=240`、`hold_steps=120`、
`minimum_bilateral_hold_steps=60`、`closed_target_m=0.0`、
`arm_joint_tolerance_rad=0.01`。夹爪力使用 Panda URDF 中的正有限力上限，手臂
每步继续通过位置控制保持 approach 的七关节目标。

闭合阶段逐步记录：

- 命令与实测双指位置；
- 实测七关节和末端位姿；
- cube 世界位姿与末端相对 cube 位置；
- 左/右/双指有效接触状态和各指最大法向力；
- 手臂保持误差、禁止环境接触数、自碰撞数和数值有限性。

阶段 4 的 `state_trace.csv` 合并 pregrasp、approach、close 和 contact_hold
四种 phase；新增 `commanded_finger_positions`、左右接触状态和法向力字段。
阶段 2/3 的同名 CSV 继续可读，新字段对运动阶段填入开放夹爪命令和零接触。

## 成功门控与失败处理

阶段 4 总科学门控必须同时满足：

- 阶段 3 的全部前置门控通过；
- 闭合控制实际执行，且至少出现一次双指同时有效目标接触；
- 运行结束时连续双指接触步数不少于 60；
- 左、右手指各至少记录一条有限正法向力目标接触；
- 手臂保持误差不超过 `0.01 rad`，全部状态有限；
- 非手指 cube 接触、其他环境碰撞和自碰撞计数均为零。

240 个闭合步结束仍未获取双指接触时，继续按最终 `0.0 m` 命令完成 120 步
保持并记录失败；不得自动调整 approach 高度、姿态或阈值。pregrasp 或 approach
失败时不发送闭合命令。任何失败都保存已执行轨迹、接触事件、最后图像和明确
`failure_stage`。

## 输出与元数据

默认输出目录为
`data/processed/pybullet/grasp_execution/stage_4_bilateral_contact/`，至少包含：

- `state_trace.csv`；
- 非空 `contact_events.csv`；
- `summary.json`、`metadata.json`；
- `start.png`、`pregrasp.png`、`approach.png`、`closed.png`。

成功时 metadata 设置 `gripper_close_commanded=true`、`gripper_closed=true`、
`contact_evaluated=true`、`target_contacted=true`；这里的 `gripper_closed` 表示
闭合阶段完成并通过双指保持门控，不表示两个关节必须到达零位。由于没有抬升，
`object_lifted` 和 `physical_grasp_executed` 必须保持 `false`。

## 测试与验收

先为夹爪配置校验和真实 PyBullet 双指接触写失败测试，再实现控制器；随后为
阶段 4 runner 写真实集成失败测试，核验四种 phase、双指接触、保持步数、非空
接触事件、零禁止碰撞和全部边界标志。正式 evidence run 通过后人工检查四张
图像，确认方块仍在桌面上且夹爪已经闭合到其两侧，再更新中文项目记录。

## 自检

- 阶段 4 只增加闭合和目标接触，没有加入抬升。
- 有效接触由真实 body/link/正法向力共同定义，不以手指位置代替接触。
- 双指接触必须同一步出现并在末段持续，不接受左右手指不同时的历史接触拼接。
- 失败时不自动改变已冻结的 approach 位姿或研究阈值。
