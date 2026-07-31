# PyBullet 抓取姿态与离线 IK 审计设计

## 目标与证据边界

在已经通过的九个“二维中心 + 米制深度 + 相机矩阵 → 目标表面世界点”结果上，
生成确定性的俯视夹爪候选，计算 Franka Panda 的离线逆运动学，并在独立
PyBullet DIRECT 状态中审计关节限位、FK 误差和碰撞。该阶段不调用关节电机
控制，不步进执行轨迹，不闭合夹爪，不改变目标状态，也不报告物理抓取成功。

允许的结论仅为：某个表面悬停姿态在数值上具有满足阈值的 IK 解，并且离散审计
的关节插值状态未发现规定类别的碰撞。它不证明连续轨迹规划完备、控制稳定、
夹爪接触合理或物体能够被抬升。

## 方案选择

采用固定俯视方案：世界坐标 $-Z$ 为工具接近方向，二维抓取矩形的长边方向为
平行夹爪指板方向。这样只使用当前二维输出确实提供的平面方向，并利用固定桌面
场景的已知竖直轴；不从单个深度像素虚构表面法向。

未采用沿相机射线接近，因为斜视相机会产生倾斜工具轴并增加桌面/侧向碰撞风险。
未采用局部深度法向，因为当前阶段没有经过验证的邻域去噪、边缘处理和法向质量
门控；加入它会同时改变姿态来源与深度使用范围。

## 坐标与姿态约定

每条输入必须来自已通过的 `BackprojectionAudit`，并与同一 target/backend 的
后端行关联。二维角度是抓取矩形长边相对图像 $+x$ 的方向；图像 $+y$ 向下。
系统在中心两侧沿该角度各取 5 pixel 的辅助点，使用中心的同一米制深度和同一
相机矩阵反投影。两辅助世界点之差投影到世界 XY 平面并归一化，得到指板轴
$x_g$。接近轴固定为 $z_g=(0,0,-1)$，闭合轴为
$y_g=z_g\times x_g$。旋转矩阵以 $[x_g,y_g,z_g]$ 为列并转换为四元数。

平行夹爪具有 $180^\circ$ 对称性，因此每条九点记录生成两个候选：原指板轴与
其反向。两个候选位置相同，只绕接近轴相差 $180^\circ$。系统不把
$90^\circ$ 旋转作为等价候选，因为那会交换指板方向与闭合方向。

世界表面点本身不是物体中心。为避免把可见表面点误称为闭合抓取中心，第一阶段
定义两个悬停姿态：

- `surface_standoff_pose`：表面点沿世界 $+Z$ 偏移 `0.02 m`；
- `pregrasp_pose`：standoff 再沿世界 $+Z$ 偏移 `0.10 m`。

二者方向相同。输出明确使用 `surface_standoff`，不使用“接触/闭合姿态”名称。

## Panda 模型解析与中立状态

运行时按 URDF 名称解析：

- 臂关节：`panda_joint1` 至 `panda_joint7`；
- 手指关节：`panda_finger_joint1`、`panda_finger_joint2`；
- 工具 link：`panda_grasptarget`。

若名称缺失、重复、关节类型不符或限位无效，审计立即失败；不得回退到硬编码
索引。手指仅为碰撞几何设为完全张开 `0.04 m`，不执行闭合。

中立臂状态取七个有效上下限的逐关节中点，而不是引用外部示例姿态。IK rest
pose 使用同一中点；range 为上限减下限。所有输入、输出和限位必须有限。

## 离线 IK 与候选选择

每个方向候选依次求解 pregrasp 与 surface-standoff：

1. 调用 PyBullet `calculateInverseKinematics`，提供工具 link、目标位置、目标
   四元数、全部九个可动关节（七个臂关节与两个手指）的上下限、range 和 rest
   pose，以及 `maxNumIterations=200`、`residualThreshold=1e-5`；手指 rest
   pose 固定为完全张开的 `0.04 m`；
2. PyBullet 对当前 URDF 返回九个可动关节值；只用其中按名称映射的七个臂关节
   作为姿态解，手指始终由静态审计显式设为 `0.04 m`；
3. 检查结果有限且位于限位内（容差 `1e-6 rad`）；
4. 在保存原始关节状态后，以 `resetJointState` 临时设置候选并用 FK 回读工具
   位姿；
5. 要求位置误差不超过 `0.005 m`、四元数最短弧方向误差不超过 `5°`；
6. 在 `finally` 中恢复全部臂与手指原始状态。

`resetJointState` 只用于独立 DIRECT 客户端中的静态几何审计。整个 runner 不
调用 `setJointMotorControl*`，不在候选状态后调用 `stepSimulation`，因此不把
这些状态称为执行轨迹。

若两个对称候选都通过，选择从中立状态到 pregrasp 再到 standoff 的归一化关节
位移平方和较小者；若只有一个通过则选该候选；若都失败则保留两个失败记录并
给出分阶段原因。

## 碰撞审计

审计使用固定多物体场景并为 Panda 启用 PyBullet self-collision 标志。每个
候选检查两段关节空间线性插值：中立状态到 pregrasp，以及 pregrasp 到
surface-standoff；每段包含端点在内固定采样 21 个状态。

每个状态设置七个臂关节与两个张开的手指关节，调用
`performCollisionDetection`，但不调用 `stepSimulation`。以下任一情况失败：

- Panda 与 plane、table 或任一场景物体之间的距离小于 `0.002 m`；
- Panda 非相邻 link 之间出现穿透或距离小于 `0.002 m`。

自碰撞过滤只忽略同一 link、直接父子 link 以及左右手指与其直接父 link 的
结构邻接对；不按碰撞结果临时添加白名单。目标物体也不豁免，因为当前姿态是
表面上方的 standoff，而不是允许接触的闭合姿态。

碰撞审计结束后必须恢复原关节状态并再次调用碰撞检测。任何 PyBullet 异常都
转换为该候选的失败原因，runner 仍保存其他候选。

## 模块与数据流

新增三个职责分离模块：

- `pose_generation.py`：纯 NumPy/矩阵逻辑，输入二维中心、角度、深度与相机
  矩阵，输出两个 `PoseCandidate`；
- `kinematic_audit.py`：解析 Panda、调用 IK/FK、临时设置状态、碰撞审计并
  恢复状态；
- `run_pose_ik_study.py`：读取已有九点 CSV 和 backend CSV，重建固定场景，
  编排 18 个候选并写出结果。

输入文件固定为：

- `backprojection_results.csv`；
- `backend_results.csv`；
- `metadata.json`（用于核对相机配置、场景 seed 和对象顺序）。

runner 在计算前验证九条 target/backend 顺序、每条原九点门控为真、中心和
角度一致、场景配置为 seed 42 与 640×480。输入不满足时整体失败且不调用 IK。

## 输出与总门控

输出目录仍为 `data/processed/pybullet/multi_object_study/`，新增：

- `pose_ik_candidates.csv`：18 行候选的姿态、关节解、FK 误差、碰撞统计与
  失败原因；
- `pose_ik_summary.json`：九条是否各有一个选中候选及各阶段计数；
- `pose_ik_metadata.json`：输入路径/哈希、PyBullet 版本、阈值和执行边界。

九点总门控要求每个 target/backend 恰有两个方向候选，且恰有一个选中候选；
选中候选必须同时通过姿态、pregrasp IK、standoff IK、FK、限位和两段碰撞
审计。未选中的对称候选允许通过或失败，但完整记录。

元数据必须包含：

```text
ik_solver_called: true
joint_states_set_for_static_audit: true
motor_control_called: false
simulation_stepped_during_candidate_audit: false
trajectory_executed: false
gripper_closed: false
physical_grasp_executed: false
```

这些布尔值描述真实代码路径；任何未来加入电机或 step 的改动必须使相关测试
失败，不能只依赖说明文字。

## 测试与验收

按 TDD 实现：

1. 纯单元测试覆盖角度到世界轴、正交旋转、180° 对称、边界辅助像素和无效
   输入；
2. 依赖注入测试覆盖名称解析、限位、FK 误差、候选选择、异常恢复和禁止电机/
   step 的调用边界；
3. 真实 PyBullet DIRECT 测试覆盖 Panda 名称解析、至少一个手工可达俯视姿态
   的 IK/FK 以及碰撞检测状态恢复；
4. runner 假数据测试固定 18 行顺序、缺失输入失败、CSV/JSON schema 和总门控；
5. 完整项目回归、`git diff --check` 和一次真实九点离线运行。

真实运行可以报告门控通过或失败，但不得为获得通过而放宽 `5 mm`、`5°`、
`2 mm`、21 状态或九点完整性要求。若固定场景存在不可达或碰撞候选，应保存
失败并把它作为下一步设计证据，而不是写成物理抓取失败率。
