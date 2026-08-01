# PyBullet 真值方块抬升与保持实施计划

> **供代理执行者使用：**必须使用 `superpowers:executing-plans` 逐项执行；步骤使用复选框（`- [ ]`）跟踪。本计划不使用子代理。

**目标：**从已通过的双指 cube 接触状态冻结夹爪命令，垂直抬升方块并保持，以高度、离桌、相对稳定和真实手指接触证明一次真值姿态仿真抓取成功。

**架构：**新增独立 `lift_control.py` 管理冻结夹爪下的手臂抬升、保持、目标/桌面接触与逐步物理状态；现有 `grasp_execution.py` 只增加阶段 5 编排、抬升 IK 和证据合并；新增薄 runner 暴露默认真值 cube CLI。

**技术栈：**Python 3、PyBullet、NumPy、OpenCV、pytest。

## 全局约束

- 文档使用中文；代码标识符、CLI 与 CSV/JSON 字段使用英文。
- 只在阶段 4 总门控通过后执行抬升；不运行感知后端，不处理 sphere/duck。
- 抬升期间逐步复用阶段 4 最终 `commanded_finger_positions`，不得继续收紧、重抓或改用力控制。
- 工具沿世界 Z 计划上移 `0.12 m`；cube 评价门槛保持为相对闭合结束质心至少上升 `0.10 m`。
- 抬升插值 240 步，终点保持 240 步；最后 240 步必须全部达到高度门槛、全部脱离桌面且相对工具漂移不超过 `0.01 m`。
- 运行结束连续双指同时接触至少 120 步；非手指 cube 接触、其他环境碰撞和自碰撞均为零。
- 阶段 5 通过时可称为“一次真值姿态仿真抓取成功”，不得称为感知后端成功率。
- 严格执行测试先行；任一门控失败就停止，不接入几何或多头后端。

---

## 文件职责

- 新建 `src/simulation/pybullet/lift_control.py`：冻结夹爪的抬升/保持控制、接触分类、目标高度/漂移和安全采样。
- 修改 `src/simulation/pybullet/grasp_execution.py`：新增阶段 5 编排、抬升 IK、七阶段证据和边界元数据。
- 新建 `src/simulation/pybullet/run_truth_lift.py`：阶段 5 配置、公共函数与 CLI。
- 新建 `tests/simulation/test_pybullet_lift_control.py`：从已建立双指接触的真实场景验证独立控制器。
- 新建 `tests/simulation/test_pybullet_truth_lift.py`：从中立位执行完整阶段 5 并验证产物。
- 修改仿真 README、项目结构、当前状态、研究计划和工作日志：只记录正式验证产物。

### 任务 1：独立冻结夹爪抬升控制器

**文件：**
- 新建：`src/simulation/pybullet/lift_control.py`
- 新建：`tests/simulation/test_pybullet_lift_control.py`

**接口：**
- `LiftConfig(lift_steps: int = 240, settle_steps: int = 240, hold_steps: int = 240, tool_lift_command_m: float = 0.12, minimum_object_lift_m: float = 0.10, maximum_hold_relative_drift_m: float = 0.01, minimum_trailing_bilateral_contact_steps: int = 120, arm_joint_tolerance_rad: float = 0.01)`。
- `LiftTraceRow`：step/phase、命令/实测臂和双指、工具/cube 位姿、cube 上升量、相对向量/漂移、左右接触/力、cube--table 接触及禁止碰撞。
- `LiftResult`：完整 trace/events、到达、末段高度/离桌/漂移/双指保持、最大误差/碰撞、有限性和 gate。
- `execute_object_lift(*, robot_id, target_body_id, table_body_id, client_id, model, lift_arm_positions, lift_target_pose, frozen_finger_positions, reference_target_position, reference_tool_relative_to_target, environment_body_ids, allowed_environment_link_pairs=(), config=LiftConfig(), physics=p) -> LiftResult`。

- [x] **步骤 1：写配置与真实抬升失败测试**

配置测试拒绝零抬升步、负 settle、非正保持步、非正抬升命令、cube 门槛不小于工具命令、
非正漂移门槛、双指保持要求大于保持步和非正关节容差；执行入口另拒绝越出
`[0.0, 0.04] m` 的冻结双指命令。

真实测试加载固定场景并稳定 60 步，用 cube 真值顶面与 `surface_standoff_m=0.005` 生成抓取深度姿态；静态设置手臂和开放双指后调用现有 `execute_gripper_close` 建立接触。测试保存闭合结束 cube 位置、工具相对 cube 向量和冻结双指命令，生成同姿态世界 Z `+0.12 m` 的 `ToolPose` 并求 IK，然后调用：

```python
result = execute_object_lift(
    robot_id=scene.bodies.robot,
    target_body_id=cube_id,
    table_body_id=scene.bodies.table,
    client_id=scene.client_id,
    model=model,
    lift_arm_positions=lift_ik.solution,
    lift_target_pose=lift_pose,
    frozen_finger_positions=gripper.trace[-1].commanded_finger_positions,
    reference_target_position=gripper.trace[-1].target_position,
    reference_tool_relative_to_target=tuple(
        tool - target
        for tool, target in zip(
            gripper.trace[-1].actual_tool_position,
            gripper.trace[-1].target_position,
        )
    ),
    environment_body_ids=(scene.bodies.plane, scene.bodies.table,
                          scene.object_body_ids["duck"],
                          scene.object_body_ids["sphere"]),
    allowed_environment_link_pairs=((-1, scene.bodies.table),),
)
assert result.object_lift_gate_passed is True
assert result.table_release_gate_passed is True
assert result.relative_stability_gate_passed is True
assert result.trailing_bilateral_contact_steps >= 120
assert result.gate_passed is True
```

- [x] **步骤 2：确认测试按预期失败**

运行：

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_lift_control.py -v
```

预期：因 `lift_control` 模块不存在而导入失败。

- [x] **步骤 3：实现配置、采样和安全分类**

每步分别查询 robot--cube、cube--table、robot--其他环境和 robot--robot 接触。
只有精确 finger link 且有限正法向力计为目标接触；robot 非手指--cube 穿透、
除底座安装豁免外的环境接触和去除相邻 link 后的自碰撞均计为禁止事件。

- [x] **步骤 4：实现抬升、保持和门控**

从真实当前七关节状态线性插值到 `lift_arm_positions`，每步同时发送固定双指
命令。抬升 240 步后最多使用 240 个同属 `lift` 的 settle 步达到原有
`0.01 rad` 终点门槛，再保持目标 240 个完整 `lift_hold` 步；不因接触丢失
提前停止。最后 240 行分别
检查 cube 上升量、table contact 和相对漂移，另从 trace 尾部计算连续双指接触。

- [x] **步骤 5：运行定向测试和编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_lift_control.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile src/simulation/pybullet/lift_control.py
git diff --check
```

预期：配置与真实抬升测试全部通过。

- [x] **步骤 6：提交**

```bash
git add src/simulation/pybullet/lift_control.py tests/simulation/test_pybullet_lift_control.py
git commit -m "feat: lift and hold truth cube"
```

### 任务 2：完整阶段 5 状态机和 runner

**文件：**
- 修改：`src/simulation/pybullet/grasp_execution.py`
- 新建：`src/simulation/pybullet/run_truth_lift.py`
- 新建：`tests/simulation/test_pybullet_truth_lift.py`

**接口：**
- `TruthExecutionStage.LIFT_HOLD = "lift_hold"`。
- `TruthLiftConfig(output_dir: Path = Path("data/processed/pybullet/grasp_execution/stage_5_truth_cube_lift"), seed: int = 42, gui: bool = False, target_name: str = "cube", stability_steps: int = 60, maximum_target_displacement_m: float = 0.001)`。
- `run_truth_lift(config: TruthLiftConfig) -> dict[str, object]`。

- [x] **步骤 1：写完整阶段 5 失败测试**

真实 runner 测试要求 `summary["stage"] == "cube_truth_lift_hold"`，阶段 2--4
全部门控为真，且：

```python
assert summary["lift_executed"] is True
assert summary["lift_reached"] is True
assert summary["object_lift_gate_passed"] is True
assert summary["table_release_gate_passed"] is True
assert summary["relative_stability_gate_passed"] is True
assert summary["lift_hold_gate_passed"] is True
assert summary["physical_grasp_success"] is True
assert summary["scientific_gate_passed"] is True
```

轨迹 phase 必须精确包含七种状态，闭合与抬升接触事件均非空；七张 PNG 均为
640×480。metadata 中真值、运动、闭合、接触、抬升和物理抓取为真，感知为假。

- [x] **步骤 2：确认测试按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_lift.py -v
```

预期：因 `run_truth_lift` 不存在而导入失败。

- [x] **步骤 3：接入 LIFT_HOLD 阶段与抬升预检**

把阶段 5 纳入阶段 4 的全部前置执行。闭合门控通过后，用抓取深度工具姿态
世界 Z `+0.12 m` 构造 `ToolPose` 并运行 `audit_pose_ik`；抬升静态路径环境
排除目标 cube，但保留 plane、table、duck、sphere 和既有安装豁免。

- [x] **步骤 4：调用控制器并合并七阶段证据**

只有 lift IK/FK 和路径预检通过才调用 `execute_object_lift`。扩展 trace 写入
cube lift、table contact、relative drift 字段；合并 gripper/lift contact
events 并给出全局 step。保存 lifted（抬升段结束）和 lift_hold（保持结束）图。

- [x] **步骤 5：实现 summary、metadata 和失败边界**

总门控要求阶段 4、lift 预检和 `LiftResult.gate_passed` 同时为真。成功时
`object_lifted/physical_grasp_executed/physical_grasp_success=true`；任何前置
失败时不得调用抬升，抬升物理门控失败时保留真实 trace 并保持成功字段为假。

- [x] **步骤 6：运行阶段 2--5 回归和编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_truth_lift.py \
  tests/simulation/test_pybullet_truth_contact.py \
  tests/simulation/test_pybullet_truth_approach.py \
  tests/simulation/test_pybullet_truth_pregrasp.py \
  tests/simulation/test_pybullet_lift_control.py \
  tests/simulation/test_pybullet_gripper_control.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/grasp_execution.py \
  src/simulation/pybullet/run_truth_lift.py
git diff --check
```

- [ ] **步骤 7：提交**

```bash
git add src/simulation/pybullet/grasp_execution.py \
  src/simulation/pybullet/run_truth_lift.py \
  tests/simulation/test_pybullet_truth_lift.py
git commit -m "feat: execute truth cube lift stage"
```

### 任务 3：正式证据与中文记录

**文件：**
- 修改：`src/simulation/pybullet/README.md`
- 修改：`docs/agent/PROJECT_STRUCTURE.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/agent/vlm_robotic_grasp_study_plan.md`
- 修改：`docs/worklog/WORKLOG.md`
- 生成但不提交：`data/processed/pybullet/grasp_execution/stage_5_truth_cube_lift/*`

- [ ] **步骤 1：运行正式阶段 5 证据**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m src.simulation.pybullet.run_truth_lift
```

预期：`physical_grasp_success` 和 `scientific_gate_passed` 均为真；否则停止并使用
系统化调试，不接入感知后端。

- [ ] **步骤 2：核验 JSON、CSV 与图像**

检查七种 phase、最后 240 个 hold 状态、cube 高度、table contact、相对漂移、
双指正法向力、零禁止碰撞和七张 640×480 PNG。人工查看 closed、lifted 和
lift_hold，确认方块离开桌面并持续位于双指之间。

- [ ] **步骤 3：更新中文记录**

记录实际抬升/保持步数、cube 最小/最终上升量、桌面解除、相对漂移、双指保持、
碰撞和边界。下一步改为冻结阶段 1--5 控制链并设计几何/多头感知后端公平 pilot。

- [ ] **步骤 4：运行完整回归和差异检查**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest -q
git diff --check
```

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md \
  docs/agent/CURRENT_STATUS.md docs/agent/vlm_robotic_grasp_study_plan.md \
  docs/worklog/WORKLOG.md \
  docs/superpowers/plans/2026-08-01-pybullet-truth-cube-lift.md
git commit -m "docs: record truth cube lift evidence"
```

## 自检结果

- 设计覆盖：独立控制、完整状态机、真实证据、失败保真和中文记录均有明确任务。
- 类型一致：任务 2 只消费任务 1 定义的 `LiftConfig/LiftResult` 接口。
- 阈值一致：工具命令 `0.12 m`，cube 成功阈值 `0.10 m`，保持 240 步，双指尾部 120 步。
- 边界明确：不调整阶段 4 夹爪命令，不接入感知，不开始批量成功率实验。
- 无占位项：所有代码文件、命令、字段和预期失败均已明确。
