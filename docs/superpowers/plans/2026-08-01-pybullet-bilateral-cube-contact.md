# PyBullet 方块双指闭合接触实施计划

> **供代理执行者使用：**必须使用 `superpowers:executing-plans` 逐项执行；每一步用复选框（`- [ ]`）跟踪。本计划不使用子代理。

**目标：**从已通过的 cube 接触前姿态让张开的夹爪短距离下探到可闭合抓取深度，再缓慢闭合 Panda 双指；首次获得双指同时目标接触后保持，并用真实 body/link/法向力证明接触成立；本阶段不抬升。

**架构：**新增独立 `gripper_control.py` 管理手臂保持、双指闭合、接触分类和逐步采样；`grasp_execution.py` 只负责在阶段 3 门控通过后调用控制器并汇总阶段 4 证据；新增薄 runner 暴露 CLI。

**技术栈：**Python 3、PyBullet、NumPy、OpenCV、pytest。

## 全局约束

- 文档使用中文；代码标识符、CLI 与 CSV/JSON 字段使用英文。
- 关闭目标从每指 `0.04 m` 线性趋近 `0.0 m`，最多 240 步；双指接触后冻结命令并保持 120 步。
- 最终连续双指接触至少 60 步；有效接触要求目标 cube、手指 link、有限且严格大于零的法向力。
- 手臂保持误差不超过 `0.01 rad`；非手指 cube 接触、其他环境接触和自碰撞均为零。
- 阶段 3 的 cube 顶面上方 `0.02 m` 是安全接触前高度；阶段 4 先保持双指
  `0.04 m` 张开并垂直下探 `0.015 m`，到顶面上方 `0.005 m` 后才闭合。
- cube 与桌面接触允许；cube 位移和高度变化只记录，不新增成功阈值。
- 阶段 4 不发送抬升命令；`object_lifted` 和 `physical_grasp_executed` 必须为假。
- 严格执行测试先行；任一门控失败就停止，不实现阶段 5。

---

## 文件职责

- 新建 `src/simulation/pybullet/gripper_control.py`：闭合控制、接触/碰撞分类、轨迹与结果类型。
- 修改 `src/simulation/pybullet/grasp_execution.py`：新增阶段 4 编排、合并轨迹和证据写入。
- 新建 `src/simulation/pybullet/run_truth_contact.py`：阶段 4 配置、公共函数和 CLI。
- 新建 `tests/simulation/test_pybullet_gripper_control.py`：从静态 approach 姿态真实验证独立控制器。
- 新建 `tests/simulation/test_pybullet_truth_contact.py`：从中立位运行完整阶段 4 集成测试。
- 修改仿真 README、项目结构、当前状态、研究计划和工作日志：只记录正式产物。

### 任务 1：独立夹爪控制器

**文件：**
- 新建：`src/simulation/pybullet/gripper_control.py`
- 新建：`tests/simulation/test_pybullet_gripper_control.py`

**接口：**
- `GripperCloseConfig(close_steps: int = 240, hold_steps: int = 120, minimum_bilateral_hold_steps: int = 60, closed_target_m: float = 0.0, arm_joint_tolerance_rad: float = 0.01)`。
- `ContactEvent(step, phase, robot_link, target_body, normal_force)`。
- `GripperTraceRow`：命令/实测关节、双指、末端、cube 位姿、左右接触、法向力、碰撞和有限性。
- `GripperCloseResult`：完整 trace/events、接触获取、末段连续保持、碰撞、有限性和 gate。
- `execute_gripper_close(*, robot_id, target_body_id, client_id, model, arm_hold_positions, environment_body_ids, allowed_environment_link_pairs=(), config=GripperCloseConfig(), physics=p) -> GripperCloseResult`。

- [x] **步骤 1：写配置与真实接触失败测试**

配置测试拒绝零闭合步、负保持步、最小保持大于总保持、闭合目标越出 Panda 手指限位，以及非正手臂容差。

真实测试加载固定场景，等待稳定，解析 Panda，使用 cube 真值顶面生成
抓取深度姿态并求 IK；测试准备阶段只用 `resetJointState` 将手臂放到该
姿态、双指放到 `0.04 m`，然后调用控制器：

```python
result = execute_gripper_close(
    robot_id=scene.bodies.robot,
    target_body_id=scene.object_body_ids["cube"],
    client_id=scene.client_id,
    model=model,
    arm_hold_positions=approach_ik.solution,
    environment_body_ids=(scene.bodies.plane, scene.bodies.table,
                          scene.object_body_ids["duck"],
                          scene.object_body_ids["sphere"]),
    allowed_environment_link_pairs=((-1, scene.bodies.table),),
)
assert result.bilateral_contact_acquired is True
assert result.trailing_bilateral_contact_steps >= 60
assert {event.robot_link for event in result.contact_events} == set(
    model.finger_joint_indices
)
assert result.gate_passed is True
```

该独立控制器测试显式设置 `surface_standoff_m=0.005`。真实诊断已经证明默认
`0.02 m` 接触前高度中，闭合后的每指到 cube 仍约有 `0.01173 m` 间隙，不能
把不可能接触归咎于控制器或通过放宽接触门槛规避。

- [x] **步骤 2：确认测试按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_gripper_control.py -v
```

预期：因 `gripper_control` 模块不存在而导入失败。

- [x] **步骤 3：实现配置、接触分类和安全采样**

每步用 `getContactPoints(robot_id, target_body_id)`；只把两个 finger link 且
`normal_force > 0` 的有限记录转为 `ContactEvent`。其他 robot→cube link 计入
禁止目标接触。对其他环境刚体和 robot→robot 接触复用现有相邻 link 与安装
豁免规则。

- [x] **步骤 4：实现闭合与保持控制**

每步用位置控制保持七臂关节；闭合命令按线性插值从当前双指位置到
`closed_target_m`。首次双指同时接触后冻结当步命令并转入 `contact_hold`；
若 240 步未获取，则用最终命令进入保持。每个真实 `stepSimulation` 后采样，
计算运行结束时连续双指接触步数和 gate。

- [x] **步骤 5：运行定向测试和编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_gripper_control.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile src/simulation/pybullet/gripper_control.py
```

预期：配置与真实接触测试全部通过。

- [ ] **步骤 6：提交**

```bash
git add src/simulation/pybullet/gripper_control.py tests/simulation/test_pybullet_gripper_control.py
git commit -m "feat: control Panda bilateral cube contact"
```

### 任务 2：完整阶段 4 状态机和 runner

**文件：**
- 修改：`src/simulation/pybullet/grasp_execution.py`
- 新建：`src/simulation/pybullet/run_truth_contact.py`
- 新建：`tests/simulation/test_pybullet_truth_contact.py`

**接口：**
- `TruthExecutionStage.CLOSE_CONTACT = "close_contact"`。
- `TruthContactConfig(output_dir: Path = Path("data/processed/pybullet/grasp_execution/stage_4_bilateral_contact"), seed: int = 42, gui: bool = False, target_name: str = "cube", stability_steps: int = 60, maximum_target_displacement_m: float = 0.001)`。
- `run_truth_contact(config: TruthContactConfig) -> dict[str, object]`。

- [ ] **步骤 1：写完整阶段 4 失败测试**

运行真实 runner 并断言：

```python
assert summary["stage"] == "cube_truth_bilateral_contact"
assert summary["pregrasp_reached"] is True
assert summary["approach_reached"] is True
assert summary["gripper_close_executed"] is True
assert summary["left_finger_contacted"] is True
assert summary["right_finger_contacted"] is True
assert summary["bilateral_contact_acquired"] is True
assert summary["trailing_bilateral_contact_steps"] >= 60
assert summary["target_contact_gate_passed"] is True
assert summary["scientific_gate_passed"] is True
```

要求轨迹包含 `pregrasp/approach/grasp_depth/close/contact_hold`，接触事件
非空且 link 集合等于两个手指；五张 PNG 为 640×480。metadata 中闭合、接触评价和目标接触为
真，抬升和物理抓取为假。

- [ ] **步骤 2：确认测试按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_contact.py -v
```

预期：因 `run_truth_contact` 不存在而导入失败。

- [ ] **步骤 3：接入 CLOSE_CONTACT 阶段**

将该阶段纳入与 OPEN_APPROACH 相同的双姿态预检和两段电机执行；approach
全部动态门控通过后，继续在同一 scene/client 中以开放夹爪垂直下探到 cube
顶面上方 `0.005 m` 的抓取深度。该短段也必须通过到达、有限性、零目标接触与
零禁止碰撞门控，之后才调用：

```python
gripper = execute_gripper_close(
    robot_id=scene.bodies.robot,
    target_body_id=cube_id,
    client_id=scene.client_id,
    model=model,
    arm_hold_positions=approach_ik.solution,
    environment_body_ids=(scene.bodies.plane, scene.bodies.table,
                          scene.object_body_ids["duck"],
                          scene.object_body_ids["sphere"]),
    allowed_environment_link_pairs=allowed_mounting_pair,
)
```

- [ ] **步骤 4：合并轨迹、事件和阶段门控**

扩展 `state_trace.csv` 字段以包含命令双指、左右接触和法向力；运动段填开放
命令与零接触，闭合段写控制器实测值。`contact_events.csv` 写真实事件。
summary 汇总闭合/保持步数、双指最终位置、各指最大法向力、cube 位移/高度
变化和所有禁止碰撞。总门控要求阶段 3 门控与 `gripper.gate_passed` 同时为真。

- [ ] **步骤 5：实现薄 runner 与边界 metadata**

runner 将配置转换为统一配置并调用 `CLOSE_CONTACT`。成功 metadata 设置
`gripper_close_commanded/gripper_closed/contact_evaluated/target_contacted`
为真，`object_lifted/physical_grasp_executed` 为假。

- [ ] **步骤 6：运行阶段 2–4 回归和编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_contact.py tests/simulation/test_pybullet_truth_approach.py tests/simulation/test_pybullet_truth_pregrasp.py tests/simulation/test_pybullet_gripper_control.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile src/simulation/pybullet/grasp_execution.py src/simulation/pybullet/run_truth_contact.py
```

预期：阶段 2–4 真实集成均通过，旧阶段边界不变。

- [ ] **步骤 7：提交**

```bash
git add src/simulation/pybullet/grasp_execution.py src/simulation/pybullet/run_truth_contact.py tests/simulation/test_pybullet_truth_contact.py
git commit -m "feat: execute bilateral cube contact stage"
```

### 任务 3：正式证据与中文记录

**文件：**
- 修改：`src/simulation/pybullet/README.md`
- 修改：`docs/agent/PROJECT_STRUCTURE.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/agent/vlm_robotic_grasp_study_plan.md`
- 修改：`docs/worklog/WORKLOG.md`
- 生成但不提交：`data/processed/pybullet/grasp_execution/stage_4_bilateral_contact/*`

- [ ] **步骤 1：运行正式阶段 4 证据**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m src.simulation.pybullet.run_truth_contact
```

预期：`scientific_gate_passed` 为真；否则停止并进行系统化调试，不实现抬升。

- [ ] **步骤 2：核验 JSON、CSV 与图像**

检查 summary/metadata、轨迹 phase、左右手指 contact event、正有限法向力、
五张 640×480 PNG。人工查看 grasp_depth 和 closed 图，确认双指先开放进入
cube 两侧，再闭合接触，且 cube 仍在桌面上。

- [ ] **步骤 3：更新中文记录**

记录实际闭合/保持步数、首次双指接触步、末段连续接触、各指法向力、cube
位移、碰撞和边界。下一步改为阶段 5 真值 cube 抬升与保持。

- [ ] **步骤 4：运行完整回归和差异检查**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest -q
git diff --check
```

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md docs/agent/CURRENT_STATUS.md docs/agent/vlm_robotic_grasp_study_plan.md docs/worklog/WORKLOG.md
git commit -m "docs: record bilateral cube contact evidence"
```

## 自检结果

- 范围完整：控制器、完整状态机、正式证据和中文记录均有独立任务。
- 类型一致：共同核心消费任务 1 的 `GripperCloseResult`，runner 只选择 `CLOSE_CONTACT`。
- 判据独立：接触由 body/link/正法向力和连续双指保持决定，不用手指位置替代。
- 边界明确：阶段 4 不抬升，任何失败都阻止阶段 5。
