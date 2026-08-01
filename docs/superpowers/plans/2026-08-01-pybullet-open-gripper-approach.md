# PyBullet 张开夹爪垂直接近实施计划

> **供代理执行者使用：**必须使用 `superpowers:executing-plans` 逐项执行；每一步用复选框（`- [ ]`）跟踪。本计划不使用子代理。

**目标：**保留阶段 2 兼容性的同时，让 Panda 从已通过的真值 cube pregrasp 垂直下降 `0.10 m`，到达 cube 顶面以上 `0.02 m` 的接触前姿态，夹爪全程保持张开。

**架构：**将阶段 2 runner 中已验证的场景、真值姿态、预检、动态门控和证据逻辑抽到 `grasp_execution.py`；阶段 2 与阶段 3 各自保留薄 CLI。共同核心按阶段决定只执行 pregrasp，或依次执行 pregrasp 与 approach，不修改静态运动学审计职责。

**技术栈：**Python 3、PyBullet、NumPy、OpenCV、pytest。

## 全局约束

- 计划、设计、当前状态和工作日志使用中文；代码标识符、CLI 与 CSV/JSON 字段保留英文。
- 固定目标为 `cube`，随机种子为 `42`，夹爪目标为每指 `0.04 m`。
- pregrasp 高于 cube 顶面 `0.12 m`，approach 终点高于 cube 顶面 `0.02 m`，世界 Z 下降量为 `0.10 m`。
- 末端门槛保持 `5 mm/5°`，XY 偏差门槛 `5 mm`，目标位移门槛 `1 mm`，双指张开误差门槛 `1 mm`，碰撞余量 `2 mm`。
- 阶段 3 不发送闭合命令，不评价接触，不抬升物体，不产生抓取成功结论。
- 必须先写失败测试并确认失败原因正确，再写最小实现；重构后先验证阶段 2 行为。
- 任一门控失败即停在阶段 3，不实现阶段 4。

---

## 文件职责

- 新建 `src/simulation/pybullet/grasp_execution.py`：共同真值执行核心、阶段枚举、统一配置、门控和证据写入。
- 修改 `src/simulation/pybullet/run_truth_pregrasp.py`：保留现有配置、公共函数和 CLI，只转换配置并调用共同核心。
- 新建 `src/simulation/pybullet/run_truth_approach.py`：阶段 3 配置、公共函数和 CLI。
- 修改 `tests/simulation/test_pybullet_truth_pregrasp.py`：保护阶段 2 输出兼容，并要求空接触事件文件。
- 新建 `tests/simulation/test_pybullet_truth_approach.py`：真实 PyBullet 阶段 3 集成测试。
- 修改仿真 README、项目结构、当前状态、研究总计划和工作日志：只记录正式输出支持的结果。

### 任务 1：抽取共同真值执行核心并保持阶段 2 兼容

**文件：**
- 新建：`src/simulation/pybullet/grasp_execution.py`
- 修改：`src/simulation/pybullet/run_truth_pregrasp.py`
- 修改：`tests/simulation/test_pybullet_truth_pregrasp.py`

**接口：**
- `TruthExecutionStage(str, Enum)`：`PREGRASP = "pregrasp"`、`OPEN_APPROACH = "open_approach"`。
- `TruthExecutionConfig(output_dir: Path, seed: int = 42, gui: bool = False, target_name: str = "cube", stability_steps: int = 60, maximum_target_displacement_m: float = 0.001)`。
- `run_truth_execution(config: TruthExecutionConfig, stage: TruthExecutionStage) -> dict[str, object]`。
- 保留 `run_truth_pregrasp(config: TruthPregraspConfig) -> dict[str, object]`。

- [ ] **步骤 1：为阶段 2 证据契约写失败测试**

在 required 文件中加入 `contact_events.csv`，并断言其只有表头：

```python
assert (tmp_path / "contact_events.csv").read_text(
    encoding="utf-8"
) == "step,phase,robot_link,target_body,normal_force\n"
```

- [ ] **步骤 2：确认测试按预期失败**

运行：

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_pregrasp.py -v
```

预期：阶段 2 运动门控仍通过，但因接触事件文件不存在而失败。

- [ ] **步骤 3：抽取共同核心并添加空接触证据**

将阶段 2 的初始化、写入、稳定性、姿态、IK/FK、余量、执行和门控移入共同核心。阶段 2 只执行：

```python
execute_joint_motion(
    segments=(MotionSegment("pregrasp", pregrasp_solution),),
    tracked_body_ids=(cube_id,),
    ...,
)
```

使用固定字段写入空接触事件 CSV；阶段 2 summary 字段、图像、轨迹 phase 和 metadata 未执行标志保持不变。薄 runner 转换配置并调用 `TruthExecutionStage.PREGRASP`。

- [ ] **步骤 4：验证阶段 2 与相关回归**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_pregrasp.py tests/simulation/test_pybullet_motion_control.py tests/simulation/test_pybullet_safe_motion_runner.py -v
```

预期：全部通过，阶段 2 科学门控仍为真，phase 仍只有 `pregrasp`。

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/grasp_execution.py src/simulation/pybullet/run_truth_pregrasp.py tests/simulation/test_pybullet_truth_pregrasp.py
git commit -m "refactor: share truth grasp execution core"
```

### 任务 2：新增阶段 3 两段真实运动和门控

**文件：**
- 修改：`src/simulation/pybullet/grasp_execution.py`
- 新建：`src/simulation/pybullet/run_truth_approach.py`
- 新建：`tests/simulation/test_pybullet_truth_approach.py`

**接口：**
- `TruthApproachConfig(output_dir: Path = Path("data/processed/pybullet/grasp_execution/stage_3_open_approach"), seed: int = 42, gui: bool = False, target_name: str = "cube", stability_steps: int = 60, maximum_target_displacement_m: float = 0.001)`。
- `run_truth_approach(config: TruthApproachConfig) -> dict[str, object]`。

- [ ] **步骤 1：写真实阶段 3 失败测试**

```python
summary = run_truth_approach(TruthApproachConfig(output_dir=tmp_path))
assert summary["stage"] == "cube_truth_open_approach"
assert summary["pregrasp_reached"] is True
assert summary["approach_reached"] is True
assert summary["approach_endpoint_pose_gate_passed"] is True
assert summary["approach_height_gate_passed"] is True
assert summary["fingers_open_gate_passed"] is True
assert summary["gripper_close_command_count"] == 0
assert summary["scientific_gate_passed"] is True
```

同时断言轨迹 phase 恰为 `{"pregrasp", "approach"}`，approach 第一行工具高度高于最后一行；三个 PNG、两个 CSV 和两个 JSON 均非空。metadata 只把 `vertical_approach_executed` 置为真，接触下降、闭合、接触评价、抬升和物理抓取均为假。

- [ ] **步骤 2：确认测试按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_approach.py -v
```

预期：因阶段 3 runner 模块不存在而在导入阶段失败。

- [ ] **步骤 3：实现双姿态预检和分段执行**

对 `candidate.pregrasp_pose` 与 `candidate.surface_standoff_pose` 分别运行 IK/FK，并用真实两个解检查中立→pregrasp→standoff 静态路径。预检通过后先执行 pregrasp，只有其动态 gate 和到达标志通过才执行：

```python
approach = execute_joint_motion(
    segments=(MotionSegment("approach", approach_ik.solution),),
    tracked_body_ids=(cube_id,),
    ...,
)
```

- [ ] **步骤 4：实现阶段 3 终点与边界门控**

```python
actual_height_above_cube_top_m = final_tool_z - cube_top_z
approach_height_error_m = abs(actual_height_above_cube_top_m - 0.02)
approach_height_gate_passed = approach_height_error_m <= 0.005
```

总门控同时要求两段到达与动态 gate、终点 `5 mm/5°`、XY `5 mm`、cube 位移 `1 mm`、双指张开误差 `1 mm`、零闭合命令、零禁止碰撞和全部状态有限。summary 分别记录两段步数和合计。

- [ ] **步骤 5：实现阶段 3 薄 runner 与 CLI**

```python
def run_truth_approach(config: TruthApproachConfig) -> dict[str, object]:
    return run_truth_execution(
        config.to_execution_config(),
        TruthExecutionStage.OPEN_APPROACH,
    )
```

- [ ] **步骤 6：运行定向测试和 Python 编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests/simulation/test_pybullet_truth_approach.py tests/simulation/test_pybullet_truth_pregrasp.py tests/simulation/test_pybullet_motion_control.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile src/simulation/pybullet/grasp_execution.py src/simulation/pybullet/run_truth_approach.py
```

预期：全部通过；阶段 2 和阶段 3 均通过各自真实集成门控。

- [ ] **步骤 7：提交**

```bash
git add src/simulation/pybullet/grasp_execution.py src/simulation/pybullet/run_truth_approach.py tests/simulation/test_pybullet_truth_approach.py
git commit -m "feat: execute open-gripper cube approach"
```

### 任务 3：正式证据、视觉检查与中文记录

**文件：**
- 修改：`src/simulation/pybullet/README.md`
- 修改：`docs/agent/PROJECT_STRUCTURE.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/agent/vlm_robotic_grasp_study_plan.md`
- 修改：`docs/worklog/WORKLOG.md`
- 生成但不提交：`data/processed/pybullet/grasp_execution/stage_3_open_approach/*`

- [ ] **步骤 1：运行正式阶段 3 证据**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m src.simulation.pybullet.run_truth_approach
```

预期：`scientific_gate_passed` 为真；否则停止并使用系统化调试，不实现阶段 4。

- [ ] **步骤 2：核验数值、文件和图像**

用 `python -m json.tool` 检查 JSON，用 `wc -l` 和 `file` 检查轨迹与三张 `640×480` PNG，并查看 start、pregrasp、approach 图像。确认夹爪靠近方块但保持张开，cube 无明显位移。

- [ ] **步骤 3：更新中文项目记录**

README 增加命令和边界；项目结构登记核心、runner 和测试；研究计划、当前状态和工作日志只记录正式产物中的步数、终点误差、高度误差、cube 位移、夹爪误差、余量与碰撞。下一步改为阶段 4 闭合与目标接触。

- [ ] **步骤 4：运行完整回归和差异检查**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest -q
git diff --check
```

预期：全部测试通过且差异检查无输出。

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md docs/agent/CURRENT_STATUS.md docs/agent/vlm_robotic_grasp_study_plan.md docs/worklog/WORKLOG.md
git commit -m "docs: record open-gripper approach evidence"
```

## 自检结果

- 设计覆盖：任务 1 保护阶段 2，任务 2 只实现阶段 3，任务 3 负责真实证据和中文记录。
- 边界一致：surface-standoff 高度 `0.02 m`、下降 `0.10 m` 与现有姿态生成默认值一致。
- 类型一致：两个薄 runner 都转换为同一个 `TruthExecutionConfig`，阶段枚举名称一致。
- 实现完整：每个任务均给出明确文件、接口、失败测试、实现动作、验证命令和提交边界。
