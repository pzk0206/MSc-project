# PyBullet 方块真值预抓取实施计划

> **供代理执行者使用：**必须使用 `superpowers:executing-plans` 按任务逐项执行；每一步用复选框（`- [ ]`）跟踪。本计划不使用子代理。

**目标：**在固定 PyBullet 场景中，仅使用方块真值位姿，让 Panda 从中立位安全移动到方块正上方的俯视 pregrasp，并用逐步轨迹证明夹爪始终张开、方块未被扰动且没有发生禁止碰撞。

**架构：**在现有纯位姿生成模块中增加“世界坐标表面点 → 俯视候选”的公共入口，继续复用同一套姿态约定；在运动控制轨迹中可选记录指定刚体的实测位姿；新增独立阶段 2 runner，负责固定场景、真值方块读取、静态预检、动态执行、科学门控和证据写入。静态审计仍留在 `kinematic_audit.py`，阶段 2 不接入 Grounding DINO、几何后端或 CNN。

**技术栈：**Python 3、PyBullet、NumPy、OpenCV、pytest。

## 全局约束

- 计划、项目状态和工作日志使用中文；代码标识符、JSON/CSV 字段和命令保留英文。
- 固定随机种子为 `42`，固定目标为 `cube`，输出目录为 `data/processed/pybullet/grasp_execution/stage_2_cube_pregrasp/`。
- 末端位姿门槛保持 `5 mm/5°`，目标上方 XY 偏差不超过 `5 mm`，碰撞余量保持 `2 mm`。
- 手指目标始终为每指 `0.04 m`，最大张开误差不得超过 `0.001 m`。
- pregrasp 使用方块 AABB 顶面以上 `0.12 m` 的固定高度：现有 `0.02 m` 表面余量加 `0.10 m` pregrasp 偏移。
- 方块质心在执行期间相对运动起点的最大位移不得超过 `0.001 m`；该门槛同时覆盖执行前稳定性检查和机械臂运动期间扰动检查。
- 本阶段不得下降到接触高度，不得闭合夹爪，不得评价接触、抬升或物理抓取成功。
- 必须先写失败测试并确认按预期失败，再写最小实现；每个任务独立提交。

---

## 文件职责

- 修改 `src/simulation/pybullet/pose_generation.py`：提供从真值世界表面点生成俯视 pregrasp 的公共纯函数，复用现有正交坐标系和四元数实现。
- 修改 `src/simulation/pybullet/motion_control.py`：可选逐步采样指定刚体的世界位姿，不改变未传入跟踪目标时的阶段 1 行为。
- 新建 `src/simulation/pybullet/run_truth_pregrasp.py`：只负责阶段 2 的场景编排、门控、CLI 与证据写入。
- 新建 `tests/simulation/test_pybullet_truth_pregrasp.py`：覆盖真值姿态生成和真实 PyBullet 阶段 2 集成结果。
- 修改 `tests/simulation/test_pybullet_motion_control.py`：覆盖运动轨迹中的刚体位姿采样契约。
- 修改 `src/simulation/pybullet/README.md`：记录阶段 2 的运行命令、产物和边界。
- 修改 `docs/agent/PROJECT_STRUCTURE.md`、`docs/agent/CURRENT_STATUS.md`、`docs/worklog/WORKLOG.md`：仅在真实输出验证通过后记录模块与结果。

### 任务 1：真值俯视 pregrasp 生成

**文件：**
- 修改：`src/simulation/pybullet/pose_generation.py`
- 新建：`tests/simulation/test_pybullet_truth_pregrasp.py`

**接口：**
- 输入：`generate_top_down_pose_from_world_point(*, target: str, backend: str, surface_point: Sequence[float], finger_axis_world: Sequence[float], surface_standoff_m: float = 0.02, pregrasp_offset_m: float = 0.10) -> PoseCandidate`
- 输出：一个 `PoseCandidate`；其 approach 轴固定为世界 `-Z`，pregrasp 比顶面高 `0.12 m`。

- [ ] **步骤 1：写失败测试**

```python
def test_truth_world_point_generates_top_down_cube_pregrasp() -> None:
    candidate = generate_top_down_pose_from_world_point(
        target="cube",
        backend="ground_truth",
        surface_point=(0.48, 0.0, 0.685),
        finger_axis_world=(0.8660254038, 0.5, 0.0),
    )
    assert candidate.surface_standoff_pose.position == pytest.approx(
        (0.48, 0.0, 0.705)
    )
    assert candidate.pregrasp_pose.position == pytest.approx(
        (0.48, 0.0, 0.805)
    )
    assert candidate.approach_axis_world == pytest.approx((0.0, 0.0, -1.0))
```

- [ ] **步骤 2：确认测试按预期失败**

运行：`pytest tests/simulation/test_pybullet_truth_pregrasp.py -v`

预期：因 `generate_top_down_pose_from_world_point` 尚不存在而在导入阶段失败。

- [ ] **步骤 3：写最小实现**

将 `_candidate` 的输入校验补齐后提升为上述公共函数；`generate_top_down_pose_candidates` 改为调用该公共函数，不复制旋转矩阵或四元数逻辑。验证表面点为有限三维向量，finger 轴在 XY 平面投影后归一化且非零，两个偏移为有限正数。

- [ ] **步骤 4：运行定向测试与既有位姿测试**

运行：`pytest tests/simulation/test_pybullet_truth_pregrasp.py tests/simulation/test_pybullet_pose_generation.py -v`

预期：全部通过，既有图像反投影候选行为不变。

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/pose_generation.py tests/simulation/test_pybullet_truth_pregrasp.py
git commit -m "feat: generate truth-based Panda pregrasp pose"
```

### 任务 2：逐步记录目标刚体位姿

**文件：**
- 修改：`src/simulation/pybullet/motion_control.py`
- 修改：`tests/simulation/test_pybullet_motion_control.py`

**接口：**
- 新增：`TrackedBodyPose(body_id: int, position: tuple[float, float, float], quaternion_xyzw: tuple[float, float, float, float])`
- 修改：`execute_joint_motion(..., tracked_body_ids: Sequence[int] = (), ...) -> MotionExecutionResult`
- 修改：`MotionTraceRow.tracked_body_poses: tuple[TrackedBodyPose, ...] = ()`

- [ ] **步骤 1：写失败测试**

在既有真实 PyBullet 控制 fixture 中创建一个固定测试刚体，传入 `tracked_body_ids=(body_id,)`，断言每个 trace row 恰有一条相同 body id 的有限三维位置和四元数；再保留一个未传参数的调用，断言默认值为空元组。

- [ ] **步骤 2：确认测试按预期失败**

运行：`pytest tests/simulation/test_pybullet_motion_control.py -v`

预期：因 `tracked_body_ids` 参数不存在而失败。

- [ ] **步骤 3：写最小实现**

每次 `stepSimulation` 后用 `getBasePositionAndOrientation` 读取指定刚体，把数值转换为不可变 tuple 并写入当前 `MotionTraceRow`；将这些数值纳入 `all_states_finite` 门控。未指定刚体时不额外调用 PyBullet。

- [ ] **步骤 4：运行测试**

运行：`pytest tests/simulation/test_pybullet_motion_control.py tests/simulation/test_pybullet_safe_motion_runner.py -v`

预期：全部通过，阶段 1 的轨迹字段和门控保持兼容。

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/motion_control.py tests/simulation/test_pybullet_motion_control.py
git commit -m "feat: track target poses during Panda motion"
```

### 任务 3：执行阶段 2 并保存证据

**文件：**
- 新建：`src/simulation/pybullet/run_truth_pregrasp.py`
- 修改：`tests/simulation/test_pybullet_truth_pregrasp.py`

**接口：**
- 新增：`TruthPregraspConfig(output_dir: Path = DEFAULT_OUTPUT_DIR, seed: int = 42, gui: bool = False, target_name: str = "cube", stability_steps: int = 60, maximum_target_displacement_m: float = 0.001)`
- 新增：`run_truth_pregrasp(config: TruthPregraspConfig) -> dict[str, object]`

- [ ] **步骤 1：写真实集成失败测试**

调用 `run_truth_pregrasp(TruthPregraspConfig(output_dir=tmp_path))`，断言生成非空的 `state_trace.csv`、`summary.json`、`metadata.json`、`start.png` 和 `pregrasp.png`；断言阶段为 `cube_truth_pregrasp`，IK/FK、静态余量、执行到位、XY、方块稳定、夹爪张开和总科学门控均通过；CSV 每行含 cube 位姿和 target-relative tool position；元数据中 perception、闭合、接触、抬升和物理抓取全部为 `false`。

- [ ] **步骤 2：确认测试按预期失败**

运行：`pytest tests/simulation/test_pybullet_truth_pregrasp.py -v`

预期：因 runner 模块不存在而失败。

- [ ] **步骤 3：实现真值输入和执行前门控**

复用 `fixed_scene_config` 并启用机器人自碰撞；settle 60 步后读取 cube body id、质心、姿态和 AABB 顶面。继续仿真 `stability_steps` 步，逐步计算相对首帧的最大质心位移，超过 `0.001 m` 时保存失败 summary/metadata/start image 且不发送电机命令。通过时以 cube 顶面中心和 cube 世界 X 轴生成俯视 pregrasp，执行 `audit_pose_ik` 和 `audit_joint_path_clearance` 后才能发送电机命令。

- [ ] **步骤 4：实现动态执行和科学门控**

调用 `execute_joint_motion` 执行唯一的 `pregrasp` 段，并用 `tracked_body_ids=(cube_id,)` 逐步记录方块。计算终点全姿态误差、末端与方块中心 XY 偏差、全过程方块最大位移、双指最大张开误差和碰撞计数。总门控必须同时要求：预稳定、IK/FK、静态余量、动态控制、`5 mm/5°`、XY `5 mm`、方块位移 `1 mm`、双指张开误差 `1 mm` 全部通过。

- [ ] **步骤 5：实现证据写入和阶段边界**

`state_trace.csv` 每行写关节、夹爪、末端、cube 真值位姿、末端相对 cube 位姿、余量和碰撞；metadata 明确写入 `truth_target_used=true`、`target_approach_executed=true`、`descent_to_contact_executed=false`、`gripper_close_commanded=false`、`contact_evaluated=false`、`object_lifted=false`、`physical_grasp_executed=false`。失败路径也必须产生 summary、metadata、空 trace 和 start image。

- [ ] **步骤 6：运行定向测试**

运行：`pytest tests/simulation/test_pybullet_truth_pregrasp.py tests/simulation/test_pybullet_motion_control.py tests/simulation/test_pybullet_safe_motion_runner.py -v`

预期：全部通过，真实固定场景阶段 2 科学门控为真。

- [ ] **步骤 7：提交**

```bash
git add src/simulation/pybullet/run_truth_pregrasp.py tests/simulation/test_pybullet_truth_pregrasp.py
git commit -m "feat: execute cube truth pregrasp stage"
```

### 任务 4：运行正式阶段 2 证据并更新中文记录

**文件：**
- 修改：`src/simulation/pybullet/README.md`
- 修改：`docs/agent/PROJECT_STRUCTURE.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/worklog/WORKLOG.md`
- 生成但不提交：`data/processed/pybullet/grasp_execution/stage_2_cube_pregrasp/*`

- [ ] **步骤 1：执行固定证据运行**

运行：`python -m src.simulation.pybullet.run_truth_pregrasp`

预期：输出 JSON 的 `scientific_gate_passed` 为 `true`；若为 `false`，停止并按 `superpowers:systematic-debugging` 查明原因，不实现阶段 3。

- [ ] **步骤 2：核验产物而非只看退出码**

运行：`python -m json.tool data/processed/pybullet/grasp_execution/stage_2_cube_pregrasp/summary.json`，并检查 CSV 行数、数值有限性、两张图像尺寸和 metadata 的未执行标志。

预期：所有门控与阶段边界一致，图像为 `640×480`，轨迹非空。

- [ ] **步骤 3：更新中文文档**

在 README 写运行命令与“不下降、不闭合”的边界；在项目结构中登记新 runner；在当前状态和工作日志中记录实际输出的步数、误差、最大方块位移、最大夹爪误差、最小余量和碰撞计数。不得写入未由本次产物支持的抓取成功结论。

- [ ] **步骤 4：运行完整回归测试**

运行：`pytest -q`

预期：全部测试通过。

- [ ] **步骤 5：提交**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md
git commit -m "docs: record cube truth pregrasp evidence"
```

## 自检结果

- 需求覆盖：计划覆盖真值 cube、俯视 pregrasp、张开夹爪、无下降/闭合、逐步目标稳定性、静态与动态碰撞、证据产物和中文项目记录。
- 边界检查：阶段 3 的下降、阶段 4 的闭合接触、阶段 5 的抬升均明确排除。
- 类型一致性：任务 2 的 `tracked_body_poses` 是任务 3 CSV 与方块位移门控的唯一动态数据来源；函数名和字段名在后续任务中一致。
- 占位符检查：所有实现步骤均已给出明确接口、操作与验收条件。
