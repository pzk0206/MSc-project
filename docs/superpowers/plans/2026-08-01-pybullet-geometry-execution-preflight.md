# PyBullet 几何感知执行计划预检实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：**在同一固定 PyBullet 场景内实时生成并静态验证一份 VLM + geometry
方块抓取执行计划，为下一阶段物理执行提供可审计输入。

**Architecture：**先把现有候选 IK/FK/碰撞审计从离线研究 runner 提升为共享函数；
再新增严格可加载的执行计划数据类型；最后新增单目标 Stage 6A runner，复用
现有相机、Grounding DINO、geometry、反投影和候选生成模块。runner 只写静态
证据，只有总门控通过才写 `execution_plan.json`。

**Tech Stack：**Python 3、PyBullet 3.2.7、NumPy、OpenCV、PyTorch、Transformers、
pytest。

## 全局约束

- 场景 seed 固定为 42，目标固定 `cube`，prompt 固定 `red cube`。
- 模型固定 `IDEA-Research/grounding-dino-tiny`，box/text threshold 均为 0.25。
- 后端固定 `geometry`，不加载 CNN，不允许回退真值或旧 CSV。
- 相机固定 640×480；深度只在二维预测后使用。
- segmentation、目标真值框和射线只作事后门控，不得修正预测。
- 接触前/抓取深度/pregrasp 偏移固定为 0.02/0.005/0.10 m。
- 静态碰撞余量固定 0.002 m；接触前与抓取深度各审计 41 个状态，每个候选
  共 82 个状态。接触前审计包含 cube，抓取深度审计只排除预期目标 cube。
- Stage 6A 不调用位置电机、连续轨迹、夹爪闭合、接触控制或抬升控制。
- 正式 VLM 证据必须使用真实 CUDA；无 CUDA 时失败，不静默回退 CPU。

---

### 任务 1：共享候选静态审计接口

**文件：**
- 修改：`src/simulation/pybullet/kinematic_audit.py`
- 修改：`src/simulation/pybullet/run_pose_ik_study.py`
- 测试：`tests/simulation/test_pybullet_kinematic_audit.py`
- 测试：`tests/simulation/test_pybullet_pose_ik_runner.py`

**接口：**
- 产生：`audit_pose_candidate(candidate: PoseCandidate, *, robot_id: int, client_id: int, model: PandaModelInfo, environment_body_ids: Sequence[int], allowed_environment_link_pairs: Sequence[tuple[int, int]] = (), physics: Any = p) -> CandidateAudit`
- 产生：`select_candidate_pair(audits: Sequence[CandidateAudit]) -> tuple[CandidateAudit, CandidateAudit]`（保持现有接口与确定性规则）。
- 消费：现有 `audit_pose_ik`、`audit_joint_path_clearance` 和 `CandidateAudit`。

- [x] **步骤 1：写共享接口失败测试**

在真实 Panda 测试场景中生成两个 cube 俯视候选，调用公开
`audit_pose_candidate`，断言 pregrasp/standoff IK 解均为七维、碰撞检查状态
为 41、输入关节状态被恢复；再构造两个同代价通过审计，断言
`select_candidate_pair` 只选 `0°`。

```python
audit = audit_pose_candidate(
    candidate,
    robot_id=scene.bodies.robot,
    client_id=scene.client_id,
    model=model,
    environment_body_ids=environment,
    allowed_environment_link_pairs=((-1, scene.bodies.table),),
)
assert len(audit.pregrasp_ik.solution) == 7
assert len(audit.standoff_ik.solution) == 7
assert audit.collision.checked_state_count == 41
```

- [x] **步骤 2：运行并确认按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_kinematic_audit.py -v
```

预期：因 `audit_pose_candidate` 尚未导出而导入失败。

- [x] **步骤 3：移动最小共享实现**

将 `run_pose_ik_study.py` 中 `_failed_collision`、`_joint_cost` 和
`_audit_candidate` 的职责移入 `kinematic_audit.py`，公开名为
`audit_pose_candidate`。保留相同公式、异常文本、关节恢复、阈值和
`select_candidate_pair` 行为；离线 runner 改为导入共享函数，不保留重复实现。

- [x] **步骤 4：运行定向与离线回归**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_kinematic_audit.py \
  tests/simulation/test_pybullet_pose_ik_runner.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/kinematic_audit.py \
  src/simulation/pybullet/run_pose_ik_study.py
git diff --check
```

- [x] **步骤 5：提交**

```bash
git add src/simulation/pybullet/kinematic_audit.py \
  src/simulation/pybullet/run_pose_ik_study.py \
  tests/simulation/test_pybullet_kinematic_audit.py
git commit -m "refactor: share pose candidate audit"
```

---

### 任务 2：严格的几何执行计划类型

**文件：**
- 新建：`src/simulation/pybullet/execution_plan.py`
- 新建：`tests/simulation/test_pybullet_execution_plan.py`

**接口：**
- 产生：`PROTOCOL_VERSION = "stage_6a_geometry_preflight_v1"`。
- 产生：`FrozenControlProtocol(approach_standoff_m: float = 0.02, grasp_depth_standoff_m: float = 0.005, pregrasp_offset_m: float = 0.10, collision_clearance_m: float = 0.002, samples_per_segment: int = 21, tool_lift_command_m: float = 0.12, minimum_object_lift_m: float = 0.10, lift_hold_steps: int = 240)`。
- 产生：`PerceptionEvidence`、`PlannedPoseCandidate`、`GeometryExecutionPlan` 冻结 dataclass。
- 产生：`write_geometry_execution_plan(path: Path, plan: GeometryExecutionPlan) -> None`。
- 产生：`load_geometry_execution_plan(path: Path) -> GeometryExecutionPlan`。
- 消费：`ToolPose`、`CandidateAudit` 及 JSON 基本类型。

- [x] **步骤 1：写配置与往返失败测试**

测试构造两个候选和一个唯一选中候选，写入再读取后要求完全相等；分别篡改
protocol、backend、seed、候选数、selected 数、NaN、IK 长度、三个位姿高度
关系或任一门控，加载必须抛出明确 `ValueError`。

```python
write_geometry_execution_plan(path, plan)
assert load_geometry_execution_plan(path) == plan

payload = json.loads(path.read_text())
payload["backend"] = "multi_head"
path.write_text(json.dumps(payload))
with pytest.raises(ValueError, match="backend must be geometry"):
    load_geometry_execution_plan(path)
```

- [x] **步骤 2：运行并确认按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_execution_plan.py -v
```

预期：因 `execution_plan` 模块不存在而导入失败。

- [x] **步骤 3：实现不可变计划、JSON 转换和加载校验**

所有浮点必须有限；四元数必须归一；每个候选必须包含七维 pregrasp、approach
和 grasp-depth IK；候选顺序必须为 `0.0, 180.0`；只有一个候选 selected 且其
全部门控为真。验证同一候选三个位姿 XY 与四元数相同，并满足：

```python
pregrasp_z - approach_z == pytest.approx(0.10)
approach_z - grasp_depth_z == pytest.approx(0.015)
```

`write_geometry_execution_plan` 使用 `allow_nan=False` 和 UTF-8，加载拒绝额外或
缺失顶层字段。冻结控制协议值必须与全局约束逐项相等，防止下一阶段消费漂移。

- [x] **步骤 4：运行定向测试和编译**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_execution_plan.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/execution_plan.py
git diff --check
```

- [x] **步骤 5：提交**

```bash
git add src/simulation/pybullet/execution_plan.py \
  tests/simulation/test_pybullet_execution_plan.py
git commit -m "feat: define frozen geometry execution plan"
```

---

### 任务 3：同场景 Stage 6A runner

**文件：**
- 新建：`src/simulation/pybullet/run_geometry_execution_preflight.py`
- 新建：`tests/simulation/test_pybullet_geometry_execution_preflight.py`

**接口：**
- 产生：`GeometryPreflightConfig(output_dir: Path = Path("data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight"), seed: int = 42, gui: bool = False, device: str = "cuda", target_name: str = "cube", prompt: str = "red cube", backend: str = "geometry", model_id: str = "IDEA-Research/grounding-dino-tiny", width: int = 640, height: int = 480, box_threshold: float = 0.25, text_threshold: float = 0.25, iou_threshold: float = 0.25)`。
- 产生：`GeometryPreflightDependencies(scene_factory, capture_frame, load_detector, localize, predict, ray_test)`。
- 产生：`run_geometry_execution_preflight(config: GeometryPreflightConfig, dependencies: GeometryPreflightDependencies | None = None) -> dict[str, object]`。
- 消费：任务 1 的 `audit_pose_candidate/select_candidate_pair`；任务 2 的计划类型；现有 `CameraConfig`、`predict_grasp`、`audit_backprojected_grasp`、`generate_top_down_pose_candidates`、`evaluate_target_selection`。

- [x] **步骤 1：写同场景真实静态失败测试**

测试注入固定 localization 和 geometry predictor，但使用真实
`PyBulletScene/capture_camera_frame`、真实深度、反投影、IK/FK 和碰撞：

```python
summary = run_geometry_execution_preflight(
    GeometryPreflightConfig(output_dir=tmp_path, device="cpu"),
    dependencies=fixed_prediction_dependencies(),
)
assert summary["scientific_gate_passed"] is True
assert summary["selected_candidate_count"] == 1
assert summary["simulation_setup_steps"] == 60
assert summary["simulation_steps_after_capture"] == 0
assert (tmp_path / "execution_plan.json").is_file()
```

读取计划并断言 backend 为 geometry、RGB SHA-256 与文件一致、候选为
`0°/180°`、所有输入有限。metadata 必须明确
`motor_control_executed/trajectory_executed/gripper_closed/contact_evaluated/object_lifted/physical_grasp_executed=false`。

- [x] **步骤 2：写失败保真测试**

先在目录放置旧 `execution_plan.json`，再分别注入无检测、geometry 异常和
目标错误 localization。每次运行后旧计划必须被删除，summary/metadata 保留
准确 `failure_stage`，所有执行标志为假。

- [x] **步骤 3：运行并确认按预期失败**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_geometry_execution_preflight.py -v
```

预期：因 runner 模块不存在而导入失败。

- [x] **步骤 4：实现配置、依赖和输出准备**

配置拒绝非 seed 42、非 cube/red cube/geometry、非法阈值或非 640×480 固定
协议。开始运行时创建目录并删除所有旧 Stage 6A 文件，尤其是旧计划。真实依赖
只加载 Grounding DINO；geometry 的 model 参数必须是 `None`。

- [x] **步骤 5：实现感知、三维与候选门控**

场景只调用 `scene.step(60)` 一次；拍摄后不再 step。保存原始帧，运行定位和
geometry，并用现有审计函数产生 target selection、backend audit 和
backprojection audit。使用 backprojection 的 sampled column/row/depth 与
geometry angle 生成两个候选；对每个候选额外用相同世界表面点/方向生成
`0.005 m` grasp-depth pose并求 IK。接触前候选对包含 cube 的完整环境审计 41
状态；抓取深度候选对只排除预期目标 cube、继续审计其他环境与自碰撞 41 状态。
合并两组门控和 82 状态统计后再执行确定性选择。

- [x] **步骤 6：实现计划、summary、metadata 和可视化**

仅当感知、反投影、两个候选审计和唯一选择门控全部满足时写计划。summary 至少
记录 detection IoU、geometry 参数、world point、两个候选门控、选择角、最小
余量和总门控。metadata 记录相机矩阵、scene object poses、数据流边界、RGB
哈希以及全 false 执行标志。保存 localization 和 geometry prediction 图。

- [x] **步骤 7：运行 Stage 6A 与既有静态审计回归**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_geometry_execution_preflight.py \
  tests/simulation/test_pybullet_execution_plan.py \
  tests/simulation/test_pybullet_pose_ik_runner.py \
  tests/simulation/test_pybullet_multi_object_runner.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/run_geometry_execution_preflight.py
git diff --check
```

- [x] **步骤 8：提交**

```bash
git add src/simulation/pybullet/run_geometry_execution_preflight.py \
  tests/simulation/test_pybullet_geometry_execution_preflight.py
git commit -m "feat: preflight geometry grasp execution plan"
```

---

### 任务 4：正式 CUDA 证据与中文记录

**文件：**
- 修改：`src/simulation/pybullet/README.md`
- 修改：`docs/agent/PROJECT_STRUCTURE.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/agent/vlm_robotic_grasp_study_plan.md`
- 修改：`docs/worklog/WORKLOG.md`
- 修改：本实施计划复选框
- 生成但不提交：`data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight/*`

- [ ] **步骤 1：运行正式真实模型证据**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m \
  src.simulation.pybullet.run_geometry_execution_preflight --device cuda
```

若沙箱隐藏 GPU，使用同一命令的受控外部执行权限；不得改为 CPU 后称正式 VLM
证据。若真实检测、反投影或静态预检失败，保存失败并停止，不开始 Stage 6B。

- [ ] **步骤 2：核验产物与图像**

独立读取 summary、metadata、candidate CSV 和计划，核对 RGB 哈希、同一场景、
零 capture 后 step、两个候选、唯一选择、每候选 82 状态、目标/射线审计和全部非执行
标志。人工查看 localization/geometry prediction 是否确实位于红色方块。

- [ ] **步骤 3：更新中文记录**

只记录真实正式输出：检测框/分数/IoU、geometry 参数、世界点、选中对称角、
IK/FK 误差、最小余量及总门控。明确 Stage 6A 尚未运动或抓取，下一步仅为
Stage 6B 消费冻结计划。

- [ ] **步骤 4：运行完整验证**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest -q
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/execution_plan.py \
  src/simulation/pybullet/run_geometry_execution_preflight.py
git diff --check
```

- [ ] **步骤 5：提交中文证据记录**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md \
  docs/agent/CURRENT_STATUS.md docs/agent/vlm_robotic_grasp_study_plan.md \
  docs/worklog/WORKLOG.md \
  docs/superpowers/plans/2026-08-01-pybullet-geometry-execution-preflight.md
git commit -m "docs: record geometry execution preflight evidence"
```

## 自检结果

- 设计中的同场景实时感知、无真值修正、静态安全审计、失败保真和非执行边界
  分别由任务 3 的步骤 4--6 覆盖。
- 任务 1 产生的 `audit_pose_candidate` 与任务 3 消费接口一致；任务 2 的计划
  类型同时供任务 3 写入和未来 Stage 6B 加载。
- 所有阈值与设计一致：0.25 检测阈值、0.02/0.005/0.10 m 位姿偏移、2 mm
  余量、两组各 41 状态、seed 42 和 CUDA 正式证据。
- 未包含多头 CNN、物理执行、协议批量化或成功率统计，范围保持为一个可独立
  验收的 Stage 6A。
