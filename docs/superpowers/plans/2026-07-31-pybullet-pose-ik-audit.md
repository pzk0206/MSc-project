# PyBullet Pose and Offline IK Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Convert each verified 2-D grasp rectangle into two deterministic top-down Panda hover-pose candidates, audit IK/FK and sampled collision clearance, and persist results without executing robot motion.

**Architecture:** Extend matrix backprojection with continuous image coordinates, keep pose generation pure in pose_generation.py, isolate all PyBullet state changes in kinematic_audit.py, and use a separate run_pose_ik_study.py runner that consumes the saved nine-point study. Static audit may reset joints in DIRECT but never calls motor control, steps candidate states, closes the gripper, or claims physical success.

**Tech Stack:** Python 3.10, NumPy, PyBullet 3.2.7, CSV/JSON, pytest, existing BackprojectionAudit, PyBulletScene, and fixed multi-object protocol.

## Global Constraints

- Fixed approach axis is world (0, 0, -1); no surface normal is estimated.
- Surface standoff is 0.02 m; pregrasp adds 0.10 m; image tangent offset is 5 px.
- Generate exactly two candidates per row: 0 and 180 degree symmetry only.
- Resolve Panda joints and panda_grasptarget by name, never hard-coded index.
- IK limits include all nine movable joints; only seven named arm values form an arm solution.
- FK thresholds: 0.005 m and 5 degrees; joint-limit tolerance: 1e-6 rad.
- Collision clearance: 0.002 m; sample 21 states per segment including endpoints.
- No setJointMotorControl calls, no stepSimulation during candidate audit, no gripper closure.
- Save and report a real failing gate; never relax thresholds to obtain a pass.

---

### Task 1: Continuous backprojection and pure pose generation

**Files:**
- Modify: src/simulation/pybullet/backprojection.py
- Create: src/simulation/pybullet/pose_generation.py
- Modify: tests/simulation/test_pybullet_backprojection.py
- Create: tests/simulation/test_pybullet_pose_generation.py

**Interfaces:**
- Produces: backproject_image_coordinate(pixel_x, pixel_y, depth_m, width, height, view_matrix, projection_matrix, near, far) -> BackprojectedPoint
- Produces: ToolPose(position, quaternion_xyzw)
- Produces: PoseCandidate(target, backend, symmetry_degrees, finger_axis_world, closing_axis_world, approach_axis_world, surface_standoff_pose, pregrasp_pose)
- Produces: generate_top_down_pose_candidates(...) -> tuple[PoseCandidate, PoseCandidate]

- [ ] **Step 1: Write continuous-coordinate RED tests**

Assert integer (0,0) matches backproject_pixel for a 1x1 camera; assert fractional coordinates round-trip; reject non-finite and outside-image values.

- [ ] **Step 2: Run RED**

    conda run -n msc-grasp python -m pytest tests/simulation/test_pybullet_backprojection.py -q

Expected: import failure for backproject_image_coordinate.

- [ ] **Step 3: Implement shared continuous backprojection**

Move NDC calculation into backproject_image_coordinate. Make backproject_pixel validate integer indices then delegate. Preserve the pixel-centre convention pixel + 0.5.

- [ ] **Step 4: Write pose-generation RED tests**

With downward and oblique cameras assert two candidates, world -Z approach, orthonormal rotation, standoff/pregrasp offsets, opposite symmetry axes, and fractional reprojection. Add 90-degree, border, NaN angle, zero tangent, and invalid-offset cases.

- [ ] **Step 5: Implement the exact interface**

    def generate_top_down_pose_candidates(
        *, target: str, backend: str, column: int, row: int,
        depth_m: float, angle_degrees: float, width: int, height: int,
        view_matrix: Sequence[float], projection_matrix: Sequence[float],
        near: float, far: float, tangent_offset_px: float = 5.0,
        surface_standoff_m: float = 0.02,
        pregrasp_offset_m: float = 0.10,
    ) -> tuple[PoseCandidate, PoseCandidate]:

Clip two continuous auxiliary points to image bounds, backproject at the centre depth, project the difference to XY, reject norm below 1e-8, and construct rotation columns [x_g, cross(z_g,x_g), z_g]. Require orthogonality and determinant within 1e-9.

- [ ] **Step 6: Run GREEN and commit**

    conda run -n msc-grasp python -m pytest tests/simulation/test_pybullet_backprojection.py tests/simulation/test_pybullet_pose_generation.py -q
    git diff --check
    git add src/simulation/pybullet/backprojection.py src/simulation/pybullet/pose_generation.py tests/simulation/test_pybullet_backprojection.py tests/simulation/test_pybullet_pose_generation.py
    git commit -m "feat: generate top-down Panda pose candidates"

---

### Task 2: Panda resolution and IK/FK audit

**Files:**
- Create: src/simulation/pybullet/kinematic_audit.py
- Create: tests/simulation/test_pybullet_kinematic_audit.py

**Interfaces:**
- Consumes: ToolPose and PoseCandidate
- Produces: PandaModelInfo with named arm, finger, movable and tool indices plus limits/ranges/rests
- Produces: IKPoseAudit(solution, limits_passed, position_error_m, orientation_error_degrees, fk_passed, failure_reason)
- Produces: resolve_panda_model(robot_id, client_id, physics=p)
- Produces: audit_pose_ik(robot_id, client_id, model, pose, physics=p)

- [ ] **Step 1: Write resolver RED tests**

Use a fake physics object with shuffled indices. Verify names determine seven arm joints, two fingers, nine movable order and tool link. Mutate a name/type/limit and require exact ValueError messages.

- [ ] **Step 2: Implement resolver**

Decode getJointInfo names, validate revolute arms and prismatic fingers, midpoint arm rests, 0.04 m finger rests, and finite ordered limits.

- [ ] **Step 3: Write IK/FK RED tests**

Cover nine-value IK mapping, limit violation, non-finite output, position error over 5 mm, orientation error over 5 degrees, and restoration after getLinkState raises. Fake motor and step methods must raise if called.

- [ ] **Step 4: Implement state-safe IK/FK**

Save nine movable states; call calculateInverseKinematics with nine limit/range/rest arrays, maxNumIterations=200 and residualThreshold=1e-5; reset seven arm and two open fingers; read FK; use 2*acos(abs(dot(q1,q2))); restore in finally.

- [ ] **Step 5: Add real DIRECT convention test**

Load the Panda, resolve names, and audit a reachable top-down pose at (0.45,0,0.85). Assert seven finite arm values, limit/FK pass, and original states restored.

- [ ] **Step 6: Run and commit**

    conda run -n msc-grasp python -m pytest tests/simulation/test_pybullet_kinematic_audit.py -q
    git diff --check
    git add src/simulation/pybullet/kinematic_audit.py tests/simulation/test_pybullet_kinematic_audit.py
    git commit -m "feat: audit Panda inverse kinematics"

---

### Task 3: Collision audit and candidate selection

**Files:**
- Modify: src/simulation/pybullet/scene.py
- Modify: src/simulation/pybullet/kinematic_audit.py
- Modify: tests/simulation/test_pybullet_smoke.py
- Modify: tests/simulation/test_pybullet_kinematic_audit.py

**Interfaces:**
- Adds: SceneConfig.robot_self_collision: bool = False
- Produces: CollisionAudit(clearance_passed, checked_state_count, minimum_clearance_m, environment_collision_count, self_collision_count, failure_reason)
- Produces: audit_joint_path_clearance(..., samples_per_segment=21, clearance_m=0.002)
- Produces: CandidateAudit and select_candidate_pair(audits)

- [ ] **Step 1: Write scene-flag RED test**

Patch loadURDF. Assert default Panda flags remain zero and opt-in passes p.URDF_USE_SELF_COLLISION.

- [ ] **Step 2: Implement opt-in self-collision loading**

Add the frozen config field and pass explicit flags only to the Panda load.

- [ ] **Step 3: Write collision RED tests**

Assert 41 unique states for two 21-state segments, open fingers, one collision detection per state, environment failures, adjacent self-pair filtering, non-adjacent self failure, forbidden step/motor calls, and state restoration after exceptions.

- [ ] **Step 4: Implement collision audit**

Use np.linspace and de-duplicate pregrasp. Query robot against plane/table/all objects with distance 0.002. Query robot against itself, ignoring identical and direct parent-child link pairs derived from joint metadata. Absence within the query radius means clearance is at least 2 mm.

- [ ] **Step 5: Implement deterministic selection**

Require symmetry order (0,180). Select fully passing candidate with minimum sum of squared arm displacement divided by squared joint range; tie-break on 0. Preserve both rows; select neither if both fail.

- [ ] **Step 6: Run and commit**

    conda run -n msc-grasp python -m pytest tests/simulation/test_pybullet_kinematic_audit.py tests/simulation/test_pybullet_smoke.py -q
    git diff --check
    git add src/simulation/pybullet/scene.py src/simulation/pybullet/kinematic_audit.py tests/simulation/test_pybullet_smoke.py tests/simulation/test_pybullet_kinematic_audit.py
    git commit -m "feat: audit Panda path collision clearance"

---

### Task 4: Independent pose/IK runner

**Files:**
- Create: src/simulation/pybullet/run_pose_ik_study.py
- Create: tests/simulation/test_pybullet_pose_ik_runner.py

**Interfaces:**
- Consumes: backend_results.csv, backprojection_results.csv and metadata.json
- Produces: pose_ik_candidates.csv, pose_ik_summary.json and pose_ik_metadata.json
- CLI: --input-dir and --output-dir, both default to the fixed multi-object directory

- [ ] **Step 1: Write input-contract RED tests**

Reject missing/reordered rows, false prior gate, target/backend mismatch, centre mismatch over 1e-9, missing angle, metadata dimensions other than 640x480, seed other than 42, and wrong object order. Assert no IK dependency is called on failure.

- [ ] **Step 2: Write output RED tests**

Inject fake scene/pose/IK/collision boundaries. Assert 18 rows in target/backend/symmetry order, nine selections on complete pass, exact counts and metadata flags. An all-fail fixture still writes 18 rows with scientific gate false.

- [ ] **Step 3: Implement runner and CLI**

Connect fixed scene with self collision, perform the same 60 setup steps before candidate audit, resolve Panda once, and audit 18 candidates. Do not call scene.step after candidate generation. Save failure metadata and always close the scene.

- [ ] **Step 4: Enforce non-execution metadata**

Record simulation_setup_steps=60 separately from simulation_stepped_during_candidate_audit=false. Record solver/static-reset true and motor/trajectory/gripper/physical flags false.

- [ ] **Step 5: Run and commit**

    conda run -n msc-grasp python -m pytest tests/simulation/test_pybullet_pose_ik_runner.py -q
    conda run -n msc-grasp python src/simulation/pybullet/run_pose_ik_study.py --help
    git diff --check
    git add src/simulation/pybullet/run_pose_ik_study.py tests/simulation/test_pybullet_pose_ik_runner.py
    git commit -m "feat: add offline pose and IK study runner"

---

### Task 5: Documentation and real audit

**Files:**
- Modify: src/simulation/pybullet/README.md
- Modify: docs/agent/PROJECT_STRUCTURE.md
- Modify after evidence: docs/agent/CURRENT_STATUS.md
- Modify: docs/worklog/WORKLOG.md
- Generate/ignore: data/processed/pybullet/multi_object_study/pose_ik_*

- [ ] **Step 1: Document command, outputs and exact boundaries**

Document 18-row schema, thresholds, setup steps, static resets, and seven non-execution flags. Update project structure for new modules/tests.

- [ ] **Step 2: Run full pre-audit verification**

    conda run -n msc-grasp python -m pytest -q
    git diff --check

Expected: at least 97 tests pass.

- [ ] **Step 3: Run real offline audit**

    conda run -n msc-grasp python src/simulation/pybullet/run_pose_ik_study.py --input-dir data/processed/pybullet/multi_object_study --output-dir data/processed/pybullet/multi_object_study

Independently check row order/count, finite values, thresholds, selection invariant, hashes and non-execution flags. Infrastructure may succeed while the scientific gate is false.

- [ ] **Step 4: Record only observed results**

Update status/worklog with actual pass/failure counts and reasons. Do not add unverified IK claims to the dissertation.

- [ ] **Step 5: Fresh final verification and commit**

    conda run -n msc-grasp python -m pytest -q
    git diff --check
    git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md
    git commit -m "docs: record offline Panda pose audit"

