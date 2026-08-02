# PyBullet 仿真感知模块

本目录用于验证现有 Grounding DINO 定位和二维抓取后端能否接收机器人式
虚拟相机图像。当前环境使用 PyBullet `3.2.7`、API `202010061`。

场景、相机和 segmentation 接口基于 Erwin Coumans、Yunfei Bai 等维护的
[Bullet Physics / PyBullet 官方项目](https://github.com/bulletphysics/bullet3)
及其随包资源。这里没有复制或改编外部抓取执行代码。

## 九种运行方式

单物体感知诊断：

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend geometry --device cuda --prompt "yellow rubber duck"
```

固定多物体目标选择研究：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py \
  --device cuda \
  --output-dir data/processed/pybullet/multi_object_study
```

在保存的九点门控结果上运行独立静态姿态/IK 审计：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_pose_ik_study.py \
  --input-dir data/processed/pybullet/multi_object_study \
  --output-dir data/processed/pybullet/multi_object_study
```

分阶段物理执行的阶段 1 安全空中运动冒烟：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_safe_motion_smoke.py \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_1_safe_motion
```

该命令先用静态 IK/FK 和 `2 mm` 碰撞余量预检一个比中立末端高 `0.05 m`
的安全空中点，只有预检通过才通过 Panda 位置电机执行“中立位→安全点→中立
位”。执行中逐步调用 `stepSimulation`，手指保持 `0.04 m` 张开，并保存状态
轨迹、碰撞统计、summary、metadata 和三个关键帧。该阶段不靠近目标、不闭合
夹爪、不评价接触或抬升，因此不能称为仿真抓取。

分阶段物理执行的阶段 2 方块真值 pregrasp：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_pregrasp.py \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_2_cube_pregrasp
```

该命令只读取 PyBullet 中红色方块的真值中心、姿态和 AABB 顶面，不运行
Grounding DINO、几何后端或 CNN。程序先检查方块稳定性、末端 IK/FK 和
`2 mm` 静态碰撞余量，再让张开的 Panda 移动到方块顶面以上 `0.12 m` 的
俯视 pregrasp；轨迹逐步保存方块真值位姿和末端相对方块的位置。该阶段不下降
到接触高度、不闭合、不评价接触或抬升，因此仍不能称为仿真抓取成功。

分阶段物理执行的阶段 3 张开夹爪垂直接近：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_approach.py \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_3_open_approach
```

该命令复用阶段 2 的真值 cube 姿态，先到达顶面以上 `0.12 m` 的 pregrasp，
只有该段动态门控通过后才沿世界 Z 方向下降到顶面以上 `0.02 m`。双指全程
保持 `0.04 m` 张开，并保存两段轨迹、空接触事件表和三个关键帧。本阶段没有
下降到接触、没有闭合或评价接触，也没有抬升物体。

分阶段物理执行的阶段 4 方块双指闭合接触：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_contact.py \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_4_bilateral_contact
```

该命令完整重放阶段 2/3 门控，再让张开的夹爪从 cube 顶面以上 `0.02 m`
短距离下探到 `0.005 m` 抓取深度。只有下探到达、方块稳定、双指张开和零禁止
碰撞门控通过后才缓慢闭合；首次双指同时产生有限正法向力后冻结闭合命令并
保持 120 步。正式运行在闭合第 93 步取得双指接触，末段连续保持 121 步，
左右最大法向力为 `20.49 N/19.15 N`，禁止碰撞为零。本阶段不抬升，所以只能
称为“双指目标接触通过”，不能称为仿真抓取成功。

分阶段物理执行的阶段 5 真值方块抬升与保持：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_truth_lift.py \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_5_truth_cube_lift
```

该命令完整重放阶段 2--4，在双指接触门控通过后冻结最终夹爪命令，使工具沿
世界 Z 上升 `0.12 m`。240 个插值步后最多使用 240 个 settle 步达到既定
`0.01 rad` 关节终点门槛，再完整保持 240 步。正式运行使用 2 个 settle 步；
保持段 cube 最小/最终上升量为 `0.116738 m/0.120003 m`，桌面接触为零，
最大末端—cube 相对漂移为 `0.001410 m`。closed、lifted、lift-hold 图已人工
确认方块被双指夹持、离桌并保持。该结果是一例真值姿态仿真抓取成功，不是
geometry 或 CNN 感知后端成功率。

Stage 6A VLM + geometry 同场景执行计划预检：

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_geometry_execution_preflight.py \
  --device cuda \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight
```

该命令在同一个固定场景中采集一次 RGB、深度和 segmentation，实时运行
`red cube` Grounding DINO 定位和 geometry 后端，再将二维中心反投影为世界
表面点。segmentation 和射线只作预测后的目标审计，不修正中心或角度。两个
`0°/180°` 候选各执行两组静态安全检查：接触前 41 状态包含 cube，抓取深度
41 状态只排除预期目标 cube；桌面、鸭、球和自碰撞继续检查。

2026-08-01 的正式 CUDA 运行中，定位框为 `[297,189,344,245]`、分数
`0.8170`、cube IoU `0.8717`；geometry 中心为 `(320.5,217.0)`、角度 `0°`。
两个候选均通过各 82 状态门控，确定性选择 `0°`，并写出严格校验的
`execution_plan.json`。该阶段没有电机、闭合、接触或抬升，只能称为“几何
感知执行计划通过静态预检”，不能称为仿真抓取成功。

Stage 6A.1 冻结产物中心偏差诊断：

```bash
conda run -n msc-grasp python -m \
  src.simulation.pybullet.run_center_bias_diagnostic \
  --source-dir \
  data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight \
  --output-dir \
  data/processed/pybullet/grasp_execution/stage_6a1_center_bias_diagnostic \
  --evidence-role formal
```

该命令只读取 Stage 6A 的 summary、metadata、冻结计划和 RGB，不加载模型、
不创建 PyBullet 场景，也不调用任何运动或抓取接口。它先校验四个来源文件的
协议、RGB 哈希、世界点和非执行边界，再使用 metadata 中拍摄时保存的 cube
质心与冻结 `0.025 m` 半高计算名义顶面参考。正式结果的 XY 偏差为
`0.0265495568 m`，超过 `0.005 m` 参考门槛；名义顶面 Z 偏差为
`0.0005091724 m`。诊断前后 Stage 6A 全目录哈希清单一致，且所有运动和抓取
标志仍为 `false`。

另一次真实 CUDA Stage 6A 重跑保存在独立的
`stage_6a_geometry_preflight_reproducibility/`，再由相同诊断器写入
`stage_6a1_center_bias_reproducibility/`。其 RGB 哈希、定位、世界表面点和
全部中心偏差数值与正式运行一致；该结果只证明当前固定协议可重复，不修改原
计划，也不构成抓取成功。

上述静态审计命令为每条二维抓取生成 `0°/180°` 两个世界 `-Z` 俯视候选，检查 Panda
七关节 IK、`5 mm/5°` FK 误差和 `2 mm` 碰撞余量。两段关节插值各采样
21 个状态，共 41 个唯一状态；状态只通过 DIRECT 中的 `resetJointState`
静态检查，不调用电机控制、不执行轨迹、不闭合夹爪。

该命令还会在三个正确目标上并列运行：

- `geometry`；
- `single`：
  `data/processed/vlm/cnn_grasp_single_head_deterministic/cnn_grasp_model_seed_42.pt`；
- `multi_head`：
  `data/processed/vlm/cnn_grasp_multi_head_deterministic/cnn_grasp_model_seed_42.pt`。

两个 CNN 各加载一次，三个后端复用完全相同的 RGB 和 Grounding DINO
`Localization`。如需显式复现其他已知权重路径，可用 `--single-weights` 和
`--multi-head-weights`；程序不会自动换 seed 或回退到 geometry。

沙箱可能隐藏 `/dev/dxg`，使同一 Conda 环境中的
`torch.cuda.is_available()` 返回 `False`。请求 `--device cuda` 时程序会
直接失败，不会静默回退 CPU；真实 GPU 运行应在可访问显卡的环境中执行。

## 固定研究协议

场景包含黄色鸭、红色方块、绿色球体和作为 distractor 的 Franka Panda。
一次渲染后只加载一次 Grounding DINO，并按固定顺序运行：

1. `yellow rubber duck`
2. `red cube`
3. `green sphere`
4. `small object`（只作 generic prompt 诊断）

PyBullet segmentation 只用于运行后的真值框和 mask 评价，不进入 Grounding
DINO、几何抓取后端或模型 prompt。主目标判为正确需同时满足：

- 预测框与指定目标真值框的 IoU 至少为 `0.25`；
- 指定目标是鸭、方块、球和 Panda 四个实体中的唯一最佳匹配。

这里的 `0.25` 是固定 pilot 的工程门槛，不是 Cornell 抓取矩形指标。generic
prompt 不计入三目标成功率。无检测、低 IoU、选错物体和并列匹配都保留为
实验结果，不通过 segmentation 修正模型输出。

只有正确选择的主目标会进入三个二维抓取后端。系统先检查抓取中心是否位于
目标 mask 内，再以最近像素深度和保存的 PyBullet view/projection 矩阵恢复
相机与世界坐标。segmentation 和 `rayTest` 只在坐标产生后作真值审计，绝不
用于移动、修正或选择三维点。

固定九点门控要求三个目标乘三个后端的结果完整且顺序确定，并逐点满足：深度
有限且位于裁剪面之间、坐标有限、重投影误差不超过 1 pixel、深度往返误差
不超过 `1e-4 m`、segmentation body 与目标一致、射线首次命中目标 body。
该门控验证二维中心到目标表面三维点的转换，不等同于抓取位姿或抓取成功。

## 输出

```text
data/processed/pybullet/multi_object_study/
├── rgb.png
├── depth.npy
├── depth_visualization.png
├── segmentation.png
├── ground_truth_boxes.png
├── results.csv
├── backend_results.csv
├── backend_comparison.json
├── backprojection_results.csv
├── backprojection_summary.json
├── pose_ik_candidates.csv
├── pose_ik_summary.json
├── pose_ik_metadata.json
├── summary.json
├── metadata.json
└── targets/
    ├── duck/
    ├── cube/
    ├── sphere/
    └── generic/evaluation.png
```

三个 main 目标目录各包含：

```text
evaluation.png
prediction.png
geometry_prediction.png
single_prediction.png
multi_head_prediction.png
backend_comparison.png
```

`prediction.png` 与 `geometry_prediction.png` 内容相同，前者保留旧输出兼容性。
`backend_results.csv` 每个目标和后端一行，记录有限性、正尺寸、中心是否在
目标 mask 以及旋转框是否在图像范围内。`backend_comparison.json` 只汇总
这些几何诊断，不使用 Cornell 真值、不生成最佳后端或性能排名。

`backprojection_results.csv` 保存九个二维中心的采样像素、米制深度、相机与
世界坐标、重投影误差、segmentation/ray 命中及失败原因。
`backprojection_summary.json` 保存九点完整性和门控计数。若目标选择或后端
输出不完整，程序仍保留已有行，但 `backprojection_complete` 与总门控均为
`false`。

`pose_ik_candidates.csv` 保存 18 个对称候选的世界姿态、七关节解、FK 误差、
41 状态碰撞统计、选择标志和失败原因。`pose_ik_summary.json` 保存分阶段计数，
`pose_ik_metadata.json` 保存输入 SHA-256、固定阈值和非执行边界。固定底座与
桌面的安装接触以及固定关节刚体内部接触不计为候选碰撞；目标物体不豁免。

2026-07-31 的真实 DIRECT 审计中，18/18 候选通过 IK/FK，12/18 通过碰撞
余量，6/9 个输入选出候选。三个 duck 输入均因环境余量失败，因此总科学
门控为 `false`；该结果不作物理抓取成功率解释。

`depth.npy` 保存米制深度；PNG 深度图只是固定近远裁剪面的诊断显示。
`metadata.json` 明确记录：

```json
{
  "segmentation_used_as_model_input": false,
  "depth_used_after_2d_prediction": true,
  "segmentation_used_as_coordinate_input": false,
  "ray_test_used_as_coordinate_input": false,
  "ik_executed": false,
  "performance_ranking_computed": false,
  "physical_grasp_executed": false
}
```

## 当前边界

本模块已经实现二维抓取中心反投影、确定性六自由度悬停姿态、离线 IK/FK、
离散碰撞余量审计、阶段 1 安全空中电机往返、阶段 2 真值方块上方 pregrasp，
阶段 3 张开夹爪的接触前垂直接近、阶段 4 的开放夹爪短下探与双指接触，以及
阶段 5 的真值方块抬升和保持、Stage 6A 的 VLM + geometry 同场景感知与静态
执行计划预检，以及 Stage 6A.1 的冻结产物离线中心偏差诊断。阶段 1--5 已支持
一例真值姿态仿真抓取成功判定；Stage 6A/6A.1 均未驱动机械臂，且已确认原始
表面点不满足 `5 mm` XY 中心参考。多头 CNN 也尚未接入，因此当前仍没有感知
后端仿真抓取成功率。下一步必须先冻结“原样执行”或“为两个后端共同修订中心
恢复规则”的选择，再进入 Stage 6B。
