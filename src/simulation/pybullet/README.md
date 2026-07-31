# PyBullet 仿真感知模块

本目录用于验证现有 Grounding DINO 定位和二维抓取后端能否接收机器人式
虚拟相机图像。当前环境使用 PyBullet `3.2.7`、API `202010061`。

场景、相机和 segmentation 接口基于 Erwin Coumans、Yunfei Bai 等维护的
[Bullet Physics / PyBullet 官方项目](https://github.com/bulletphysics/bullet3)
及其随包资源。这里没有复制或改编外部抓取执行代码。

## 两种运行方式

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

本模块已经实现二维抓取中心的深度反投影与事后目标表面审计，但尚未把单点
扩展为完整六自由度抓取位姿，也未实现逆运动学、碰撞规划、机械臂控制、夹爪
闭合或物理抓取成功判定。任何九点门控通过都只能作为进入姿态与 IK 设计的
前置证据。
