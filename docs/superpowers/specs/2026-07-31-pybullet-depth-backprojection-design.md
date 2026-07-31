# PyBullet 深度反投影与九点门控设计

日期：2026-07-31

## 目标

在固定 PyBullet 多物体研究中，把三目标、三后端产生的九个二维抓取中心，
结合预测完成后取得的米制深度、相机 view matrix 和 projection matrix，转换为
相机坐标与世界坐标。该阶段只验证感知输出能否形成可信三维表面点，不实现
夹爪姿态、IK、碰撞检查、机械臂运动或物理抓取。

## 证据边界

- Grounding DINO 和 CNN 继续只接收 RGB。
- 深度只在二维抓取框已经产生后使用。
- segmentation 和 `rayTest` 只作事后真值审计，不参与深度采样、矩阵反演或
  世界坐标计算。
- 三维表面点通过门控不等于姿态可达、无碰撞或能够稳定抓取。
- 实现沿用 PyBullet 官方相机 API 的 OpenGL 矩阵与深度约定：
  <https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf>。

## 方案选择

### 采用：通用矩阵反投影

将米制深度恢复为 OpenGL depth buffer，再转换为 NDC 深度；随后使用
`inverse(projection @ view)` 恢复世界齐次坐标。PyBullet 返回的 16 元素矩阵
按 OpenGL column-major 顺序解释，因此 NumPy 使用 `order="F"` 重建 `4×4`
矩阵。

采用该方案是因为它直接使用运行时保存的 view/projection matrix，不把当前
固定 FOV、相机位置或主点假设写死到实现中，更适合后续更换相机。

### 未采用：由 FOV 手工推导内参

该方案代码较短，也便于解释，但依赖对称透视投影、固定主点和当前相机配置。
它可作为独立诊断公式，不作为正式输出来源。

### 未采用：邻域或 mask-aware 深度

邻域中位数能抗噪，mask-aware 采样还能避开背景，但会使用抓取中心之外的像素，
并可能让 segmentation 参与输入。正式结果固定采用最近像素；邻域方法不进入
第一版实现。

## 模块边界

新增 `src/simulation/pybullet/backprojection.py`，只负责纯坐标转换和一行结果
审计。现有 `run_multi_object_study.py` 继续负责场景生命周期、九个后端结果的
编排和产物保存。

主要接口：

```text
sample_nearest_depth(depth_m, center_x, center_y, near, far)
    -> sampled column、sampled row、metric depth

backproject_pixel(sampled_column, sampled_row, depth_m,
                  image_width, image_height,
                  view_matrix, projection_matrix, near, far)
    -> camera xyz、world xyz

reproject_world_point(world_xyz, image_width, image_height,
                      view_matrix, projection_matrix)
    -> floating pixel centre、metric camera depth

audit_backprojected_grasp(...)
    -> one immutable diagnostic record

summarize_backprojection_rows(rows)
    -> exact-nine-row gate summary
```

## 数学与坐标约定

对二维预测中心 `(x, y)`，最近像素固定为：

```text
column = floor(x + 0.5)
row    = floor(y + 0.5)
```

不使用 Python 的 banker rounding。中心必须在 `[0, width-1] × [0, height-1]`
内，采样深度必须有限且严格位于 near/far 裁剪面之间；等于 far 的背景深度视为
无效。

整数像素代表其像素中心，转换为 NDC：

```text
x_ndc =  2 * (column + 0.5) / width  - 1
y_ndc =  1 - 2 * (row + 0.5) / height
```

现有 `linearize_depth` 使用：

```text
z = far * near / (far - (far - near) * buffer)
```

正式反投影使用其代数逆式：

```text
buffer = (far - far * near / z) / (far - near)
z_ndc  = 2 * buffer - 1
```

世界点通过以下公式恢复：

```text
clip  = [x_ndc, y_ndc, z_ndc, 1]
world = inverse(projection @ view) @ clip
world = world[:3] / world[3]
```

相机坐标使用 `inverse(projection)` 以相同方式恢复。重投影必须执行相反过程，
用于捕获行列顺序、上下翻转、pixel-centre 和 column-major 解释错误。

## 九点数据流

固定研究先完成三条明确 prompt 和九个二维后端结果。随后对每个
`(target, backend)`：

1. 读取保存于该行的浮点抓取中心。
2. 用最近像素从同一 `CameraFrame.depth_m` 采样米制深度。
3. 仅使用深度和相机矩阵计算 camera/world xyz。
4. 将 world xyz 重投影，计算相对于被采样像素中心的像素误差和深度误差。
5. 计算完成后，才查询 segmentation 中的 body ID。
6. 从相机 eye 沿重建点方向发射稍微越过表面的 ray，记录命中的 body ID 与
   hit position。由于渲染 visual mesh 与 collision mesh 可能不同，hit-position
   距离只报告，不设硬阈值；目标 body 是否一致进入门控。

行顺序沿用现有 `TARGET_ORDER × BACKEND_ORDER`，不得缺行、重复或重排。

## 门控判据

第一版要求全部九行同时满足：

- 二维中心和采样索引有效；
- 深度有限且位于 `(near, far)`；
- camera/world xyz 和齐次除数有限；
- 重投影像素误差不超过 `1.0` pixel；
- 重投影深度绝对误差不超过 `1e-4` metre；
- 采样像素的 segmentation body ID 等于指定目标；
- `rayTest` 首次命中的 body ID 等于指定目标。

汇总写入 `backprojection_gate_passed`。门控失败不删除已经完成的二维结果，也
不把整个感知研究伪装成异常退出；它明确阻止后续 IK 阶段，并保存逐行失败原因。

## 输出

在现有多物体研究目录新增：

```text
backprojection_results.csv
backprojection_summary.json
```

CSV 每行至少包含：target、backend、原始中心、采样行列、depth、camera xyz、
world xyz、重投影坐标、pixel/depth error、segmentation body、ray body、
各布尔检查和失败原因。

JSON 至少包含：协议名、九行计数、各检查通过数、总门控状态、阈值、
`depth_used_after_2d_prediction: true`、
`segmentation_used_as_coordinate_input: false`、
`ray_test_used_as_coordinate_input: false`、
`ik_executed: false` 和 `physical_grasp_executed: false`。

主 metadata 增加上述两个输出路径和相同边界标志。

## 错误处理

纯函数对矩阵长度错误、奇异矩阵、非有限值、越界中心、无效深度和接近零的
齐次除数抛出带明确原因的 `ValueError`。九点编排层逐行捕获可预期的转换失败，
写出失败行与汇总，不丢失其他八行或原有二维产物。PyBullet API/场景生命周期
错误继续由现有 runner 的阶段化失败 metadata 处理。

## 测试策略

严格使用测试驱动开发：

1. 单元测试最近像素规则、边界与无效深度。
2. 单元测试 clip-plane 逆公式和 column-major 矩阵解释。
3. 使用已知矩阵与世界点验证 backproject/reproject 往返。
4. 测试奇异矩阵、非有限输入和零齐次除数。
5. 测试 exact-nine-row 顺序、计数、阈值和失败原因汇总。
6. runner 测试确认反投影只在九个二维结果完成后发生，输出 CSV/JSON，并保留
   segmentation/ray 为 audit-only 的 metadata。
7. DIRECT PyBullet 集成测试验证可见目标像素反投影后能重投影到原像素，并由
   segmentation/ray 命中同一 body。
8. 完整回归通过后，使用固定真实多物体产物运行九点审计；只有实际输出满足
   门控才更新项目状态与论文证据。

## 非目标

- 不把二维矩形角度直接解释为机械臂下降方向。
- 不从二维宽高推断夹爪物理开口。
- 不实现抓取姿态、IK、碰撞检查、轨迹、夹爪闭合或抬升检测。
- 不修改 Cornell 正式实验、模型权重、prompt 或阈值。
