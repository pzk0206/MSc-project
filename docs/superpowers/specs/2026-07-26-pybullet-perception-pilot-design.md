# PyBullet 感知 Pilot 设计

日期：2026-07-26

## 目标

建立一个范围受控、可重复运行的 PyBullet 仿真感知演示：

```text
PyBullet 场景
→ 固定虚拟相机采集 RGB、深度和 segmentation
→ Grounding DINO 定位目标
→ 现有几何或 CNN 后端预测二维抓取矩形
→ 保存预测可视化和运行元数据
```

该 pilot 用于回应导师关于“虚拟相机图像接入现有抓取流程”的建议。第一阶段
只验证感知管线和可视化，不进行像素到机器人坐标转换、逆运动学、碰撞规划、
夹爪闭合或物理抓取执行。

## 范围

### 本次实现

- 在 PyBullet 中加载平面、桌面、Franka Panda 和默认测试物体
  `duck_vhacd.urdf`。
- 默认使用无窗口的 `DIRECT` 模式，并通过 `--gui` 支持交互窗口。
- DIRECT 模式固定使用 `ER_TINY_RENDERER`，GUI 模式使用
  `ER_BULLET_HARDWARE_OPENGL`。
- 使用固定相机参数采集 RGB、线性深度和 segmentation。
- 把 RGB 图像交给现有 Grounding DINO 单图推理接口。
- 使用固定通用提示 `small object`，并沿用当前框阈值、文本阈值与 10% 扩框。
- 支持 `geometry`、`single` 和 `multi_head` 三种抓取后端选择。
- 默认后端为 `geometry`，用于最先验证不依赖 CNN 权重的完整数据流。
- 在完整 RGB 图像上绘制定位框、抓取矩形、中心和方向。
- 保存图像、数值结果、配置、版本和随机种子。
- 提供不依赖 GUI 的单元测试和 PyBullet 集成冒烟测试。

### 本次不实现

- 不训练或微调 Grounding DINO。
- 不使用仿真数据重新训练 CNN。
- 不把 segmentation 真值作为感知算法输入。
- 不把仿真目标真值框替代 Grounding DINO 输出。
- 不实施像素坐标到世界坐标的抓取执行。
- 不实现预抓取、闭环重估、失败恢复或抓取成功判定。
- 不把单个仿真案例解释为真实机器人成功率或泛化证据。

## 目录设计

代码放在：

```text
src/
└── simulation/
    ├── __init__.py
    └── pybullet/
        ├── __init__.py
        ├── scene.py
        ├── camera.py
        ├── perception.py
        ├── visualization.py
        ├── run_pilot.py
        └── README.md
```

不直接创建 `src/pybullet/`，避免在 `PYTHONPATH=src` 等运行方式下遮蔽官方
`pybullet` 包。

测试放在：

```text
tests/
└── simulation/
    ├── test_pybullet_camera.py
    ├── test_pybullet_visualization.py
    └── test_pybullet_smoke.py
```

输出放在被 Git 忽略的数据目录：

```text
data/processed/pybullet/pilot/
├── rgb.png
├── depth.npy
├── depth_visualization.png
├── segmentation.png
├── localization.png
├── prediction.png
└── metadata.json
```

## 组件职责

### `scene.py`

负责仿真连接和场景生命周期：

- `SceneConfig` 保存连接模式、时间步、重力、机器人和物体姿态。
- `PyBulletScene` 显式保存 `physics_client_id`。
- 加载 `plane.urdf`、`table/table.urdf`、`franka_panda/panda.urdf` 和
  PyBullet 自带的默认测试物体 `duck_vhacd.urdf`。
- 使用固定 seed 和显式初始姿态，避免每次运行场景不同。
- 提供 `step()`、`reset()` 和 `close()`。
- 所有 PyBullet API 调用显式传入 client ID，避免多连接时污染状态。

第一阶段机器人仅作为场景上下文，不执行运动。

### `camera.py`

负责相机模型与输出转换：

- `CameraConfig` 保存宽、高、视点、目标点、上方向、视场角、近远裁剪面。
- 生成 view matrix 和 projection matrix。
- 调用 PyBullet 相机 API 返回 RGB、原始深度缓冲和 segmentation。
- 按 PyBullet 透视深度公式把深度缓冲转换为米制线性深度。
- 校验数组形状、数值有限性和深度范围。

相机默认俯视桌面，分辨率使用 `640×480`，与通用二维感知和结果展示兼容。

### `perception.py`

只做仿真图像到现有感知接口的适配，不复制模型实现：

- 将 RGB 数组保存为本次运行的图像文件。
- 复用 `src.vlm.run_grounding_dino_localization.run_grounding_dino_on_image`。
- 复用现有提示规范化、候选选择、框解析和 10% 扩框逻辑。
- `geometry` 后端复用
  `src.vlm.run_vlm_assisted_grasp.predict_grasp_with_vlm_box`。
- `single` 和 `multi_head` 复用现有 CNN 类、
  `crop_to_tensor` 和 `predict_from_crop`。
- CNN 权重必须由命令行显式指定，或使用文档中列出的正式默认权重；
  加载时校验架构与 state dict。
- 返回统一的 `PilotPrediction`，包含定位框、置信度、后端、抓取中心、尺寸和角度。

Grounding DINO 模型在一次运行中只加载一次。设备参数支持 `cuda` 和 `cpu`；
请求 CUDA 但不可用时直接报错，不静默改用 CPU。

### `visualization.py`

负责纯 OpenCV 绘制：

- 黄色定位框；
- 蓝色抓取矩形；
- 抓取中心点；
- 方向线；
- prompt、置信度和后端名称。

绘制函数不依赖 PyBullet 或 Torch，方便独立测试。输入矩形无效或越界时抛出
清晰异常，不生成具有误导性的图片。

### `run_pilot.py`

提供单一命令行入口：

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend geometry \
  --device cuda
```

主要参数：

- `--gui`：使用 PyBullet GUI；默认 DIRECT。
- `--backend {geometry,single,multi_head}`。
- `--device {cpu,cuda}`。
- `--prompt`，默认 `small object`。
- `--object-urdf`，默认 `duck_vhacd.urdf`，仅接受
  `pybullet_data` 中可解析的资源。
- `--model-weights`，CNN 后端时使用。
- `--output-dir`。
- `--seed`。
- `--width`、`--height`。

入口按“场景→相机→感知→绘图→元数据”的固定顺序执行，并通过 `try/finally`
保证关闭仿真连接。

## 数据与坐标约定

- PyBullet 相机返回 RGBA；保存和传入模型前转换为 RGB。
- OpenCV 绘制前显式转换为 BGR，避免颜色通道颠倒。
- 所有二维框使用图像坐标：原点在左上，$x$ 向右，$y$ 向下。
- Grounding DINO 框使用 `(x1, y1, x2, y2)`。
- 抓取结果使用 `(center_x, center_y, width, height, angle_degrees)`。
- 深度数组保存为米，无法观测的像素接近远裁剪面。
- Segmentation 只用于检查物体是否进入画面和未来诊断，不进入预测。

## 元数据

`metadata.json` 至少记录：

- 运行时间和随机种子；
- PyBullet API/version；
- 连接模式和 renderer；
- 机器人、桌面和物体资源名称及初始姿态；
- 相机内外参数、近远裁剪面和图像尺寸；
- prompt、阈值、设备和 Grounding DINO checkpoint；
- 后端名称及 CNN 权重路径；
- 定位框、置信度和抓取矩形；
- 所有输出文件路径；
- 明确的 `physical_grasp_executed: false`。

## 错误处理

- 未安装 PyBullet：给出在 `msc-grasp` 环境安装依赖的命令。
- URDF 或 `pybullet_data` 不可用：报告具体资源路径。
- 相机输出形状错误或含非有限值：停止运行。
- Grounding DINO 未检测到目标：仍保存原始相机输出和失败元数据，返回非零退出码。
- CNN 后端缺少权重或架构不匹配：在模型推理前停止。
- 输出目录已存在：覆盖本次固定文件，但不删除目录中的其他文件。
- GUI 在无显示环境不可用：提示改用默认 DIRECT。

## 测试策略

### 单元测试

- `CameraConfig` 拒绝非正尺寸、非法 near/far 和无效视场角。
- 深度缓冲转换在 near/far 边界与中间值上正确、有限且单调。
- RGBA→RGB 和 RGB→BGR 通道转换正确。
- 抓取矩形顶点与方向线在已知角度下正确。
- 无效框、NaN 和负尺寸被拒绝。
- 元数据包含全部必需字段且可 JSON 序列化。

### PyBullet 冒烟测试

使用 DIRECT 模式：

- 成功连接和断开。
- 场景包含机器人、桌面和物体。
- 相机返回 `480×640×3` RGB、`480×640` 深度和 segmentation。
- RGB、深度均为有限值，深度位于 near/far 范围。
- 简单物体在 segmentation 中可见。

模型级冒烟不进入普通快速测试，因为 Grounding DINO 需要模型缓存和较多显存；
它通过显式命令单独运行并保存产物。

## 依赖与来源

- 将 PyBullet 安装到现有 `msc-grasp` Conda 环境，并在完成后记录准确版本。
- 只使用 PyBullet 官方 Python API、`pybullet_data` 自带 URDF 和项目现有模型。
- 若实现参考或改编 PyBullet 官方示例中的相机、深度或场景代码，在相关文件
  顶部注明官方来源链接和改编范围。
- 不复制未注明出处的第三方抓取执行代码。

## 验收标准

完成需同时满足：

1. `src/simulation/pybullet/` 模块职责清晰，可从仓库根目录运行。
2. DIRECT 冒烟测试稳定通过，且运行结束后无残留连接。
3. 生成 RGB、深度、segmentation、定位、预测和 metadata 六类产物。
4. Grounding DINO 在仿真 RGB 上返回可见定位框。
5. 几何后端在定位框内返回有限、合法的抓取矩形。
6. 预测图清楚显示定位框、抓取中心、尺寸和方向。
7. CNN 后端至少完成权重加载与单图推理接口测试；如果仿真域预测质量较差，
   如实记录，不据此修改既有 Cornell 实验结论。
8. 普通项目测试和新增单元测试全部通过。
9. `PROJECT_STRUCTURE.md`、`CURRENT_STATUS.md` 和 `WORKLOG.md` 更新。

## 后续阶段

只有第一阶段验收通过后，才单独设计第二阶段：

```text
像素 + 深度
→ 相机坐标
→ PyBullet 世界坐标
→ 预抓取姿态
→ Panda 逆运动学
→ 一次简单夹取
```

第二阶段必须增加坐标回投测试、碰撞与抓取成功判定，不从本设计自动扩张。
