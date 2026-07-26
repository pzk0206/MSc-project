# VLM-guided grasp detection pipeline

这个文件夹包含 VLM-guided 抓取检测相关代码。

当前流程：

```text
RGB image + text prompt
    ↓
Grounding DINO / open-vocabulary detector
    ↓
object bounding box
    ↓
geometry backend / CNN backend
    ↓
2D grasp rectangle
    ↓
Cornell-style evaluation
```

当前已经完成三部分：

1. VLM localization：使用 Grounding DINO 根据 prompt 定位目标物体；
2. VLM-guided grasp：在 VLM box 内使用几何后端生成 2D 抓取矩形；
3. CNN grasp regression：使用单头或多头轻量 CNN 回归 2D 抓取参数。

主要脚本：

```text
run_grounding_dino_localization.py
    运行 Grounding DINO 目标定位。

run_vlm_assisted_grasp.py
    读取 VLM 定位结果，并生成 / 评估抓取矩形。

cnn_grasp_models.py
    定义兼容旧权重的单头模型、多头模型及多头损失。

run_cnn_grasp.py
    训练、评估并重复运行单头或多头 CNN 抓取后端。

prompts.py
    保存主实验使用的 prompt 设置。
```

CNN 命令：

```bash
# 复现原单头模型
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py \
  --mode all --architecture single --device cuda

# 训练并评估多头模型
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py \
  --mode all --architecture multi_head \
  --output-dir data/processed/vlm/cnn_grasp_multi_head \
  --seed 42 --device cuda

# 多头模型五次重复实验；各 seed 的权重、历史、预测和汇总分别保存
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py \
  --mode multi --architecture multi_head \
  --output-dir data/processed/vlm/cnn_grasp_multi_head \
  --num-runs 5 --device cuda
```

多头模型共享四个卷积块，并分别使用中心、尺寸和方向回归头。三个参数组均
采用 Smooth L1 损失，方向头另加权重为 0.1 的单位范数约束。推理阶段仍输出
与原流程相同的 `[cx, cy, width, height, sin(2θ), cos(2θ)]` 六参数格式。
正式训练会固定 Python、NumPy、PyTorch、DataLoader 和 CUDA 算法；同一软件、
硬件和 seed 下应产生相同训练历史与权重。跨 PyTorch/CUDA 版本仍不保证逐位
一致，因此每批正式实验必须记录环境版本。

输出位置：

```text
data/processed/vlm/
├── localization/
│   ├── grounding_dino_generic_small_object_predictions.csv
│   └── grounding_dino_generic_small_object_summary.json
├── grasp/
│   ├── vlm_assisted_grasp_predictions.csv
│   └── vlm_assisted_grasp_summary.json
├── cnn_grasp/
│   ├── cnn_grasp_model.pt
│   ├── training_history.json
│   ├── cnn_grasp_predictions.csv
│   └── cnn_grasp_summary.json
├── cnn_grasp_multi_head/
│   ├── cnn_grasp_model.pt
│   ├── training_history.json
│   ├── cnn_grasp_predictions.csv
│   └── cnn_grasp_summary.json
└── visualizations/
    └── localization_checks/
```

主实验 prompt：

```text
small object
```
