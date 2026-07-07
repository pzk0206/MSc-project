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
geometry backend
    ↓
2D grasp rectangle
    ↓
Cornell-style evaluation
```

当前已经完成两部分：

1. VLM localization：使用 Grounding DINO 根据 prompt 定位目标物体；
2. VLM-guided grasp：在 VLM box 内使用几何后端生成 2D 抓取矩形。

主要脚本：

```text
run_grounding_dino_localization.py
    运行 Grounding DINO 目标定位。

run_vlm_assisted_grasp.py
    读取 VLM 定位结果，并生成 / 评估抓取矩形。

prompts.py
    保存主实验使用的 prompt 设置。
```

输出位置：

```text
data/processed/vlm/
├── localization/
│   ├── grounding_dino_generic_small_object_predictions.csv
│   └── grounding_dino_generic_small_object_summary.json
├── grasp/
│   ├── vlm_assisted_grasp_predictions.csv
│   └── vlm_assisted_grasp_summary.json
└── visualizations/
    └── localization_checks/
```

主实验 prompt：

```text
small object
```
