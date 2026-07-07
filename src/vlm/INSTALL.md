# VLM 运行说明

当前 VLM 实现使用 Hugging Face Grounding DINO。

当前 `msc-grasp` conda 环境已经完成依赖安装，可以直接运行 VLM localization。

运行小批量 localization：

```bash
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py
```

运行全量 Cornell localization：

```bash
conda run -n msc-grasp python src/vlm/run_grounding_dino_localization.py --all --device cuda
```

默认设置：

```text
model: IDEA-Research/grounding-dino-tiny
prompt: small object
samples: 2 per Cornell subdirectory
output: data/processed/vlm/
```
