# 实验结果溯源

日期：2026-07-24

## 使用规则

- 论文数字必须能够追溯到实际保存的 CSV 或 JSON。
- 五次实验聚合结果与单次/最后一轮结果分开表述。
- 已被覆盖且无法从当前产物复核的旧单次结果不进入论文主表。

## 结果与产物

| 论文结果 | 实际文件 | 方法字段 | 可复核状态 | 使用决定 |
|---|---|---|---|---|
| Traditional CV | `data/processed/baseline_cv/cv_baseline_summary.json` | `opencv_contour_min_area_rect_rgb` | 可复核 | 主表 |
| VLM + geometry | `data/processed/vlm/grasp/vlm_assisted_grasp_summary.json` | `vlm_assisted_opencv_contour_min_area_rect_rgb` | 可复核 | 主表 |
| CNN five-run aggregate | `data/processed/vlm/cnn_grasp/multi_run_summary.json` | `vlm_cnn_multi_run` | 聚合指标可复核；旧 `best_val_loss` 错误 | 主表只使用 aggregate |
| CNN saved prediction rows | `data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv` | 五次实验最后一轮，seed 46 | 可复核 | 定性图和逐样本比较 |
| Legacy CNN single run 73.11% | 原始独立 JSON/CSV 已被覆盖 | 无 | 当前不可复核 | 不进入论文主表 |

## SHA-256

```text
e16579c0403ebceb919216055058e02f7691ce28181843846a517a772bb88822  data/processed/baseline_cv/cv_baseline_summary.json
61f4511bef06ae3279413f474bea31d44aa01c6763cd379802a9a42c7fa4d589  data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
4e71a556fafd5cc3bcec978251b69f18626d19f2ea9c2f769fe67a71cad3190d  data/processed/vlm/cnn_grasp/cnn_grasp_summary.json
15cb742a435a55455538fdd6cf3d1a5d21783baa9622f662f6e154edb2fdb32d  data/processed/vlm/cnn_grasp/multi_run_summary.json
```

## 已验证统计口径

- Baseline 和 VLM + geometry 的 summary 均覆盖 885 个样本。
- CNN five-run aggregate 使用 seeds 42–46。
- `cnn_grasp_predictions.csv` 的目录 09–10 包含 85 个样本，其中 68 个成功；
  该文件只代表最后一轮，不能替代五轮均值。
