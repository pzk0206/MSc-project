# 实验结果溯源

日期：2026-07-26

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
| Cornell image-wise fold manifest | `data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.json` | `cornell_image_wise_5_fold` | 885 个样本覆盖、五个 177 测试 fold、SHA-256 已审计 | 单头/多头正式五折共同使用 |
| 单头 image-wise 五折 | `data/processed/vlm/cnn_cross_validation/single/cross_validation_summary.json` | `vlm_cnn_single_image_wise_5_fold` | 五折、885 条合并预测和历史最小验证损失均已审计 | 论文五折主表 |
| 多头 image-wise 五折 | `data/processed/vlm/cnn_cross_validation/multi_head/cross_validation_summary.json` | `vlm_cnn_multi_head_image_wise_5_fold` | 五折、885 条合并预测和历史最小验证损失均已审计 | 论文五折主表 |
| Image-wise 架构成对比较 | `data/processed/vlm/cnn_cross_validation/architecture_comparison.json` | `multi_head_minus_single` | ID 集合和 manifest 哈希一致 | 论文架构比较 |

## SHA-256

```text
e16579c0403ebceb919216055058e02f7691ce28181843846a517a772bb88822  data/processed/baseline_cv/cv_baseline_summary.json
61f4511bef06ae3279413f474bea31d44aa01c6763cd379802a9a42c7fa4d589  data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
4e71a556fafd5cc3bcec978251b69f18626d19f2ea9c2f769fe67a71cad3190d  data/processed/vlm/cnn_grasp/cnn_grasp_summary.json
15cb742a435a55455538fdd6cf3d1a5d21783baa9622f662f6e154edb2fdb32d  data/processed/vlm/cnn_grasp/multi_run_summary.json
b7d3e22a145f50add6d57a70bf0abb87b4b12ee674541deab0d7fee9a286bc2d  data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.json
282f2a275b8593ad8484df4dc41d65ee5dcea40dddbe4edb99a512f766ba5324  data/processed/vlm/cnn_cross_validation/single/cross_validation_summary.json
f0c6987b7217589af4e892e6b1ffb9e6bdc20980ff15f23050ef450b9753e073  data/processed/vlm/cnn_cross_validation/multi_head/cross_validation_summary.json
1b598d3a891bee50f29c2a523813d76aa68eab2dc45bbd2f2bfc02f92fe2fb1b  data/processed/vlm/cnn_cross_validation/architecture_comparison.json
```

## 已验证统计口径

- Baseline 和 VLM + geometry 的 summary 均覆盖 885 个样本。
- CNN five-run aggregate 使用 seeds 42–46。
- `cnn_grasp_predictions.csv` 的目录 09–10 包含 85 个样本，其中 68 个成功；
  该文件只代表最后一轮，不能替代五轮均值。
- Image-wise manifest 使用 seed 42；每折为 566/142/177，五个测试 fold
  两两互斥并覆盖 885 个唯一 sample ID。
- 单头五折 pooled：635/885，成功率 71.75%，平均 IoU 0.4390，平均角度
  误差 17.74°。
- 多头五折 pooled：647/885，成功率 73.11%，平均 IoU 0.4580，平均角度
  误差 17.40°。
- 多头减单头：成功率 +1.36 个百分点、IoU +0.0190、角度误差 -0.34°。
- 两个架构各五个 fold 均有 177 条测试预测；合并后各有 885 个唯一 ID，
  所有正式数值有限，且每折 `best_val_loss` 等于保存历史的最小值。
