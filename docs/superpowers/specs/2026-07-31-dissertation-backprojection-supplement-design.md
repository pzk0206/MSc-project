# 论文深度反投影补充验证设计

## 目标

把已经完成并审计的 PyBullet 三目标乘三后端深度反投影实验写入论文，作为从
二维抓取中心到目标表面世界坐标的补充系统可行性证据。该增补不改变三个研究
问题，不改变 Cornell RGB-only 主实验，也不把坐标门控表述为 IK、抓取位姿或
物理抓取成功。

## 证据来源

论文中的反投影数字只允许来自以下真实 CUDA 运行产物：

- `data/processed/pybullet/multi_object_study/backprojection_results.csv`
- `data/processed/pybullet/multi_object_study/backprojection_summary.json`
- `data/processed/pybullet/multi_object_study/summary.json`
- `data/processed/pybullet/multi_object_study/metadata.json`

允许报告的核心结果为：九条结果完整；有限坐标、有效深度、重投影通过、
segmentation 目标匹配和射线目标匹配均为 `9/9`；最大像素误差
`8.04e-13 px`；最大深度往返误差 `3.57e-7 m`；门槛分别为 `1 px` 和
`1e-4 m`。最大反投影点到射线命中点距离 `0.00676 m` 只作为诊断量，不增加
新的通过条件。

## 论文结构

### 范围与方法

引言保留 Cornell 主研究为 RGB-only 的限定，同时说明论文另含一个不参与
Cornell 性能排名的 PyBullet 补充验证。方法章新增独立小节，描述：

1. 三目标、三后端复用同一 RGB 帧与定位框；
2. 对二维中心使用最近像素、半向上取整读取米制深度；
3. 使用运行时保存的 PyBullet view/projection 矩阵完成一般矩阵反投影；
4. 世界点重投影回同一采样像素；
5. segmentation 与 `rayTest` 仅在坐标生成后作真值审计；
6. 不执行姿态生成、IK、碰撞规划、关节控制或夹爪闭合。

PyBullet 相机与深度约定链接到 Bullet 官方 Quickstart Guide；不声称本项目
原创 PyBullet 的矩阵或深度缓冲约定。项目自己的九点协议、阈值组合和审计
实现明确写为工程设计。

### 结果

结果章在三个研究问题之后新增“补充 PyBullet 坐标验证”小节，不编号为第四个
研究问题。用一张紧凑表汇总五个 `9/9` 计数、两个误差阈值、两个最大误差和
总门控状态。正文说明这是单个固定场景、九个相关样本点的确定性集成检查，
不是具有置信区间的成功率实验，也不用于后端性能排序。

### 讨论、结论与附录

讨论章把结果解释为数据流与坐标约定已连通，并明确三项剩余缺口：单点不能
确定完整六自由度姿态、运动学可达不等于无碰撞、无物理闭合与抬升验证。
结论只增加一句有边界的进展陈述。复现附录增加真实运行命令、两份新输出文件、
门控字段和 `ik_executed: false`、`physical_grasp_executed: false` 元数据。

## 不做的内容

- 不新增研究问题或修改 Cornell 主结果表；
- 不重新运行数值未受影响的 Cornell 实验；
- 不从九点结果推断三个后端的性能优劣；
- 不加入 IK、轨迹或抓取成功数字；
- 不使用 segmentation 或射线结果修正世界坐标；
- 不新增未经实际输出验证的实验结论。

## 验收

- 自动核对论文中的九点核心数字与保存的 JSON/CSV；
- 检查所有 citation key 均存在且无未定义引用；
- 用 Tectonic 重新编译 PDF，确认无 LaTeX 错误；
- 回读新增方法、结果、讨论、结论和附录文字；
- 视觉检查新增表格所在页面无越界或不可读内容；
- 运行完整项目测试并通过 `git diff --check`。
