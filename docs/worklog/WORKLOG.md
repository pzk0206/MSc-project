# 项目工作日志

本文是回顾已完成项目工作的入口。详细内容保留在按日期命名的周报中。

## 2026-07-27 — PyBullet 虚拟相机感知 pilot 与失败诊断

- 安装 PyBullet `3.2.7`（API `202010061`）到 `msc-grasp` Conda 环境。
- 在 `src/simulation/pybullet/` 实现场景生命周期、固定虚拟相机、米制深度
  转换、现有 Grounding DINO/几何/CNN 后端适配、OpenCV 绘图、CLI 和失败
  元数据；第一阶段不包含机器人运动或物理抓取。
- 使用测试驱动开发新增 24 项 simulation 测试；完整回归为 `56 passed`，
  `git diff --check` 通过。
- 验证 DIRECT + `ER_TINY_RENDERER` 生成 640×480 RGB、深度和 segmentation，
  且小鸭在 segmentation 中可见。
- 确认沙箱内无法访问 CUDA，但沙箱外同一环境可识别 GTX 1650 Ti 并成功运行
  Grounding DINO。
- 真实运行发现 `small object` 错误定位 Panda 末端；只改变 prompt 为
  `yellow rubber duck` 后正确定位小鸭，并生成合法几何抓取框。
- 明确该结果是系统集成与 prompt 歧义的单案例可行性证据，不是仿真抓取成功率，
  metadata 保持 `physical_grasp_executed: false`。
- 设计与实施计划分别保存于 `docs/superpowers/specs/` 和
  `docs/superpowers/plans/`；实现检查点提交为 `adc2c84`、`ba250fd`、
  `c49ad30`、`1e68f8a` 和 `4ce9762`。
- 增加向后兼容的命名多物体场景、纯目标选择评价、四实体真值框、逐 prompt
  可视化和固定研究 CLI；segmentation 明确只作事后评价。
- 固定场景使用黄色鸭、红色方块、绿色球体和 Panda distractor；一次渲染、
  一次模型加载后运行三条明确 prompt 和一条 `small object` 诊断。
- 首次真实运行发现 Panda 基座与桌体初始穿透并将球体弹出画面。通过初始
  contact 和逐 body 像素数定位根因，将固定研究场景 Panda 基座设置为
  `z=0.625`，并加入穿透与 60 步可见性回归测试。
- 在 `msc-grasp` 环境和 GTX 1650 Ti 上重新运行后，三条主 prompt 全部正确：
  鸭、方块和球体 IoU 分别为 `0.8597`、`0.8717` 和 `0.8485`，平均
  `0.8600`；三个抓取中心均位于目标 mask 内。
- `small object` 诊断选择红色方块，score `0.5933`，不计入三目标成功率。
  人工图像审计确认定位框正确，但抓取框宽度普遍超出物体轮廓，因此不声称
  物理抓取成功。
- 新增模块 README 并记录 PyBullet 官方来源、命令、评价协议、输出和明确
  非目标；当前 38 项 simulation 测试、完整回归 `70 passed`。
- 多物体实现检查点提交为 `5d6cfac`、`17b3c20`、`8b32934` 和
  `66a9611`；真实产物位于被 Git 忽略的
  `data/processed/pybullet/multi_object_study/`。

## 2026-07-26 — 完整中文论文初稿与最终验证

- 按 `docs/agent/DISSERTATION_WRITING_GUIDE.md` 将英文 LaTeX 工作稿结构性
  改写为完整中文初稿；英文旧稿仍可从 Git 提交 `a4d62d6` 恢复。
- 完成 312 字摘要、1,114 字引言、3,191 字文献综述、1,809 字方法论、
  3,020 字结果和 1,003 字结论；全文无 `\todo{}`。
- 文献综述按主题组织，重点批判分析 Lenz、Redmon、GG-CNN、Grounding DINO
  和 Vuong 等工作，并明确 RGB/RGB-D、image-wise/object-wise、密集/单矩形
  与离线/物理抓取的有限可比边界。
- 方法论补充研究选择理由、控制变量和 Research Ethics，明确公开数据、外部
  模型与代码出处、结果不夸大、确定性、哈希及计算资源责任。
- 结果章按三个研究问题重组，保留全数据五 seed、固定 85 样本配对和共同
  image-wise 五折三种统计口径；所有核心数字由保存 JSON/CSV 复核。
- 中文化摘要、目录、章节、图表标签、致谢和附录；保留英文作者、文献标题、
  期刊、BibTeX 条目和 citation key。
- 旧 `agsm.bst` 在当前 BibTeX 引擎中产生 36 条 entry-mutation errors，改用
  natbib 兼容的 `plainnat` 作者—年份样式后，引用与参考文献无未定义项。
- Tectonic 成功编译 27 页中文 PDF；PDF 回读覆盖摘要、目录、七个主要章节、
  表/图标签和参考文献，关键页面视觉检查无缺字或表格越界。
- 12 个 citation key、12 个核心数字和章节字符范围全部通过自动检查；
  最终代码回归为 32 passed，`git diff --check` 通过。

## 2026-07-26 — Cornell image-wise 五折实现与训练前审计

- 设计并实现确定性的 image-wise 五折 manifest；seed 42 下每折固定为
  566/142/177，五个测试 fold 两两互斥并覆盖全部 885 张图。
- 新增 manifest CSV/JSON 稳定持久化、SHA-256、角色重叠、样本遗漏和
  非完整测试覆盖检查；正式 JSON 哈希为
  `b7d3e22a145f50add6d57a70bf0abb87b4b12ee674541deab0d7fee9a286bc2d`。
- 将 CNN crop 构建与数据角色划分解耦，保留旧固定目录入口，同时支持由
  manifest 按 sample ID 显式划分。
- 修复 `evaluate_model` 忽略传入样本集合的问题；用历史 seed 42 权重确认
  旧测试集合严格只输出 85 条结果。
- 实现逐 fold 独立模型、历史、预测和 summary 路径，以及五折完整性审计、
  pooled/fold 统计、单头与多头共同 manifest 校验和成对比较。
- 实现可恢复 CLI：可单独生成 manifest、运行任意 fold、从保存产物聚合或
  比较架构；修复直接执行脚本时的项目根路径导入问题。
- 完整测试为 32 passed；Python 编译与 diff 检查通过。
- CPU 冒烟以及 GTX 1650 Ti 上的单头/多头 CUDA 冒烟均完成；严格确定性
  已启用。冒烟产物写入 `/tmp`，未作为正式实验结果记录。
- Object-wise 仍因缺少 885 图到物体实例的权威映射而阻塞；`01`–`10`
  不作为 object ID。
- 完成单头和多头共十次正式 image-wise fold 训练。单头 pooled 为
  635/885（71.75%）、IoU 0.4390、角度 17.74°；多头为
  647/885（73.11%）、IoU 0.4580、角度 17.40°。
- 独立复核两个架构各 885 个唯一 ID、每折 177 行、共同 manifest 哈希、
  最佳验证损失与历史最小值以及所有数值有限性，全部通过。
- 整理论文写作结构与字数指南，明确 Abstract、Introduction、Literature
  Review、Methodology、Findings 和 Conclusion 的内容重点、批判性分析要求
  与推荐写作顺序。

## 2026-07-24 — 实验溯源与数据划分审计

- 修复 CNN 多轮汇总错误记录验证损失的问题，并移除重复 CLI 入口。
- 建立实验结果溯源清单，确认旧 73.11% 单次结果缺少当前独立产物。
- 审计目录 01–06、07–08、09–10，确认样本数为 600/200/85。
- 在固定 85 样本上验证几何后端为 64/85，CNN 最后一轮为 68/85。
- 生成三个 split 各 12 张代表图，并收紧未见物体泛化表述。
- 完成固定测试子集逐样本比较：共同成功 52、仅 CNN 成功 16、仅几何成功
  12、共同失败 5，并生成三类代表失败图。
- 核对九项二维抓取研究，形成统一的输入、划分、指标、输出和可比性矩阵。
- 将当前 CNN 定位为 432,454 参数的轻量受控基线，逐项区分文献依据和自主
  工程选择，并生成可复现的矢量结构图。
- 修正论文 BibTeX 条目；明确 Park et al. (2018) 预印本已撤回，不用于正式
  性能排名。
- 完成论文 Methodology 和 Results 初稿，插入主结果表、固定测试子集表及
  三张证据图，并删除这两个章节的核心占位符。
- 用最小样例定位 `newtxmath` 与 `amssymb` 的重复符号冲突，修正论文模板
  兼容性；Tectonic 成功编译 22 页 PDF，并完成方法与结果页视觉检查。
- 完成 evidence-first 的 General Discussion，按定位增益、IoU、角度、
  后端互补性和测试集限制解释结果，并区分 Cornell、RGB-D、密集预测和物理
  抓取指标。
- 扩展固定 85 样本失败分析，为 12 个代表案例分别记录观测与可能原因；
  完成环境、超参数、核心命令和原始输出路径复现附录。
- 实现共享主干及中心、尺寸、方向三个回归头的多头 CNN，保留单头旧权重
  兼容性，并增加独立输出目录、分项损失历史和多次运行产物隔离。
- 完成 seed 42 多头 CNN 可行性门控：第 43 个 epoch 早停，完整 885 样本
  成功率 77.18%、平均 IoU 0.4630、平均角度误差 15.63°；固定 85 样本
  成功率 82.35%。验证全部损失有限、885 条预测齐全且输出框合法。
- 确认正式评估范围：保留旧单头产物作为历史证据，在独立目录重跑单头和
  多头 seeds 42–46；验证物体实例分组后优先增加 object-wise 五折，再增加
  image-wise 五折。RGB-D 不纳入本月主实验，作为未来深度融合方向记录。
- 发现仅设置 seed 仍会因 CUDA 非确定性导致同 seed 结果漂移；关闭 cuDNN
  benchmark、启用严格确定性算法、固定 DataLoader generator，并将
  AdaptiveAvgPool2d 换为数学等价的固定 `7×7` 平均池化。两次三轮 GPU
  训练得到相同历史和逐位一致权重。

## 2026-07-23 — 文档结构整理

- 按读者和用途重新整理项目文档。
- 增加供 AI 恢复上下文的文档和代码组织规范。
- 将项目文档的标题、说明、导航和回顾内容统一为中文。
- 生成三页中英文导师项目进展汇报 PDF，并保留可编辑生成脚本。
- 编写与英文汇报对应的双语讲稿和导师可能追问的英文回答。
- 增加严格对应三页汇报内容的 3–4 分钟简洁双语讲稿。
- 将简洁双语讲稿转换为适合 iPad 竖屏阅读的九页 PDF。

## 2026-07-16 — 失败分析与 CNN 抓取后端

- 完成几何实验流程的失败案例分析。
- 新增并评估 VLM 引导的 CNN 抓取后端。
- 记录单次实验和五次重复实验结果。
- 详情：[2026-07-16 周报](weekly_progress_2026-07-16.md)

## 2026-07-06 — 基线与 VLM 引导的几何实验流程

- 完成 Cornell 数据解析和传统计算机视觉基线。
- 完成 Grounding DINO 定位及 VLM 辅助的几何抓取检测。
- 详情：[2026-07-06 周报](weekly_progress_2026-07-06.md)
