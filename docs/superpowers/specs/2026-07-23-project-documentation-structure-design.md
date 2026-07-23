# 项目文档结构整理设计

日期：2026-07-23

## 1. 目标

将项目现有文档按用途重新分类，使两类读者都能快速获得所需信息：

- 后续 AI 会话通过 `docs/agent/` 快速恢复项目背景、当前状态和代码结构。
- 项目作者通过 `docs/debugging/`、`docs/planning/` 和 `docs/worklog/` 回顾调试过程、研究计划及阶段进展。

本次工作只整理文档，不修改实验代码、模型实现、数据或实验结果。

## 2. 目标目录

```text
AGENTS.md
docs/
├── agent/
│   ├── CODE_ORGANIZATION.md
│   ├── CURRENT_STATUS.md
│   ├── PROJECT_OVERVIEW.md
│   └── PROJECT_STRUCTURE.md
├── debugging/
│   ├── BUGLOG.md
│   └── FAILURE_ANALYSIS.md
├── planning/
│   └── vlm_robotic_grasp_study_plan.md
├── superpowers/
│   └── specs/
│       └── 2026-07-23-project-documentation-structure-design.md
└── worklog/
    ├── WORKLOG.md
    ├── weekly_progress_2026-07-06.md
    └── weekly_progress_2026-07-16.md
```

## 3. 文件职责

### `AGENTS.md`

作为 Codex 自动发现的仓库级持久指令。Codex 每次运行或新会话开始时会自动读取根目录的 `AGENTS.md`；普通的 `docs/agent/*.md` 不会被自动读取，因此由该文件明确规定阅读和维护顺序：

1. 开始任何项目任务前先读 `docs/agent/CURRENT_STATUS.md`。
2. 首次接触项目或上下文不足时再读 `PROJECT_OVERVIEW.md`。
3. 创建、移动或重构代码前读 `PROJECT_STRUCTURE.md` 和 `CODE_ORGANIZATION.md`。
4. 完成产生实际变更的任务后，按需更新 `CURRENT_STATUS.md` 和 `WORKLOG.md`。
5. 不在状态文件中记录未经验证的结果。

`AGENTS.md` 保持简短，只保存强制工作约定，并通过相对路径指向详细文档。

### `docs/agent/CODE_ORGANIZATION.md`

规定后续代码和目录应如何组织，重点包括：

- 按功能域分目录：共享数据与几何逻辑放入 `src/shared/`，传统视觉方法放入 `src/baseline_cv/`，VLM 定位和抓取后端放入 `src/vlm/`。
- 同一功能的实现、配置和专用辅助代码优先放在同一功能目录中。
- 跨两条或以上 pipeline 复用的稳定逻辑放入 `src/shared/`，避免复制实现。
- 入口脚本负责参数解析和流程编排；数据读取、模型定义、训练、评估与可视化逻辑应保持清晰边界。
- 一个模块应有单一、可说明的主要职责；当文件同时承担多个独立职责且难以单独测试时，应拆分模块。
- 新增文件前先检查是否已有职责相同的模块，避免建立平行实现。
- 实验输出只写入 `data/processed/<pipeline>/`，不混入源码目录。
- 新增或移动主要模块后同步更新 `PROJECT_STRUCTURE.md`。
- 结构调整应保持现有命令和输出格式兼容；确需破坏兼容性时先记录迁移方式。

### `docs/agent/CURRENT_STATUS.md`

作为后续 AI 会话的首要入口。文件顶部明确说明 AI 开始工作时应优先阅读此文件。内容包括：

- 当前研究阶段和完成度。
- 三条实验 pipeline 的最新指标。
- 已确认的主要研究结论。
- 当前未完成事项和建议的下一步。
- 继续工作前建议阅读的文件。
- 最后更新时间。

该文件只保存当前有效状态，不累积详细历史。

### `docs/agent/PROJECT_OVERVIEW.md`

提供相对稳定的项目背景，包括：

- 研究问题与目标。
- Cornell Grasping Dataset 的用途。
- Traditional CV、VLM + geometric 和 VLM + CNN 三种方法。
- 统一评估标准。
- 项目贡献、关键发现和范围边界。

### `docs/agent/PROJECT_STRUCTURE.md`

解释仓库结构和模块职责，包括：

- `src/shared/`、`src/baseline_cv/`、`src/vlm/` 和论文目录。
- 主要入口脚本及其输入、输出。
- 数据与实验产物的默认位置。
- 常用运行命令。
- 文档目录的阅读顺序。

### `docs/debugging/BUGLOG.md`

由现有 `docs/debug_log_cornell_baseline.md` 移动并改名而来。保留完整调试记录，在文件开头增加简短导航，避免丢失已有技术细节。

### `docs/debugging/FAILURE_ANALYSIS.md`

由现有 `docs/failure_analysis.md` 移动并统一命名而来，保留 VLM-guided geometric pipeline 的失败模式、统计和结论。

### `docs/planning/vlm_robotic_grasp_study_plan.md`

由根目录的 `MSc_项目初步计划书_庞镇坤.md` 移动并改名而来。保留原始研究规划及后续调整记录。当前不生成 PDF，以免引入额外构建依赖和重复维护。

### `docs/worklog/WORKLOG.md`

作为人工回顾入口，按时间倒序列出阶段记录。每条记录包括日期、完成事项、主要结果和指向详细周报的链接。

### 周报文件

现有两份周报移动到 `docs/worklog/`，保留原文件名和内容，保证日期语义清楚。

## 4. 移动与链接规则

- 使用 Git 可识别的文件移动，历史内容不复制、不删除。
- 更新仓库内所有指向旧路径的 Markdown 链接和目录树。
- `README.md` 的项目结构部分改为新的文档布局。
- 不覆盖工作区内与本次文档整理无关的修改。
- 文档内链接使用相对路径，便于 GitHub 和本地编辑器浏览。

## 5. 内容来源与一致性

三个 agent 文档以以下资料为准：

- 根目录 `README.md` 的最新实验概览。
- 两份 weekly progress 文档的阶段记录。
- 调试日志与失败分析中的技术结论。
- 当前源码结构和脚本参数。

如果不同文档中的数字不一致，以当前 `README.md` 中已更新的汇总结果为最新状态，并在 `CURRENT_STATUS.md` 标注结果口径，例如 single run 或 5-run mean ± std。

## 6. 验证

整理完成后执行以下检查：

1. 确认目标目录和文件全部存在。
2. 搜索旧文档路径，确保没有遗留的失效引用。
3. 检查所有相对 Markdown 链接指向存在的文件。
4. 对照当前源码文件列表，检查 `PROJECT_STRUCTURE.md` 没有遗漏主要模块。
5. 检查 `CURRENT_STATUS.md` 中的指标与最新 `README.md` 一致。
6. 检查根目录 `AGENTS.md` 明确要求读取状态和代码组织文档。
7. 查看 Git diff，确认没有实验代码或数据被意外改动。

## 7. 完成标准

- AI 只需先阅读 `CURRENT_STATUS.md`，即可了解项目处于什么阶段以及下一步应做什么。
- Codex 在新运行或新会话中自动读取根目录 `AGENTS.md`，并按其中约定加载相应的 agent 文档。
- 后续新增代码有明确的功能分组、模块边界和共享逻辑放置规则。
- 项目作者可从 `WORKLOG.md` 快速浏览时间线，并进入详细周报。
- 调试、失败分析和研究计划均有清晰且唯一的归档位置。
- README 和文档内部不存在旧路径造成的失效链接。
- 原有详细内容完整保留。
