# 项目文档结构实施计划与完成记录

> 本文件记录项目文档重组的实施范围、执行步骤和验证结果。路径、命令及 Git 提交信息保持原样，便于复查。

**目标：** 将项目文档整理为稳定的 AI 上下文、调试、计划和工作日志区域，并通过仓库指令确保 Codex 在后续任务开始前读取正确的项目上下文。

**结构设计：** 根目录的 `AGENTS.md` 是 Codex 自动发现的指令入口。`docs/agent/` 中的四个文档分别负责当前状态、稳定背景、仓库结构和代码组织规则；详细历史文档按调试、计划和工作日志分类保存。

**使用技术：** Markdown、Git、Bash 路径与链接检查。

## 全局约束

- 不修改实验代码、模型行为、数据集或实验结果文件。
- 完整保留已有详细文档内容。
- 移动原文档，不保留重复副本。
- 保留用户已有的无关未提交修改。
- Markdown 文档之间使用相对链接。
- 以当前 `README.md` 的结果表作为汇总指标的最新来源。
- 本次整理不生成 PDF。

## 文件结构

### 新建文件

- `AGENTS.md`：Codex 自动读取的仓库级工作说明。
- `docs/agent/CODE_ORGANIZATION.md`：新代码放置和模块拆分规则。
- `docs/agent/CURRENT_STATUS.md`：简明的当前项目状态。
- `docs/agent/PROJECT_OVERVIEW.md`：稳定的研究背景和方法概览。
- `docs/agent/PROJECT_STRUCTURE.md`：当前仓库结构、模块职责和命令。
- `docs/worklog/WORKLOG.md`：按时间倒序排列的工作日志入口。

### 移动文件

- `docs/debug_log_cornell_baseline.md` → `docs/debugging/BUGLOG.md`
- `docs/failure_analysis.md` → `docs/debugging/FAILURE_ANALYSIS.md`
- `MSc_项目初步计划书_庞镇坤.md` → `docs/planning/vlm_robotic_grasp_study_plan.md`
- `docs/weekly_progress_2026-07-06.md` → `docs/worklog/weekly_progress_2026-07-06.md`
- `docs/weekly_progress_2026-07-16.md` → `docs/worklog/weekly_progress_2026-07-16.md`

### 修改文件

- `README.md`：展示新的文档结构和入口。
- `docs/debugging/BUGLOG.md`：增加导航说明，不删除历史内容。
- `docs/worklog/weekly_progress_2026-07-16.md`：更新失败分析文件路径。

---

## 任务一：移动现有文档

**涉及文件：** 五份调试、分析、计划和周报文档。

- [x] 移动前记录五份文件的 SHA-256 哈希和行数。
- [x] 创建 `docs/agent/`、`docs/debugging/`、`docs/planning/` 和 `docs/worklog/`。
- [x] 将五份文档移动到目标路径。
- [x] 再次检查哈希和行数，确认内容完全一致。
- [x] 提交文档移动。

验证结果：

```text
BUGLOG.md                         800 行
FAILURE_ANALYSIS.md              142 行
vlm_robotic_grasp_study_plan.md  251 行
weekly_progress_2026-07-06.md    226 行
weekly_progress_2026-07-16.md    152 行
```

对应提交：

```text
838741f：按用途整理项目历史文档
```

## 任务二：增加持久项目指令和代码组织规则

**新建文件：**

- `AGENTS.md`
- `docs/agent/CODE_ORGANIZATION.md`

- [x] 在 `AGENTS.md` 中规定每次任务开始前的阅读顺序。
- [x] 规定工作过程中应保护用户修改并只记录已验证结论。
- [x] 规定任务结束后何时更新状态、工作日志和项目结构。
- [x] 在代码组织规范中说明各源码目录的功能边界。
- [x] 说明共享逻辑、入口脚本和模块拆分规则。
- [x] 验证所有指令引用的目标路径。

对应提交：

```text
bad2ad7：增加持久项目组织说明
```

## 任务三：建立 AI 项目上下文

**新建文件：**

- `docs/agent/CURRENT_STATUS.md`
- `docs/agent/PROJECT_OVERVIEW.md`
- `docs/agent/PROJECT_STRUCTURE.md`

### 当前状态文档

- [x] 记录当前研究阶段和三条已完成实验流程。
- [x] 区分单次 CNN 实验与五次重复实验统计。
- [x] 记录完整数据集和未见物体测试集结果。
- [x] 记录已确认结论和下一步优先事项。
- [x] 增加指向项目概览、结构、调试、计划和工作日志的链接。

必须与 README 一致的指标：

| 方法 | 成功率 | 平均最佳 IoU | 平均角度误差 |
|---|---:|---:|---:|
| 传统计算机视觉基线 | 56.95% | 0.3360 | 29.62° |
| VLM + 几何后端 | 73.33% | 0.4182 | 14.81° |
| VLM + CNN 后端（单次实验） | 73.11% | 0.4476 | 15.97° |
| VLM + CNN 后端（五次实验） | 74.51% ± 1.38% | 0.4510 ± 0.0081 | 16.49° ± 0.72° |

### 项目概览文档

- [x] 说明研究主题、问题、数据集和三种对比方法。
- [x] 明确成功标准为 `IoU >= 0.25` 且角度误差 `<= 30°`。
- [x] 说明主要发现、研究范围和局限。

### 项目结构文档

- [x] 记录 `src/`、`docs/` 和 `uog_dissertation_outline/` 的目录结构。
- [x] 说明各 Python 文件的主要职责。
- [x] 记录数据输入、实验输出和数据流。
- [x] 收录当前常用运行命令。
- [x] 规定 AI 上下文文档的阅读顺序。

对应提交：

```text
46c077e：增加聚焦的 AI 项目上下文
```

## 任务四：建立人工回顾入口并更新导航

**涉及文件：**

- `docs/worklog/WORKLOG.md`
- `docs/debugging/BUGLOG.md`
- `docs/worklog/weekly_progress_2026-07-16.md`
- `README.md`

- [x] 建立按日期倒序排列的项目工作日志。
- [x] 在调试记录顶部增加当前状态和失败分析链接。
- [x] 更新 README 的仓库结构。
- [x] 更新周报中的旧失败分析路径。
- [x] 只暂存 README 中属于文档结构的修改，保留用户已有结果表修改。

对应提交：

```text
982578b：增加项目工作日志与导航
```

## 任务五：验证最终文档系统

### 必需文件检查

检查以下 11 个文件存在且非空：

```text
AGENTS.md
docs/agent/CODE_ORGANIZATION.md
docs/agent/CURRENT_STATUS.md
docs/agent/PROJECT_OVERVIEW.md
docs/agent/PROJECT_STRUCTURE.md
docs/debugging/BUGLOG.md
docs/debugging/FAILURE_ANALYSIS.md
docs/planning/vlm_robotic_grasp_study_plan.md
docs/worklog/WORKLOG.md
docs/worklog/weekly_progress_2026-07-06.md
docs/worklog/weekly_progress_2026-07-16.md
```

验证结果：`11 / 11` 个文件存在且非空。

### 旧路径检查

在 `docs/superpowers/` 历史设计和计划记录之外搜索以下旧路径：

```text
docs/debug_log_cornell_baseline.md
docs/failure_analysis.md
MSc_项目初步计划书_庞镇坤.md
docs/weekly_progress_2026-07-06.md
docs/weekly_progress_2026-07-16.md
```

验证结果：实际项目文档中没有遗留旧路径。

### Markdown 链接检查

链接检查规则：

1. 扫描全部 Markdown 文件。
2. 忽略代码块、网页链接、邮件链接和纯锚点链接。
3. 将本地链接相对于所在文档解析。
4. 报告目标不存在的链接。

验证结果：没有失效的本地 Markdown 文件链接。

### 变更范围检查

- [x] 实施提交没有包含 Python 源码或 `data/` 下的文件。
- [x] 用户原有的 `README.md` 结果修改保持未提交。
- [x] 用户原有的 `src/vlm/run_cnn_grasp.py` 修改保持未提交。

链接检查器忽略代码块示例的修正提交：

```text
3e9546a：链接验证时忽略代码示例
```

## 完成状态

文档结构整理已经完成。后续 Codex 会话通过根目录 `AGENTS.md` 获得持久指令，并按任务需要读取 `docs/agent/` 中的当前状态、项目概览、项目结构和代码组织规范。项目作者可从 `docs/worklog/WORKLOG.md` 按时间回顾工作，并从 `docs/debugging/` 和 `docs/planning/` 查看详细记录。
