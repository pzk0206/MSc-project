# 仓库工作说明

## 每次任务开始前

1. 阅读 `docs/agent/CURRENT_STATUS.md`。
2. 阅读 `docs/agent/vlm_robotic_grasp_study_plan.md`，确认当前研究范围、阶段门控和后续顺序。
3. 如果不清楚项目背景，阅读 `docs/agent/PROJECT_OVERVIEW.md`。
4. 创建、移动或重构代码前，阅读：
   - `docs/agent/PROJECT_STRUCTURE.md`
   - `docs/agent/CODE_ORGANIZATION.md`

## 工作过程中

- 保留工作区中与当前任务无关的用户修改。
- 按功能组织代码，并通过 `src/shared/` 复用共享逻辑。
- 未经实际输出验证的实验结论不得写入项目记录。
- 引用、复制或改编他人的代码时，必须明确标注出处，包括原作者、
  项目或论文名称以及可访问的来源链接；不得将外部代码表述为本项目原创。

## 任务完成后

- 项目状态、实验结果或下一步计划发生变化时，更新 `docs/agent/CURRENT_STATUS.md`。
- 完成实质性工作后，在 `docs/worklog/WORKLOG.md` 中添加简明记录。
- 新增、移动或删除主要模块后，更新 `docs/agent/PROJECT_STRUCTURE.md`。
