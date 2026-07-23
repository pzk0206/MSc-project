# Project Documentation Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the project documentation into stable AI-context, debugging, planning, and worklog areas, with repository instructions that make Codex load the right context before future tasks.

**Architecture:** A root `AGENTS.md` acts as the automatically discovered instruction entry point. Focused documents under `docs/agent/` separate current state, stable project background, current repository layout, and future code-organization rules; detailed historical documents are moved without losing their content.

**Tech Stack:** Markdown, Git, Bash path/link validation.

## Global Constraints

- Do not modify experiment code, model behavior, datasets, or result files.
- Preserve all existing detailed documentation content.
- Move existing documents instead of leaving duplicate copies.
- Keep the user's unrelated uncommitted changes intact.
- Use relative links between Markdown files.
- Treat the current `README.md` result table as the latest source for summary metrics.
- Do not generate a PDF in this change.

## File Structure

**Create**

- `AGENTS.md` — durable repository instructions automatically loaded by Codex.
- `docs/agent/CODE_ORGANIZATION.md` — rules for placing and splitting future code.
- `docs/agent/CURRENT_STATUS.md` — concise, current AI handoff.
- `docs/agent/PROJECT_OVERVIEW.md` — stable research and method overview.
- `docs/agent/PROJECT_STRUCTURE.md` — current repository map and commands.
- `docs/worklog/WORKLOG.md` — reverse-chronological worklog index.

**Move**

- `docs/debug_log_cornell_baseline.md` → `docs/debugging/BUGLOG.md`
- `docs/failure_analysis.md` → `docs/debugging/FAILURE_ANALYSIS.md`
- `MSc_项目初步计划书_庞镇坤.md` → `docs/planning/vlm_robotic_grasp_study_plan.md`
- `docs/weekly_progress_2026-07-06.md` → `docs/worklog/weekly_progress_2026-07-06.md`
- `docs/weekly_progress_2026-07-16.md` → `docs/worklog/weekly_progress_2026-07-16.md`

**Modify**

- `README.md` — show the new documentation layout and point readers to the entry documents.
- `docs/debugging/BUGLOG.md` — add a navigation preface without removing history.
- Markdown files containing old paths — replace only the affected references.

---

### Task 1: Move Existing Documents into Purpose-Based Directories

**Files:**

- Move: `docs/debug_log_cornell_baseline.md` → `docs/debugging/BUGLOG.md`
- Move: `docs/failure_analysis.md` → `docs/debugging/FAILURE_ANALYSIS.md`
- Move: `MSc_项目初步计划书_庞镇坤.md` → `docs/planning/vlm_robotic_grasp_study_plan.md`
- Move: `docs/weekly_progress_2026-07-06.md` → `docs/worklog/weekly_progress_2026-07-06.md`
- Move: `docs/weekly_progress_2026-07-16.md` → `docs/worklog/weekly_progress_2026-07-16.md`

**Interfaces:**

- Consumes: the five existing Markdown documents and their unchanged content.
- Produces: stable categorized paths used by all later tasks.

- [ ] **Step 1: Record hashes and line counts before moving**

Run:

```bash
sha256sum \
  docs/debug_log_cornell_baseline.md \
  docs/failure_analysis.md \
  MSc_项目初步计划书_庞镇坤.md \
  docs/weekly_progress_2026-07-06.md \
  docs/weekly_progress_2026-07-16.md

wc -l \
  docs/debug_log_cornell_baseline.md \
  docs/failure_analysis.md \
  MSc_项目初步计划书_庞镇坤.md \
  docs/weekly_progress_2026-07-06.md \
  docs/weekly_progress_2026-07-16.md
```

Expected: all five paths exist and produce a hash and non-zero line count.

- [ ] **Step 2: Create the target directories and move the files**

Run:

```bash
mkdir -p docs/agent docs/debugging docs/planning docs/worklog
mv docs/debug_log_cornell_baseline.md docs/debugging/BUGLOG.md
mv docs/failure_analysis.md docs/debugging/FAILURE_ANALYSIS.md
mv MSc_项目初步计划书_庞镇坤.md docs/planning/vlm_robotic_grasp_study_plan.md
mv docs/weekly_progress_2026-07-06.md docs/worklog/weekly_progress_2026-07-06.md
mv docs/weekly_progress_2026-07-16.md docs/worklog/weekly_progress_2026-07-16.md
```

Expected: every source path is absent and every target path exists.

- [ ] **Step 3: Verify that the moves preserved content**

Run the same `sha256sum` and `wc -l` commands against the five target paths.

Expected: hashes and line counts match Step 1 exactly.

- [ ] **Step 4: Commit only the document moves**

```bash
git add \
  docs/debug_log_cornell_baseline.md \
  docs/failure_analysis.md \
  MSc_项目初步计划书_庞镇坤.md \
  docs/weekly_progress_2026-07-06.md \
  docs/weekly_progress_2026-07-16.md \
  docs/debugging/BUGLOG.md \
  docs/debugging/FAILURE_ANALYSIS.md \
  docs/planning/vlm_robotic_grasp_study_plan.md \
  docs/worklog/weekly_progress_2026-07-06.md \
  docs/worklog/weekly_progress_2026-07-16.md
git diff --cached --check
git commit -m "docs: organize project history by purpose"
```

Expected: one commit containing only moves of the five documents.

### Task 2: Add Durable Agent and Code-Organization Instructions

**Files:**

- Create: `AGENTS.md`
- Create: `docs/agent/CODE_ORGANIZATION.md`

**Interfaces:**

- Consumes: target paths established by Task 1 and current source layout under `src/`.
- Produces: persistent task-start instructions and future module-placement rules.

- [ ] **Step 1: Create `AGENTS.md`**

Create a concise repository instruction file containing these exact requirements:

```markdown
# Repository Instructions

## Start every task

1. Read `docs/agent/CURRENT_STATUS.md`.
2. If the project background is unclear, read `docs/agent/PROJECT_OVERVIEW.md`.
3. Before creating, moving, or refactoring code, read:
   - `docs/agent/PROJECT_STRUCTURE.md`
   - `docs/agent/CODE_ORGANIZATION.md`

## While working

- Preserve unrelated user changes in the working tree.
- Keep code grouped by feature and reuse shared logic through `src/shared/`.
- Do not record experimental claims unless they are supported by verified output.

## Finish tasks

- Update `docs/agent/CURRENT_STATUS.md` when the project state, results, or next steps change.
- Add a concise entry to `docs/worklog/WORKLOG.md` for material completed work.
- Update `docs/agent/PROJECT_STRUCTURE.md` after adding, moving, or removing a major module.
```

- [ ] **Step 2: Create `CODE_ORGANIZATION.md`**

Include the following sections and rules:

```markdown
# Code Organization Guidelines

## Purpose

This document defines where new code belongs and when an existing module should be split.

## Functional boundaries

- `src/shared/`: stable dataset, geometry, evaluation, and visualization utilities reused by two or more pipelines.
- `src/baseline_cv/`: traditional OpenCV baseline implementation and baseline-only diagnostics.
- `src/vlm/`: Grounding DINO localization, VLM-guided geometric grasping, CNN grasping, and VLM-only analysis.
- `uog_dissertation_outline/`: dissertation LaTeX sources and dissertation assets.
- `data/processed/<pipeline>/`: generated models, predictions, summaries, and plots; never place generated outputs under `src/`.

## Module design rules

1. Group files by feature or pipeline, not by arbitrary file type.
2. Give each module one clear primary responsibility.
3. Keep command-line entry points focused on argument parsing and orchestration.
4. Separate reusable dataset, model, training, evaluation, and visualization logic when each can be understood or tested independently.
5. Move logic to `src/shared/` only after two or more pipelines genuinely reuse it.
6. Search for an existing module with the same responsibility before creating a new file.
7. Avoid parallel implementations of grasp geometry or evaluation metrics.
8. Preserve existing commands and output schemas during refactors, or document a migration explicitly.

## When to split a file

Split a file when it combines multiple independently changing responsibilities, such as model definition, dataset preparation, training loop, evaluation, and plotting. Do not split solely to reduce line count.

## Documentation maintenance

- Update `PROJECT_STRUCTURE.md` when major modules move or change responsibility.
- Update `CURRENT_STATUS.md` only when current state or next actions change.
- Add completed milestones to `../worklog/WORKLOG.md`.
```

- [ ] **Step 3: Validate instruction paths**

Run:

```bash
for path in \
  docs/agent/CURRENT_STATUS.md \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/agent/CODE_ORGANIZATION.md \
  docs/worklog/WORKLOG.md
do
  test -f "$path" || printf 'missing: %s\n' "$path"
done
```

Expected at this point: only the three agent summaries and `WORKLOG.md` are reported missing; `CODE_ORGANIZATION.md` exists. Those missing paths are delivered in Task 3 and Task 4.

- [ ] **Step 4: Commit the durable instructions**

```bash
git add AGENTS.md docs/agent/CODE_ORGANIZATION.md
git diff --cached --check
git commit -m "docs: add persistent project organization guidance"
```

Expected: one commit containing only the two instruction documents.

### Task 3: Create the Agent Context Documents

**Files:**

- Create: `docs/agent/CURRENT_STATUS.md`
- Create: `docs/agent/PROJECT_OVERVIEW.md`
- Create: `docs/agent/PROJECT_STRUCTURE.md`

**Interfaces:**

- Consumes: current `README.md`, source files under `src/`, moved historical documents, and current Git status.
- Produces: three focused context layers used through `AGENTS.md`.

- [ ] **Step 1: Create `CURRENT_STATUS.md` from verified current facts**

Use this section order:

```markdown
# Current Project Status

> AI task entry point: read this file before starting project work.

Last updated: 2026-07-23

## Current phase
## Completed pipelines
## Latest verified results
## Confirmed findings
## Current repository state
## Next priorities
## Read next
## Maintenance rule
```

The results table must reproduce the current `README.md` values exactly:

- Traditional CV baseline: 504/885, 56.95%, mean best IoU 0.3360, mean angle error 29.62°.
- VLM + geometric: 649/885, 73.33%, mean best IoU 0.4182, mean angle error 14.81°.
- VLM + CNN single run: 647/885, 73.11%, mean best IoU 0.4476, mean angle error 15.97°.
- VLM + CNN five-run mean: success rate 74.51% ± 1.38%, IoU 0.4510 ± 0.0081, angle error 16.49° ± 0.72°.
- Unseen-object test: geometric 75.3% (64/85), CNN single 81.2% (69/85), CNN five-run mean 82.35% ± 4.53%.

State explicitly that Grounding DINO with prompt `small object` localized all 885 Cornell samples in the current experiment.

- [ ] **Step 2: Create `PROJECT_OVERVIEW.md`**

Use this section order:

```markdown
# Project Overview

## Research topic
## Research question
## Dataset
## Compared methods
### 1. Traditional CV baseline
### 2. VLM-guided geometric pipeline
### 3. VLM-guided CNN pipeline
## Evaluation protocol
## Main findings
## Scope and limitations
## Related documents
```

Describe the Cornell rectangle success criterion exactly as `IoU >= 0.25` and angular error `<= 30°`. Keep this file stable: put dates and short-term next steps in `CURRENT_STATUS.md`, not here.

- [ ] **Step 3: Create `PROJECT_STRUCTURE.md` from the actual repository**

Include:

1. A current directory tree covering `src/`, `docs/`, and `uog_dissertation_outline/`.
2. A table mapping each Python entry point to its responsibility.
3. The input location `data/raw/cornell/`.
4. Output locations under `data/processed/baseline_cv/` and `data/processed/vlm/`.
5. The existing commands from `README.md`.
6. A documentation reading order linking the four `docs/agent/` files.

Verify every listed source file with:

```bash
rg --files src docs uog_dissertation_outline | sort
```

Expected: every path included in the tree exists; generated data paths are described but need not exist in Git.

- [ ] **Step 4: Compare current-status metrics with README**

Run:

```bash
rg -n '56\.95|73\.33|73\.11|74\.51|82\.35|0\.4510|16\.49' \
  README.md docs/agent/CURRENT_STATUS.md
```

Expected: each current metric appears in both files with the same method label and result scope.

- [ ] **Step 5: Commit the agent context documents**

```bash
git add \
  docs/agent/CURRENT_STATUS.md \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/PROJECT_STRUCTURE.md
git diff --cached --check
git commit -m "docs: add focused agent project context"
```

Expected: one commit containing only the three agent context files.

### Task 4: Add the Worklog Entry Point and Update Documentation Navigation

**Files:**

- Create: `docs/worklog/WORKLOG.md`
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `README.md`
- Modify: `docs/worklog/weekly_progress_2026-07-16.md`

**Interfaces:**

- Consumes: all paths and summaries created in Tasks 1–3.
- Produces: human-facing timeline navigation and valid repository-wide links.

- [ ] **Step 1: Create `WORKLOG.md`**

Use reverse chronological order and this structure:

```markdown
# Project Worklog

This file is the entry point for reviewing completed project work. Detailed reports remain in dated files.

## 2026-07-23 — Documentation structure

- Reorganized project documentation by audience and purpose.
- Added durable AI context and code-organization guidance.

## 2026-07-16 — Failure analysis and CNN grasp backend

- Completed geometric-pipeline failure analysis.
- Added and evaluated the VLM-guided CNN grasp backend.
- Recorded single-run and five-run results.
- Details: [Weekly progress — 2026-07-16](weekly_progress_2026-07-16.md)

## 2026-07-06 — Baseline and VLM-guided geometric pipeline

- Completed Cornell parsing and the traditional CV baseline.
- Completed Grounding DINO localization and VLM-assisted geometric grasp detection.
- Details: [Weekly progress — 2026-07-06](weekly_progress_2026-07-06.md)
```

- [ ] **Step 2: Add a navigation preface to `BUGLOG.md`**

Immediately below the title, add:

```markdown
> 本文件保留完整调试历史。当前状态见
> [`../agent/CURRENT_STATUS.md`](../agent/CURRENT_STATUS.md)，失败模式汇总见
> [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)。
```

Do not rewrite or delete the existing debugging sections.

- [ ] **Step 3: Update the README documentation tree and entry links**

Replace the old flat `docs/` tree with the new categorized tree. Add a short paragraph after the tree:

```markdown
后续 AI 会话从 `docs/agent/CURRENT_STATUS.md` 开始；人工回顾从
`docs/worklog/WORKLOG.md` 开始。代码放置规则见
`docs/agent/CODE_ORGANIZATION.md`。
```

Preserve all unrelated README edits, including the five-run CNN results.

- [ ] **Step 4: Find and repair old path references**

Run:

```bash
rg -n \
  'docs/debug_log_cornell_baseline\.md|docs/failure_analysis\.md|MSc_项目初步计划书_庞镇坤\.md|docs/weekly_progress_2026-07-(06|16)\.md' \
  -g '*.md' \
  -g '!docs/superpowers/**' \
  .
```

Replace the two occurrences of `docs/failure_analysis.md` in
`docs/worklog/weekly_progress_2026-07-16.md` with
`docs/debugging/FAILURE_ANALYSIS.md`. Do not alter historical paths recorded
inside the approved design and implementation plan.

Expected: rerunning the command outside `docs/superpowers/` returns no matches.

- [ ] **Step 5: Commit navigation and link updates**

```bash
git add \
  docs/debugging/BUGLOG.md \
  docs/worklog/WORKLOG.md \
  docs/worklog/weekly_progress_2026-07-16.md
git add -p README.md
git diff --cached --check
git commit -m "docs: add project worklog and navigation"
```

At the `git add -p README.md` prompt, stage only the documentation-tree and
entry-link hunk created by this task; leave pre-existing result-table changes
unstaged. Before committing, inspect `git diff --cached --name-status` and
unstage any experiment-code path or unrelated file. Expected: only Markdown
navigation/content files are committed.

### Task 5: Validate the Finished Documentation System

**Files:**

- Verify: `AGENTS.md`
- Verify: all Markdown files under `docs/`
- Verify: `README.md`

**Interfaces:**

- Consumes: the completed documentation structure.
- Produces: evidence that entry points, content, and links are consistent.

- [ ] **Step 1: Verify required files**

Run:

```bash
for path in \
  AGENTS.md \
  docs/agent/CODE_ORGANIZATION.md \
  docs/agent/CURRENT_STATUS.md \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/debugging/BUGLOG.md \
  docs/debugging/FAILURE_ANALYSIS.md \
  docs/planning/vlm_robotic_grasp_study_plan.md \
  docs/worklog/WORKLOG.md \
  docs/worklog/weekly_progress_2026-07-06.md \
  docs/worklog/weekly_progress_2026-07-16.md
do
  test -s "$path" || printf 'missing-or-empty: %s\n' "$path"
done
```

Expected: no output.

- [ ] **Step 2: Verify old paths are gone**

Run the old-path `rg` command from Task 4.

Expected: no output and exit code 1 outside `docs/superpowers/`, meaning no
obsolete operational reference remains. Historical source-path descriptions in
the design and plan are expected.

- [ ] **Step 3: Validate local Markdown links**

Run this read-only link checker over every Markdown file:

```bash
python3 - <<'PY'
from pathlib import Path
import re
from urllib.parse import unquote

broken = []
for source in Path(".").rglob("*.md"):
    if ".git" in source.parts:
        continue
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        destination = raw.strip().split()[0].strip("<>")
        if destination.startswith(("http://", "https://", "mailto:", "#")):
            continue
        destination = unquote(destination.split("#", 1)[0])
        if not destination:
            continue
        target = (source.parent / destination).resolve()
        if not target.exists():
            broken.append(f"{source}: {raw}")

print("\n".join(broken))
raise SystemExit(1 if broken else 0)
PY
```

Expected: no broken local-file links. If the repository already contains unrelated broken links, list them separately and verify all links changed by this implementation.

- [ ] **Step 4: Verify no source code was changed by this documentation task**

Run:

```bash
git status --short
git log --oneline -5
git show --stat --oneline HEAD
```

Expected: commits from this plan contain only `AGENTS.md` and Markdown documentation. Pre-existing user modifications such as `src/vlm/run_cnn_grasp.py` remain intact and are not included in documentation commits.

- [ ] **Step 5: Review the two task entry paths**

Read:

```bash
sed -n '1,220p' AGENTS.md
sed -n '1,260p' docs/agent/CURRENT_STATUS.md
sed -n '1,220p' docs/worklog/WORKLOG.md
```

Expected:

- `AGENTS.md` directs future agents to current context and organization rules.
- `CURRENT_STATUS.md` is sufficient to identify current progress and next priorities.
- `WORKLOG.md` provides a concise human-readable timeline with valid report links.
