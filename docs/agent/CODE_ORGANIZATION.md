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
