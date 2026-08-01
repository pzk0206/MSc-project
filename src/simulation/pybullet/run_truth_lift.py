"""Execute stage 5: lift and hold the contacted truth cube."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.grasp_execution import (
    TruthExecutionConfig,
    TruthExecutionStage,
    run_truth_execution,
)


DEFAULT_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/stage_5_truth_cube_lift"
)


@dataclass(frozen=True)
class TruthLiftConfig:
    """Deterministic stage-5 scene, target, and stability settings."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    gui: bool = False
    target_name: str = "cube"
    stability_steps: int = 60
    maximum_target_displacement_m: float = 0.001

    def __post_init__(self) -> None:
        if self.target_name != "cube":
            raise ValueError("stage 5 target_name must be cube")
        if self.stability_steps <= 0:
            raise ValueError("stability_steps must be positive")
        if (
            not math.isfinite(self.maximum_target_displacement_m)
            or self.maximum_target_displacement_m <= 0.0
        ):
            raise ValueError(
                "maximum_target_displacement_m must be finite and positive"
            )

    def to_execution_config(self) -> TruthExecutionConfig:
        """Convert the public stage-5 config to the shared core config."""

        return TruthExecutionConfig(
            output_dir=self.output_dir,
            seed=self.seed,
            gui=self.gui,
            target_name=self.target_name,
            stability_steps=self.stability_steps,
            maximum_target_displacement_m=(
                self.maximum_target_displacement_m
            ),
        )


def run_truth_lift(config: TruthLiftConfig) -> dict[str, object]:
    """Execute the full truth-cube contact, lift, and hold chain."""

    return run_truth_execution(
        config.to_execution_config(),
        TruthExecutionStage.LIFT_HOLD,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute stage-5 Panda truth-cube lift and hold."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    summary = run_truth_lift(
        TruthLiftConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            gui=args.gui,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
