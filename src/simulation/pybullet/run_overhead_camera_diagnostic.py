"""Overhead-camera diagnostic for single-pixel backprojection XY bias.

Replicates the Stage 6A perception pipeline (fixed seed-42 scene with a red
cube, Grounding DINO "red cube" localization, geometry backend grasp
prediction) but replaces the oblique camera with an overhead camera and
performs plain single-pixel depth backprojection.

Reference pipeline mirrored here:
    src/simulation/pybullet/run_stage6a2_recovery_preflight.py

Differences from the reference:
  * camera is overhead: eye=(0.5, 0.0, 1.3), target=(0.5, 0.0, 0.62)
  * single-pixel backprojection only -- center_recovery is NOT called
  * no files are written; every result is printed to stdout

The baseline (oblique camera, single-pixel backprojection of the geometry
grasp centre) had an XY deviation of 26.55 mm between the reconstructed world
surface point and the live cube-truth centroid.  This script checks whether a
top-down camera removes that bias.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.backprojection import (
    backproject_pixel,
    sample_nearest_depth,
)
from src.simulation.pybullet.camera import (
    CameraConfig,
    capture_camera_frame,
)
from src.simulation.pybullet.perception import (
    Localization,
    load_grounding_dino,
    predict_grasp,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene
from src.simulation.pybullet.target_selection import (
    box_iou,
    mask_to_box,
    segmentation_mask_for_body,
)
from src.vlm.run_grounding_dino_localization import (
    select_best_detection,
)
from src.vlm.prompts import normalize_grounding_prompt


# ---------------------------------------------------------------------------
# Frozen protocol values (identical to the Stage 6A reference)
# ---------------------------------------------------------------------------

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SEED = 42
TARGET_NAME = "cube"
PROMPT = "red cube"
DEVICE = "cuda"
WIDTH = 640
HEIGHT = 480
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Baseline oblique result for comparison (stage_6a1_center_bias_diagnostic).
BASELINE_XY_DEVIATION_MM = 26.55
BASELINE_WORLD_SURFACE_POINT = (0.506456, 0.002225, 0.675478)
BASELINE_CUBE_TRUTH_CENTER = (0.480000, 0.0, 0.649970)

# Overhead camera: directly above the cube, optical axis vertical.
# A top-down view must use a non-vertical up vector: the default up=(0,0,1)
# is parallel to the view direction and makes computeViewMatrix degenerate
# (NaNs, nothing rendered).  up=(0,1,0) gives a proper top-down frame.
CAMERA_EYE = (0.5, 0.0, 1.3)
CAMERA_TARGET = (0.5, 0.0, 0.62)
CAMERA_UP = (0.0, 1.0, 0.0)

MILLIMETRES = 1000.0


def _run_grounding_dino_in_memory(
    processor: object,
    model: object,
    device: str,
    rgb: np.ndarray,
) -> dict | None:
    """Replicate run_grounding_dino_on_image using an in-memory RGB image.

    The vlm helper hard-codes ``Image.open(image_path)``, which would require
    writing a temporary image file.  This diagnostic is forbidden from creating
    any file, so the exact same processor/model/post-processing calls are made
    against a PIL image built directly from the captured frame.
    """

    pil_image = Image.fromarray(rgb)
    prompt = normalize_grounding_prompt(PROMPT)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    post_process = processor.post_process_grounded_object_detection
    signature = inspect.signature(post_process)
    post_process_kwargs: dict[str, object] = {
        "outputs": outputs,
        "input_ids": inputs.get("input_ids"),
        "text_threshold": TEXT_THRESHOLD,
        "target_sizes": [pil_image.size[::-1]],
    }
    if "threshold" in signature.parameters:
        post_process_kwargs["threshold"] = BOX_THRESHOLD
    else:
        post_process_kwargs["box_threshold"] = BOX_THRESHOLD
    results = post_process(**post_process_kwargs)
    return select_best_detection(results[0])


def _detection_to_localization(detection: dict) -> Localization:
    """Convert a raw detection dict to a Localization exactly like localize_object."""

    values = np.asarray(detection["box"], dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Grounding DINO returned an invalid box")
    x1, y1, x2, y2 = (float(value) for value in values)
    left = int(np.clip(math.floor(min(x1, x2)), 0, WIDTH - 1))
    top = int(np.clip(math.floor(min(y1, y2)), 0, HEIGHT - 1))
    right = int(np.clip(math.ceil(max(x1, x2)), 0, WIDTH - 1))
    bottom = int(np.clip(math.ceil(max(y1, y2)), 0, HEIGHT - 1))
    if right <= left or bottom <= top:
        raise ValueError("Grounding DINO returned a zero-area box")
    return Localization(
        box=(left, top, right, bottom),
        score=float(detection["score"]),
        label=str(detection["label"]),
    )


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _print_deviation(
    label: str,
    center: tuple[float, float],
    depth_m: float,
    world: tuple[float, float, float],
    truth: tuple[float, float, float],
) -> None:
    dx = world[0] - truth[0]
    dy = world[1] - truth[1]
    dz = world[2] - truth[2]
    xy_dev_mm = math.hypot(dx, dy) * MILLIMETRES
    z_dev_mm = dz * MILLIMETRES
    nominal_top_z = truth[2] + 0.025
    z_above_top_mm = (world[2] - nominal_top_z) * MILLIMETRES
    print("=" * 78)
    print(f"[{label}]")
    print(f"  2D centre pixel            : (u, v) = ({center[0]:.3f}, {center[1]:.3f})")
    print(f"  depth at pixel             : {depth_m * 1000.0:.3f} mm")
    print(
        f"  backprojected world point   : x={world[0]:.6f}  y={world[1]:.6f}  "
        f"z={world[2]:.6f}"
    )
    print(f"  cube truth centroid        : x={truth[0]:.6f}  y={truth[1]:.6f}  "
          f"z={truth[2]:.6f}")
    print(f"  XY deviation               : {xy_dev_mm:8.3f} mm")
    print(f"  Z deviation (world-truth)  : {z_dev_mm:8.3f} mm")
    print(f"  Z offset above nominal top : {z_above_top_mm:8.3f} mm")


def main() -> None:
    print("#" * 78)
    print("# Overhead-camera single-pixel backprojection diagnostic")
    print("#" * 78)
    print(f"camera eye    = {CAMERA_EYE}")
    print(f"camera target = {CAMERA_TARGET}")
    print(
        f"baseline (oblique) XY deviation to beat: "
        f"{BASELINE_XY_DEVIATION_MM} mm"
    )
    print()

    camera_config = CameraConfig(
        width=WIDTH,
        height=HEIGHT,
        eye=CAMERA_EYE,
        target=CAMERA_TARGET,
        up=CAMERA_UP,
    )
    scene_config = replace(
        fixed_scene_config(
            MultiObjectStudyConfig(
                gui=False,
                seed=SEED,
                device="cpu",
            )
        ),
        robot_self_collision=True,
    )

    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        frame = capture_camera_frame(
            scene.client_id,
            camera_config,
            scene.renderer,
        )

        entity_ids = {
            **scene.object_body_ids,
            "robot": scene.bodies.robot,
        }
        masks = {
            name: segmentation_mask_for_body(frame.segmentation, body_id)
            for name, body_id in entity_ids.items()
        }
        # Some entities (e.g. the robot at x=0) are outside the overhead
        # camera frustum and have no visible pixels; only box non-empty masks.
        boxes = {
            name: mask_to_box(mask)
            for name, mask in masks.items()
            if bool(np.any(mask))
        }

        cube_body_id = entity_ids[TARGET_NAME]
        if TARGET_NAME not in boxes:
            raise RuntimeError(
                f"cube has no visible pixels under the overhead camera"
            )
        cube_truth = tuple(float(v) for v in scene.object_poses()[TARGET_NAME]["position"])
        print(f"live cube-truth centroid (object_poses) = {cube_truth}")
        print()

        # -- Grounding DINO "red cube" localization (in-memory, no file) ----
        processor, model = load_grounding_dino(MODEL_ID, DEVICE)
        detection = _run_grounding_dino_in_memory(
            processor,
            model,
            DEVICE,
            frame.rgb,
        )
        if detection is None:
            print("Grounding DINO: no detection; falling back to the "
                  "segmentation mask box for localization.")
            localization = Localization(
                box=boxes[TARGET_NAME],
                score=0.0,
                label="mask_fallback",
            )
        else:
            localization = _detection_to_localization(detection)
        gdino_box = localization.box
        gdino_iou_cube = box_iou(gdino_box, boxes[TARGET_NAME])
        print(
            f"Grounding DINO box {gdino_box} score={localization.score:.3f} "
            f"label={localization.label!r}  IoU(cube mask box)={gdino_iou_cube:.3f}"
        )

        # -- Geometry backend grasp prediction (in-memory) ------------------
        image_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        prediction = predict_grasp(
            image_bgr,
            localization,
            "geometry",
            DEVICE,
            None,
        )
        grasp = prediction.grasp
        grasp_center = (float(grasp["center_x"]), float(grasp["center_y"]))
        print(
            f"geometry grasp centre = ({grasp_center[0]:.3f}, {grasp_center[1]:.3f})  "
            f"failure_reason={prediction.failure_reason!r}"
        )
        print()

        # -- 2-D centre candidates ------------------------------------------
        mask_center = _box_center(boxes[TARGET_NAME])
        centers = (
            ("segmentation_mask_center", mask_center),
            ("grounding_dino_box_center", _box_center(gdino_box)),
            ("geometry_grasp_center", grasp_center),
        )

        # -- single-pixel backprojection (NO center_recovery) ---------------
        for label, (u, v) in centers:
            sample = sample_nearest_depth(
                frame.depth_m,
                u,
                v,
                camera_config.near,
                camera_config.far,
            )
            point = backproject_pixel(
                sample.column,
                sample.row,
                sample.depth_m,
                WIDTH,
                HEIGHT,
                frame.view_matrix,
                frame.projection_matrix,
                camera_config.near,
                camera_config.far,
            )
            _print_deviation(
                label,
                center=(float(sample.column), float(sample.row)),
                depth_m=sample.depth_m,
                world=tuple(float(value) for value in point.world_xyz),
                truth=cube_truth,
            )

    print()
    print("=" * 78)
    print("done (no files written)")


if __name__ == "__main__":
    main()
