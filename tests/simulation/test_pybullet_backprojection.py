import numpy as np
import pybullet as p
import pytest

from src.simulation.pybullet.backprojection import (
    audit_backprojected_grasp,
    backproject_image_coordinate,
    backproject_pixel,
    build_ray_segment,
    metric_depth_to_buffer,
    reproject_world_point,
    sample_nearest_depth,
    summarize_available_backprojection_rows,
    summarize_backprojection_rows,
)
from src.simulation.pybullet.camera import CameraConfig, capture_camera_frame
from src.simulation.pybullet.scene import PyBulletScene, SceneConfig


def test_nearest_depth_uses_half_up_pixel_rounding() -> None:
    depth = np.arange(20, dtype=np.float32).reshape(4, 5) / 10 + 0.1

    sample = sample_nearest_depth(
        depth,
        center_x=1.5,
        center_y=2.5,
        near=0.05,
        far=3.0,
    )

    assert (sample.column, sample.row) == (2, 3)
    assert sample.depth_m == pytest.approx(float(depth[3, 2]))


@pytest.mark.parametrize(
    ("center_x", "center_y"),
    [(-0.01, 0.0), (5.0, 0.0)],
)
def test_nearest_depth_rejects_centres_outside_image(
    center_x: float,
    center_y: float,
) -> None:
    with pytest.raises(ValueError, match="inside image"):
        sample_nearest_depth(
            np.ones((4, 5), dtype=np.float32),
            center_x,
            center_y,
            near=0.05,
            far=3.0,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.05, 3.0])
def test_nearest_depth_rejects_non_surface_depth(value: float) -> None:
    depth = np.full((2, 2), 0.8, dtype=np.float32)
    depth[1, 1] = value

    with pytest.raises(ValueError, match="depth"):
        sample_nearest_depth(
            depth,
            center_x=1.0,
            center_y=1.0,
            near=0.05,
            far=3.0,
        )


def test_metric_depth_inverse_maps_clip_planes() -> None:
    assert metric_depth_to_buffer(0.1, 0.1, 10.0) == pytest.approx(0.0)
    assert metric_depth_to_buffer(10.0, 0.1, 10.0) == pytest.approx(1.0)


def test_backprojection_recovers_hand_derived_world_point() -> None:
    view = p.computeViewMatrix(
        cameraEyePosition=(0.0, 0.0, 1.0),
        cameraTargetPosition=(0.0, 0.0, 0.0),
        cameraUpVector=(0.0, 1.0, 0.0),
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=10.0,
    )

    point = backproject_pixel(
        column=0,
        row=0,
        depth_m=0.5,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
        near=0.1,
        far=10.0,
    )

    assert point.camera_xyz == pytest.approx(
        (0.0, 0.0, -0.5), abs=1e-6
    )
    assert point.world_xyz == pytest.approx(
        (0.0, 0.0, 0.5), abs=1e-6
    )

    reprojection = reproject_world_point(
        point.world_xyz,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
    )
    assert (reprojection.pixel_x, reprojection.pixel_y) == pytest.approx(
        (0.0, 0.0), abs=1e-6
    )
    assert reprojection.depth_m == pytest.approx(0.5, abs=1e-6)


def test_fractional_image_coordinate_round_trips() -> None:
    view, projection = _downward_camera_matrices()

    point = backproject_image_coordinate(
        pixel_x=1.25,
        pixel_y=2.75,
        depth_m=0.5,
        width=4,
        height=4,
        view_matrix=view,
        projection_matrix=projection,
        near=0.1,
        far=10.0,
    )
    reprojection = reproject_world_point(
        point.world_xyz,
        width=4,
        height=4,
        view_matrix=view,
        projection_matrix=projection,
    )

    assert reprojection.pixel_x == pytest.approx(1.25, abs=1e-6)
    assert reprojection.pixel_y == pytest.approx(2.75, abs=1e-6)
    assert reprojection.depth_m == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("pixel_x", [-0.01, 4.0, np.nan])
def test_fractional_backprojection_rejects_invalid_x(pixel_x: float) -> None:
    view, projection = _downward_camera_matrices()

    with pytest.raises(ValueError, match="image coordinate"):
        backproject_image_coordinate(
            pixel_x=pixel_x,
            pixel_y=1.0,
            depth_m=0.5,
            width=4,
            height=4,
            view_matrix=view,
            projection_matrix=projection,
            near=0.1,
            far=10.0,
        )


def _downward_camera_matrices() -> tuple[tuple[float, ...], tuple[float, ...]]:
    view = p.computeViewMatrix(
        cameraEyePosition=(0.0, 0.0, 1.0),
        cameraTargetPosition=(0.0, 0.0, 0.0),
        cameraUpVector=(0.0, 1.0, 0.0),
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=10.0,
    )
    return tuple(view), tuple(projection)


def test_audit_accepts_matching_surface_point() -> None:
    view, projection = _downward_camera_matrices()

    def ray_test(ray_from, ray_to, client_id):
        assert ray_from == (0.0, 0.0, 1.0)
        assert client_id == 7
        return 3, (0.0, 0.0, 0.5)

    audit = audit_backprojected_grasp(
        backend_row={
            "target": "duck",
            "backend": "geometry",
            "center_x": 0.0,
            "center_y": 0.0,
        },
        depth_m=np.array([[0.5]], dtype=np.float32),
        segmentation=np.array([[3]], dtype=np.int32),
        expected_body_id=3,
        camera_eye=(0.0, 0.0, 1.0),
        image_width=1,
        image_height=1,
        near=0.1,
        far=10.0,
        view_matrix=view,
        projection_matrix=projection,
        client_id=7,
        ray_test=ray_test,
    )

    assert audit.sampled_column == 0
    assert audit.sampled_row == 0
    assert audit.depth_m == pytest.approx(0.5)
    assert audit.world_z == pytest.approx(0.5, abs=1e-6)
    assert audit.pixel_error == pytest.approx(0.0, abs=1e-6)
    assert audit.depth_error_m == pytest.approx(0.0, abs=1e-6)
    assert audit.coordinates_finite
    assert audit.valid_depth
    assert audit.reprojection_passed
    assert audit.segmentation_target_match
    assert audit.ray_target_match
    assert audit.gate_passed
    assert audit.failure_reason == ""


def test_exact_nine_summary_rejects_reordered_rows() -> None:
    view, projection = _downward_camera_matrices()
    rows = []
    for target, body_id in (("duck", 3), ("cube", 4), ("sphere", 5)):
        for backend in ("geometry", "single", "multi_head"):
            rows.append(
                audit_backprojected_grasp(
                    backend_row={
                        "target": target,
                        "backend": backend,
                        "center_x": 0.0,
                        "center_y": 0.0,
                    },
                    depth_m=np.array([[0.5]], dtype=np.float32),
                    segmentation=np.array([[body_id]], dtype=np.int32),
                    expected_body_id=body_id,
                    camera_eye=(0.0, 0.0, 1.0),
                    image_width=1,
                    image_height=1,
                    near=0.1,
                    far=10.0,
                    view_matrix=view,
                    projection_matrix=projection,
                    client_id=7,
                    ray_test=lambda _start, _end, _client, hit=body_id: (
                        hit,
                        (0.0, 0.0, 0.5),
                    ),
                )
            )

    summary = summarize_backprojection_rows(rows)
    assert summary["backprojection_result_count"] == 9
    assert summary["backprojection_gate_passed"] is True

    with pytest.raises(ValueError, match="exact target/backend order"):
        summarize_backprojection_rows(list(reversed(rows)))


def test_available_summary_never_passes_an_incomplete_gate() -> None:
    view, projection = _downward_camera_matrices()
    row = audit_backprojected_grasp(
        backend_row={
            "target": "duck",
            "backend": "geometry",
            "center_x": 0.0,
            "center_y": 0.0,
        },
        depth_m=np.array([[0.5]], dtype=np.float32),
        segmentation=np.array([[3]], dtype=np.int32),
        expected_body_id=3,
        camera_eye=(0.0, 0.0, 1.0),
        image_width=1,
        image_height=1,
        near=0.1,
        far=10.0,
        view_matrix=view,
        projection_matrix=projection,
        client_id=7,
        ray_test=lambda _start, _end, _client: (
            3,
            (0.0, 0.0, 0.5),
        ),
    )

    summary = summarize_available_backprojection_rows([row])

    assert summary["backprojection_result_count"] == 1
    assert summary["backprojection_complete"] is False
    assert summary["backprojection_gate_passed"] is False


def test_real_pybullet_frame_backprojects_to_the_rendered_target() -> None:
    camera = CameraConfig(width=160, height=120)
    with PyBulletScene(
        SceneConfig(
            gui=False,
            object_urdf="cube_small.urdf",
            object_name="cube",
            object_position=(0.55, 0.0, 0.66),
        )
    ) as scene:
        scene.step(10)
        frame = capture_camera_frame(
            scene.client_id,
            camera,
            scene.renderer,
        )
        target_body = scene.bodies.target_object
        decoded = frame.segmentation & ((1 << 24) - 1)
        target_pixels = np.argwhere(
            (frame.segmentation >= 0) & (decoded == target_body)
        )
        assert len(target_pixels) > 0
        centroid = target_pixels.mean(axis=0)
        row, column = target_pixels[
            np.argmin(np.sum((target_pixels - centroid) ** 2, axis=1))
        ]

        point = backproject_pixel(
            column=int(column),
            row=int(row),
            depth_m=float(frame.depth_m[row, column]),
            width=camera.width,
            height=camera.height,
            view_matrix=frame.view_matrix,
            projection_matrix=frame.projection_matrix,
            near=camera.near,
            far=camera.far,
        )
        reprojection = reproject_world_point(
            point.world_xyz,
            width=camera.width,
            height=camera.height,
            view_matrix=frame.view_matrix,
            projection_matrix=frame.projection_matrix,
        )
        ray_from, ray_to = build_ray_segment(camera.eye, point.world_xyz)
        hit = p.rayTest(
            ray_from,
            ray_to,
            physicsClientId=scene.client_id,
        )[0]

        assert int(decoded[row, column]) == target_body
        assert reprojection.pixel_x == pytest.approx(column, abs=1e-5)
        assert reprojection.pixel_y == pytest.approx(row, abs=1e-5)
        assert reprojection.depth_m == pytest.approx(
            float(frame.depth_m[row, column]),
            abs=1e-5,
        )
        assert hit[0] == target_body
