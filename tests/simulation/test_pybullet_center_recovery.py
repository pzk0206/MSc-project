"""Tests for windowed top-surface depth centre recovery."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.simulation.pybullet.backprojection import (
    backproject_pixel,
)
from src.simulation.pybullet.center_recovery import (
    CENTER_RECOVERY_PROTOCOL,
    CENTER_RECOVERY_WINDOW_SIZE,
    recover_center_via_windowed_depth,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_matrices() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a simple view+projection pair for synthetic tests.

    The matrices produce a well-behaved pinhole camera looking along the
    world -Z axis from (0, 0, 1) with reasonable clipping.
    """
    # View: identity (camera at origin looking -Z).
    view = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, -1.0,
        0.0, 0.0, 0.0, 1.0,
    )
    # Projection: simple perspective, fov ~90°, near=0.01, far=100.
    f = 1.0
    near, far = 0.01, 100.0
    projection = (
        f, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far),
        0.0, 0.0, -1.0, 0.0,
    )
    return view, projection


def _make_depth(
    height: int = 10,
    width: int = 10,
    base_depth: float = 0.5,
) -> np.ndarray:
    """Return a constant depth image at *base_depth* metres."""
    return np.full((height, width), base_depth, dtype=np.float32)


def _make_seg(
    body_id: int,
    height: int = 10,
    width: int = 10,
) -> np.ndarray:
    """Return a segmentation image where every pixel belongs to *body_id*."""
    return np.full((height, width), body_id, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecoverCenterValidation:
    """Input validation rejections."""

    def test_even_window_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                _make_depth(),
                _make_seg(1),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
                window_size=4,
            )

    def test_zero_window_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                _make_depth(),
                _make_seg(1),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
                window_size=0,
            )

    def test_negative_window_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                _make_depth(),
                _make_seg(1),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
                window_size=-3,
            )

    def test_out_of_bounds_centre_rejected(self) -> None:
        with pytest.raises(ValueError, match="inside image"):
            recover_center_via_windowed_depth(
                -0.5, 2.0,
                _make_depth(),
                _make_seg(1),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
            )

    def test_non_finite_centre_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            recover_center_via_windowed_depth(
                math.nan, 2.0,
                _make_depth(),
                _make_seg(1),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
            )

    def test_negative_body_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_body_id"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                _make_depth(),
                _make_seg(1),
                -1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
            )

    def test_depth_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="depth_m"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                np.zeros((5, 5), dtype=np.float32),
                _make_seg(1, 10, 10),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
            )

    def test_seg_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="segmentation"):
            recover_center_via_windowed_depth(
                5.0, 5.0,
                _make_depth(10, 10),
                np.zeros((5, 5), dtype=np.int32),
                1, 10, 10,
                *_default_matrices(),
                0.01, 100.0,
            )


class TestRecoverCenterBasic:
    """Happy-path behaviour on synthetic data."""

    def test_selects_min_depth_in_window(self) -> None:
        """Pixels in a 5×5 window have varying depths; the shallowest wins."""
        depth = np.full((10, 10), 0.8, dtype=np.float32)
        depth[3:6, 3:6] = 0.3  # some pixels shallower
        depth[4, 4] = 0.2       # this is the shallowest
        seg = _make_seg(7, 10, 10)
        view, proj = _default_matrices()

        pixel, corrected_d, world = recover_center_via_windowed_depth(
            4.5, 4.5, depth, seg, 7, 10, 10,
            view, proj, 0.01, 100.0,
        )

        # Pixel (4, 4) is the shallowest.
        assert pixel == (4, 4)
        assert corrected_d == pytest.approx(0.2)

        # Backproject centre pixel (5, 5) with corrected depth ~0.2.
        ref = backproject_pixel(
            5, 5, corrected_d, 10, 10, view, proj, 0.01, 100.0,
        )
        assert world == pytest.approx(ref.world_xyz)

    def test_filters_by_target_mask(self) -> None:
        """Pixels not belonging to the target are ignored."""
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        depth[5, 5] = 0.1  # shallow but wrong body
        seg = np.full((10, 10), 2, dtype=np.int32)
        seg[5, 5] = 99       # different object
        view, proj = _default_matrices()

        # The shallowest pixel (5,5) is NOT target body 2.
        # The window contains other pixels at depth 0.5 belonging to body 2.
        pixel, corrected_d, world = recover_center_via_windowed_depth(
            5.0, 5.0, depth, seg, 2, 10, 10,
            view, proj, 0.01, 100.0,
        )

        assert corrected_d == pytest.approx(0.5)
        # The selected pixel should not be (5, 5).
        assert pixel != (5, 5)

    def test_handles_negative_segmentation_values(self) -> None:
        """PyBullet uses -1 for 'no object'; those pixels are skipped."""
        depth = np.full((10, 10), 0.6, dtype=np.float32)
        seg = np.full((10, 10), -1, dtype=np.int32)
        seg[4:7, 4:7] = 3  # small target island
        view, proj = _default_matrices()

        pixel, corrected_d, world = recover_center_via_windowed_depth(
            5.0, 5.0, depth, seg, 3, 10, 10,
            view, proj, 0.01, 100.0,
        )

        assert corrected_d == pytest.approx(0.6)
        # The centre pixel (5,5) is inside the island.
        col, row = pixel
        assert 4 <= col <= 6 and 4 <= row <= 6

    def test_clips_window_at_image_boundary(self) -> None:
        """The centre is near the top-left corner; window is clipped."""
        depth = np.full((10, 10), 0.7, dtype=np.float32)
        depth[0, 0] = 0.1
        seg = _make_seg(5, 10, 10)
        view, proj = _default_matrices()

        # Centre at (0.1, 0.1) -> nearest pixel (0, 0).
        # 5x5 window would extend to (-2, -2) but should clip.
        pixel, corrected_d, world = recover_center_via_windowed_depth(
            0.1, 0.1, depth, seg, 5, 10, 10,
            view, proj, 0.01, 100.0,
        )

        assert pixel == (0, 0)
        assert corrected_d == pytest.approx(0.1)

    def test_raises_when_no_target_pixel_in_window(self) -> None:
        """All target-mask pixels are excluded from the window."""
        depth = _make_depth(10, 10, 0.5)
        seg = _make_seg(99, 10, 10)  # none match target_body_id=7
        view, proj = _default_matrices()

        with pytest.raises(ValueError, match="no valid target pixel"):
            recover_center_via_windowed_depth(
                5.0, 5.0, depth, seg, 7, 10, 10,
                view, proj, 0.01, 100.0,
            )

    def test_raises_when_all_depths_out_of_clip(self) -> None:
        """All depths in window are at or beyond clipping planes."""
        depth = np.full((10, 10), 0.001, dtype=np.float32)  # too shallow
        seg = _make_seg(1, 10, 10)
        view, proj = _default_matrices()

        with pytest.raises(ValueError, match="no valid target pixel"):
            recover_center_via_windowed_depth(
                5.0, 5.0, depth, seg, 1, 10, 10,
                view, proj, 0.01, 100.0,
            )

    def test_window_size_one_is_valid(self) -> None:
        """window_size=1 means exact single-pixel sampling."""
        depth = np.full((10, 10), 0.42, dtype=np.float32)
        seg = _make_seg(3, 10, 10)
        view, proj = _default_matrices()

        pixel, corrected_d, world = recover_center_via_windowed_depth(
            3.3, 6.7, depth, seg, 3, 10, 10,
            view, proj, 0.01, 100.0,
            window_size=1,
        )

        assert pixel == (3, 7)  # nearest pixel to (3.3, 6.7)
        assert corrected_d == pytest.approx(0.42)


class TestRecoverCenterConstants:
    """Frozen constants match expectations."""

    def test_window_size_is_odd_positive(self) -> None:
        assert CENTER_RECOVERY_WINDOW_SIZE >= 1
        assert CENTER_RECOVERY_WINDOW_SIZE % 2 == 1

    def test_protocol_is_non_empty(self) -> None:
        assert len(CENTER_RECOVERY_PROTOCOL) > 0
