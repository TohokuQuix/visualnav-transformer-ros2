import math

import numpy as np
import pytest

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem


def plane_grid(offset=0.0, start=(0, 0)):
    x = (np.arange(20) - 9.5) * 0.1
    y = (np.arange(20) - 9.5) * 0.1
    logical = 0.2 * x[None, :] + 0.3 * y[:, None] + offset
    physical = np.roll(logical, shift=(start[1], start[0]), axis=(0, 1))
    return SourceGrid(
        height=physical.astype(np.float32),
        valid=np.ones((20, 20), dtype=bool),
        origin_xy=(-0.95, -0.95),
        resolution_m=0.1,
        start_index_xy=start,
        frame_id="map",
        stamp_ns=1_000_000_000,
    )


def test_dem_axis_geometry_and_scalar_offset_preserve_slope():
    out = robot_dem(plane_grid(), (0.0, 0.0, 0.0), EATConfig())

    assert out.shape == (2, 20, 20)
    assert out[1].all()
    assert out[0, 10, 10] == pytest.approx(0.025, abs=1e-5)
    assert out[0, 10, 11] - out[0, 10, 10] == pytest.approx(0.02, abs=1e-5)
    assert out[0, 11, 10] - out[0, 10, 10] == pytest.approx(0.03, abs=1e-5)


@pytest.mark.parametrize(
    ("yaw", "expected_center"),
    [(0.0, 0.025), (math.pi / 4, 0.0212132034), (math.pi / 2, 0.005)],
)
def test_robot_yaw_rotates_xy_samples_without_leveling_terrain(yaw, expected_center):
    out = robot_dem(plane_grid(), (0.0, 0.0, yaw), EATConfig())

    assert out[0, 10, 10] == pytest.approx(expected_center, abs=1e-5)
    if yaw == math.pi / 4:
        assert out[1, 0, 0] == 0


def test_circular_buffer_start_index_does_not_change_physical_terrain():
    plain = robot_dem(plane_grid(), (0.0, 0.0, 0.0), EATConfig())
    wrapped = robot_dem(plane_grid(start=(7, 13)), (0.0, 0.0, 0.0), EATConfig())

    np.testing.assert_allclose(wrapped, plain, atol=1e-6)


def test_translation_and_absolute_height_offset_do_not_change_relative_dem():
    reference = robot_dem(plane_grid(), (0.0, 0.0, 0.0), EATConfig())
    shifted = robot_dem(plane_grid(offset=11.0), (0.1, -0.2, 0.0), EATConfig())

    common = (reference[1] != 0) & (shifted[1] != 0)
    np.testing.assert_allclose(shifted[0, common], reference[0, common], atol=1e-5)


def test_nan_and_masked_cells_remain_unknown_without_extrapolation():
    grid = plane_grid()
    height = grid.height.copy()
    height[10, 10] = np.nan
    grid = SourceGrid(height, grid.valid, grid.origin_xy, grid.resolution_m,
                      grid.start_index_xy, grid.frame_id, grid.stamp_ns)
    out = robot_dem(grid, (0.0, 0.0, 0.0), EATConfig())

    assert out[1, 10, 10] == 0
    assert out[0, 10, 10] == 0
    assert out[1, 10, 11] == 1
    assert np.isfinite(out[0]).all()


def test_insufficient_valid_cells_are_rejected():
    grid = plane_grid()
    valid = np.zeros((20, 20), dtype=bool)
    grid = SourceGrid(grid.height, valid, grid.origin_xy, grid.resolution_m,
                      grid.start_index_xy, grid.frame_id, grid.stamp_ns)

    with pytest.raises(ValueError, match="valid"):
        robot_dem(grid, (0.0, 0.0, 0.0), EATConfig())


def test_source_grid_must_match_real_grid_dimensions():
    grid = SourceGrid(
        np.zeros((21, 20), dtype=np.float32),
        np.ones((21, 20), dtype=bool),
        (-0.95, -1.0),
        0.1,
        (0, 0),
        "map",
        0,
    )

    with pytest.raises(ValueError, match="shape"):
        robot_dem(grid, (0.0, 0.0, 0.0), EATConfig())
