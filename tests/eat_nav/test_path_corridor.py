import numpy as np

from visualnav_transformer.eat_nav.path_corridor import (
    bilinear_world, corridor_points, local_to_world, resample_path, sample_source_grid,
)


def test_bilinear_rejects_invalid_contributor_and_outside():
    height = np.arange(4, dtype=float).reshape(2, 2)
    mask = np.ones((2, 2), dtype=float)
    mask[1, 1] = 0
    assert bilinear_world(height, mask, (0, 0), 1, (0.5, 0.5)) == (0.0, False)
    assert bilinear_world(height, mask, (0, 0), 1, (-0.1, 0.5)) == (0.0, False)
    assert bilinear_world(height, mask, (0, 0), 1, (0.5, 0.0)) == (0.5, True)


def test_local_to_world_rotates_and_translates():
    xy = local_to_world(np.array([[1.0, 0.0], [0.0, 1.0]]),
                        (4.0, -2.0), np.pi / 2)
    np.testing.assert_allclose(xy, [[4, -1], [3, -2]], atol=1e-12)


def test_corridor_sampling_includes_origin_and_lateral_offsets():
    path = np.column_stack([np.linspace(0.1, 0.8, 8), np.zeros(8)])
    stations, tangents, distance = resample_path(path, 16)
    points = corridor_points(stations, tangents, np.linspace(-0.4, 0.4, 7))
    assert points.shape == (16, 7, 2)
    np.testing.assert_allclose(points[0, 3], [0, 0], atol=1e-12)
    np.testing.assert_allclose(points[-1, 0], [0.8, -0.4], atol=1e-12)
    np.testing.assert_allclose(distance[-1], 0.8, atol=1e-12)


def test_rotated_anchor_query_samples_world_grid_at_correct_location():
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    dem = {'source_height': x + 10 * y, 'source_mask': np.ones((5, 5)),
           'origin_xy': [0, 0], 'resolution_m': 1.0}
    anchor_local = np.array([[[1.0, 0.0]]])
    world = local_to_world(anchor_local, (2, 2), np.pi / 2)
    sampled = sample_source_grid(dem, world, reference_height=0)
    assert sampled.shape == (2, 1, 1)
    np.testing.assert_allclose(sampled[:, 0, 0], [32, 1])
