from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav import demo_motion_predictor as motion_adapter
from visualnav_transformer.eat_nav.demo_motion_predictor import (
    DEMO_CHECKPOINT_SHA256,
    load_demo_motion_predictor,
    prepare_demo_motion_inputs,
)


CHECKPOINT = Path(
    '/home/kojima/lib/visualnav-transformer-ros2/'
    '.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/'
    'training/full-seed456/best.pt'
)


def test_three_forward_paths_are_padded_to_six_without_mutating_inputs():
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    state = np.arange(22, dtype=np.float32)
    paths = np.arange(48, dtype=np.float32).reshape(3, 8, 2) / 100
    original = dem.copy(), state.copy(), paths.copy()
    norm = {'dem_height_mean': .1, 'dem_height_std': .2,
            'state_mean': [1.] * 22, 'state_std': [2.] * 22}

    normalized_dem, normalized_state, six_paths, direction, mask = (
        prepare_demo_motion_inputs(dem, state, paths, norm)
    )

    assert normalized_dem.shape == (1, 4, 2, 20, 20)
    assert normalized_state.shape == (1, 22)
    assert six_paths.shape == (1, 6, 8, 2)
    assert direction.shape == (1, 6, 1)
    assert mask.tolist() == [[True, True, True, False, False, False]]
    assert np.array_equal(six_paths[0, :3], paths)
    assert np.array_equal(six_paths[0, 3:], np.zeros((3, 8, 2)))
    assert np.array_equal(direction[0, :3, 0], np.ones(3))
    assert np.allclose(normalized_dem[0, :, 0], -.5)
    assert np.allclose(normalized_state[0], (state - 1) / 2)
    for actual, expected in zip((dem, state, paths), original):
        assert np.array_equal(actual, expected)


def test_seven_front_arcs_are_scored_across_two_model_batches():
    if not CHECKPOINT.is_file():
        pytest.skip('sealed MI-B checkpoint unavailable')
    predictor = load_demo_motion_predictor(CHECKPOINT)
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    paths = np.zeros((7, 8, 2), np.float32)
    paths[:, :, 0] = np.linspace(0, .75, 8)
    paths[1:, :, 1] = np.arange(1, 7, dtype=np.float32)[:, None] * .02
    state = np.zeros(22, np.float32)
    normed_dem, normed_state, slots, directions, mask = (
        prepare_demo_motion_inputs(dem, state, paths, predictor.normalization)
    )
    assert normed_dem.shape == (2, 4, 2, 20, 20)
    assert normed_state.shape == (2, 22)
    assert slots.shape == (2, 6, 8, 2)
    assert np.array_equal(slots.reshape(12, 8, 2)[:7], paths)
    assert mask.tolist() == [[True] * 6, [True] + [False] * 5]
    assert np.all(directions[:, :, 0] == 1)

    motion, scores = predictor.predict(dem, state, paths,
                                       np.asarray([0., 0., 0.]),
                                       np.asarray([2., 0.]))
    assert motion.shape == (7, 2) and np.isfinite(motion).all()
    assert scores.shape == (7,) and np.isfinite(scores).all()
    assert np.allclose(scores, motion_adapter.global_goal_progress(motion, [2., 0.]))


def test_identical_forward_path_predictions_do_not_depend_on_model_slot():
    if not CHECKPOINT.is_file():
        pytest.skip('sealed MI-B checkpoint unavailable')
    predictor = load_demo_motion_predictor(CHECKPOINT)
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    path = np.zeros((8, 2), np.float32)
    path[:, 0] = np.linspace(0, .75, 8)
    paths = np.repeat(path[None], 7, axis=0)
    motion, _ = predictor.predict(dem, np.zeros(22, np.float32), paths,
                                  np.asarray([0., 0., 0.]), np.asarray([2., 0.]))
    assert np.allclose(motion, motion[0], atol=1e-7)


def test_global_goal_is_transformed_to_current_body_frame_without_distance_clipping():
    goal = motion_adapter.global_goal_body_xy(np.asarray([2., 3., np.pi / 2]),
                                              np.asarray([2., 5.]))
    assert np.allclose(goal, [2., 0.], atol=1e-6)


def test_global_goal_progress_uses_full_goal_distance():
    predicted = np.asarray([[.4, 0], [.25, .15], [-.2, 0]], np.float32)
    scores = motion_adapter.global_goal_progress(predicted, np.asarray([2., 0], np.float32))
    assert np.allclose(scores, [.4, 2. - np.hypot(1.75, -.15), -.2])
    with pytest.raises(ValueError):
        motion_adapter.global_goal_progress(predicted, np.asarray([0, 0], np.float32))


def test_pinned_full_checkpoint_loads_and_predicts_finite_motion():
    if not CHECKPOINT.is_file():
        pytest.skip('sealed MI-B checkpoint unavailable')
    predictor = load_demo_motion_predictor(CHECKPOINT)
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    paths = np.zeros((3, 8, 2), np.float32)
    paths[:, :, 0] = np.linspace(.09375, .75, 8)
    paths[1, :, 1] = .1
    paths[2, :, 1] = -.1
    pose = np.asarray([1., 2., np.pi / 2], np.float32)
    goal_map = np.asarray([1., 4.], np.float32)
    motion, scores = predictor.predict(dem, np.zeros(22, np.float32), paths,
                                       pose, goal_map)
    assert predictor.checkpoint_sha256 == DEMO_CHECKPOINT_SHA256
    assert motion.shape == (3, 2) and np.isfinite(motion).all()
    assert scores.shape == (3,) and np.isfinite(scores).all()
    assert np.allclose(scores, motion_adapter.global_goal_progress(motion, [2., 0.]), atol=1e-6)


def test_checkpoint_sha_mismatch_fails_closed(tmp_path):
    path = tmp_path / 'wrong.pt'
    path.write_bytes(b'not the frozen MI-B checkpoint')
    with pytest.raises(ValueError, match='SHA'):
        load_demo_motion_predictor(path)
