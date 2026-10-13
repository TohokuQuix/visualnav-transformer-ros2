import json
from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav.demo_contract import (
    candidates_for_goal, load_bank, select_best, sensor_stamp_valid, validate_evaluator_sample,
)


BANK = Path(__file__).resolve().parents[1] / 'config/eat_nav_demo_p1_bank.json'


def test_raycaster_float_stamp_slack_and_staleness():
    now = 2_250_000_000
    assert sensor_stamp_valid(now, now + 15_497)
    assert sensor_stamp_valid(now, now - 250_000_000)
    assert not sensor_stamp_valid(now, now + 1_000_001)
    assert not sensor_stamp_valid(now, now - 300_000_001)


def test_frozen_three_path_bank_and_goal_rotation():
    bank = load_bank(BANK)
    assert set(bank) == {'direct', 'left_20', 'right_20'}
    body, mapped, meta = candidates_for_goal(bank, (2., 3., 0.), (2., 3.75))
    assert meta['subgoal_distance_m'] == .75
    assert np.allclose(body['direct'][-1], [0, .75], atol=1e-6)
    assert np.allclose(mapped['direct'][-1], [2., 3.75], atol=1e-6)
    assert body['left_20'].shape == (8, 2)
    assert select_best([.1, .5, .3]) == 'left_20'


def test_evaluator_input_and_stale_time_fail_closed():
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    state = np.zeros(22, np.float32)
    paths = np.zeros((3, 8, 2), np.float32)
    stamps = [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]
    validate_evaluator_sample(dem, state, paths, dem_stamps_ns=stamps,
                              state_stamp_ns=stamps[-1], now_ns=stamps[-1])
    with pytest.raises(ValueError, match='timestamp'):
        validate_evaluator_sample(dem, state, paths, dem_stamps_ns=stamps,
                                  state_stamp_ns=stamps[-1] + 50_000_000,
                                  now_ns=stamps[-1] + 50_000_000)
    with pytest.raises(ValueError, match='timestamp'):
        validate_evaluator_sample(dem, state, paths, dem_stamps_ns=stamps,
                                  state_stamp_ns=stamps[-1], now_ns=stamps[-1] + 400_000_000)
    dem[0, 1] = 0
    with pytest.raises(ValueError, match='mask'):
        validate_evaluator_sample(dem, state, paths, dem_stamps_ns=stamps,
                                  state_stamp_ns=stamps[-1], now_ns=stamps[-1])


def test_frozen_bank_is_not_mutated():
    before = BANK.read_bytes()
    bank = load_bank(BANK)
    candidates_for_goal(bank, (0, 0, 0), (.75, 0))
    assert BANK.read_bytes() == before
    assert json.loads(before)['source_goal_base_xy_m'] == [.75, 0.]


def test_pilot_checkpoint_scores_three_goal_candidates():
    from visualnav_transformer.eat_nav.demo_contract import DATASET_SHA256
    from visualnav_transformer.eat_nav.evaluator.inference import load_checkpoint

    checkpoint = Path('/home/kojima/eatnav-experiments/baseline-evaluator-pilot-20261013/train-common-goal-v1/checkpoint.pt')
    if not checkpoint.is_file():
        pytest.skip('local pilot checkpoint unavailable')
    evaluator, _ = load_checkpoint(checkpoint, expected_label_type='common_goal_progress',
                                   expected_dataset_sha256=DATASET_SHA256)
    bank = load_bank(BANK)
    body, _, _ = candidates_for_goal(bank, (0., 0., 0.), (.75, 0.))
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    state = np.zeros(22, np.float32)
    paths = np.stack([body[role] for role in ('direct', 'left_20', 'right_20')])
    scores = evaluator.predict(dem, state, paths, np.ones((1, 3), bool))
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
    assert select_best(scores[0]) in bank
