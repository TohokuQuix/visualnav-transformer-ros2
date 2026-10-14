from pathlib import Path
import importlib.util

import numpy as np
import pytest

from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.motion_14_predictor import CANDIDATE_IDS
from visualnav_transformer.eat_nav import demo_motion_14


CHECKPOINT = Path(__file__).resolve().parents[1]/'checkpoints/eat_nav/motion14/training/full-seed123/best.pt'


def test_demo_uses_frozen_fourteen_paths_and_body_to_map_transform():
    pose = np.asarray([2., 3., np.pi / 2])
    body, mapped, info = demo_motion_14.candidates_from_frozen_bank(pose)
    assert tuple(body) == CANDIDATE_IDS
    assert tuple(mapped) == CANDIDATE_IDS
    assert info['candidate_count'] == 14
    assert info['bank'] == 'eatnav_forward_reverse_path_bank_v1'
    for role, item in zip(CANDIDATE_IDS, bank()):
        assert np.array_equal(body[role], item.waypoints)
    assert np.allclose(mapped['forward_direct'][-1], [2., 3.75])
    assert np.allclose(mapped['reverse_direct'][-1], [2., 2.25])


def test_demo_normalization_matches_training_and_keeps_negative_dem():
    dem = np.zeros((4, 2, 20, 20), dtype=np.float32)
    dem[:, 1] = 1
    dem[0, 0, 0, 0] = -.25
    dem[0, 1, 0, 1] = 0
    dem[0, 0, 0, 1] = 99
    before = dem.copy()
    state = np.arange(22, dtype=np.float32)
    norm = {'fitted_from': 'train_only', 'dem_height_mean': .05, 'dem_height_std': .1,
            'state_mean': [1.] * 22, 'state_std': [2.] * 22,
            'target_mean': [0., 0.], 'target_std': [1., 1.]}
    body, _, _ = demo_motion_14.candidates_from_frozen_bank([0., 0., 0.])
    values = demo_motion_14.prepare_motion_14_inputs(dem, state, body, norm)
    assert [v.shape for v in values] == [(1, 4, 2, 20, 20), (1, 22),
                                           (1, 14, 8, 2), (1, 14, 1), (1, 14)]
    assert values[0][0, 0, 0, 0, 0] == pytest.approx(-3.)
    assert values[0][0, 0, 0, 0, 1] == 0
    assert np.array_equal(values[0][0, :, 1], dem[:, 1])
    assert np.array_equal(values[3][0, :, 0], [1.] * 7 + [-1.] * 7)
    assert values[4].all()
    assert np.array_equal(dem, before)


def test_sealed_adopted_checkpoint_scores_all_fourteen_global_goal_paths():
    if not CHECKPOINT.is_file():
        pytest.skip('adopted checkpoint absent')
    predictor = demo_motion_14.load_motion_14_demo_predictor(CHECKPOINT)
    dem = np.zeros((4, 2, 20, 20), dtype=np.float32)
    dem[:, 1] = 1
    body, _, _ = demo_motion_14.candidates_from_frozen_bank([0., 0., 0.])
    motion, scores = predictor.predict(dem, np.zeros(22, dtype=np.float32), body,
                                       np.asarray([0., 0., 0.]), np.asarray([3., 1.]))
    assert motion.shape == (14, 2) and np.isfinite(motion).all()
    assert scores.shape == (14,) and np.isfinite(scores).all()
    expected = np.linalg.norm([3., 1.]) - np.linalg.norm(
        np.asarray([3., 1.])[None] - motion, axis=1)
    assert np.allclose(scores, expected, atol=1e-6)
    assert predictor.checkpoint_sha256 == demo_motion_14.DEMO_CHECKPOINT_SHA256


def test_checkpoint_provenance_rejects_wrong_bytes(tmp_path):
    corrupt = tmp_path/'best.pt'
    corrupt.write_bytes(b'not the adopted checkpoint')
    with pytest.raises(ValueError, match='SHA'):
        demo_motion_14.load_motion_14_demo_predictor(corrupt)


def test_demo_launcher_pins_adopted_checkpoint_and_fourteen_candidate_source_contract():
    source = Path(__file__).resolve().parents[1]/'scripts/eat_nav_demo_launch.py'
    spec = importlib.util.spec_from_file_location('eat_nav_demo_launch_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CHECKPOINT == CHECKPOINT
    assert module.DEMO_CANDIDATE_COUNT == 14
    assert module.DEMO_CANDIDATE_ORDER == CANDIDATE_IDS
