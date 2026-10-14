"""Adopted 14-Path evaluation from sealed DEM/Actor ROS messages."""

from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.history import DemHistory
from visualnav_transformer.eat_nav.ros2_phase2.nodes import build_model_evaluation
from visualnav_transformer.eat_nav.ros2_phase2.replay import load_sealed_prepath
from visualnav_transformer.eat_nav.ros2_phase2.selection import score_global_goal
from visualnav_transformer.eat_nav.ros2_phase2.wire import CANDIDATE_IDS, BANK_SHA


COLLECTION = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/'
                  'motion-14-collection-20261014-05/'
                  '0000-pyramid_stairs-level1-seed1456788336-front_folded-prepath-0')
TRAINING = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/'
                'motion-14-training-20261014-04')
PYTHON = Path('/home/kojima/anaconda3/envs/tinyvla/bin/python')


def test_adopted_worker_selects_one_of_fourteen_from_sealed_ros_messages():
    source = load_sealed_prepath(COLLECTION, env_id=0, episode_id=1)
    history = DemHistory()
    for message in source['dem_messages']:
        history.push(message)
    actor = source['actor_message']
    root = source['root_tile_local']
    goal = (float(root[0]) + 2., float(root[1]))
    result = build_model_evaluation(
        history, actor, 1_000_000_000, 1,
        (float(root[0]), float(root[1]), 0.), goal,
        mode='learned14_adopted',
        checkpoint=TRAINING/'training/full-seed123/best.pt',
        python=PYTHON, results=TRAINING/'results.json',
        normalization=TRAINING/'normalization.json')
    assert result.predictor_mode == 'learned14_adopted'
    assert result.bank_source_sha256 == BANK_SHA
    assert len(result.predicted_body_xy_m) == 28
    assert len(result.scores_m) == 14
    assert 0 <= result.selected_index < len(CANDIDATE_IDS)
    prediction = np.asarray(result.predicted_body_xy_m, np.float32).reshape(14, 2)
    expected = score_global_goal(prediction,
                                 (float(root[0]), float(root[1]), 0.), goal,
                                 np.ones(14, bool))
    np.testing.assert_allclose(result.scores_m, expected['scores_m'], atol=1e-6, rtol=0)
    assert result.selected_index == expected['index']
