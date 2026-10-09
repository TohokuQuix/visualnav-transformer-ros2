import math
import json
from pathlib import Path

from visualnav_transformer.eat_nav.p1_statistical_pilot import (
    paths_for_slots, local_pose_from_root, validate_bridge_response,
)


def test_paths_for_slots_keep_assignment_and_eighth_monitor_duplicate():
    rows = [{'slot': i, 'path_sha256': str(i)} for i in range(7)]
    paths = {str(i): {'waypoints': [[0, 0], [1, i / 10]]} for i in range(7)}
    poses = [[10 + i, 20, math.pi / 2] for i in range(7)]
    mapped = paths_for_slots(rows, paths, poses)
    assert len(mapped) == 8
    assert mapped[4][0] == [14, 20]
    assert mapped[4][1] == [13.6, 21]
    assert mapped[7] == mapped[0]


def test_local_pose_and_exact_stamp_response():
    pose = local_pose_from_root([12, 23, .3, math.cos(.2), 0, 0, math.sin(.2)], [10, 20])
    assert all(abs(a-b) < 1e-8 for a, b in zip(pose, [2, 3, .4]))
    response = {'fresh': True, 'stamp_ns': 123, 'clock_now_ns': 123,
                'tf_stamp_ns': 123, 'commands': [[.1, 0, 0]] * 8}
    assert validate_bridge_response(response, 123, 8) == response['commands']
    response['tf_stamp_ns'] = 122
    try:
        validate_bridge_response(response, 123, 8)
    except ValueError as exc:
        assert 'stamp' in str(exc)
    else:
        raise AssertionError('stale TF accepted')


def test_corrected_protocol_freezes_120_steps_and_keeps_v1_immutable():
    repo=Path(__file__).resolve().parents[1]
    old=json.loads((repo/'config/eat_nav_p1_statistical_v1.json').read_text())
    corrected=json.loads((repo/'config/eat_nav_p1_statistical_v2.json').read_text())
    assert old['diagnostic_control_steps']==80
    assert corrected['primary_control_steps']==40
    assert corrected['diagnostic_control_steps']==120
    assert corrected['schema']=='p1_statistical_protocol_v2'
    for key in ('batch_seeds','assignment_randomization_seed','reward_gamma_per_step',
                'minimum_meaningful_progress_m','minimum_meaningful_discounted_reward',
                'resource_cap','initialization_tolerances'):
        assert corrected[key]==old[key]
