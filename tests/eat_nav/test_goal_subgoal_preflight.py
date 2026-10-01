"""Check frozen pilot identities and the recorded runtime stop without simulator startup."""
import collections
import hashlib
import json
import pathlib

import numpy as np
import pytest

DOC = pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator')
OFF = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood')
RAW = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def frozen():
    if not (DOC / 'run-identities.json').exists():
        pytest.skip('frozen pilot artifacts unavailable')
    return load(DOC / 'run-identities.json')['runs'], load(DOC / 'pilot-input-freeze.json')


def test_fixed_44_run_schedule_and_shared_nominal(frozen):
    runs, freeze = frozen
    assert len(runs) == 44
    assert [r['order'] for r in runs] == list(range(44))
    assert [r['kind'] for r in runs] == (['duplicate'] * 3 + ['candidate'] * 8) * 4
    assert collections.Counter(r['state_id'] for r in runs) == {
        'train-003': 11, 'train-004': 11, 'train-019': 11, 'train-021': 11,
    }
    assert digest(DOC / 'run-identities.json') == freeze['run_identities_sha256']
    for j in range(4):
        group = runs[11 * j:11 * (j + 1)]
        assert len({r['base_path_sha256'] for r in group[:3]}) == 1
        assert len({r['logical_t0_ns'] for r in group}) == 1
        assert [(r['goal_id'], r['seed']) for r in group[3:]] == [
            (goal, seed) for goal in ('logged', 'id_forward_long', 'near_left_lateral', 'far_diag_right')
            for seed in (1001, 1002)
        ]


def test_all_candidate_paths_match_existing_offline_array(frozen):
    runs, _ = frozen
    sweep = load('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood/goal-sweep-inputs.json')
    for j in range(4):
        sid = runs[11*j]['state_id']
        with np.load(OFF / 'inference' / f'{sid}.npz') as output:
            for row in runs[11*j+3:11*(j+1)]:
                folder = RAW / 'inputs' / f'state-{j:03d}'
                path = folder / f"{row['candidate_id']}-base-path.json"
                assert digest(path) == row['base_path_sha256']
                points = np.asarray(load(path))
                assert points.shape == (9, 2)
                assert np.array_equal(points[0], [0, 0])
                assert np.allclose(points[1:], output['paths'][row['offline_goal_order'], row['offline_seed_index']])
                frozen_goal = sweep['inputs'][int(sid.split('-')[1])*13 + row['offline_goal_order']]['goal_xy']
                assert np.allclose(row['goal_xy'], frozen_goal)


def test_runtime_stop_contains_no_performance_claim(frozen):
    _, freeze = frozen
    status = load(RAW / 'status.json')
    decision = load(DOC / 'gate-decision.json')
    resource = load(DOC / 'runtime-storage.json')
    assert status['stage'] == 'STOPPED_RUNTIME_FORECAST'
    assert status['candidate_runs'] == 0
    assert decision['GS1'] == decision['GS2'] == decision['GS3'] == 'NA'
    assert resource['candidate_runs_complete'] == resource['duplicate_runs_complete'] == 0
    assert resource['conservative_44_live_prefix_hours'] > 4
    assert freeze['outcomes_used'] is False


def test_logging_parity_and_raw_effort_mapping():
    parity = load(DOC / 'parity-smoke.json')
    assert parity['status'] == 'pass'
    assert parity['compared_control_steps'] == 120
    assert parity['first_difference'] is None
    assert parity['raw_joint_effort_dim'] == 52
    assert parity['raw_effort_grouped_dim'] == 6
    assert parity['raw_to_prenoise_effort_max_abs'] < 1e-5
