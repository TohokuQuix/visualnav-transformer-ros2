from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.mi_a_v3 import (
    build_collection_assignment, build_evaluation_schedule,
    configure_run_mode, validate_collection_manifest, validate_evaluation_schedule,
)


def cfg():
    return SimpleNamespace(terminations=SimpleNamespace(time_out=object(), non_finite_state=object(),
                                                         tilt_over=object()), events=SimpleNamespace(
        reset_flipper_fold_mode=SimpleNamespace(params={'modes': (0.,), 'probabilities': (1.,)}),
        reset_robot_joints=SimpleNamespace(params={
            'flipper_joint_names': ['rr', 'rl', 'fr', 'fl'],
            'pose_probabilities': (0., .5, .5),
            'folded_angle': 3.141592653589793,
            'front_rest_angle': 0.,
        }),
    ))


def groups():
    return [dict(state_group_id=f'{terrain}-s{seed}', terrain_type=terrain,
                 terrain_seed=1000+100*terrain_index+seed, terrain_level=seed, env_tile_ids=[0, 1, 2, 3])
            for terrain_index, terrain in enumerate(('pyramid_stairs', 'pyramid_stairs_inv',
                            'random_rough', 'k_rail_step_field'))
            for seed in (1, 2, 3)]


def test_demo_stays_front_folded_after_collection_assignment():
    assignment = build_collection_assignment(groups(), 123)
    assert assignment == build_collection_assignment(groups(), 123)
    a = cfg()
    assert configure_run_mode(a, 'demo') == 'front_folded'
    assert a.events.reset_robot_joints.params['pose_probabilities'] == (0., 1., 0.)
    assert a.events.reset_flipper_fold_mode.params == {'modes': (0.,), 'probabilities': (1.,)}
    assert sorted(assignment.values()).count('front_folded') == 6
    assert sorted(assignment.values()).count('rear_folded') == 6
    for terrain in {g['terrain_type'] for g in groups()}:
        modes = [assignment[g['state_group_id']] for g in groups() if g['terrain_type'] == terrain]
        assert set(modes) == {'front_folded', 'rear_folded'}


def test_collection_and_evaluation_require_explicit_group_assignment():
    g = groups()[0]
    for mode in ('collection', 'evaluation'):
        with pytest.raises(ValueError):
            configure_run_mode(cfg(), mode, state_group_id=g['state_group_id'])
    c = cfg()
    assert configure_run_mode(c, 'collection', state_group_id=g['state_group_id'],
                              assignment={g['state_group_id']: 'rear_folded'}) == 'rear_folded'
    assert c.events.reset_robot_joints.params['pose_probabilities'] == (0., 0., 1.)
    with pytest.raises(ValueError):
        configure_run_mode(cfg(), 'evaluation', state_group_id='other',
                           assignment={g['state_group_id']: 'rear_folded'})
    manifest = {'schema': 'eatnav_mi_a_v3_collection_assignment',
                'state_groups': groups(), 'randomization_seed': 123,
                'assignment': build_collection_assignment(groups(), 123)}
    assert validate_collection_manifest(manifest) == manifest['assignment']
    manifest['assignment'][g['state_group_id']] = 'invalid'
    with pytest.raises(ValueError):
        validate_collection_manifest(manifest)


def test_schedule_balanced_and_candidate_pairing():
    rows = build_evaluation_schedule(groups())
    assert len(rows) == 192
    validate_evaluation_schedule(rows, groups())
    for group in groups():
        for condition in ('front_folded', 'rear_folded'):
            subset = [r for r in rows if r['state_group_id'] == group['state_group_id']
                      and r['flipper_initialization_mode'] == condition]
            assert len(subset) == 8
            assert [r['stage'] for r in subset[:2]] == ['prepath', 'prepath']
            assert len({r['candidate_id'] for r in subset[2:]}) == 6
            assert all(r['terrain_seed'] == group['terrain_seed'] for r in subset)
    broken = [dict(row) for row in rows]
    broken[-1]['candidate_id'] = 'forward_direct'
    with pytest.raises(ValueError):
        validate_evaluation_schedule(broken, groups())


def test_frozen_execution_manifest_and_collection_allocation():
    from scripts.eat_nav_mi_a_v3_run import load_frozen
    manifest = load_frozen()
    assert manifest['num_processes'] == 192
    assert manifest['candidate_trials'] == 576
    assert manifest['independent_seed_count'] == 12
    assert len(set(manifest['smoke_state_groups'])) == 2
    assert len({g['terrain_type'] for g in manifest['state_groups']
                if g['state_group_id'] in manifest['smoke_state_groups']}) == 2
    path = Path(__file__).resolve().parents[1]/'docs/eat-nav/motion-predictor-vnext/collection_randomization_audit.json'
    from visualnav_transformer.eat_nav.mi_a_v3 import validate_collection_manifest
    assignment = validate_collection_manifest(json.loads(path.read_text()))
    assert len(assignment) == 12


def test_seed_cluster_bootstrap_uses_twelve_seed_units_not_tiles():
    import numpy as np
    from scripts.eat_nav_mi_a_v3_analyze import _cluster_ci, _difference
    terrain_indices = [list(range(i, i+3)) for i in (0, 3, 6, 9)]
    values = np.tile([.06, .01, .1], (12, 1))
    result = _cluster_ci(values, terrain_indices, np.random.default_rng(3), 100)
    assert len(result['per_seed_mean']) == 12
    assert result['seeds_above_0_03m'] == 12
    assert result['xy_projected_95_ci_m'][0] > .03
    assert np.isclose(_difference(np.array([[0., 0., -3.13]]),
                                  np.array([[0., 0., 3.13]]))[0, 2], .023185307179586445)
