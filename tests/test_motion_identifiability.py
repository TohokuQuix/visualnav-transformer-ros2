import math

import numpy as np
import pytest


def test_relative_motion_uses_same_direction_direct_and_wraps_yaw():
    from visualnav_transformer.eat_nav.motion_identifiability import relative_motion

    direct = {'delta_body_xy_m': [-.2, .01], 'delta_body_yaw_rad': -3.13}
    arc = {'delta_body_xy_m': [-.18, -.03], 'delta_body_yaw_rad': 3.13}
    result = relative_motion(arc, direct, direction=-1, angle_deg=30)
    np.testing.assert_allclose(result['delta_xy_m'], [.02, -.04])
    assert result['delta_yaw_rad'] == pytest.approx(-.02318530718, abs=1e-8)
    assert result['signed_travel_lateral_m'] == pytest.approx(.04)


def test_repeatability_keeps_seed_as_cluster_not_four_env_samples():
    from visualnav_transformer.eat_nav.motion_identifiability import seed_effects

    rows = []
    for seed in range(6):
        for repeat in (0, 1):
            for env in range(4):
                rows.append({'seed': seed, 'repeat': repeat, 'env': env,
                             'condition': 'rough' if seed < 4 else 'mirror',
                             'candidate': 'forward_direct',
                             'xy': [0., 0.], 'yaw': 0.})
                rows.append({'seed': seed, 'repeat': repeat, 'env': env,
                             'condition': 'rough' if seed < 4 else 'mirror',
                             'candidate': 'forward_+30',
                             'xy': [0., .04 + .001 * seed], 'yaw': .09 + .001 * seed})
    result = seed_effects(rows, 'forward_+30')
    assert len(result['seed_values']) == 6
    assert result['seed_values'][0]['signed_yaw_rad'] == pytest.approx(.09)
    assert result['seed_values'][0]['signed_lateral_m'] == pytest.approx(.04)
    assert result['repeat_p90_yaw_rad'] == pytest.approx(0.)


def test_seed_effects_rejects_missing_pair_or_duplicate():
    from visualnav_transformer.eat_nav.motion_identifiability import seed_effects

    direct = {'seed': 1, 'repeat': 0, 'env': 0, 'condition': 'rough',
              'candidate': 'reverse_direct', 'xy': [-.2, 0.], 'yaw': 0.}
    arc = dict(direct, candidate='reverse_+30', xy=[-.2, -.04], yaw=.1)
    with pytest.raises(ValueError, match='six seed'):
        seed_effects([direct, arc], 'reverse_+30')
    with pytest.raises(ValueError, match='duplicate'):
        seed_effects([direct, direct, arc], 'reverse_+30')


def test_candidate_decision_distinguishes_pass_fail_and_uncertainty():
    from visualnav_transformer.eat_nav.motion_identifiability import candidate_decision
    base = {'seed_values': [{'seed': n, 'condition': 'rough' if n < 4 else 'mirror',
                             'signed_yaw_rad': .08, 'signed_lateral_m': .01}
                            for n in range(6)],
            'repeat_p90_yaw_rad': .005, 'repeat_p90_lateral_m': .005}
    assert candidate_decision(base)['status'] == 'pass'
    weak = {**base, 'seed_values': [{**x, 'signed_yaw_rad': .001} for x in base['seed_values']]}
    assert candidate_decision(weak)['status'] == 'fail'
    uncertain = {**base, 'seed_values': [{**x, 'signed_yaw_rad': .08 if x['seed'] < 3 else -.02}
                                          for x in base['seed_values']]}
    assert candidate_decision(uncertain)['status'] == 'inconclusive'


def test_mirror_audit_uses_observed_dem_and_pose():
    from visualnav_transformer.eat_nav.motion_identifiability import mirror_audit
    root = np.zeros((4, 13))
    root[:, 3] = 1
    dem = np.zeros((4, 4, 2, 20, 20), dtype=np.float32)
    dem[:, :, 1] = 1
    assert mirror_audit(root, dem)['pass']
    dem[0, 0, 0, 0, 0] = .1
    assert not mirror_audit(root, dem)['pass']


def test_frozen_schedule_has_paired_direct_and_six_independent_seeds():
    import json
    from pathlib import Path
    from scripts.eat_nav_motion_identifiability_schedule import assignment_rows
    manifest = json.loads((Path(__file__).parents[1]/'docs/eat-nav/motion-predictor-vnext/'
                           'motion_identifiability_manifest.json').read_text())
    rows = assignment_rows(manifest)
    assert len(rows) == 72
    assert len({row['seed'] for row in rows}) == 6
    for seed in {row['seed'] for row in rows}:
        for repeat in (0, 1):
            roles = {row['candidate'] for row in rows
                     if row['seed'] == seed and row['repeat'] == repeat}
            assert roles == set(manifest['candidate_order'])
