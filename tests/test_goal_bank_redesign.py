import pytest

from visualnav_transformer.eat_nav.goal_bank_redesign import (
    AMPLITUDE_POOL_M, OFFSET_POOL_M, FOLLOWER_SCREEN_PER_M,
    follower_screen_reasons, rank_triples, rollout_readiness,
)


def test_frozen_redesign_pools_and_screen():
    assert AMPLITUDE_POOL_M == (.04, .08, .12, .16, .20)
    assert OFFSET_POOL_M == (.04, .06, .08, .10, .12)
    assert FOLLOWER_SCREEN_PER_M == 1.236


def test_screen_reason_keeps_curvature_separate_from_command_mode():
    follower = {'command_mode': 'tracking', 'reverse_estimate': False,
                'turn_in_place_entry': False, 'angular_saturation_estimate': False,
                'initial_command_estimate': {'linear_x': .3, 'angular_z': .1}}
    row = {'maximum_curvature_per_m': 1.3}
    reasons = follower_screen_reasons(row, follower)
    assert reasons['compatible'] is False
    assert reasons['failure_flags'] == ['curvature_screen_exceeded']
    assert reasons['diagnostic_flags'] == []
    follower['angular_saturation_estimate'] = True
    assert follower_screen_reasons(row, follower)['diagnostic_flags'] == ['angular_saturation_estimate']


def test_rank_prefers_follower_supply_then_symmetry_before_geometry():
    base = {'hard_valid_groups': 40, 'follower_compatible': 120,
            'side_balance': 0, 'geometry_mode_total': 280,
            'known_terrain_separated_pairs': 10, 'unknown_dependence': 0,
            'maximum_curvature_per_m': 2, 'turn_in_place_estimates': 0,
            'endpoint_residual_m_sum': 0, 'parameters': [.04,.08,.12]}
    more_geometry = dict(base, follower_compatible=119, geometry_mode_total=999,
                         parameters=[.04,.08,.16])
    assert rank_triples([base, more_geometry])['parameters'] == base['parameters']
    unbalanced = dict(base, side_balance=-2, known_terrain_separated_pairs=999,
                      parameters=[.04,.10,.12])
    assert rank_triples([base, unbalanced])['parameters'] == base['parameters']


def test_readiness_requires_five_in_every_group_and_both_seed_terrain_support():
    groups = {f'g{i}': {'hard_valid': True, 'geometry_modes': 7,
                       'compatible': 5, 'compatible_required_l1': True,
                       'reverse': 0, 'self_intersections': 0}
              for i in range(40)}
    support = {cls: {'first': 1, 'second': 1}
               for cls in ('transition','differential','graded')}
    assert rollout_readiness(groups, support)['ready']
    groups['g7']['compatible'] = 3
    assert not rollout_readiness(groups, support)['ready']
    groups['g7']['compatible'] = 5
    groups['g7']['compatible_required_l1'] = False
    assert not rollout_readiness(groups, support)['ready']
    groups['g7']['compatible_required_l1'] = True
    support['graded']['second'] = 0
    assert not rollout_readiness(groups, support)['ready']
