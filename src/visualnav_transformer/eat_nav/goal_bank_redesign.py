"""Frozen, outcome-blind selection and gating for the follower-compatible review."""

from __future__ import annotations

import math

from .goal_bank_offline import FOLLOWER_CURVATURE_SCREEN_PER_M

AMPLITUDE_POOL_M = (.04, .08, .12, .16, .20)
OFFSET_POOL_M = (.04, .06, .08, .10, .12)
FOLLOWER_SCREEN_PER_M = FOLLOWER_CURVATURE_SCREEN_PER_M
INTERACTION_CLASSES = ('transition', 'differential', 'graded')
SELECTION_ORDER = (
    'hard_valid_groups', 'follower_compatible', 'side_balance',
    'geometry_mode_total', 'known_terrain_separated_pairs',
    'unknown_dependence', 'maximum_curvature_per_m',
    'turn_in_place_estimates', 'endpoint_residual_m_sum',
    'parameter_simplicity',
)


def follower_screen_reasons(row: dict, follower: dict) -> dict:
    curvature = row['maximum_curvature_per_m']
    failures = []
    if curvature is None or not math.isfinite(curvature) or curvature > FOLLOWER_SCREEN_PER_M:
        failures.append('curvature_screen_exceeded')
    if follower['command_mode'] != 'tracking':
        failures.append('command_mode_' + follower['command_mode'])
    if follower['reverse_estimate']:
        failures.append('reverse_estimate')
    diagnostic = []
    if follower['angular_saturation_estimate']:
        diagnostic.append('angular_saturation_estimate')
    if follower['turn_in_place_entry']:
        diagnostic.append('turn_in_place_estimate')
    return {'compatible': not failures, 'failure_flags': failures,
            'diagnostic_flags': diagnostic,
            'curvature_limit_per_m': FOLLOWER_SCREEN_PER_M,
            'screen_is_not_physical_vehicle_limit': True}


def rank_triples(scores: list[dict]) -> dict:
    if not scores:
        raise ValueError('no parameter triples')

    def key(row: dict):
        return (row['hard_valid_groups'], row['follower_compatible'],
                row['side_balance'], row['geometry_mode_total'],
                row['known_terrain_separated_pairs'], -row['unknown_dependence'],
                -row['maximum_curvature_per_m'], -row['turn_in_place_estimates'],
                -row['endpoint_residual_m_sum'], -sum(row['parameters']))

    return max(scores, key=key)


def rollout_readiness(group_support: dict, class_seed_terrain_support: dict) -> dict:
    if len(group_support) != 40:
        return {'ready': False, 'reason': 'state_group_count',
                'groups_at_least_five': 0}
    required = {cls: {seed: class_seed_terrain_support.get(cls, {}).get(seed, 0)
                      for seed in ('first', 'second')}
                for cls in INTERACTION_CLASSES}
    group_gate = all(row['hard_valid'] and row['geometry_modes'] == 7
                     and row['compatible'] >= 5 and row['compatible_required_l1']
                     and row['reverse'] == 0
                     and row['self_intersections'] == 0
                     for row in group_support.values())
    terrain_gate = all(value > 0 for seeds in required.values() for value in seeds.values())
    return {'ready': group_gate and terrain_gate,
            'groups_at_least_five': sum(row['compatible'] >= 5 for row in group_support.values()),
            'groups_all_hard_valid': sum(row['hard_valid'] for row in group_support.values()),
            'groups_seven_geometry_modes': sum(row['geometry_modes'] == 7 for row in group_support.values()),
            'class_seed_terrain_support': required,
            'terrain_gate': terrain_gate,
            'group_gate': group_gate}
