"""MI-FI v1 initialization assignment and paired-state contract."""

from __future__ import annotations

import math

from .mi_a_v2 import CANDIDATES, compare_gate_p as compare_mi_a_gate_p


SCHEMA = 'mi_fi_v1'
CONDITIONS = ('front_folded', 'rear_folded')


def configure_initialization(cfg, condition: str) -> tuple[float, float, float, float]:
    """Select one existing reset pose; retain the demo's unconstrained fold mode."""
    if condition not in CONDITIONS:
        raise ValueError('unknown flipper initialization condition')
    joint = cfg.events.reset_robot_joints.params
    mode = cfg.events.reset_flipper_fold_mode.params
    names = tuple(joint.get('flipper_joint_names', ()))
    if len(names) != 4 or not all(fragment in name for fragment, name in
                                   zip(('rr', 'rl', 'fr', 'fl'), names)):
        raise ValueError('flipper joint mapping differs from the frozen reset event')
    folded = float(joint['folded_angle'])
    rest = float(joint['front_rest_angle'])
    if not math.isfinite(folded) or not math.isfinite(rest) or \
            not math.isclose(rest, 0.0, abs_tol=1e-12) or \
            not math.isclose(folded, math.pi, abs_tol=1e-6):
        raise ValueError('flipper reset angle contract changed')
    if mode != {'modes': (0.,), 'probabilities': (1.,)}:
        raise ValueError('flipper mode must remain zero')
    joint['pose_probabilities'] = (0., 1., 0.) if condition == 'front_folded' else (0., 0., 1.)
    return (rest, rest, folded, folded) if condition == 'front_folded' else \
           (folded, folded, rest, rest)


def compare_gate_p(reference: dict, current: dict) -> dict:
    expected = reference.get('identity', {}).get('flipper_initialization_mode')
    actual = current.get('identity', {}).get('flipper_initialization_mode')
    result = compare_mi_a_gate_p(reference, current)
    same = expected in CONDITIONS and actual == expected
    result['schema'] = SCHEMA
    result['fields']['flipper_initialization_mode'] = {
        'kind': 'identity', 'pass': same, 'reference': expected, 'current': actual}
    result['pass'] = result['pass'] and same
    return result


def pilot_plan(seed: int) -> list[dict]:
    if not isinstance(seed, int) or seed <= 0:
        raise ValueError('positive terrain seed required')
    rows = []
    for condition in CONDITIONS:
        for repeat in (0, 1):
            rows.append({'index': len(rows), 'condition': condition, 'stage': 'prepath',
                         'candidate': None, 'repeat': repeat, 'seed': seed})
        for candidate in CANDIDATES:
            rows.append({'index': len(rows), 'condition': condition, 'stage': 'candidate',
                         'candidate': candidate, 'repeat': 0, 'seed': seed})
    return rows
