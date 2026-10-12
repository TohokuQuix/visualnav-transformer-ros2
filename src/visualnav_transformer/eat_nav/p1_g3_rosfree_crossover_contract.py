"""Pre-Path identity contract for the independent ROS-free crossover series."""

from __future__ import annotations

from .p1_g3_crossover_runtime import compare_prepath


def validate_crossover_prepath(current: dict, reference: dict | None,
                               execution_index: int) -> dict:
    """Compare Path-independent state with sealed period 0 before activation."""
    if execution_index not in (0, 1, 2):
        raise ValueError('crossover period invalid')
    follower = current.get('follower_pre_activation')
    if follower != {'instance_count': 0, 'path_activations': 0}:
        raise ValueError('follower state contaminated before Path activation')
    if execution_index == 0:
        if reference is not None:
            raise ValueError('period 0 must not use another period as reference')
        return {'pass': True, 'mode': 'period_zero_baseline',
                'cross_period_equality_assessed': False}
    if reference is None:
        raise ValueError('sealed period-0 pre-Path reference required')
    if reference.get('follower_pre_activation') != follower:
        raise ValueError('follower state differs across periods')
    comparison = compare_prepath(reference, current)
    if not comparison['pass']:
        raise ValueError('cross-period strict pre-Path mismatch: '+repr(comparison['strict_failures']))
    return {'pass': True, 'mode': 'strict_against_sealed_period_zero',
            'cross_period_equality_assessed': True, 'comparison': comparison}
