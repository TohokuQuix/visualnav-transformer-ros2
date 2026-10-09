"""Pure comparisons for candidate-free P1 stepwise diagnostics."""

from __future__ import annotations

import numpy as np


FIELDS = {
    'root_position': ('root_state_w', slice(0, 3)),
    'root_orientation': ('root_state_w', slice(3, 7)),
    'root_velocity': ('root_state_w', slice(7, 13)),
    'joint_position': ('joint_position', None),
    'joint_velocity': ('joint_velocity', None),
    'joint_target': ('joint_position_target', None),
    'joint_velocity_target': ('joint_velocity_target', None),
    'applied_torque': ('applied_torque', None),
    'action': ('action', None),
    'previous_action': ('previous_action', None),
    'flipper_previous_target': ('flipper_previous_target', None),
    'contact_force': ('contact_force', None),
}


def first_divergence(source: list[dict], target: list[dict], field: str, tolerance: float) -> dict:
    if field not in FIELDS:
        raise ValueError(f'unknown diagnostic field: {field}')
    if len(source) != len(target) or [x['physics_step'] for x in source] != [x['physics_step'] for x in target]:
        raise ValueError('step alignment mismatch')
    name, selector = FIELDS[field]
    result = {'field': field, 'status': 'observed', 'first_nonzero_step': None,
              'first_nonzero_max_abs': None, 'first_threshold_step': None,
              'first_threshold_max_abs': None, 'max_abs_overall': 0.0}
    for a, b in zip(source, target):
        if a.get(name) is None or b.get(name) is None:
            result['status'] = 'unobserved'
            return result
        av, bv = np.asarray(a[name], dtype=np.float64), np.asarray(b[name], dtype=np.float64)
        if selector is not None:
            av, bv = av[selector], bv[selector]
        if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError(f'invalid {field} at step {a["physics_step"]}')
        magnitude = float(np.max(np.abs(av-bv)))
        result['max_abs_overall'] = max(result['max_abs_overall'], magnitude)
        if magnitude > 0 and result['first_nonzero_step'] is None:
            result['first_nonzero_step'] = a['physics_step']
            result['first_nonzero_max_abs'] = magnitude
        if magnitude > tolerance and result['first_threshold_step'] is None:
            result['first_threshold_step'] = a['physics_step']
            result['first_threshold_max_abs'] = magnitude
    return result


def resource_violation(sample: dict, cap: dict) -> str | None:
    """Return the first frozen resource/process cap exceeded, or None."""
    bounds = (
        ('gpu_device_total', sample['gpu_mib'], cap['gpu_device_total_mib'], 'max'),
        ('host_available_ram', sample['host_available_gib'], cap['host_available_min_gib'], 'min'),
        ('active_storage', sample['active_storage_gib'], cap['active_storage_gib'], 'max'),
        ('heartbeat_stale', sample['heartbeat_age_s'], cap['heartbeat_stale_s'], 'max'),
        ('process_timeout', sample['elapsed_s'], cap['process_timeout_s'], 'max'),
    )
    for name, value, limit, direction in bounds:
        if value is None or (direction == 'max' and value > limit) or (direction == 'min' and value < limit):
            return name
    return None
