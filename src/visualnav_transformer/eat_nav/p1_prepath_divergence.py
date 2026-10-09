"""Read-only field and direct-clone comparisons for P1 pre-Path evidence."""

from __future__ import annotations

import math

import numpy as np


def _array(value):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite diagnostic input')
    return a


def _max_abs(a, b):
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise ValueError('diagnostic shape mismatch')
    return float(np.max(np.abs(a-b)))


def _orientation(a, b):
    a, b = _array(a), _array(b)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError('quaternion shape mismatch')
    cosine = min(1.0, abs(float(np.dot(a, b))) / (np.linalg.norm(a)*np.linalg.norm(b)))
    return float(2*math.acos(cosine))


def initial_state_contract(source: dict) -> dict:
    """Describe what v2 writes, retains, or omits without inventing live step-0 data."""
    return {
        'root_xy': {'status': 'source_tile_local_xy_written_not_measured_at_step_0',
                    'source_value': source['root_state_w'][:2]},
        'root_z': {'status': 'retained_from_batch_reset_not_source_fixture',
                   'source_value': source['root_state_w'][2]},
        'root_orientation': {'status': 'source_yaw_written_not_measured_at_step_0',
                             'source_value': source['root_state_w'][3:7]},
        'root_velocity': {'status': 'retained_from_batch_reset_not_source_fixture',
                          'source_value': source['root_state_w'][7:13]},
        'joint_position': {'status': 'source_fixture_written_not_measured_at_step_0',
                           'source_value': source['joint_position']},
        'joint_velocity': {'status': 'source_fixture_written_not_measured_at_step_0',
                           'source_value': source['joint_velocity']},
        'flipper_latch': {'status': 'source_fixture_written_not_measured_at_step_0',
                          'source_value': source['flipper_fold_mode_latch']},
        'action_manager': {'status': 'not_restored_or_recorded'},
        'flipper_previous_target': {'status': 'not_restored_or_recorded'},
        'latency_buffers': {'status': 'not_restored_or_recorded'},
        'observation_rng': {'status': 'not_restored_or_recorded'},
        'actuator_target_and_internal_state': {'status': 'not_restored_or_recorded'},
        'contact_solver_state': {'status': 'not_restored_or_recorded'},
    }


def pair_metrics(a: dict, b: dict) -> dict:
    """Compare two settled batch records in their own tile-local frames."""
    aa, bb = _array(a['root_state_w']), _array(b['root_state_w'])
    if aa.shape != (13,) or bb.shape != (13,):
        raise ValueError('root state shape mismatch')
    axy = aa[:2]-_array(a['target_tile_origin_xy'])
    bxy = bb[:2]-_array(b['target_tile_origin_xy'])
    result = {
        'root_local_max_abs_m': _max_abs(np.r_[axy, aa[2]], np.r_[bxy, bb[2]]),
        'root_orientation_rad': _orientation(aa[3:7], bb[3:7]),
        'root_velocity_max_abs': _max_abs(aa[7:13], bb[7:13]),
        'joint_position_max_abs': _max_abs(a['joint_position'], b['joint_position']),
        'joint_velocity_max_abs': _max_abs(a['joint_velocity'], b['joint_velocity']),
        'actor_max_abs': _max_abs(a['actor_input'], b['actor_input']),
        'dem_frames': [],
    }
    if len(a['dem_frames']) != 4 and len(a['dem_frames']) != 1:
        raise ValueError('unexpected DEM frame count')
    if len(a['dem_frames']) != len(b['dem_frames']):
        raise ValueError('DEM history length mismatch')
    for af, bf in zip(a['dem_frames'], b['dem_frames']):
        da, db = _array(af['transformed']), _array(bf['transformed'])
        if da.shape != db.shape or da.shape[0] != 2:
            raise ValueError('DEM shape mismatch')
        mask_a, mask_b = da[1] > .5, db[1] > .5
        union = int(np.logical_or(mask_a, mask_b).sum())
        delta = da[0]-db[0]
        result['dem_frames'].append({
            'height_rms_m': float(np.sqrt(np.mean(delta**2))),
            'height_max_abs_m': float(np.max(np.abs(delta))),
            'mask_iou': float(np.logical_and(mask_a, mask_b).sum()/union) if union else 1.0,
        })
    return result


def gate_checks(metrics: dict, tolerance: dict) -> dict:
    return {
        'root_local_position': metrics['root_local_max_abs_m'] <= tolerance['root_position_m'],
        'root_orientation': metrics['root_orientation_rad'] <= tolerance['root_orientation_rad'],
        'root_velocity': metrics['root_velocity_max_abs'] <= tolerance['root_velocity_mps_or_radps'],
        'joint_position': metrics['joint_position_max_abs'] <= tolerance['joint_position_rad'],
        'joint_velocity': metrics['joint_velocity_max_abs'] <= tolerance['joint_velocity_radps'],
        'actor': metrics['actor_max_abs'] <= tolerance['actor_post_delay_max_abs'],
        'dem': all(f['height_rms_m'] <= tolerance['dem_height_rms_m'] and
                   f['height_max_abs_m'] <= tolerance['dem_height_max_abs_m'] and
                   f['mask_iou'] >= tolerance['dem_mask_iou_min']
                   for f in metrics['dem_frames']),
    }
