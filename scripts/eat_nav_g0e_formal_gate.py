"""Pure, frozen safety and anchor comparison gates for G0-E formal replay."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


HISTORICAL_NA = ['joint_position','joint_velocity','previous_action',
    'actor_observation','actor_latent','critic_observation','raw_joint_effort']


def formal_roots(table, container_id, remote_output):
    return {pid for pid,process in table.items() if container_id in process['cgroup']
        and '/tmp/eatnav-g0e-formal-anchor-20261003/formal_sim.py' in process['command_line']
        and '--output '+remote_output in process['command_line']}


def baseline_gate(samples, limits):
    for sample in samples:
        if sample['residual_processes']:return 'residual_process'
        if sample['unresolved_pid_mib']>0:return 'unresolved_gpu_pid'
        if sample['device_total_used_mib']>=limits['GPU_cap_mib']:return 'external_gpu_occupancy'
        if abs(sample['unattributed_used_mib'])>limits['unattributed_cap_mib']:return 'unattributed_gpu'
        if sample['anon_shmem_bytes']>limits['anon_shmem_cap_bytes']:return 'anon_shmem_cap'
        if sample['host_available_bytes']<limits['host_memory_floor_bytes']:return 'host_memory_floor'
    return None


def runtime_gate(sample, limits):
    if sample['device_total_used_mib']>limits['GPU_cap_mib']:
        return ('device_GPU_cap_external_load' if
            sample['simulator_process_tree_used_mib']<=limits['GPU_cap_mib'] else 'device_GPU_cap_simulator_load')
    if sample['unresolved_pid_mib']>0:return 'unresolved_gpu_pid'
    if abs(sample['unattributed_used_mib'])>limits['unattributed_cap_mib']:return 'unattributed_gpu'
    if sample['oom_events_delta']>0:return 'OOM_event'
    if sample['anon_shmem_bytes']>limits['anon_shmem_cap_bytes']:return 'anon_shmem_cap'
    if sample['host_available_bytes']<limits['host_memory_floor_bytes']:return 'host_memory_floor'
    return None


def allow_second_anchor(first_result, first_cleanup):
    return first_result.get('status')=='pass' and first_cleanup.get('status')=='pass'


def finite(value):
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return not isinstance(value,(int,float)) or bool(np.isfinite(value))


def compare_anchor(summary, logged, state, selected_goal, limits):
    saved=summary['state'];anchor=summary['anchor'];error=summary['anchor_error_vs_logged']
    reasons=[]
    if (saved['state_id']!=state['state_id'] or saved['episode_key']!=state['episode_key'] or
        saved['step']!=state['global_step'] or saved['local_step']!=state['local_step'] or
        [logged['session_id'],logged['env_id'],logged['episode_id']]!=state['episode_key'] or
        logged['step']!=state['global_step'] or logged.get('pre_reset',False)):
        reasons.append('state_episode_step_identity')
    goal_diff=float(np.max(np.abs(np.asarray(selected_goal,dtype=float)-np.asarray(state['logged_goal'],dtype=float))))
    if goal_diff>0:reasons.append('frozen_goal_identity')
    pose=np.asarray(anchor['root_state'][:7],dtype=float);reference_pose=np.asarray(logged['pose'],dtype=float)
    twist=np.asarray(anchor['root_state'][7:13],dtype=float);reference_twist=np.asarray(logged['twist'],dtype=float)
    position=float(np.linalg.norm(pose[:3]-reference_pose[:3]))
    orientation=float(2*math.acos(min(1.,abs(float(np.dot(pose[3:7],reference_pose[3:7]))))))
    velocity=float(np.max(np.abs(twist-reference_twist)))
    command=float(np.max(np.abs(np.asarray(anchor['command_manager'])-np.asarray(logged['command']))))
    action=float(np.max(np.abs(np.asarray(anchor['action_manager_action'])-np.asarray(logged['action']))))
    metrics={'position_m':position,'orientation_rad':orientation,'velocity_max_abs':velocity,
             'command_max_abs':command,'action_max_abs':action,'goal_max_abs':goal_diff,
             'prefix_command_max_abs':error['prefix_command_max_abs'],
             'current_dem_rms_m':error['dem_rms_m'],'current_mask_iou':error['mask_iou'],
             'critic_value_abs_vs_logged':abs(anchor['critic_value']-logged['critic_value']) if 'critic_value' in logged else None}
    for key,limit in [('position_m',limits['position_m']),('orientation_rad',limits['orientation_rad']),
                      ('velocity_max_abs',limits['velocity_max_abs']),('command_max_abs',limits['command_max_abs']),
                      ('prefix_command_max_abs',limits['command_max_abs']),
                      ('current_dem_rms_m',limits['dem_rms_m'])]:
        if metrics[key]>limit:reasons.append(key)
    if metrics['action_max_abs']!=0:reasons.append('action_manager_action')
    if metrics['current_mask_iou']<limits['mask_iou_min']:reasons.append('current_mask_iou')
    if not finite(anchor) or not finite(metrics):reasons.append('nonfinite_anchor')
    return {'status':'pass' if not reasons else 'fail','reasons':reasons,'metrics':metrics,
            'historical_reference_NA':HISTORICAL_NA}


def cleanup_gate(results, limits):
    for result in results:
        if result['own_residual'] or result['zombie_count'] or result['unrelated_signaled']:
            return 'fail','residual_zombie_or_unrelated_signal'
        if result['cleanup_elapsed_sec']>limits['cleanup_max_sec']:
            return 'fail','cleanup_timeout'
        if not result['log_flushed'] or not result['marker_flushed']:
            return 'fail','cleanup_evidence_missing'
    if len(results)>=2 and all(result['kill_used'] for result in results):
        return 'fail','both_anchors_required_kill'
    if any(result['kill_used'] for result in results):return 'pass','one_kill_warning'
    return 'pass','clean_native_or_term_exit'


def validate_process_accounting(rows):
    if len(rows)!=20 or [r['order'] for r in rows]!=list(range(20)):
        raise ValueError('20 ordered processes required')
    ids={r['state_group_id'] for r in rows}
    if len(ids)!=2 or len({r['process_id'] for r in rows})!=20:
        raise ValueError('StateGroup or process identity mismatch')
    roles=Counter(r['role'] for r in rows)
    if roles!={'anchor_only':2,'nominal_duplicate':6,'candidate_outcome':12}:
        raise ValueError('process role arithmetic mismatch')
    for state in ids:
        group=[r for r in rows if r['state_group_id']==state]
        if len(group)!=10 or sum(r['role']=='nominal_duplicate' for r in group)!=3:
            raise ValueError('per-group process arithmetic mismatch')
        if len({r['path_id'] for r in group if r['path_id']})!=7 or sum(r['oracle_denominator'] for r in group)!=7:
            raise ValueError('oracle denominator or unique Path count mismatch')
    return {'total':20,'anchor':2,'nominal_duplicate':6,'candidate_outcome':12,
            'unique_path_per_group':7}
