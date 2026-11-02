"""Causal, independently masked 2 s pose and event teachers from 20 Hz traces.

Only observed posts enter the forty-point trajectory. Unknown values are NaN,
with explicit masks and reasons. A trace index is distinct from its global
control_step, which can include a prefix/anchor offset.
"""
from __future__ import annotations
import copy
import numpy as np
from . import fall_stall_targets as legacy
from .moving_pose2s_geometry import normalize_quaternion, relative_endpoint, rotation_geodesic, quat_to_matrix

SCHEMA='eatnav_moving_pose2s_targets_v1'
DT_NS=50_000_000
STEPS=40
EVENT_NAMES=legacy.EVENT_NAMES


def _root(value):
    try:
        root=np.asarray(value,dtype=np.float64)
        if root.shape!=(13,) or not np.isfinite(root).all():return None
        norm=np.linalg.norm(root[3:7])
        if not np.isfinite(norm) or abs(norm-1)>1e-3:return None
        root=root.copy();root[3:7]=normalize_quaternion(root[3:7])
        return root
    except (ValueError,TypeError,OverflowError):return None


def _same_root(a,b,atol):
    return (np.allclose(a[:3],b[:3],atol=atol,rtol=0) and
            np.allclose(a[7:],b[7:],atol=atol,rtol=0) and
            rotation_geodesic(quat_to_matrix(a[3:7]),quat_to_matrix(b[3:7]))<=atol)


def _intervention(row,index,candidate):
    if index and row.get('candidate_replan'):return 'internal_replan'
    if candidate is not None and row.get('candidate_id') is not None and row['candidate_id']!=candidate:return 'candidate_changed'
    if row.get('append_events') or row.get('path_append') or row.get('append_count',0):return 'path_append'
    path_id=str(row.get('path_id',''))
    if ':' in path_id and path_id.rsplit(':',1)[-1].isdigit() and int(path_id.rsplit(':',1)[-1])>0:
        return 'path_append'
    for key in ('external_intervention','external_stop','manual_stop','guard_stopped','boundary_crossed',
                'boundary_violation','boundary_censor','reset','recovery_command','recovery_intervention'):
        if row.get(key):return key
    diagnostic=row.get('follower_diagnostic') or {}
    if any(diagnostic.get(k) for k in ('linear_recovery_started','angular_recovery_started')):
        return 'recovery_intervention'
    terminal=row.get('terminal') or {}
    reasons=legacy.termination_reasons(row)
    if any(any(token in reason.lower() for token in ('manual','boundary','parallel_env','external','censor','reset')) for reason in reasons):
        return 'censored_terminal'
    if terminal and not reasons:return 'unknown_terminal'
    return None


def build_window(rows,start,snapshot,contract=None):
    """Build targets for one executed candidate, never its unobserved alternatives.

    ``snapshot`` is a mapping with root_state, t0_ns, candidate_id,
    path_body_xy [8,2], and direction (+1/-1). Root/time/candidate may be inferred
    from the first row when omitted; supplied values must agree. ``contract``
    optionally sets root_atol (default 1e-5); the horizon and cadence are fixed.
    Noncausal/malformed intervals are censored with reasons rather than filled.
    """
    contract={} if contract is None else contract
    if contract.get('horizon_sec',2.)!=2. or contract.get('control_hz',20)!=20:
        raise ValueError('moving_pose2s requires 2 s at 20 Hz')
    if not isinstance(start,(int,np.integer)) or start<0 or start>=len(rows):
        raise ValueError('start must index an observed interval')
    snapshot=dict(snapshot or {})
    first=rows[start];root0=_root(snapshot.get('root_state',first.get('root_state')))
    first_root=_root(first.get('root_state'));t0=snapshot.get('t0_ns',first.get('logical_sim_time_ns'))
    candidate=snapshot.get('candidate_id',first.get('candidate_id'))
    tolerance=float(contract.get('root_atol',1e-5))
    reasons=[];observed=[];positions=np.full((STEPS,3),np.nan);quaternions=np.full((STEPS,4),np.nan)
    trajectory_mask=np.zeros(STEPS,dtype=bool)
    if root0 is None or first_root is None:reasons.append('invalid_snapshot_root')
    elif not _same_root(root0,first_root,tolerance):reasons.append('snapshot_root_mismatch')
    if not isinstance(t0,(int,np.integer)) or isinstance(t0,bool) or t0!=first.get('logical_sim_time_ns'):
        reasons.append('snapshot_t0_mismatch')
    if snapshot.get('candidate_id') is not None and first.get('candidate_id') is not None and snapshot['candidate_id']!=first['candidate_id']:
        reasons.append('snapshot_candidate_mismatch')
    path=snapshot.get('path_body_xy',snapshot.get('path_body_xy_m'))
    try:
        path=np.asarray(path,dtype=float)
        if path.shape!=(8,2) or not np.isfinite(path).all() or np.linalg.norm(np.diff(path,axis=0),axis=-1).sum()<=1e-12:
            raise ValueError()
        path=path.tolist()
    except (ValueError,TypeError):
        path=None;reasons.append('invalid_snapshot_path')
    if snapshot.get('direction') not in (-1,1):reasons.append('invalid_snapshot_direction')
    step0=first.get('control_step')
    if not isinstance(step0,(int,np.integer)):reasons.append('invalid_start_control_step')
    initial_reasons=list(reasons)
    previous=None;terminal_seen=False;terminal_tip_without_pose=False
    if not initial_reasons:
        for i,row in enumerate(rows[start:start+STEPS]):
            bad=_intervention(row,i,candidate)
            for identity in ('episode_id','reset_id','episode_index'):
                if identity in first and row.get(identity)!=first[identity]:bad='episode_or_reset_changed'
            if bad:reasons.append(bad);break
            if row.get('control_step')!=step0+i or row.get('logical_sim_time_ns')!=t0+i*DT_NS:
                reasons.append('noncontiguous_20hz_intervals');break
            # An explicit dt/post timestamp is checked when available.
            if ('dt' in row and not np.isclose(row['dt'],.05,atol=1e-9,rtol=0)) or (
                'post_logical_sim_time_ns' in row and row['post_logical_sim_time_ns']!=t0+(i+1)*DT_NS):
                reasons.append('invalid_interval_duration');break
            pre=_root(row.get('root_state'))
            if pre is None:reasons.append('invalid_pre_root');break
            if previous is not None and not _same_root(previous,pre,tolerance):
                reasons.append('root_discontinuity_or_reset');break
            terminal=row.get('terminal') or {}
            # Natural termination has a captured pre-reset state; reset output is never a substitute.
            post_raw=(terminal.get('pre_reset_state') or {}).get('root_state') if terminal else row.get('post_root_state')
            post=_root(post_raw)
            if post is None:
                terminal_tip_without_pose=bool(terminal and 'tilt_over' in legacy.termination_reasons(row))
                terminal_seen=bool(terminal)
                reasons.append('missing_pre_reset_root' if terminal else 'invalid_post_root');break
            relative=relative_endpoint(root0,post)
            positions[i]=relative['delta_position_body_m'];quaternions[i]=relative['delta_quaternion_wxyz'];trajectory_mask[i]=True
            local=copy.deepcopy(row)
            local.update(control_step=i,logical_sim_time_ns=int(t0)+i*DT_NS,candidate_cycle=0,candidate_replan=False,
                         root_state=pre.tolist(),post_root_state=post.tolist())
            if terminal:local['terminal']['pre_reset_state']['root_state']=post.tolist()
            observed.append(local);previous=post
            if terminal:
                terminal_seen=True
                if i<STEPS-1:reasons.append('early_natural_terminal')
                break
    complete=len(observed)==STEPS
    if not complete and not reasons:reasons.append('trace_cutoff')
    primary=not initial_reasons and all(r=='early_natural_terminal' for r in reasons)
    # True observed positives survive censoring; absent events require a complete horizon.
    values=np.full(4,np.nan);masks=np.zeros(4,dtype=bool);event_reasons={name:'no_valid_observations' for name in EVENT_NAMES}
    evidence=None
    if observed:
        evidence=legacy.build_targets(observed,path,decision_stamp_ns=int(t0))
        heads=evidence['horizons']['2s']['events']
        for j,name in enumerate(EVENT_NAMES):
            measurement=heads[name];masks[j]=measurement['target_mask']
            if masks[j]:values[j]=measurement['label']
            event_reasons[name]=measurement['reason']
    if terminal_tip_without_pose:
        values[0]=1.;masks[0]=True;event_reasons['tip_over']='natural_tilt_over_terminal_without_pose'
    endpoint=relative_endpoint(root0,previous) if complete else dict(delta_position_body_m=np.full(3,np.nan),
                delta_rotation_6d=np.full(6,np.nan),delta_quaternion_wxyz=np.full(4,np.nan))
    return dict(schema=SCHEMA,pos3=endpoint['delta_position_body_m'],rot6=endpoint['delta_rotation_6d'],
                quat4=endpoint['delta_quaternion_wxyz'],pose_mask=bool(complete),events4=values,event_mask4=masks,
                trajectory_position=positions,trajectory_quaternion=quaternions,trajectory_mask=trajectory_mask,
                trajectory_time_offset_sec=np.arange(1,STEPS+1)*.05,primary_eligible=bool(primary),
                reasons=dict(window=reasons,pose='complete_2s' if complete else (reasons[-1] if reasons else 'missing'),events=event_reasons),
                metadata=dict(t0_ns=t0,start_index=int(start),start_control_step=step0,candidate_id=candidate,
                              interval_count=len(observed),horizon_sec=2.,terminal_observed=terminal_seen,
                              pose_supervision='complete_2s_only',event_names=EVENT_NAMES,
                              event_evidence=evidence),**endpoint)
