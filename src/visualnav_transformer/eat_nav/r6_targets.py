"""Observed future teachers. World-up retained; only initial position/yaw removed."""
from __future__ import annotations
import math
import numpy as np
from .fall_stall_targets import build_targets, post_state, quaternion, root_yaw

EVENT_HORIZON_SEC = 2
TIMES_SEC = np.arange(1, 21, dtype=np.float64) / 10
EVENT_NAMES = ('tip_over', 'near_fall', 'linear_stall', 'turn_stall')
TRAJECTORY_LAYOUT = ('dx_m','dy_m','dz_m','sin_relative_yaw','cos_relative_yaw',
                     'sin_roll','cos_roll','sin_pitch','cos_pitch')


def future_targets(rows):
    """Label first 2 s of one actual candidate; never infer unobserved negatives.

    Row i contains the [i*.05,(i+1)*.05] action interval. A boundary-crossing
    interval is excluded; a natural terminal's pre-reset snapshot is observed.
    Entries after any terminal/reset/manual stop are ignored.
    """
    accepted=[]
    for row in rows[:40]:
        if row.get('boundary_censor') or row.get('reset') or row.get('manual_stop_before_step') or \
                (row.get('terminal') or {}).get('termination_reason')=='parallel_env_censored_by_other_env':
            break
        row=dict(row)
        row.setdefault('candidate_cycle',row['control_step']//40)
        row.setdefault('candidate_replan',row['control_step']>0 and row['control_step']%40==0)
        accepted.append(row)
        if row.get('terminal') or row.get('external_intervention') or row.get('manual_stop'):
            break
    trajectory=np.zeros((20,9),np.float32); mask=np.zeros(20,bool)
    events=np.zeros(4,np.float32); event_mask=np.zeros(4,bool)
    contact=dict(available=False,first_observed_contact_sec=None,kind='unknown',
                 reason='per_step_contact_not_recorded')
    if not accepted:
        return dict(trajectory=trajectory,trajectory_mask=mask,event_targets=events,event_mask=event_mask,contact_metadata=contact)
    # Original teacher validates all 20 Hz boundaries and conservatively handles
    # complete stall cycles and per-head right censoring.
    teacher=build_targets(accepted, [[0.,0.],[.75,0.]])
    # Stall path geometry must be supplied by the caller where it differs from
    # straight travel; future_targets accepts row-level candidate_path for this.
    path=accepted[0].get('candidate_path')
    if path is not None:
        teacher=build_targets(accepted,path)
    for j,name in enumerate(EVENT_NAMES):
        target=teacher['horizons']['2s']['events'][name]
        event_mask[j]=target['target_mask']
        if target['label'] is not None:events[j]=target['label']
    initial=np.asarray(accepted[0]['root_state'],dtype=float)
    yaw0=root_yaw(accepted[0]['root_state'])
    if initial.shape!=(13,) or not np.isfinite(initial).all() or yaw0 is None:
        raise ValueError('invalid initial root')
    c,s=math.cos(yaw0),math.sin(yaw0)
    inverse=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    for slot,index in enumerate(range(1,40,2)):
        if index>=len(accepted):break
        root=post_state(accepted[index]);q=quaternion(root)
        if q is None or len(root)!=13 or not np.isfinite(root).all():continue
        w,x,y,z=q
        yaw=root_yaw(root)-yaw0
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=math.asin(np.clip(2*(w*y-z*x),-1,1))
        trajectory[slot]=np.r_[inverse@(np.asarray(root[:3])-initial[:3]),
            math.sin(yaw),math.cos(yaw),math.sin(roll),math.cos(roll),math.sin(pitch),math.cos(pitch)]
        mask[slot]=True
    # Net contact is not obstacle contact: only an explicit obstacle signal can
    # establish obstacle-contact onset. Never interpret ordinary ground support.
    explicit=[(i,r) for i,r in enumerate(accepted) if 'obstacle_contact' in r]
    if explicit:
        onset=next(((int(r['obstacle_contact_time_ns'])-int(accepted[0]['logical_sim_time_ns']))*1e-9 if r.get('obstacle_contact_time_ns') is not None else (i+1)*.05 for i,r in explicit if r['obstacle_contact']),None)
        contact=dict(available=True,first_observed_contact_sec=onset,kind='explicit_obstacle_contact',
                     reason='observed',observed_intervals=len(explicit),fully_observed=len(explicit)==len(accepted),timestamp_resolution_sec=.005 if any(r.get('obstacle_contact_time_ns') is not None for _,r in explicit) else .05)
    elif any('contact_net_forces_w' in r for r in accepted):
        contact['reason']='net_forces_present_but_obstacle_identity_unknown'
    return dict(trajectory=trajectory,trajectory_mask=mask,event_targets=events,event_mask=event_mask,contact_metadata=contact)
