"""Fixed two-second labels from observed P1 control traces."""

from __future__ import annotations

import math
import numpy as np


SCHEMA = 'eatnav_baseline_labels_v1'
LABELS = ('negative_motion_error', 'common_goal_progress', 'realized_reward', 'negative_pain')
LABEL_DIRECTIONS = (1, 1, 1, 1)
DT = .05
WINDOW = 40


def pain_from_trace(rows, allowables):
    """Return separate normalized load components; require explicit engineering provenance."""
    if allowables is None:
        return {'status':'missing_allowables','joint_component_mean':None,
                'body_component_mean':None,'combined_mean':None}
    if allowables.get('schema')!='eatnav_pain_allowables_v1':
        raise ValueError('Pain allowables schema')
    names_j=allowables.get('joint_names',[]);names_b=allowables.get('body_names',[])
    tau_ref=allowables.get('joint_torque_ref_Nm',{});force_ref=allowables.get('body_force_ref_N',{})
    joint_w=allowables.get('joint_weights',{});body_w=allowables.get('body_weights',{})
    if not names_j or not names_b or any(name not in tau_ref or name not in joint_w for name in names_j) or \
            any(name not in force_ref or name not in body_w for name in names_b):
        return {'status':'missing_allowables','joint_component_mean':None,
                'body_component_mean':None,'combined_mean':None}
    if any(not math.isfinite(float(value)) or value<=0 for table in
           (tau_ref,force_ref,joint_w,body_w) for value in table.values()):
        raise ValueError('Pain allowable nonpositive/nonfinite')
    joint=[];body=[];exceed=[]
    for row in rows:
        torques=row['applied_torque'];forces=row['contact_net_forces_w']
        if len(torques)!=len(names_j) or len(forces)!=len(names_b):
            raise ValueError('Pain load/index mapping')
        ju=[abs(float(value))/tau_ref[name] for value,name in zip(torques,names_j)]
        bu=[float(np.linalg.norm(value))/force_ref[name] for value,name in zip(forces,names_b)]
        if not all(math.isfinite(value) for value in ju+bu):
            raise ValueError('Pain nonfinite load')
        joint.append(sum(joint_w[name]*value**2 for name,value in zip(names_j,ju)))
        body.append(sum(body_w[name]*value**2 for name,value in zip(names_b,bu)))
        exceed.append(max(ju+bu)>=1.)
    combination=allowables.get('combination_weights')
    combined=None
    if combination is not None:
        if set(combination)!={'joint','body'} or any(not math.isfinite(float(v)) or v<0
              for v in combination.values()):
            raise ValueError('Pain combination weights')
        combined=float(combination['joint']*np.mean(joint)+combination['body']*np.mean(body))
    return {'status':('engineering_verified' if allowables.get('provenance')=='engineering_verified'
                      else 'synthetic_or_unverified'),
            'joint_component_mean':float(np.mean(joint)),
            'body_component_mean':float(np.mean(body)),
            'joint_component_peak':float(np.max(joint)),
            'body_component_peak':float(np.max(body)),
            'joint_component_p95':float(np.percentile(joint,95)),
            'body_component_p95':float(np.percentile(body,95)),
            'exceedance_duration_s':float(sum(exceed)*DT),
            'combined_mean':combined}


def _yaw(root):
    w,x,y,z = root[3:7]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def labels_from_trace(rows, *, common_goal_xy, env_origin_xy, gamma=1.0,
                      forward_scale_mps=.3, yaw_scale_radps=1.0, pain_allowables=None):
    if not 0 < gamma <= 1 or not rows or len(common_goal_xy)!=2 or len(env_origin_xy)!=2 or \
            not math.isfinite(forward_scale_mps) or forward_scale_mps<=0 or \
            not math.isfinite(yaw_scale_radps) or yaw_scale_radps<=0:
        raise ValueError('label input contract')
    first_stamp=rows[0]['logical_sim_time_ns']
    observed=rows[:WINDOW]
    for step,row in enumerate(observed):
        if row['control_step']!=step or row['logical_sim_time_ns']!=first_stamp+step*50_000_000:
            raise ValueError('label trace stamp/step')
    arrival=next((r['control_step'] for r in observed if r['follower_diagnostic']['goal_reached']),None)
    complete=len(observed)==WINDOW and not any(r['terminal'] is not None for r in observed)
    result={'schema':SCHEMA,'status':'complete_2s' if complete else 'censored_before_2s',
            'window_steps':len(observed),'arrival_step':arrival,
            'values':[None]*4,'availability':[False]*4,
            'pain_status':'missing_allowables',
            'pain_components':None,
            'motion_components':None,
            'active_command_fraction':None,'zero_cap_fraction':None,
            'reward_term_availability':'total_only_for_rosfree_crossover'}
    if not complete:
        return result
    pre=np.asarray(observed[0]['root_state'][:2],dtype=float)-env_origin_xy
    post=np.asarray(observed[-1]['post_root_state'][:2],dtype=float)-env_origin_xy
    goal=np.asarray(common_goal_xy,dtype=float)
    progress=float(np.linalg.norm(pre-goal)-np.linalg.norm(post-goal))
    reward=sum(float(r['reward'])*(gamma**i) for i,r in enumerate(observed))
    forward_signed=[]; yaw_signed=[]; active=0; zero_cap=0
    for row in observed:
        a,b=row['root_state'],row['post_root_state']
        angle=_yaw(a)
        dx,dy=b[0]-a[0],b[1]-a[1]
        actual_forward=(dx*math.cos(angle)+dy*math.sin(angle))/DT
        actual_yaw=_wrap(_yaw(b)-angle)/DT
        command=row['follower_command']
        forward_signed.append(float(command[0])-actual_forward)
        yaw_signed.append(float(command[2])-actual_yaw)
        active+=int(abs(float(command[0]))>1e-9 or abs(float(command[2]))>1e-9)
        diagnostic=row['follower_diagnostic']
        zero_cap+=int(bool(diagnostic.get('goal_reached')) or
                      abs(float(diagnostic.get('heading_error',0.))) >= .6)
    forward=np.asarray(forward_signed,dtype=float)
    yaw=np.asarray(yaw_signed,dtype=float)
    if not np.isfinite(forward).all() or not np.isfinite(yaw).all() or \
            not math.isfinite(progress) or not math.isfinite(reward):
        raise ValueError('nonfinite label')
    motion=-(float(np.abs(forward).mean())/forward_scale_mps+
             float(np.abs(yaw).mean())/yaw_scale_radps)
    pain=pain_from_trace(observed,pain_allowables)
    pain_available=pain['status']=='engineering_verified' and pain['combined_mean'] is not None
    result.update(values=[motion,progress,reward,-pain['combined_mean'] if pain_available else None],
                  availability=[True,True,True,pain_available],
                  pain_status=pain['status'],pain_components=pain,
                  motion_components={'forward_signed_mean_mps':float(forward.mean()),
                                     'forward_abs_mean_mps':float(np.abs(forward).mean()),
                                     'forward_rmse_mps':float(np.sqrt(np.square(forward).mean())),
                                     'yaw_signed_mean_radps':float(yaw.mean()),
                                     'yaw_abs_mean_radps':float(np.abs(yaw).mean()),
                                     'yaw_rmse_radps':float(np.sqrt(np.square(yaw).mean())),
                                     'forward_scale_mps':forward_scale_mps,
                                     'yaw_scale_radps':yaw_scale_radps},
                  active_command_fraction=active/WINDOW,zero_cap_fraction=zero_cap/WINDOW)
    return result
