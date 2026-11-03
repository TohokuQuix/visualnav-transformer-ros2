"""Exploratory A/B selection: XY goal progress and optional predicted tilt only."""
import numpy as np
from .demo_terrain import TERRAIN_TYPES


def validate_exploratory_authorization(a,checkpoint_sha):
    if a.get('user_explicitly_authorized') is not True or a.get('scope')!='exploratory_selector_and_offline_bag' or a.get('checkpoint_sha256')!=checkpoint_sha or a.get('strict_accuracy_achieved') is not False or a.get('defer_github_push') is not True:
        raise ValueError('explicit provisional-model authorization required; strict accuracy is not achieved')


def select_candidate(position,rotation,current_position,current_rotation,goal_xy,condition):
    p=np.asarray(position,float);R=np.asarray(rotation,float);p0=np.asarray(current_position,float);R0=np.asarray(current_rotation,float);goal=np.asarray(goal_xy,float)
    if condition not in ['A','B'] or p.shape!=(14,3) or R.shape!=(14,3,3) or p0.shape!=(3,) or R0.shape!=(3,3) or goal.shape!=(2,) or not all(np.isfinite(x).all() for x in [p,R,p0,R0,goal]):raise ValueError('invalid selector prediction/input')
    if not np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-5) or np.any(np.linalg.det(R)<0) or not np.allclose(R0.T@R0,np.eye(3),atol=1e-5):raise ValueError('rotation outside SO3')
    future=p0+p@R0.T;future_R=R0[None]@R
    tilt=np.rad2deg(np.arccos(np.clip(future_R[:,2,2],-1,1)))
    scores=np.linalg.norm(p0[:2]-goal)-np.linalg.norm(future[:,:2]-goal,axis=1)
    allowed=np.ones(14,bool) if condition=='A' else tilt<=60.+1e-10
    ids=np.flatnonzero(allowed);chosen=None if not len(ids) else int(ids[np.argmax(scores[ids])])
    return dict(candidate_id=chosen,score_m=scores.tolist(),predicted_tilt_deg=tilt.tolist(),allowed=allowed.tolist(),all_filtered=chosen is None,predicted_world_positions=future.tolist())


def make_selector_manifest(seed=20261104,conditions=('A','B')):
    if not conditions or len(set(conditions))!=len(conditions) or any(c not in ['A','B'] for c in conditions):raise ValueError('conditions must be unique A/B')
    batches=[];used=set()
    for fi,field in enumerate(TERRAIN_TYPES):
        for level in range(5):
            pair=[]
            for i in range(10):
                s=int(np.random.SeedSequence([seed,fi,level,i,710]).generate_state(1)[0]);initial=int(np.random.SeedSequence([s,1]).generate_state(1)[0])
                if s in used:raise ValueError('seed collision')
                used.add(s);pair.append(dict(pair_id=f'{field}-l{level}-s{i:02d}',field=field,level=level,seed=s,initial_seed=initial,trial_index=i,dem_phase_steps=i%5,candidates=[0]*5))
            for c in conditions:
                parents=[dict(p,parent_id=c+'-'+p['pair_id'],role='selector_'+c,condition=c) for p in pair]
                batches.append(dict(tag=f'{c}-{field}-l{level}',field=field,level=level,condition=c,parents=parents))
    return dict(schema='eatnav_pose2s_exploratory_selector_manifest_v1',seed=seed,batches=batches,planned_runs=350*len(conditions),goal_forward_m=3.,goal_tolerance_m=.2,time_limit_s=10.,replan_seconds=2.,tilt_threshold_deg=60.,strict_accuracy_achieved=False,known_terrain_independent_matched_runs=True)


def decision_input(raw,odometry):
    """Current prefix -> public K14 payload, using only causal actual-LIO pose differences."""
    from .single_pose2s_contract import INPUT_SCHEMA,PROFILE_SHA,validate_input
    from .single_pose2s_fastlio import odom_velocity_at
    from .single_pose2s_collection import causal_dem_context
    from .bidirectional_bank import bank
    t0=int(raw['stamps_ns'][-1]);frames=np.flatnonzero(np.asarray(raw['dem_stamp_ns'])<=t0)
    if not len(frames):raise ValueError('no current causal DEM')
    k=int(frames[-1]);stamp=int(raw['dem_stamp_ns'][k]);state=np.asarray(raw['state'][-1]).copy();mask=np.asarray(raw['state_mask'][-1]).copy();age=np.asarray(raw['state_age'][-1]).copy()
    v,w,dt,ids=odom_velocity_at(t0,**odometry);state[18:21]=v;state[21:24]=w;mask[18:24]=1;age[18:24]=dt
    times=np.asarray(odometry['stamps_ns']);received=np.asarray(odometry['received_ns']);available=np.flatnonzero((times<=stamp)&(received<=stamp))
    if not len(available):raise ValueError('no causal LIO frame pose')
    fid=int(available[np.argmax(times[available])]);poses=np.asarray(odometry['poses'])
    s=dict(schema=INPUT_SCHEMA,dem=np.asarray(raw['dem'][k]),state=state,state_mask=mask,state_age=age,dem_context=causal_dem_context(poses[fid],poses[ids[-1]],stamp,t0),paths=np.asarray([c.waypoints for c in bank()]),directions=np.asarray([[c.direction] for c in bank()]),t0_ns=t0,dem_stamp_ns=stamp,dem_received_ns=stamp)
    validate_input(s)
    a={k:np.asarray([val]) for k,val in s.items() if k!='schema'}
    a.update(input_schema=np.array(INPUT_SCHEMA),profile_sha256=np.array(PROFILE_SHA),q0_wxyz=np.asarray(raw['root'][-1,3:7])[None])
    return a


def validate_run_identity(expected,actual):
    if not expected or expected!=actual:raise ValueError('model/selector/runtime identity changed; do not mix resumed rollouts')
