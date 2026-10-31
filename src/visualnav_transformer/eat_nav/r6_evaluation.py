"""Pure fixed-duration closed-loop metrics with explicit censoring."""
from __future__ import annotations
import math
import numpy as np
from .fall_stall_targets import build_targets,post_state,tilt,termination_reasons
from .r6_policy import wilson

DT=.05
RESIDUAL_POLICY='penalize_as_unavailable_stop'


def summarize_trial(rows,initial_xy,goal_xy,*,duration_sec=60.,goal_reached_time_sec=None):
    if duration_sec<=0 or len(rows)*DT>duration_sec+1e-8:raise ValueError('fixed positive evaluation duration required')
    initial=np.asarray(initial_xy,float);goal=np.asarray(goal_xy,float);delta=goal-initial;length=float(np.linalg.norm(delta))
    if length<=0:raise ValueError('nonzero fixed route required')
    direction=delta/length;distance=0.;progress=0.;tips=0;near=0;high_run=0;tipped=False
    linear=turn=unknown=0;terminal=None
    for i,row in enumerate(rows):
        if row['control_step']!=i:raise ValueError('contiguous trial intervals required')
        before=np.asarray(row['root_state'][:2]);after=np.asarray(post_state(row)[:2])
        distance+=float(np.linalg.norm(after-before))
        arc=lambda p:float(np.clip((p-initial)@direction,0,length))
        progress+=arc(after)-arc(before)
        angle=tilt(post_state(row))
        high_run=high_run+1 if angle is not None and angle>=60 else 0
        if high_run==5:near+=1
        if not tipped and ((angle is not None and angle>=85) or 'tilt_over' in termination_reasons(row)):
            tips+=1;tipped=True
        if row.get('terminal'):
            terminal=row['terminal']
            if i!=len(rows)-1:raise ValueError('trial continues beyond reset')
    for start in range(0,len(rows),40):
        chunk=[]
        for j,row in enumerate(rows[start:start+40]):
            chunk.append(dict(row,control_step=j,candidate_cycle=0,candidate_replan=False))
        path=chunk[0].get('candidate_path',[[0,0],[.75,0]])
        teacher=build_targets(chunk,path)
        targets=teacher['horizons']['2s']['events']
        if not all(targets[name]['target_mask'] for name in ('linear_stall','turn_stall')):unknown+=1
        linear+=targets['linear_stall']['label']==1;turn+=targets['turn_stall']['label']==1
    observed=len(rows)*DT;residual=max(0.,duration_sec-observed)
    last=np.asarray(post_state(rows[-1])[:2]) if rows else initial
    stop=sum(bool(r.get('protective_stop')) for r in rows)*DT;hold=sum(bool(r.get('goal_hold')) for r in rows)*DT
    return dict(schema='eatnav_r6_fixed_duration_trial_metrics_v1',duration_sec=float(duration_sec),observed_sec=observed,
        residual_unavailable_sec=residual,residual_policy=RESIDUAL_POLICY,complete=residual<1e-8,
        tip_count=tips,fall_outcome_unknown=bool(residual>1e-8 and tips==0),ranking_fall_count_upper_bound=int(tips or residual>1e-8),near_fall_count=near,protective_stop_sec=stop,goal_hold_sec=hold,
        all_zero_command_sec=sum(np.max(np.abs(r['follower_command']))<=1e-12 for r in rows)*DT,
        ranking_STOP_upper_bound_sec=stop+residual,route_progress_m=progress,route_progress_definition='sum signed changes in clamped arc on fixed initial-to-2m-goal line',
        physical_distance_m=distance,net_goal_approach_m=length-float(np.linalg.norm(goal-last)),goal_distance_m=float(np.linalg.norm(goal-last)),
        goal_reached=goal_reached_time_sec is not None,goal_reached_time_sec=goal_reached_time_sec,
        linear_windows=int(linear),turn_windows=int(turn),linear_stall_sec=linear*2.,turn_stall_sec=turn*2.,unknown_stall_windows=unknown,
        stall_window_sec=2,stall_stride_sec=2,stall_seconds_overlap='disjoint windows',terminal=terminal,
        all_candidates_excluded_decisions=sum(bool(r.get('decision_all_excluded')) for r in rows),
        timeout=not bool(goal_reached_time_sec is not None) and residual<1e-8,intervention=any(r.get('external_intervention',False) for r in rows))


def ranking_key(metrics,scales=None):
    if metrics.get('residual_policy')!=RESIDUAL_POLICY:raise ValueError('fixed residual policy required for ranking')
    names=('route_progress_m','linear_windows','turn_windows','linear_stall_sec','turn_stall_sec')
    scales=scales or {name:1. for name in names}
    if any(not math.isfinite(scales[name]) or scales[name]<=0 for name in names):raise ValueError('positive frozen validation scales required')
    third=sum((-1 if name=='route_progress_m' else 1)*metrics[name]/scales[name] for name in names)/5
    return (max(metrics['tip_count'],metrics.get('ranking_fall_count_upper_bound',metrics['tip_count'])),metrics['ranking_STOP_upper_bound_sec'],third)


def aggregate_trials(trials):
    if not trials:raise ValueError('actual nonempty trials required')
    groups={}
    for row in trials:groups.setdefault((row['field'],row['level']),[]).append(row)
    fields=('protective_stop_sec','goal_hold_sec','route_progress_m','physical_distance_m','net_goal_approach_m','linear_windows','turn_windows','linear_stall_sec','turn_stall_sec','residual_unavailable_sec','ranking_STOP_upper_bound_sec')
    def summarise(rows):
        n=len(rows);tips=sum(r['tip_count'] for r in rows);unknown=sum(bool(r.get('fall_outcome_unknown',False)) for r in rows)
        lower=wilson(tips,n);upper=wilson(min(n,tips+unknown),n)
        return dict(trials=n,tip_count=tips,tip_rate=tips/n,tip_rate_wilson95=[lower[0],upper[1]],fall_outcome_unknown_trials=unknown,ranking_fall_count_upper_bound=tips+unknown,tip_rate_lower_bound=tips/n,tip_rate_upper_bound=(tips+unknown)/n,
            goal_reached_count=sum(r['goal_reached'] for r in rows),incomplete_trials=sum(not r['complete'] for r in rows),
            **{'mean_'+name:float(np.mean([r[name] for r in rows])) for name in fields})
    cells={f'{field}/level{level}':summarise(rows) for (field,level),rows in groups.items()}
    result=summarise(trials)
    result.update(total_trials=len(trials),cells=cells,macro_tip_rate=float(np.mean([r['tip_rate'] for r in cells.values()])),
        **{'macro_mean_'+name:float(np.mean([r['mean_'+name] for r in cells.values()])) for name in fields})
    return result
