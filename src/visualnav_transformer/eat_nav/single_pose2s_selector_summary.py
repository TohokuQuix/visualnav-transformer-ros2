"""Descriptive rollout outcomes; these are never candidate-selection constraints."""
import numpy as np
from .moving_pose2s_geometry import quat_to_matrix
from .single_pose2s_geometry import endpoint_target,path_errors


def summarize_rollout(raw,metadata,decisions,goal,goal_step):
    parent=metadata['parent'];terminal=metadata.get('terminal');end=min(200,len(raw['command']))
    if terminal is not None:end=min(end,int(terminal['step']))
    # Recompute arrival from observed prefix, including exact t=10s endpoint.
    distances=np.linalg.norm(np.asarray(raw['root'])[:end+1,:2]-np.asarray(goal),axis=1)
    arrivals=np.flatnonzero(distances<=.2+1e-10)
    if len(arrivals):goal_step=int(arrivals[0])
    elif not 0<=goal_step<=end:goal_step=-1
    if goal_step>=0:end=min(end,int(goal_step))
    terminal_in_scope=terminal if terminal is not None and int(terminal['step'])<=end else None
    root=np.asarray(raw['root'])[:end+1];R=quat_to_matrix(root[:,3:7]);tilt=np.rad2deg(np.arccos(np.clip(R[:,2,2],-1,1)));distance=np.linalg.norm(root[:,:2]-np.asarray(goal),axis=1)
    chosen=[d for d in decisions if d['parent_id']==parent['parent_id'] and d['step']<end]
    stop=sum(min(40,end-d['step'])*.05 for d in chosen if d['all_filtered'])
    linear=turn=complete=0;observed_windows=[]
    for start in range(0,end,40):
        finish=start+40
        if finish>end:continue
        complete+=1;cid=parent['candidates'][start//40];commands=np.asarray(raw['command'])[start:finish];actual=float(np.linalg.norm(np.diff(root[start:finish+1,:2],axis=0),axis=1).sum());requested=float(np.abs(commands[:,0]).sum()*.05)
        progress=None
        if cid>=0:
            target=endpoint_target(root[start],root[finish]);progress=path_errors(target['position'],target['position'],root[start,3:7],cid)['projection_s_m']
        ls=bool(requested>=.20-1e-8 and actual/max(requested,1e-12)<=.25 and progress is not None and progress<=.05)
        rates=np.abs(commands[:,2]);gyro=np.abs(np.asarray(raw['state'])[start+1:finish+1,5]);ts=bool(np.all(rates>=.01)&np.all(gyro<=rates*.50))
        linear+=ls;turn+=ts;observed_windows.append(dict(stage=start//40,linear_stall=ls,turn_stall=ts,commanded_distance_m=requested,actual_distance_m=actual,arc_progress_m=progress))
    return dict(parent_id=parent['parent_id'],field=parent['field'],level=parent['level'],condition=parent.get('condition',parent['parent_id'].split('-')[0]),goal_reached=goal_step>=0,tip_over_observed=bool((tilt>=85.).any() or (terminal_in_scope and 'tilt_over' in terminal_in_scope.get('reasons',[]))),duration_seconds=end*.05,route_progress_m=float(distance[0]-distance[-1]),remaining_goal_distance_m=float(distance[-1]),travel_distance_m=float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),filter_stop_seconds=float(stop),follower_zero_command_seconds=float(np.sum(np.all(np.abs(np.asarray(raw['command'])[:end])<1e-9,axis=1))*.05),linear_stall_windows=linear,turn_stall_windows=turn,linear_stall_positive_seconds=linear*2.,turn_stall_positive_seconds=turn*2.,max_measured_tilt_deg=float(tilt.max()),candidate_switches=sum(a['candidate_id']!=b['candidate_id'] for a,b in zip(chosen,chosen[1:])),complete_2s_windows=complete,window_outcomes=observed_windows)
