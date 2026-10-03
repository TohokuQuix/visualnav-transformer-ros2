"""Frozen E1 duplicate and P→C→R→Y analysis at the StateGroup level."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np
import yaml

from eat_nav_e1_prepare import G0E,sha
from eat_nav_g0e_gpu_probe import atomic


def load_trace(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def rms(value):
    array=np.asarray(value,dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def post_xy(row):
    terminal=row.get('terminal')
    root=terminal.get('root_state') if terminal and terminal.get('root_state') is not None else row['post_root_state']
    return np.asarray(root[:2],dtype=np.float64)


def relative_xy(trace):
    start=np.asarray(trace[0]['root_state'][:2],dtype=np.float64)
    return np.stack([post_xy(row)-start for row in trace])


def duplicate_metrics(traces,threshold):
    if len(traces)!=3 or not traces[0] or any(len(t)!=len(traces[0]) for t in traces):
        raise ValueError('three equal-length direct repeats required')
    keys=('follower_command','actor_input','actor_latent','actor_action','critic_observation',
          'critic_value','reward','joint_position','joint_velocity','contact_force')
    measurements={'xy_rms_m':0.,'command_rms_m_per_s':0.,'actor_input_max_abs':0.,
        'actor_action_max_abs':0.,'critic_value_max_abs':0.,'reward_max_abs':0.,
        'DEM_RMS_max_m':0.,'DEM_mask_IoU_min':1.}
    first=None;reason=[];other_rms={key:[] for key in keys}
    for a,b in ((0,1),(0,2),(1,2)):
        left,right=traces[a],traces[b]
        measurements['xy_rms_m']=max(measurements['xy_rms_m'],rms(relative_xy(left)-relative_xy(right)))
        for step,(x,y) in enumerate(zip(left,right)):
            for key in keys:
                if key not in x or key not in y:continue
                xa=np.asarray(x[key],dtype=np.float64);ya=np.asarray(y[key],dtype=np.float64)
                if xa.shape!=ya.shape or not np.isfinite(xa).all() or not np.isfinite(ya).all():
                    reason.append('signal_shape_or_finite_'+key);continue
                delta=xa-ya;other_rms[key].extend(delta.ravel().tolist())
                if first is None and np.any(delta):
                    first={'step':step,'signal':key,'max_abs':float(np.max(np.abs(delta))),'pair':[a,b]}
                if key=='actor_input':measurements['actor_input_max_abs']=max(measurements['actor_input_max_abs'],float(np.max(np.abs(delta))))
                if key=='actor_action':measurements['actor_action_max_abs']=max(measurements['actor_action_max_abs'],float(np.max(np.abs(delta))))
                if key=='critic_value':measurements['critic_value_max_abs']=max(measurements['critic_value_max_abs'],float(np.max(np.abs(delta))))
                if key=='reward':measurements['reward_max_abs']=max(measurements['reward_max_abs'],float(np.max(np.abs(delta))))
            if x.get('terminal')!=y.get('terminal'):reason.append('termination_difference')
            for stamp in ('logical_sim_time_ns','tf_stamp_ns','follower_command_stamp_ns','clock_stamp_ns','path_stamp_ns'):
                if x.get(stamp)!=y.get(stamp):reason.append('exact_stamp_difference_'+stamp)
            da=np.asarray(x['dem']['transformed'],dtype=np.float32)
            db=np.asarray(y['dem']['transformed'],dtype=np.float32)
            if da.shape!=db.shape or da.shape[0]!=2 or not np.isfinite(da).all() or not np.isfinite(db).all():
                reason.append('DEM_shape_or_finite');continue
            va,vb=da[1]>.5,db[1]>.5;common=va&vb
            if not common.any():reason.append('DEM_no_common_valid');continue
            measurements['DEM_RMS_max_m']=max(measurements['DEM_RMS_max_m'],rms(da[0][common]-db[0][common]))
            measurements['DEM_mask_IoU_min']=min(measurements['DEM_mask_IoU_min'],float(common.sum()/(va|vb).sum()))
    measurements['command_rms_m_per_s']=rms(other_rms['follower_command'])
    for key in ('follower_command','actor_input','actor_latent','actor_action','critic_observation',
                'critic_value','reward','joint_position','joint_velocity','contact_force'):
        measurements[key+'_rms']=rms(other_rms[key]) if other_rms[key] else None
    for metric,limit in [('xy_rms_m',threshold['xy_rms_max_m']),
        ('command_rms_m_per_s',threshold['command_rms_max_m_per_s']),
        ('actor_input_max_abs',threshold['actor_input_max_abs']),
        ('actor_action_max_abs',threshold['actor_action_max_abs']),
        ('critic_value_max_abs',threshold['critic_value_max_abs']),
        ('reward_max_abs',threshold['reward_max_abs']),
        ('DEM_RMS_max_m',threshold['dem_rms_max_m'])]:
        if measurements[metric]>limit:reason.append(metric+'_above_frozen_limit')
    if measurements['DEM_mask_IoU_min']<threshold['dem_mask_iou_min']:
        reason.append('DEM_mask_IoU_below_frozen_limit')
    return {'status':'pass' if not reason else 'fail','reason':sorted(set(reason)),
        'first_divergence':first,'step_count':len(traces[0]),**measurements}


def duplicate_gate(rows,root,gates):
    if len(rows)!=3 or [r['role'] for r in rows]!=['nominal_duplicate']*3:
        raise ValueError('three fixed direct repeats required')
    folders=[Path(root)/'runs'/r['process_id'] for r in rows]
    summaries=[json.loads((f/'live-0.json').read_text()) for f in folders]
    traces=[load_trace(f/'live-0.trace.json.gz') for f in folders]
    if any(len(t)!=120 or any(row['terminal'] for row in t) for t in traces):
        return {'status':'fail','reason':['direct_repeat_not_120_nonterminal'],
            'trace_sha256':[sha(f/'live-0.trace.json.gz') for f in folders]}
    if len({s['fixed_map_path_sha256'] for s in summaries})!=1 or \
            any(s['anchor_gate']['status']!='pass' for s in summaries):
        return {'status':'fail','reason':['direct_Path_or_anchor_identity'],
            'trace_sha256':[sha(f/'live-0.trace.json.gz') for f in folders]}
    result=duplicate_metrics(traces,gates['duplicate'])
    result['state_group_id']=rows[0]['state_group_id']
    result['frozen_gate_sha256']=sha(Path(root)/'e1-gates.yaml')
    result['trace_sha256']=[sha(f/'live-0.trace.json.gz') for f in folders]
    result['run_complete_marker_sha256']=[sha(f/'run-complete.json') for f in folders]
    return result


def unique_candidate_outcomes(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['candidate_id']].append(row['outcome'])
    result={}
    for name,values in grouped.items():
        keys=set().union(*(v.keys() for v in values))
        result[name]={key:(float(statistics.median(x[key] for x in values if isinstance(x.get(key),(int,float))
            and not isinstance(x[key],bool))) if any(isinstance(x.get(key),(int,float)) and not isinstance(x[key],bool) for x in values)
            else values[0].get(key)) for key in keys}
    return result


def outcome(trace,summary,path):
    root=np.asarray([post_xy(r) for r in trace],dtype=np.float64)
    start=np.asarray(trace[0]['root_state'][:2],dtype=np.float64)
    planned=np.asarray(path,dtype=np.float64)
    goal=planned[-1]
    dist=np.sqrt(((root[:,None,:]-planned[None,:,:])**2).sum(axis=2)).min(axis=1)
    critic=np.asarray([r['critic_value'] for r in trace],dtype=np.float64)
    reward=np.asarray([r['reward'] for r in trace],dtype=np.float64)
    n=len(trace);end=summary['suffix_end_critic_value']
    delta=float(end-critic[0]) if end is not None else None
    advantage=float(np.sum((.99**np.arange(n))*reward)+(.99**n)*end-critic[0]) if end is not None else None
    motion=np.diff(np.vstack([start,root]),axis=0)
    speed_cmd=np.asarray([r['follower_command'][0] for r in trace],dtype=float)
    stack_proxy=bool(n>=20 and np.linalg.norm(root[-1]-root[-20])<.02 and np.mean(np.abs(speed_cmd[-20:]))>.1)
    contact=max(float(np.max(np.linalg.norm(np.asarray(r['contact_force'],dtype=float),axis=-1))) for r in trace)
    effort=max(float(np.max(np.abs(np.asarray(r.get('raw_joint_effort') or [0.],dtype=float)))) for r in trace)
    tilt=[]
    for r in trace:
        q=np.asarray(r['root_state'][3:7],dtype=float)
        w,x,y,z=q
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
        tilt.append(max(abs(roll),abs(pitch)))
    terminal=trace[-1]['terminal']
    return {'control_steps':n,'physical_horizon_s':n/20,
        'initial_critic':float(critic[0]),'final_critic':float(end) if end is not None else None,
        'delta_v':delta,'H_step_advantage':advantage,
        'drawdown':float(min(critic.min(),end if end is not None else critic[-1])-critic[0]),
        'recovery':float((end if end is not None else critic[-1])-critic.min()),
        'progress_m':float(np.linalg.norm(goal-start)-np.linalg.norm(goal-root[-1])),
        'endpoint_error_m':float(np.linalg.norm(goal-root[-1])),
        'cross_track_p95_m':float(np.quantile(dist,.95)),
        'path_length_m':float(np.linalg.norm(motion,axis=1).sum()),
        'maximum_roll_pitch_rad':max(tilt),'maximum_contact_n':contact,
        'maximum_raw_joint_effort':effort,'stack_proxy':stack_proxy,
        'verified_stack':stack_proxy and n>=20 and bool(np.mean(np.abs(speed_cmd[-20:]))>.1),
        'termination':bool(terminal),'termination_reason':terminal.get('termination_reason',[]) if terminal else [],
        'censored':bool(terminal),'infrastructure_valid':True,
        'timeout_alone_is_failure':False}


def distance_matrix(series):
    names=list(series)
    matrix=[]
    for a in names:
        row=[]
        for b in names:
            x=np.asarray(series[a],dtype=float);y=np.asarray(series[b],dtype=float)
            n=min(len(x),len(y));row.append(rms(x[:n]-y[:n]))
        matrix.append(row)
    return {'names':names,'rms':matrix}


def clusters(matrix,threshold):
    distances=np.asarray(matrix['rms']);parent=list(range(len(distances)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(distances)):
        for j in range(i):
            if distances[i,j]<=threshold:parent[root(i)]=root(j)
    labels=[root(i) for i in range(len(distances))]
    return len(set(labels)),labels


def group_analysis(group,root,gates,duplicate):
    root=Path(root);records=[];trace_by={};summary_by={}
    for row in group:
        folder=root/'runs'/row['process_id']
        summary=json.loads((folder/'live-0.json').read_text())
        trace=load_trace(folder/'live-0.trace.json.gz')
        path=summary['map_path']
        records.append({'process_id':row['process_id'],'candidate_id':row['path_id'],
            'role':row['role'],'repeat_id':row['repeat_id'],'path_sha256':row['map_path_sha256'],
            'trace_sha256':sha(folder/'live-0.trace.json.gz'),
            'outcome':outcome(trace,summary,path)})
        trace_by[row['process_id']]=trace;summary_by[row['process_id']]=summary
    unique=unique_candidate_outcomes(records)
    representative={}
    for row in group:
        representative.setdefault(row['path_id'],row['process_id'])
    paths={name:np.asarray(summary_by[pid]['map_path'],dtype=float) for name,pid in representative.items()}
    commands={name:np.asarray([r['follower_command'] for r in trace_by[pid]],dtype=float)
        for name,pid in representative.items()}
    realized={name:relative_xy(trace_by[pid]) for name,pid in representative.items()}
    direct_pids=[r['process_id'] for r in group[:3]]
    commands['direct_reference']=np.median(np.stack([np.asarray([r['follower_command'] for r in trace_by[p]],dtype=float)
        for p in direct_pids]),axis=0)
    realized['direct_reference']=np.median(np.stack([relative_xy(trace_by[p]) for p in direct_pids]),axis=0)
    path_matrix=distance_matrix(paths)
    command_matrix=distance_matrix(commands)
    realized_matrix=distance_matrix(realized)
    command_modes,command_labels=clusters(command_matrix,gates['P_to_C']['follower_command_rms_above_m_per_s'])
    realized_modes,realized_labels=clusters(realized_matrix,gates['C_to_R']['realized_xy_rms_above_m'])
    names=path_matrix['names'];direct_index=names.index('direct_reference')
    command_effect=[command_matrix['rms'][direct_index][i] for i,name in enumerate(names) if name!='direct_reference']
    realized_effect=[realized_matrix['rms'][direct_index][i] for i,name in enumerate(names) if name!='direct_reference']
    duplicate_xy=max(duplicate['xy_rms_m'],gates['C_to_R']['ratio_denominator_floor_m'])
    outcome_spread={key:float(np.ptp([x[key] for x in unique.values() if isinstance(x.get(key),(int,float))]))
        for key in ('delta_v','H_step_advantage','progress_m','endpoint_error_m','cross_track_p95_m')
        if any(isinstance(x.get(key),(int,float)) for x in unique.values())}
    p_to_c=(sum(x>gates['P_to_C']['follower_command_rms_above_m_per_s'] for x in command_effect)>=2
        and command_modes>=gates['P_to_C']['minimum_command_clusters_per_group'])
    c_to_r=(sum(x>gates['C_to_R']['realized_xy_rms_above_m'] for x in realized_effect)>=2
        and realized_modes>=gates['C_to_R']['minimum_realized_modes_per_group']
        and max(realized_effect)/duplicate_xy>=gates['C_to_R']['candidate_effect_to_duplicate_noise_ratio_min'])
    ry=(outcome_spread.get('delta_v',0)>gates['R_to_Y']['minimum_critic_value_difference']
        or outcome_spread.get('progress_m',0)>gates['R_to_Y']['minimum_progress_or_endpoint_difference_m']
        or outcome_spread.get('endpoint_error_m',0)>gates['R_to_Y']['minimum_progress_or_endpoint_difference_m']
        or len({str(x['termination_reason']) for x in unique.values()})>1)
    result={'state_group_id':group[0]['state_group_id'],'candidate_order':names,
        'path_pairwise':path_matrix,'command_pairwise':command_matrix,
        'realized_pairwise':realized_matrix,'command_mode_count':command_modes,
        'realized_mode_count':realized_modes,'command_mode_labels':command_labels,
        'realized_mode_labels':realized_labels,'direct_to_nondirect_command_rms':command_effect,
        'direct_to_nondirect_realized_rms':realized_effect,
        'candidate_to_duplicate_xy_effect_ratio':max(realized_effect)/duplicate_xy,
        'outcome_spread':outcome_spread,'P_to_C_pass':p_to_c,'C_to_R_pass':c_to_r,'R_to_Y_difference':ry,
        'all_same_tracking_failure':all(x['verified_stack'] or x['termination'] for x in unique.values()),
        'raw_records':records,'unique_candidate_outcomes':unique}
    return result,paths,commands,realized


def figures(root,groups,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/'figures';folder.mkdir(exist_ok=True)
    for sid,(paths,commands,realized) in data.items():
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for name,points in paths.items():ax[0].plot(points[:,0],points[:,1],label=name)
        for name,points in realized.items():ax[1].plot(points[:,0],points[:,1],label=name)
        ax[0].set_title('7 planned Paths');ax[1].set_title('realized XY')
        for a in ax:a.axis('equal');a.grid();a.legend(fontsize=6)
        fig.tight_layout();fig.savefig(folder/f'{sid}-path-realized.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(1,3,figsize=(13,3.5))
        group=groups[sid]
        for i,key in enumerate(('path_pairwise','command_pairwise','realized_pairwise')):
            matrix=np.asarray(group[key]['rms']);ax[i].imshow(matrix);ax[i].set_title(key)
            ax[i].set_xticks(range(7),group['candidate_order'],rotation=75,fontsize=6)
        fig.tight_layout();fig.savefig(folder/f'{sid}-P-C-R-distances.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        names=list(commands)
        for name,series in commands.items():ax[0].plot(np.arange(len(series))*.05,series[:,0],label=name)
        ax[0].set_title('follower linear command');ax[0].legend(fontsize=6)
        group=groups[sid]
        ax[1].bar(['duplicate XY','candidate XY max'],[group['duplicate_xy_rms_m'],
            max(group['direct_to_nondirect_realized_rms'])]);ax[1].set_title('duplicate vs candidate effect')
        fig.tight_layout();fig.savefig(folder/f'{sid}-command-effect.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    ids=list(groups)
    for j,metric in enumerate(('delta_v','progress_m','cross_track_p95_m')):
        for i,sid in enumerate(ids):
            vals=groups[sid]['unique_candidate_outcomes']
            ax[j].bar(np.arange(len(vals))+.38*i,[x.get(metric) or 0 for x in vals.values()],width=.36,label=sid)
        ax[j].set_title(metric);ax[j].set_xticks(range(7),list(groups[ids[0]]['unique_candidate_outcomes']),rotation=75,fontsize=6)
    ax[0].legend(fontsize=6);fig.tight_layout();fig.savefig(folder/'outcome-spread-and-ranks.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5))
    x=np.arange(len(ids));ax.bar(x-.25,[7]*len(ids),width=.25,label='Path modes')
    ax.bar(x,[groups[s]['command_mode_count'] for s in ids],width=.25,label='command modes')
    ax.bar(x+.25,[groups[s]['realized_mode_count'] for s in ids],width=.25,label='realized modes')
    ax.set_xticks(x,ids);ax.set_ylim(0,8);ax.legend();fig.tight_layout();fig.savefig(folder/'P-C-R-mode-transition.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3))
    value=np.asarray([[groups[s][k] for k in ('P_to_C_pass','C_to_R_pass','R_to_Y_difference')] for s in ids],dtype=int)
    ax.imshow(value,vmin=0,vmax=1,cmap='RdYlGn');ax.set_yticks(range(len(ids)),ids)
    ax.set_xticks(range(3),['P→C','C→R','R→Y']);fig.tight_layout();fig.savefig(folder/'state-gate-summary.png',dpi=140);plt.close(fig)


def final_analysis(root):
    root=Path(root);manifest=json.loads((root/'continuation-manifest.json').read_text())
    gates=yaml.safe_load((root/'e1-gates.yaml').read_text())
    groups={};plot_data={}
    for sid in ('g0e-train-000','g0e-train-001'):
        row=[r for r in manifest['pending_rows'] if r['state_group_id']==sid]
        duplicate=json.loads((root/f'duplicate-gate-{sid}.json').read_text())
        if duplicate['status']!='pass':raise ValueError('duplicate gate not passed')
        group,paths,commands,realized=group_analysis(row,root,gates,duplicate)
        group['duplicate_xy_rms_m']=duplicate['xy_rms_m']
        groups[sid]=group;plot_data[sid]=(paths,commands,realized)
        atomic(root/f'P-C-{sid}.json',{'path_pairwise':group['path_pairwise'],
            'command_pairwise':group['command_pairwise'],'command_modes':group['command_mode_count'],
            'pass':group['P_to_C_pass']})
        atomic(root/f'C-R-{sid}.json',{'command_pairwise':group['command_pairwise'],
            'realized_pairwise':group['realized_pairwise'],'realized_modes':group['realized_mode_count'],
            'effect_to_duplicate_ratio':group['candidate_to_duplicate_xy_effect_ratio'],
            'pass':group['C_to_R_pass']})
        atomic(root/f'R-Y-{sid}.json',{'outcome_spread':group['outcome_spread'],
            'unique_candidate_outcomes':group['unique_candidate_outcomes'],
            'difference':group['R_to_Y_difference']})
    all_good=all(g['P_to_C_pass'] and g['C_to_R_pass'] and not g['all_same_tracking_failure'] for g in groups.values())
    if all_good and any(g['R_to_Y_difference'] for g in groups.values()):
        status='candidate_generator_e1_supported'
    elif all(not g['P_to_C_pass'] for g in groups.values()) or all(not g['C_to_R_pass'] for g in groups.values()) or \
            all(g['all_same_tracking_failure'] for g in groups.values()):
        status='candidate_generator_e1_fail'
    else:status='inconclusive'
    ranking={}
    for sid,g in groups.items():
        values=g['unique_candidate_outcomes']
        available={name:x['delta_v'] for name,x in values.items() if x.get('delta_v') is not None}
        ranking[sid]={'oracle_denominator':7,'direct_repeat_aggregation':'median',
            'delta_v_order':sorted(available,key=available.get,reverse=True),
            'delta_v_oracle':max(available.values()) if available else None,
            'direct_delta_v':values['direct_reference']['delta_v'],
            'direct_regret':(max(available.values())-values['direct_reference']['delta_v'])
                if available and values['direct_reference']['delta_v'] is not None else None}
    all_records=[r for g in groups.values() for r in g['raw_records']]
    atomic(root/'candidate-outcome-table.json',{'records':all_records,'unique_candidates_per_group':7,
        'state_group_is_statistical_unit':True})
    atomic(root/'oracle-ranking-regret.json',ranking)
    atomic(root/'final-gate-decision.json',{'status':status,'group_gates':{
        sid:{k:g[k] for k in ('P_to_C_pass','C_to_R_pass','R_to_Y_difference','all_same_tracking_failure',
            'command_mode_count','realized_mode_count')} for sid,g in groups.items()},
        'processes_completed':18,'StateGroup_count':2,'terrain_generalization_claim':False,
        'Evaluator_performance_claim':False,'real_robot_claim':False,
        'scaled_collection_authorized':False,'learning_authorized':False,
        'frozen_gates_sha256':sha(root/'e1-gates.yaml')})
    figures(root,groups,plot_data)
    lines=['# E1 Bezier candidate feasibility pilot','',f'**Decision:** {status}.',
        '18 fresh live processes completed after reusing two SHA-verified G0-E anchors. '
        'Each StateGroup passed three direct-repeat duplicate runs before its six lateral candidates.',
        '', '| StateGroup | Path modes | command modes | realized modes | duplicate XY RMS m | max candidate/duplicate ratio | P→C | C→R | R→Y |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for sid,g in groups.items():
        lines.append(f"| {sid} | 7 | {g['command_mode_count']} | {g['realized_mode_count']} | "
            f"{g['duplicate_xy_rms_m']:.5f} | {g['candidate_to_duplicate_xy_effect_ratio']:.2f} | "
            f"{g['P_to_C_pass']} | {g['C_to_R_pass']} | {g['R_to_Y_difference']} |")
    lines += ['', 'The direct repeats count as one candidate for oracle ranking. Candidate pairs are descriptive, not independent statistical samples. '
        'The 2-second eight-waypoint Path is observed through a 6-second 120-step response, including post-endpoint behavior.',
        'Timeout alone was not classified as locomotion failure. Infrastructure failures were excluded from outcomes.',
        'Two StateGroups do not establish terrain generalization, Evaluator performance or real-robot performance.',
        '', 'No learning, NoMaD/Evaluator inference, scaled collection or hardware trial was run. '
        'Further stages require separate approval.']
    (root/'final-report.md').write_text('\n'.join(lines)+'\n')
    entries=[]
    for p in root.rglob('*'):
        if p.is_file() and p.name not in ('artifact-sha-ledger.json','status.json','timing-heartbeat.json'):
            entries.append({'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size})
    atomic(root/'artifact-sha-ledger.json',{'entries':entries,'excluded_mutable':['status.json','timing-heartbeat.json']})
    return status
