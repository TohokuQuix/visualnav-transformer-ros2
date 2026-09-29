"""Train-only feasible screen, mode recall, and oracle regret from frozen 91 rollouts."""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np
import yaml
from visualnav_transformer.eat_nav.feasible_transfer import (
    cluster_labels,command_cluster_labels,feasible_mode_recall,map_xy_to_base,
    integrate_commands,oracle_regret,
)
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest
from scripts.eat_nav_seven_state_analyze import path_tracking

ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'feasible-transfer'
CONFIG=Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/feasible-transfer/analysis-config.yaml')

def read(path):return json.loads(path.read_text())

def quat_yaw(q):
    w,x,y,z=map(float,q)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def rms_xy(a,b):
    x=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))

def command_distance(a,b):
    x=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return {'linear_rms_mps':float(np.sqrt(np.mean(x[:,0]**2))),
            'angular_rms_radps':float(np.sqrt(np.mean(x[:,2]**2)))}

def medoid_ids(rows,commands,labels):
    selected=[]
    for label in sorted(set(labels)):
        indices=[i for i,v in enumerate(labels) if v==label]
        distances={i:sum(command_distance(commands[i],commands[j])['linear_rms_mps']+
                         command_distance(commands[i],commands[j])['angular_rms_radps']
                         for j in indices) for i in indices}
        chosen=min(indices,key=lambda i:(distances[i],rows[i]['candidate_id']))
        selected.append(rows[chosen]['candidate_id'])
    return selected

def run():
    start=time.time();stage0=read(OUT/'stage-0-complete.json')
    if stage0['status']!='complete' or digest(ROOT/'analysis/oracle-outcomes.json')!=stage0['input_sha256']['outcomes']:
        raise ValueError('Stage 0 source changed')
    spec=yaml.safe_load(CONFIG.read_text())
    rows=read(ROOT/'analysis/oracle-outcomes.json')['rows']
    selection=read(ROOT/'stage0/frozen/pilot-selection.json')['selected']
    by_state=defaultdict(list)
    for row in rows:by_state[row['state_id']].append(row)
    candidate_details=[];coverage=[];regret=[];bank_winners=[];k32=[]
    for state in selection:
        sid=state['state_id'];cases=by_state[sid]
        if len(cases)!=13:raise ValueError(f'incomplete state: {sid}')
        index=int(sid.rsplit('-',1)[1]);folder=ROOT/'pilot-inputs'/f'state-{index:03d}'
        offline=read(ROOT/'stage1'/f'{sid}.json');follower=read(ROOT/'stage1'/f'{sid}-follower.json')
        pilot=[];realized40=[];command40=[];planned8=[]
        for row in cases:
            trace_path=ROOT/'pilot/runs'/row['run_id']/'live-0.trace.json.gz'
            with gzip.open(trace_path,'rt') as stream:trace=json.load(stream)
            if len(trace)!=40 or len(row['root_xy'])!=40 or len(row['command_series'])!=40:
                raise ValueError(f'horizon mismatch: {row["run_id"]}')
            anchor=trace[0]['root_state'];yaw=quat_yaw(anchor[3:7]);origin=anchor[:2]
            real=map_xy_to_base(row['root_xy'],anchor_xy=origin,yaw_rad=yaw)
            commands=np.asarray(row['command_series'],dtype=float)
            command_path=integrate_commands(commands,dt_sec=.05)
            if not np.isfinite(real).all() or not np.isfinite(command_path).all():
                raise ValueError(f'nonfinite trajectory: {row["run_id"]}')
            path=np.asarray(row['path_points'],dtype=float)
            map_path=read(folder/f"{row['candidate_id']}-map-path.json")
            track=path_tracking(trace,map_path)
            noncat=bool(not row['terminal'] and not row['stack'] and not row['tilt_over_85deg']
                         and np.isfinite(real).all())
            progress=bool(real[-1,0]>=spec['feasibility']['progress_min_m'] or
                          row['progress_m']>=spec['feasibility']['progress_min_m'])
            trackable=bool(track<=spec['feasibility']['path_tracking_error_max_m'])
            constraint=bool(row['proposal_valid'])
            conditions={'short_horizon_noncatastrophic':noncat,'progress_producing':progress,
                        'trackable':trackable,'constraint_valid':constraint,
                        'joint_observed_screen':bool(noncat and progress and trackable and constraint)}
            detail={'state_id':sid,'terrain':state['terrain_type'],'candidate_id':row['candidate_id'],
                'generator':row['kind'],'seed':row['seed'],'run_id':row['run_id'],
                'condition':conditions,'path_tracking_error_m':track,
                'forward_displacement_m':float(real[-1,0]),'progress_m':row['progress_m'],
                'delta_v':row['delta_v'],'drawdown':row['drawdown'],'recovery':row['recovery'],
                'tilt_rad':row['body_tilt_max_rad'],'contact_max_n':row['contact_max_n'],
                'ood':row['proposal_ood'],'planned_path':path.tolist(),
                'command_derived_path':command_path[[4,9,14,19,24,29,34,39]].tolist(),
                'realized_path':real[[4,9,14,19,24,29,34,39]].tolist(),
                'source_trace_sha256':row['source_sha256']['trace_sha256']}
            pilot.append(detail);candidate_details.append(detail)
            planned8.append(path);command40.append(commands);realized40.append(real)
        ids=[r['candidate_id'] for r in pilot]
        nomad_order=[f'nomad-seed-{seed}' for seed in range(1001,1009)]
        if not set(nomad_order).issubset(ids):raise ValueError(f'NoMaD candidate ID missing: {sid}')
        path_labels=cluster_labels(planned8,threshold=spec['cluster']['planned_path_rms_m'])
        command_labels=command_cluster_labels(command40,
            linear_threshold=spec['cluster']['command_linear_rms_mps'],
            angular_threshold=spec['cluster']['command_angular_rms_radps'])
        realized_labels=cluster_labels(realized40,threshold=spec['cluster']['realized_xy_rms_m'])
        for detail,pl,cl,rl in zip(pilot,path_labels,command_labels,realized_labels):
            detail['mode_id']={'planned':pl,'command':cl,'realized':rl}
        precisions={}
        for generator in ('nomad','bank','all'):
            selected=[r for r in pilot if generator=='all' or r['generator']==generator]
            precisions[generator]={'candidate_count':len(selected),'condition':{
                name:{'count':sum(r['condition'][name] for r in selected),
                      'precision':sum(r['condition'][name] for r in selected)/len(selected)}
                for name in pilot[0]['condition']}}
        recall={}
        for condition in pilot[0]['condition']:
            mask=[r['condition'][condition] for r in pilot]
            recall[condition]={space:feasible_mode_recall(ids,labels,mask,nomad_order,ks=spec['nomad_k'])
                for space,labels in [('command',command_labels),('realized',realized_labels)]}
        representatives=medoid_ids(pilot,command40,command_labels)
        by_id={r['candidate_id']:r for r in pilot}
        regret_row={'state_id':sid,'terrain':state['terrain_type'],'candidate_ids':ids,
                    'command_mode_representative_ids':representatives,'regret':{}}
        for label,selected in [('all',pilot),('constraint_valid',[r for r in pilot if r['condition']['constraint_valid']]),
                               ('progress_producing',[r for r in pilot if r['condition']['progress_producing']]),
                               ('command_mode_representatives',[by_id[cid] for cid in representatives])]:
            selected_ids={r['candidate_id'] for r in selected}
            regret_row['regret'][label]={str(k):oracle_regret(selected,[cid for cid in nomad_order[:k] if cid in selected_ids])
                                         for k in spec['nomad_k']}
        regret.append(regret_row)
        bank=max((r for r in pilot if r['generator']=='bank'),key=lambda r:r['delta_v'])
        nomad=max((r for r in pilot if r['generator']=='nomad'),key=lambda r:r['delta_v'])
        if bank['delta_v']>nomad['delta_v']+spec['oracle']['tie_epsilon']:
            b_index=ids.index(bank['candidate_id'])
            nearest=min((r for r in pilot if r['generator']=='nomad'),
                        key=lambda r:rms_xy(bank['planned_path'],r['planned_path']))
            n_index=ids.index(nearest['candidate_id'])
            bank_winners.append({'state_id':sid,'terrain':state['terrain_type'],
                'bank_candidate_id':bank['candidate_id'],'nearest_nomad_candidate_id':nearest['candidate_id'],
                'path_rms_m':rms_xy(bank['planned_path'],nearest['planned_path']),
                'command_distance':command_distance(command40[b_index],command40[n_index]),
                'realized_xy_rms_m':rms_xy(realized40[b_index],realized40[n_index]),
                'same_command_mode':command_labels[b_index]==command_labels[n_index],
                'same_realized_mode':realized_labels[b_index]==realized_labels[n_index],
                'bank':{k:bank[k] for k in ('delta_v','drawdown','recovery','progress_m','tilt_rad','contact_max_n','ood','condition')},
                'nearest_nomad':{k:nearest[k] for k in ('delta_v','drawdown','recovery','progress_m','tilt_rad','contact_max_n','ood','condition')}})
        coverage.append({'state_id':sid,'terrain':state['terrain_type'],'candidate_ids':ids,
            'mode_id_by_candidate':{cid:{'planned':pl,'command':cl,'realized':rl}
              for cid,pl,cl,rl in zip(ids,path_labels,command_labels,realized_labels)},
            'cluster_count':{'planned':len(set(path_labels)),'command':len(set(command_labels)),
                             'realized':len(set(realized_labels))},'precision':precisions,'recall':recall,
            'path_same_command_multi':len(set(path_labels))==1 and len(set(command_labels))>1,
            'command_same_realized_multi':len(set(command_labels))==1 and len(set(realized_labels))>1,
            'bank_only_command_modes':sorted(set(command_labels[i] for i,r in enumerate(pilot) if r['generator']=='bank')-
                                             set(command_labels[i] for i,r in enumerate(pilot) if r['generator']=='nomad')),
            'bank_only_realized_modes':sorted(set(realized_labels[i] for i,r in enumerate(pilot) if r['generator']=='bank')-
                                              set(realized_labels[i] for i,r in enumerate(pilot) if r['generator']=='nomad'))})
        candidates={r['candidate_id']:r for r in offline['candidates']}
        k32_ids=[f'nomad-seed-{seed}' for seed in range(1001,1033)]
        k32_paths=np.asarray([candidates[cid]['points'] for cid in k32_ids],dtype=float)
        if not np.isfinite(k32_paths).all():raise ValueError(f'nonfinite K32 path: {sid}')
        k32_labels=cluster_labels(k32_paths,threshold=spec['cluster']['planned_path_rms_m'])
        bank_path=np.asarray(bank['planned_path'])
        nearest_k8=min((rms_xy(bank_path,candidates[cid]['points']),cid) for cid in k32_ids[:8])
        nearest_k32=min((rms_xy(bank_path,candidates[cid]['points']),cid) for cid in k32_ids)
        follower_by_id={r['candidate_id']:r for r in follower['candidates']}
        initial=np.asarray([follower_by_id[cid]['initial_command'] for cid in k32_ids],dtype=float)
        k32.append({'state_id':sid,'terrain':state['terrain_type'],'candidate_ids':k32_ids,
            'seed_by_candidate':{cid:candidates[cid]['seed'] for cid in k32_ids},
            'path_cluster_count':len(set(k32_labels)),'added_path_modes_beyond_k8':sorted(set(k32_labels)-set(k32_labels[:8])),
            'path_mode_by_candidate':dict(zip(k32_ids,k32_labels)),
            'bank_winner_candidate_id':bank['candidate_id'],'nearest_bank_winner_k8':nearest_k8,
            'nearest_bank_winner_k32':nearest_k32,
            'initial_command_cluster_count':len(set(command_cluster_labels(initial[:,None,:],
                linear_threshold=spec['cluster']['command_linear_rms_mps'],
                angular_threshold=spec['cluster']['command_angular_rms_radps']))),
            'full_moving_command_mode_known':False,'simulator_feasibility_known':False,
            'ood_count':sum(candidates[cid].get('ood_over_train_p95',False) for cid in k32_ids),
            'constraint_failure_count':sum(not candidates[cid]['constraints']['valid'] for cid in k32_ids),
            'path_metrics_by_candidate':{cid:candidates[cid]['path_metrics'] for cid in k32_ids},
            'ood_distance_by_candidate':{cid:candidates[cid]['ood_distance'] for cid in k32_ids}})
    if len(candidate_details)!=91 or len(bank_winners)!=4:raise ValueError('Stage 1 frozen count mismatch')
    atomic_once(OUT/'feasibility-by-candidate.json',{'rows':candidate_details,'candidate_count':len(candidate_details)})
    atomic_once(OUT/'nomad-feasible-coverage.json',{'states':coverage,'unit':'state','state_count':7})
    atomic_once(OUT/'nomad-oracle-regret.json',{'states':regret,'bank_winner_comparisons':bank_winners})
    atomic_once(OUT/'k32-offline-analysis.json',{'states':k32,'rollout_count':0})
    atomic_once(OUT/'stage-1a-complete.json',{'stage':'1a','status':'complete','input_sha256':{
        'stage0':digest(OUT/'stage-0-complete.json'),'config':digest(CONFIG)},'output_sha256':{
        name:digest(OUT/name) for name in ('feasibility-by-candidate.json','nomad-feasible-coverage.json',
                    'nomad-oracle-regret.json','k32-offline-analysis.json')},'elapsed_sec':time.time()-start})
    print('FEASIBLE_TRANSFER_STAGE1A_COMPLETE',len(candidate_details),'candidates',round(time.time()-start,1),'sec',flush=True)

if __name__=='__main__':run()
