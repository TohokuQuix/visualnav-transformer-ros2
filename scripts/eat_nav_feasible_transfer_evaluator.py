"""Frozen GT transfer: planned, command-derived, realized and simple baselines."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
import torch
import yaml
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.feasible_transfer import to_increments
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator,SEEDS
from eat_nav_forward_evaluator import HEADS,load_split,sha

ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'feasible-transfer'
BASE=ROOT/'baseline-worktree-8e8f4fd'


def read(path):return json.loads(path.read_text())

def write_once(path,value):
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise FileExistsError(f'prior result differs: {path}')
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)


def rank_metrics(prediction,truth,ids,*,epsilon):
    p=np.asarray(prediction,dtype=float);y=np.asarray(truth,dtype=float)
    if len(p)!=len(y) or len(ids)!=len(y):raise ValueError('ranking length mismatch')
    order=np.argsort(-p,kind='stable')
    best=float(np.max(y));winners={i for i,v in enumerate(y) if v>=best-epsilon}
    pairs=[(i,j) for i,j in combinations(range(len(y)),2) if abs(y[i]-y[j])>epsilon]
    correct=sum((p[i]-p[j])*(y[i]-y[j])>0 for i,j in pairs)
    rho=spearmanr(y,p).statistic if len(p)>2 and np.ptp(y)>1e-8 and np.ptp(p)>1e-8 else None
    return {'top1_candidate_id':ids[int(order[0])],'top1_recall':bool(order[0] in winners),
            'top3_recall':bool(any(i in winners for i in order[:3])),
            'pairwise_support':len(pairs),'pairwise_accuracy':correct/len(pairs) if pairs else None,
            'spearman':float(rho) if rho is not None and np.isfinite(rho) else None,
            'predicted_spread':float(np.ptp(p)),'true_delta_v_spread':float(np.ptp(y)),
            'calibration_rmse':float(np.sqrt(np.mean((p-y)**2))),
            'oracle_regret':best-float(y[order[0]]),'oracle_best_candidate_ids':[ids[i] for i in sorted(winners)]}


def pairwise_strata(rows,p):
    result={}
    definitions={
        'all':lambda a,b:True,
        'nomad_nomad':lambda a,b:a['generator']==b['generator']=='nomad',
        'bank_bank':lambda a,b:a['generator']==b['generator']=='bank',
        'constraint_valid_pair':lambda a,b:a['condition']['constraint_valid'] and b['condition']['constraint_valid'],
        'constraint_invalid_involved':lambda a,b:not (a['condition']['constraint_valid'] and b['condition']['constraint_valid']),
        'same_realized_mode':lambda a,b:a['mode_id']['realized']==b['mode_id']['realized'],
        'distinct_realized_mode':lambda a,b:a['mode_id']['realized']!=b['mode_id']['realized'],
    }
    for name,include in definitions.items():
        pairs=[(i,j) for i,j in combinations(range(len(rows)),2)
               if include(rows[i],rows[j]) and abs(rows[i]['delta_v']-rows[j]['delta_v'])>.01]
        correct=sum((p[i]-p[j])*(rows[i]['delta_v']-rows[j]['delta_v'])>0 for i,j in pairs)
        result[name]={'pair_count':len(pairs),'accuracy':correct/len(pairs) if pairs else None}
    return result


def run():
    started=time.time();phase=read(OUT/'phase-b-reproduction.json')
    if phase['status']!='pass':raise ValueError('Phase B reproduction gate failed')
    spec=yaml.safe_load((BASE/'docs/quality-identifiability-gate/candidate-coverage/feasible-transfer/analysis-config.yaml').read_text())
    selection=read(ROOT/'stage0/frozen/pilot-selection.json')['selected']
    details=read(OUT/'feasibility-by-candidate.json')['rows']
    source=read(OUT/'source-provenance.json')
    if len(details)!=91 or source['candidate_count']!=91:raise ValueError('91 candidate source mismatch')
    view=CompleteOnlyView(EATDataset(Path('/tmp/eatnav-extended/output/manifest.json'),'train',
        load_eat_config(BASE/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml'),split_seed=123),8)
    if len(view)!=49598:raise ValueError('complete train view changed')
    frozen=FrozenEvaluator('cuda' if torch.cuda.is_available() else 'cpu')
    if frozen.checkpoint_sha256!={str(seed):source['container'][f'evaluator_GT_{seed}']['sha256'] for seed in SEEDS}:
        raise ValueError('Evaluator checkpoint changed')
    if frozen.scaler_sha256!=source['container']['evaluator_train_scalers']['sha256']:
        raise ValueError('Evaluator scaler changed')
    phase_spec=yaml.safe_load(Path('/tmp/eatnav-train/forward-evaluator-config.yaml').read_text())
    heldout=load_split('test',phase_spec)
    heldout_endpoint=np.cumsum(heldout['action'],axis=1)[:,-1]
    heldout_goal_distance=np.linalg.norm(heldout_endpoint-heldout['goal'],axis=1)
    heldout_heading=np.arctan2(heldout['action'][:,-1,1],heldout['action'][:,-1,0])
    heldout_goal_heading=np.arctan2(heldout['goal'][:,1],heldout['goal'][:,0])
    heldout_heading_diff=np.abs(np.angle(np.exp(1j*(heldout_heading-heldout_goal_heading))))
    by_state=defaultdict(list)
    for r in details:by_state[r['state_id']].append(r)
    states=[];goal_endpoint=[];per_candidate=[];baselines=[]
    for state in selection:
        sid=state['state_id'];rows=by_state[sid]
        if len(rows)!=13:raise ValueError(f'candidate count mismatch: {sid}')
        w=view.dataset.windows[view.indices[state['view_index']]]
        dem=np.asarray(w['dem'],dtype=np.float32);goal=np.asarray(w['goal'],dtype=np.float32)
        if hashlib.sha256(dem.tobytes()).hexdigest()!=state['dem_sha256'] or not np.allclose(goal,state['goal'],atol=1e-7):
            raise ValueError(f'input DEM/goal changed: {sid}')
        ids=[r['candidate_id'] for r in rows]
        paths={kind:np.stack([np.asarray(r[key],dtype=np.float32) for r in rows])
            for kind,key in [('P','planned_path'),('C','command_derived_path'),('R','realized_path')]}
        for kind,p in paths.items():
            if p.shape!=(13,8,2) or not np.isfinite(p).all():raise ValueError(f'{kind} shape/nonfinite')
            if not np.allclose(np.cumsum(np.stack([to_increments(a) for a in p]),axis=1),p,atol=1e-6):
                raise ValueError(f'{kind} increment/cumulative mismatch')
        predictions={kind:frozen.predict(dem,goal,p) for kind,p in paths.items()}
        y=np.asarray([r['delta_v'] for r in rows],dtype=float)
        goal_replaced=np.stack([frozen.predict(dem,paths['P'][i,-1],paths['P'][i:i+1])[:,0,0]
                                for i in range(13)],axis=1)
        result={'state_id':sid,'terrain':state['terrain_type'],'candidate_ids':ids,'goal':goal.tolist(),
                'trajectory_interval_sec':.25,'prediction_by_variant_seed_candidate_head':{
                    kind:predictions[kind].tolist() for kind in ('P','C','R')},
                'goal_replaced_prediction_by_seed_candidate_delta_v':goal_replaced.tolist(),
                'gt':{},'pairwise_strata':{},'goal_replacement_score_spread':float(np.ptp(goal_replaced.mean(0))),
                'goal_fixed_score_spread':float(np.ptp(predictions['P'][:,:,0].mean(0)))}
        for kind in ('P','C','R'):
            mean=predictions[kind].mean(0)[:,0]
            result['gt'][kind]=rank_metrics(mean,y,ids,epsilon=spec['oracle']['tie_epsilon'])
            result['gt'][kind]['ensemble_disagreement_mean']=float(np.std(predictions[kind][:,:,0],axis=0).mean())
            result['pairwise_strata'][kind]=pairwise_strata(rows,mean)
        fixed_scores={
            'candidate_index_or_seed_order':np.arange(13,0,-1,dtype=float),
            'goal_endpoint_distance':-np.linalg.norm(paths['P'][:,-1]-goal,axis=1),
            'path_length':-np.linalg.norm(np.stack([to_increments(a) for a in paths['P']]),axis=2).sum(1),
            'lateral_abs':-np.abs(paths['P'][:,-1,1]),
            'ood_distance':-np.asarray([r['ood']['ood_distance'] for r in rows]),
        }
        offline_by_id={c['candidate_id']:c for c in read(ROOT/'stage1'/f'{sid}.json')['candidates']}
        fixed_scores['curvature']=-np.asarray([offline_by_id[cid]['path_metrics']['curvature_rad'] for cid in ids])
        # Initial follower command is fixed before each rollout and available for all candidates.
        follower_by_id={c['candidate_id']:c for c in read(ROOT/'stage1'/f'{sid}-follower.json')['candidates']}
        fixed_scores['initial_linear_command']=np.asarray([follower_by_id[cid]['initial_command'][0] for cid in ids])
        fixed_scores['initial_abs_angular_command']=-np.abs([follower_by_id[cid]['initial_command'][2] for cid in ids])
        baseline_result={name:rank_metrics(scores,y,ids,epsilon=spec['oracle']['tie_epsilon'])
                         for name,scores in fixed_scores.items()}
        nominal_index=ids.index('bank-nominal');best=float(np.max(y))
        baseline_result['nominal_fixed']={'top1_candidate_id':'bank-nominal',
            'top1_recall':bool(y[nominal_index]>=best-spec['oracle']['tie_epsilon']),
            'oracle_regret':best-float(y[nominal_index])}
        baselines.append({'state_id':sid,'terrain':state['terrain_type'],'metrics':baseline_result})
        for i,row in enumerate(rows):
            endpoint=paths['P'][i,-1]
            tangent=paths['P'][i,-1]-paths['P'][i,-2]
            heading=math.atan2(float(tangent[1]),float(tangent[0]))
            goal_heading=math.atan2(float(goal[1]),float(goal[0]))
            goal_endpoint.append({'state_id':sid,'terrain':state['terrain_type'],
                'candidate_id':ids[i],'generator':row['generator'],'seed':row['seed'],
                'goal_endpoint_distance_m':float(np.linalg.norm(endpoint-goal)),
                'goal_endpoint_heading_delta_rad':abs(math.atan2(math.sin(heading-goal_heading),math.cos(heading-goal_heading))),
                'goal_from_candidate_endpoint_exact':bool(np.linalg.norm(endpoint-goal)<1e-5)})
            per_candidate.append({'state_id':sid,'terrain':state['terrain_type'],'candidate_id':ids[i],
                'generator':row['generator'],'seed':row['seed'],'delta_v':row['delta_v'],
                'constraint_valid':row['condition']['constraint_valid'],
                'mode_id':row['mode_id'],'P':predictions['P'][:,i].tolist(),
                'C':predictions['C'][:,i].tolist(),'R':predictions['R'][:,i].tolist(),
                'P_goal_replaced_delta_v':goal_replaced[:,i].tolist()})
        states.append(result)
    audit_rows=[
        ('trajectory representation','raw base-frame XY increments, 8 points','P/C/R cumulative base waypoints converted by first difference','match after conversion'),
        ('waypoint order','temporal future 1..8','temporal future 1..8','match'),
        ('trajectory frame','robot-local base','P base; C integrated base; R map XY rotated into anchor base','match'),
        ('x/y axis and units','x forward, y left, meters','x forward, y left, meters','match'),
        ('trajectory normalization','no action normalization in GT tensors','raw increments, no action normalization','match'),
        ('trajectory clipping','none at GT input','none','match'),
        ('trajectory interval','0.25 s','P 0.25 s; C/R sampled every 5 of 0.05 s control steps','match'),
        ('DEM channel and mask','4 frames × [height,valid mask]','same EATDataset tensor, selected DEM SHA verified','match'),
        ('height normalization','robot-relative scalar offset','same EATDataset tensor','match'),
        ('context frame order','oldest to current','same selected EATDataset tensor','match'),
        ('relative goal','robot-local meters, clipped at 0.8 m','same selected EATDataset goal, SHA/values checked','match'),
        ('Evaluator output heads',list(HEADS),list(HEADS),'match'),
        ('Delta V sign','V_end minus V_start; larger better','V_end minus V_start; larger better','match'),
        ('score direction','descending for best','descending','match'),
        ('target standardization','train-only z-score','frozen train-only inverse scaler SHA verified','match'),
        ('gamma','0.99 per 0.05 s control step','0.99 per 0.05 s control step','match'),
        ('horizon','40 control steps / 2.0 s','40 control steps / 2.0 s','match'),
        ('terminal bootstrap','true terminal zero; timeout PPO preterminal; complete labels endpoint','pilot has no terminal/timeout; endpoint Critic','compatible for observed 91'),
        ('checkpoint and ensemble',[11,29,47],'same three SHA-verified GT checkpoints, arithmetic mean','match'),
        ('goal endpoint coupling','logged trajectory endpoint, clipped at 0.8 m','one fixed logged goal with 13 alternative candidates','distribution shift to evaluate'),
    ]
    audit={'status':'pass_for_input_and_label_compatibility_with_known_goal_endpoint_shift',
        'phase_b_reproduction_sha256':sha(OUT/'phase-b-reproduction.json'),
        'checks':[{'item':item,'phase_b_training':training,'pilot_scoring':pilot,'judgment':judgment}
                  for item,training,pilot,judgment in audit_rows],
        'selected_state_sample_sha256':{s['state_id']:s['dem_sha256'] for s in selection},
        'heldout_used_for_model_selection':False,'pilot_candidate_count':91}
    write_once(OUT/'evaluator-input-audit.json',audit)
    comparison={'states':states,'candidate_rows':per_candidate,
        'privileged_R_online_selection_allowed':False,'state_count':7,'candidate_count':91,
        'goal_endpoint':{'phase_b_heldout':{
            'count':len(heldout_goal_distance),'exact_fraction':float(np.mean(heldout_goal_distance<1e-5)),
            'distance_quantiles_m':np.quantile(heldout_goal_distance,[.1,.5,.9]).tolist(),
            'heading_delta_quantiles_rad':np.quantile(heldout_heading_diff,[.1,.5,.9]).tolist()},
            'pilot_rows':goal_endpoint}}
    write_once(OUT/'planned-command-realized-comparison.json',comparison)
    aggregate={}
    for kind in ('P','C','R'):
        metrics=[s['gt'][kind] for s in states]
        aggregate[kind]={'state_count':7,'top1_recall_count':sum(x['top1_recall'] for x in metrics),
            'top3_recall_count':sum(x['top3_recall'] for x in metrics),
            'state_macro_pairwise_accuracy':float(np.mean([x['pairwise_accuracy'] for x in metrics if x['pairwise_accuracy'] is not None])),
            'state_macro_spearman':float(np.mean([x['spearman'] for x in metrics if x['spearman'] is not None])),
            'state_macro_oracle_regret':float(np.mean([x['oracle_regret'] for x in metrics]))}
    baseline_aggregate={name:{'top1_recall_count':sum(x['metrics'][name]['top1_recall'] for x in baselines),
        'state_macro_oracle_regret':float(np.mean([x['metrics'][name]['oracle_regret'] for x in baselines]))}
        for name in baselines[0]['metrics']}
    result={'stage':'counterfactual_transfer','aggregate':aggregate,'baselines':baselines,
        'baseline_aggregate':baseline_aggregate,'states':states,
        'statistical_unit':'state','state_count':7,'candidate_count':91,
        'no_candidate_level_ci':True,'heldout_used_for_model_selection':False,
        'checkpoint_sha256':frozen.checkpoint_sha256,'normalization_sha256':frozen.scaler_sha256,
        'elapsed_sec':time.time()-started,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
    write_once(OUT/'evaluator-counterfactual-results.json',result)
    write_once(OUT/'stage-2b-complete.json',{'stage':'2b','status':'complete',
        'input_sha256':{'phase_b':sha(OUT/'stage-2a-complete.json'),
                        'feasibility':sha(OUT/'feasibility-by-candidate.json')},
        'output_sha256':{'audit':sha(OUT/'evaluator-input-audit.json'),
                         'comparison':sha(OUT/'planned-command-realized-comparison.json'),
                         'results':sha(OUT/'evaluator-counterfactual-results.json')},
        'unix_sec':time.time()})
    print('FEASIBLE_TRANSFER_EVALUATOR_COMPLETE',aggregate,'sec',round(time.time()-started,1),flush=True)

if __name__=='__main__':run()
