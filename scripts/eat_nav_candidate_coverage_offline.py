"""Outcome-blind NoMaD K32 plus fixed-bank candidate and geometry freeze."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.candidate_coverage_offline import (
    candidate_constraints, path_distribution, path_metrics, train_constraint_limits,
)
from visualnav_transformer.eat_nav.candidate_coverage_sampling import prepare_state_candidates
from visualnav_transformer.eat_nav.candidate_coverage_selection import validate_frozen_selection
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from scripts.eat_nav_candidate_coverage_select import EXPECTED_SHA, atomic_once, digest


def evaluate(records, dem, config, limits, train_norm, model, ood_limit):
    by_id={}
    finite=[]
    for row in records['hybrid_k32']:
        item=dict(row)
        if not item['finite']:
            item['constraints']={'valid':False,'reasons':['nonfinite']}
            item['path_metrics']=None
            item['ood_distance']=None
        else:
            path=np.asarray(item['points'],dtype=np.float32)
            item['path_metrics']=path_metrics(path)
            item['constraints']=candidate_constraints(path,dem,resolution_m=config.dem_resolution_m,limits=limits)
            delta=np.diff(np.vstack((np.zeros((1,2),dtype=np.float32),path)),axis=0)
            norm=(delta-model.action_mean.detach().cpu().numpy())/model.action_std.detach().cpu().numpy()
            finite.append((item['candidate_id'],norm.reshape(-1)))
            item['ros_base_path']=[[0.,0.]]+item['points']
            item['ood_distance']=None
        by_id[item['candidate_id']]=item
    if finite:
        tensor=torch.as_tensor(np.stack([value for _,value in finite]),device=train_norm.device,dtype=train_norm.dtype)
        distance=(torch.cdist(tensor,train_norm).min(dim=1).values/np.sqrt(tensor.shape[1])).detach().cpu().numpy()
        for (candidate_id,_),value in zip(finite,distance):
            by_id[candidate_id]['ood_distance']=float(value)
            by_id[candidate_id]['ood_over_train_p95']=bool(value>ood_limit)
    sets={}
    for name,rows in records.items():
        ids=[row['candidate_id'] for row in rows]
        valid=[np.asarray(by_id[cid]['points'],dtype=float) for cid in ids if by_id[cid]['finite']]
        sets[name]={'candidate_ids':ids,'candidate_count':len(ids),
                    'nonfinite_count':len(ids)-len(valid),
                    'constraint_failure_count':sum(not by_id[cid]['constraints']['valid'] for cid in ids),
                    'ood_count':sum(bool(by_id[cid].get('ood_over_train_p95',False)) for cid in ids),
                    'geometry':path_distribution(np.stack(valid),cluster_rms_m=.05) if len(valid)>1 else None}
    return {'candidates':list(by_id.values()),'sets':sets}


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','reference-config','checkpoint','selection','pilot-config','maneuver-thresholds','ood-thresholds','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();started=time.monotonic()
    sources={'manifest':a.manifest,'reference_config':a.reference_config,'reference_checkpoint':a.checkpoint}
    sha={name:digest(path) for name,path in sources.items()}
    if sha!=EXPECTED_SHA:raise ValueError('source SHA mismatch')
    selection=json.loads(a.selection.read_text())
    validate_frozen_selection(selection,expected_source_sha=sha)
    if digest(a.selection)!='1bb7865e81d2f0f52fff550037bb2e07b560b73f5de553b6183203bdaa969ddc':
        raise ValueError('frozen selection changed')
    pilot=yaml.safe_load(a.pilot_config.read_text())
    if pilot['approved_wall_clock_sec']!=14400 or pilot['max_full_prefix_steps']!=40:
        raise ValueError('pilot budget/cap changed')
    if [r['terrain_type'] for r in selection['selected']]!=pilot['terrain_order']:
        raise ValueError('terrain order changed')
    cfg=load_eat_config(a.reference_config)
    if cfg.quality_enabled or cfg.len_traj_pred!=8 or cfg.context_size!=3:
        raise ValueError('reference NoMaD configuration changed')
    view=CompleteOnlyView(EATDataset(a.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(view)!=49598:raise ValueError('complete train count changed')
    actions=np.stack([view[i]['action'] for i in range(len(view))]).astype(np.float32)
    limits=train_constraint_limits(actions)
    limits['train_actions_sha256']=hashlib.sha256(actions.tobytes()).hexdigest()
    limits['selection_sha256']=digest(a.selection)
    atomic_once(a.output/'train-constraint-limits.json',limits)
    bank=yaml.safe_load(a.maneuver_thresholds.read_text())['lateral_scale_m'] if a.maneuver_thresholds.suffix=='.yaml' else json.loads(a.maneuver_thresholds.read_text())['lateral_scale_m']
    prior=json.loads(a.ood_thresholds.read_text())
    if prior['source_split']!='complete_only_train':raise ValueError('OOD threshold not train-only')
    ood_limit=float(prior['train_ood_nearest_l2_per_dimension_p95'])
    model,stats,_=load_checkpoint(a.checkpoint,cfg)
    model=model.to('cuda').eval()
    mean=model.action_mean.detach().cpu().numpy();std=model.action_std.detach().cpu().numpy()
    if not np.allclose(mean,prior['train_action_mean'],atol=1e-6) or not np.allclose(std,prior['train_action_std'],atol=1e-6):
        raise ValueError('normalization changed')
    train_norm=torch.as_tensor(((actions-mean)/std).reshape(len(actions),-1),device='cuda')
    provenance={'input_sha256':{**sha,'selection':digest(a.selection),'pilot_config':digest(a.pilot_config),
        'maneuver_thresholds':digest(a.maneuver_thresholds),'ood_thresholds':digest(a.ood_thresholds)},
        'train_constraint_limits_sha256':digest(a.output/'train-constraint-limits.json'),
        'heldout_used':False,'quality_conditioning_used':False,'candidate_outcomes_used':False}
    atomic_once(a.output/'source-provenance.json',provenance)
    with torch.inference_mode():
        for selected in selection['selected']:
            index=selected['view_index'];w=view.dataset.windows[view.indices[index]]
            if list(w['episode_key'])!=selected['episode_key'] or int(w['step'])!=selected['step']:
                raise ValueError('selected window identity changed')
            dem=np.asarray(w['dem'],dtype=np.float32)
            if hashlib.sha256(dem.tobytes()).hexdigest()!=selected['dem_sha256']:
                raise ValueError('selected DEM changed')
            goal=np.asarray(w['goal'],dtype=np.float32)
            if not np.allclose(goal,selected['goal'],atol=1e-7):raise ValueError('selected goal changed')
            target=a.output/f"{selected['state_id']}.json"
            if target.exists():
                previous=json.loads(target.read_text())
                if previous['source_sha256']!=provenance['input_sha256'] or previous['state']!=selected:
                    raise ValueError('invalid candidate resume state')
                print('OFFLINE_RESUME_SKIP',selected['state_id'],flush=True)
                continue
            dem_tensor=torch.as_tensor(dem[None],device='cuda')
            goal_tensor=torch.as_tensor(goal[None],device='cuda')
            records=prepare_state_candidates(model,dem_tensor,goal_tensor,pilot['nomad_seeds_k32'],
                slight_scale=float(bank['slight']),strong_scale=float(bank['strong']))
            measured=evaluate(records,dem,cfg,limits,train_norm,model,ood_limit)
            row={'state':selected,'source_sha256':provenance['input_sha256'],
                 'ood_limit':ood_limit,'train_constraint_limits_sha256':provenance['train_constraint_limits_sha256'],
                 **measured,'elapsed_sec_since_start':time.monotonic()-started}
            atomic_once(target,row)
            print('OFFLINE_STATE',selected['state_id'],
                  'K8_clusters',row['sets']['nomad_k8']['geometry']['path_cluster_count'],
                  'sec',round(time.monotonic()-started,1),flush=True)
    atomic_once(a.output/'status.json',{'stage':'offline_candidates_complete','state_count':len(selection['selected']),
                                      'pilot_rollout_started':False})

if __name__=='__main__':main()
