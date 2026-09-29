"""Generate the frozen endpoint-matched five-Path bank from reference NoMaD."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.feasible_set import maneuver_bank
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)


def self_intersects(path):
    p=np.concatenate((np.zeros((1,2)),np.asarray(path,dtype=float)))
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for i in range(len(p)-1):
        for j in range(i+2,len(p)-1):
            if i==0 and j==len(p)-2:continue
            a,b,c,d=p[i],p[i+1],p[j],p[j+1]
            ab=b-a;cd=d-c
            den=cross(ab,cd)
            if abs(den)<1e-10:continue
            t=cross(c-a,cd)/den;u=cross(c-a,ab)/den
            if 1e-6<t<1-1e-6 and 1e-6<u<1-1e-6:return True
    return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--thresholds',type=Path,required=True)
    p.add_argument('--prior-ood-thresholds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    paths_path=a.output/'maneuver-bank.npz'
    report_path=a.output/'maneuver-bank.json'
    if paths_path.exists() or report_path.exists():raise FileExistsError('maneuver bank output already exists')
    spec=yaml.safe_load(a.spec.read_text());frozen=json.loads(a.selection.read_text())
    thresholds=json.loads(a.thresholds.read_text())
    ood=json.loads(a.prior_ood_thresholds.read_text())
    if thresholds['source']!='complete_only_train' or ood['source_split']!='complete_only_train':
        raise ValueError('maneuver scales or OOD threshold are not train-only')
    cfg=load_eat_config(a.reference_config)
    model,stats,_=load_checkpoint(a.reference,cfg)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(device).eval()
    views={split:CompleteOnlyView(EATDataset(a.manifest,split,cfg,split_seed=spec['selection']['split_seed']),cfg.len_traj_pred)
           for split in ('train','val')}
    train_actions=np.stack([views['train'][i]['action'] for i in range(len(views['train']))]).astype(np.float32)
    normalized=torch.from_numpy(((train_actions-stats.mean)/stats.std).reshape(len(train_actions),-1)).to(device)
    names=spec['maneuvers']['names']
    rows=[];all_paths=[]
    footprint=.5*min(cfg.dem_height_cells,cfg.dem_width_cells)*cfg.dem_resolution_m
    with torch.inference_mode():
        for split,ids in [('train',frozen['train']),('val',frozen['validation'])]:
            view=views[split]
            for state_index,state in enumerate(ids):
                w=view.dataset.windows[view.indices[state['view_index']]]
                if list(w['episode_key'])!=state['episode_key'] or int(w['step'])!=state['step'] or not np.array_equal(w['goal'],np.asarray(state['goal'],dtype=np.float32)):
                    raise ValueError('frozen state window mismatch')
                if hashlib.sha256(np.asarray(w['dem']).tobytes()).hexdigest()!=state['dem_sha256']:
                    raise ValueError('frozen DEM hash mismatch')
                dem=torch.from_numpy(w['dem'][None]).to(device)
                goal=torch.from_numpy(w['goal'][None]).to(device)
                nominal=model.sample(dem,goal,None,seed=spec['maneuvers']['nominal_diffusion_seed'])[0].cpu().numpy()
                bank=maneuver_bank(nominal,w['goal'],thresholds['lateral_scale_m']['slight'],thresholds['lateral_scale_m']['strong'])
                paths=np.stack([bank[name] for name in names]).astype(np.float32)
                if not np.allclose(paths[:,-1],w['goal'],atol=1e-7):raise ValueError('maneuver endpoint mismatch')
                delta=np.diff(np.concatenate((np.zeros((len(paths),1,2),dtype=np.float32),paths),axis=1),axis=1)
                query=torch.from_numpy(((delta-stats.mean)/stats.std).reshape(len(paths),-1)).to(device)
                nearest=(torch.cdist(query,normalized).min(dim=1).values/np.sqrt(query.shape[1])).cpu().numpy()
                metrics=[]
                for i,path in enumerate(paths):
                    segments=np.diff(np.concatenate((np.zeros((1,2)),path),axis=0),axis=0)
                    length=float(np.linalg.norm(segments,axis=1).sum())
                    speed=float(np.linalg.norm(segments,axis=1).max()/cfg.trajectory_interval_sec)
                    fits_dem=bool(np.max(np.abs(path))<=footprint)
                    crossing=self_intersects(path)
                    metrics.append({'name':names[i],'path_length_m':length,'max_segment_speed_mps':speed,
                                    'in_dem_footprint':fits_dem,'self_intersects':crossing,
                                    'ood_nearest_l2_per_dimension':float(nearest[i]),
                                    'ood_over_train_p95':bool(nearest[i]>ood['train_ood_nearest_l2_per_dimension_p95']),
                                    'speed_over_frozen_max':bool(speed>spec['maneuvers']['speed_max_mps'])})
                all_paths.append(paths)
                rows.append({'split':split,'state_index':state_index,'state':state,'metrics':metrics,
                             'nominal_endpoint_before_alignment':nominal[-1].astype(float).tolist(),
                             'endpoint_alignment_delta_m':float(np.linalg.norm(nominal[-1]-w['goal']))})
                if len(rows)%10==0:print('MANEUVER_GENERATED',len(rows),'of',len(frozen['train'])+len(frozen['validation']),flush=True)
    array=np.asarray(all_paths,dtype=np.float32)
    temp=paths_path.with_name(paths_path.stem+'.tmp.npz')
    np.savez_compressed(temp,paths=array,names=np.asarray(names))
    temp.replace(paths_path)
    source={'selection_sha256':sha(a.selection),'thresholds_sha256':sha(a.thresholds),
            'spec_sha256':sha(a.spec),'reference_sha256':sha(a.reference),
            'manifest_sha256':sha(a.manifest),'prior_ood_thresholds_sha256':sha(a.prior_ood_thresholds),
            'generator_source_sha256':sha(Path(__file__))}
    atomic(report_path,{'source':source,'paths_sha256':sha(paths_path),'states':rows,
                        'train_count':len(frozen['train']),'validation_count':len(frozen['validation']),
                        'point_count':cfg.len_traj_pred,'candidate_names':names,'dem_half_extent_m':footprint,
                        'post_result_tuning':False})
    print('MANEUVER_BANK_COMPLETE',json.dumps({'shape':array.shape,'sha256':sha(paths_path)}),flush=True)

if __name__=='__main__':main()
