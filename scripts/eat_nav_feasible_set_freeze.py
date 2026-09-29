"""Freeze input-selected feasible-set anchors and train-only maneuver scales."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.feasible_set import dem_difficulty, select_stratified_states
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)


def lateral_residual(window):
    xy=np.cumsum(np.asarray(window['action'],dtype=np.float64),axis=0)
    goal=np.asarray(window['goal'],dtype=np.float64)
    t=np.arange(1,len(xy)+1,dtype=np.float64)/len(xy)
    norm=np.linalg.norm(goal)
    left=np.array([-goal[1],goal[0]])/norm if norm>1e-8 else np.array([0.0,1.0])
    return float(np.max(np.abs((xy-t[:,None]*goal)@left)))


def selected_window_rows(view, split, spec):
    rule=spec['selection']
    result=[]
    for view_index, index in enumerate(view.indices):
        w=view.dataset.windows[index]
        if int(w['episode_id'])>rule['max_episode_id'] or not rule['min_local_step']<=int(w['local_step'])<=rule['max_local_step']:
            continue
        score=dem_difficulty(w['dem'],w['goal'])
        if not np.isfinite(score):
            continue
        result.append({'split':split,'terrain_type':w['terrain_type'],'terrain_seed':int(w['terrain_seed']),
                       'episode_key':list(map(int,w['episode_key'])),'view_index':view_index,
                       'step':int(w['step']),'local_step':int(w['local_step']),
                       'goal':np.asarray(w['goal'],dtype=float).tolist(),
                       'dem_sha256':hashlib.sha256(np.asarray(w['dem']).tobytes()).hexdigest(),
                       'difficulty_score':score})
    return result


def assign_strata(rows, thresholds):
    for row in rows:
        low,high=thresholds[row['terrain_type']]
        value=row['difficulty_score']
        row['stratum']='open' if value<low else 'intermediate' if value<high else 'constrained'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--rl-checkpoint',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'freeze output already exists: {args.output}')
    spec=yaml.safe_load(args.spec.read_text())
    if not all(spec['prohibitions'].values()):
        raise ValueError('required prohibitions are not frozen')
    manifest=json.loads(args.manifest.read_text())
    meta=manifest['metadata']
    if meta['num_envs']!=100 or meta['base_seed']!=123 or sha(args.rl_checkpoint)!=meta['checkpoint_sha256']:
        raise ValueError('source collection/checkpoint mismatch')
    config=load_eat_config(args.reference_config)
    if config.len_traj_pred!=spec['maneuvers']['point_count'] or config.context_size!=3:
        raise ValueError('reference dimensions mismatch')
    train=CompleteOnlyView(EATDataset(args.manifest,'train',config,split_seed=spec['selection']['split_seed']),config.len_traj_pred)
    val=CompleteOnlyView(EATDataset(args.manifest,'val',config,split_seed=spec['selection']['split_seed']),config.len_traj_pred)
    if (len(train),len(val))!=(spec['selection']['complete_train_expected'],spec['selection']['complete_validation_expected']):
        raise ValueError(f'complete window count mismatch: {(len(train),len(val))}')
    if any(not np.asarray(train[i]['action_mask']).all() for i in range(len(train))) or any(not np.asarray(val[i]['action_mask']).all() for i in range(len(val))):
        raise ValueError('partial windows in complete-only view')
    print('FEASIBLE_FREEZE_VIEWS',len(train),len(val),flush=True)
    train_rows=selected_window_rows(train,'train',spec)
    val_rows=selected_window_rows(val,'val',spec)
    q=spec['selection']['within_terrain_score_quantiles']
    terrains=sorted({row['terrain_type'] for row in train_rows})
    thresholds={terrain:np.quantile([r['difficulty_score'] for r in train_rows if r['terrain_type']==terrain],q).astype(float).tolist()
                for terrain in terrains}
    if set(r['terrain_type'] for r in val_rows)-set(terrains):
        raise ValueError('validation has terrain absent from train threshold fit')
    assign_strata(train_rows,thresholds)
    assign_strata(val_rows,thresholds)
    train_ids=select_stratified_states(train_rows,spec['selection']['train_per_terrain'],spec['selection']['seed'],spec['selection']['episode_cap'])
    val_ids=select_stratified_states(val_rows,spec['selection']['validation_per_terrain'],spec['selection']['seed']+1,spec['selection']['episode_cap'])
    train_ep={tuple(row['episode_key']) for row in train_ids}
    val_ep={tuple(row['episode_key']) for row in val_ids}
    train_seeds={row['terrain_seed'] for row in train_ids}
    val_seeds={row['terrain_seed'] for row in val_ids}
    if train_ep&val_ep or train_seeds&val_seeds:
        raise ValueError('train/validation state leakage')
    if len(train_ids)>56 or len(val_ids)>28:
        raise ValueError('state count exceeds frozen upper bound')
    all_terrain=sorted({t['terrain_type'] for t in meta['terrain']})
    if terrains!=all_terrain:
        raise ValueError('train candidate terrain support mismatch')
    residual=np.fromiter((lateral_residual(train.dataset.windows[index]) for index in train.indices),dtype=np.float64,count=len(train))
    slight,strong=np.quantile(residual,[spec['maneuvers']['slight_lateral_train_quantile'],spec['maneuvers']['strong_lateral_train_quantile']])
    if not 0<slight<strong:
        raise ValueError(f'train lateral scales degenerate: {slight},{strong}')
    source_files={
        'manifest':args.manifest,'reference_config':args.reference_config,'reference_checkpoint':args.reference,
        'rl_checkpoint':args.rl_checkpoint,'spec':args.spec,
        'freeze_source':args.source_root/'eat_nav_feasible_set_freeze.py',
        'feasible_set_source':args.source_root/'src/visualnav_transformer/eat_nav/feasible_set.py',
        'dataset_source':args.source_root/'src/visualnav_transformer/eat_nav/dataset.py',
        'sim_source':args.source_root/'eat_nav_feasible_set_sim.py',
        'bridge_source':args.source_root/'eat_nav_phase_c_ros_bridge.py',
        'collector_source':Path('/tmp/eatnav-extended/collect_eat_nav.py'),
        'collection_env_source':Path('/tmp/eatnav-extended/isaac-source/onix_isaac_lab/tasks/locomotion/velocity/config/onix/eat_nav_env_cfg.py'),
        'follower_config':Path('/workspace/onix_dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
    }
    source_hashes={key:sha(path) for key,path in source_files.items()}
    chunk_ids=[{'file':chunk['file'],'sha256':chunk['sha256'],'rows':chunk['rows']}
               for chunk in manifest['chunks']]
    if sum(row['rows'] for row in chunk_ids)<72871:
        raise ValueError('collection manifest unexpectedly short')
    args.output.mkdir(parents=True)
    atomic(args.output/'state-selection.json',{'train':train_ids,'validation':val_ids,'heldout':[],
        'candidate_rows':{'train':len(train_rows),'validation':len(val_rows)},
        'selected_counts':{'train':len(train_ids),'validation':len(val_ids)},
        'terrain_stratum_support':{split:{terrain:{s:sum(r['terrain_type']==terrain and r['stratum']==s for r in rows)
            for s in ('open','intermediate','constrained')} for terrain in terrains}
            for split,rows in [('train',train_rows),('validation',val_rows)]}})
    atomic(args.output/'maneuver-thresholds.json',{'source':'complete_only_train','terrain_score_quantiles':thresholds,
        'lateral_scale_m':{'slight':float(slight),'strong':float(strong)},
        'lateral_residual_train_count':len(residual),
        'lateral_residual_quantiles':{'p50':float(np.quantile(residual,.5)),'p75':float(np.quantile(residual,.75)),'p90':float(np.quantile(residual,.9)),'p99':float(np.quantile(residual,.99))}})
    atomic(args.output/'source-provenance.json',{'source_sha256':source_hashes,'dataset_chunks':chunk_ids,
        'train_complete_windows':len(train),'validation_complete_windows':len(val),
        'collection_metadata':{'base_seed':meta['base_seed'],'num_envs':meta['num_envs'],'step_dt':meta['step_dt'],
                               'observation_toggles':meta['observation_toggles']},
        'heldout_used':False})
    atomic(args.output/'artifact-hashes.json',{name:sha(args.output/name) for name in
        ('state-selection.json','maneuver-thresholds.json','source-provenance.json')})
    atomic(args.output/'status.json',{'stage':'frozen','train_states':len(train_ids),'validation_states':len(val_ids),
        'measurement_gate':'anchor_reconstruction_pending','formal_phase_c_allowed':False})
    print('FEASIBLE_FREEZE_COMPLETE',json.dumps({'train':len(train_ids),'validation':len(val_ids),'scales':[slight,strong],
          'output':str(args.output)}),flush=True)

if __name__=='__main__':
    main()
