"""Freeze outcome-blind, short-prefix states from the entire complete-only train view."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def rank(row,seed):
    identity=[seed,row['terrain_type'],row['episode_key'],row['view_index']]
    return hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()


def select_cost_aware(rows,*,terrains,seed,min_goal,max_local_step):
    if min_goal<.30 or max_local_step<0:raise ValueError('frozen v2 limits invalid')
    support={t:{'complete_train':0,'eligible':0,'goal_rejected':0,'prefix_rejected':0} for t in terrains}
    eligible={t:[] for t in terrains}
    for row in rows:
        if row['split']!='train' or row['terrain_type'] not in support or not row.get('complete',True):
            raise ValueError('v2 input must be complete-only train within declared terrains')
        terrain=row['terrain_type'];support[terrain]['complete_train']+=1
        goal=math.hypot(*map(float,row['goal']))
        if goal<min_goal:support[terrain]['goal_rejected']+=1;continue
        if int(row['local_step'])>max_local_step:support[terrain]['prefix_rejected']+=1;continue
        support[terrain]['eligible']+=1
        eligible[terrain].append({**row,'goal_distance_m':goal,'rank_sha256':rank(row,seed)})
    selected=[];used=set();na={}
    for terrain in sorted(terrains):
        for row in sorted(eligible[terrain],key=lambda x:(x['rank_sha256'],x['view_index'])):
            key=tuple(row['episode_key'])
            if key not in used:
                selected.append(dict(row,state_id=f'v2-{len(selected):03d}'))
                used.add(key);break
        else:na[terrain]='NA / no state within runtime budget'
    def distribution(values):
        return {'count':len(values),'min':min(values) if values else None,
                'p50':float(np.median(values)) if values else None,
                'p90':float(np.quantile(values,.9)) if values else None,
                'max':max(values) if values else None}
    return {'seed':seed,'minimum_goal_m':min_goal,'episode_local_prefix_max_steps':max_local_step,
            'selected':selected,'na_terrains':na,'support':support,
            'eligible_prefix_distribution':distribution([int(r['local_step']) for group in eligible.values() for r in group]),
            'selected_prefix_distribution':distribution([int(r['local_step']) for r in selected]),
            'episode_cap':1,'selection_fields':['terrain_type','episode_key','view_index','goal','local_step','split'],
            'outcome_fields_used':[]}


def atomic_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise FileExistsError('v2 freeze would overwrite differing artifact')
        return
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--gate-config',type=Path,required=True)
    parser.add_argument('--selection-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    import yaml
    from visualnav_transformer.eat_nav.config import load_eat_config
    from visualnav_transformer.eat_nav.dataset import EATDataset
    from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
    spec=yaml.safe_load(a.gate_config.read_text())['v2_selection_if_failed']
    if spec['split']!='train' or not spec['complete_only'] or spec['episode_cap']!=1:
        raise ValueError('v2 selection spec changed')
    cfg=load_eat_config(a.reference_config)
    view=CompleteOnlyView(EATDataset(a.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(view)!=49598:raise ValueError(f'complete-only train count differs: {len(view)}')
    all_terrains=sorted({r['terrain_type'] for r in json.loads(a.selection_source.read_text())['train']})
    rows=[]
    for view_index,index in enumerate(view.indices):
        w=view.dataset.windows[index]
        if not np.asarray(w['action_mask']).all():raise ValueError('partial window in complete view')
        rows.append({'split':'train','complete':True,'terrain_type':w['terrain_type'],
                     'episode_key':list(map(int,w['episode_key'])),'terrain_seed':int(w['terrain_seed']),
                     'view_index':view_index,'step':int(w['step']),'local_step':int(w['local_step']),
                     'goal':np.asarray(w['goal'],dtype=float).tolist(),
                     'dem_sha256':hashlib.sha256(np.asarray(w['dem'],dtype=np.float32).tobytes()).hexdigest()})
    result=select_cost_aware(rows,terrains=all_terrains,seed=spec['seed'],
        min_goal=spec['goal_distance_min_m'],max_local_step=spec['episode_local_prefix_max_steps'])
    result['complete_train_windows']=len(view)
    result['source_sha256']={key:sha(path) for key,path in {'manifest':a.manifest,
        'reference_config':a.reference_config,'gate_config':a.gate_config,
        'original_selection':a.selection_source,'selector_source':Path(__file__),
        'dataset_source':Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/dataset.py')}.items()}
    result['rollout_started']=False;result['approval_required_before_rollout']=True
    atomic_once(a.output/'selection-v2.json',result)
    atomic_once(a.output/'status.json',{'stage':'selection_v2_frozen','state':'awaiting_approval',
                                      'selected_count':len(result['selected']),'rollout_started':False})
    print('COST_SELECTION_V2_FROZEN',[(x['terrain_type'],x['state_id'],x['local_step']) for x in result['selected']],result['na_terrains'],flush=True)

if __name__=='__main__':main()
