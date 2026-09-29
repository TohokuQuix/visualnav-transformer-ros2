"""Secondary teacher-path and quality summaries; no gate thresholds are changed."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_phase_c_candidate_diversity import pairwise_metrics
from eat_nav_phase_c_coverage import atomic,digest,write_marker
from eat_nav_phase_c_coverage_stage1 import residual


def command_mode_labels(commands,linear_threshold,angular_threshold):
    x=np.asarray(commands,dtype=float)
    labels=list(range(len(x)))
    def find(i):
        while labels[i]!=i:i=labels[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if abs(x[i,0]-x[j,0])<linear_threshold and abs(x[i,2]-x[j,2])<angular_threshold:labels[find(j)]=find(i)
    roots=[find(i) for i in range(len(x))];order={v:k for k,v in enumerate(sorted(set(roots)))}
    return [order[x] for x in roots]


def run(args):
    cfg=load_eat_config(args.reference_config)
    train=CompleteOnlyView(EATDataset(args.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(train)!=49598:raise ValueError('complete-only train count mismatch')
    with gzip.open(args.output/'train-neighborhoods.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
    details=[];strata=defaultdict(list)
    for row in rows:
        members=row['member_view_indices']
        windows=[train.dataset.windows[train.indices[i]] for i in members]
        paths=np.stack([np.cumsum(w['action'],axis=0) for w in windows])
        metrics=pairwise_metrics(paths)
        commands=np.asarray(row['follower_commands'])
        labels=command_mode_labels(commands,.03,.05)
        means=[float(np.mean([row['quality_raw'][i] for i in range(len(labels)) if labels[i]==mode])) for mode in set(labels)]
        quality_gap=float(max(means)-min(means)) if len(means)>1 else 0.
        residuals=np.stack([residual(p) for p in paths])
        ix=np.triu_indices(len(paths),1)
        shape=np.sqrt(np.mean((residuals[:,None]-residuals[None,:])**2,axis=(2,3)))[ix]
        ep=Counter(tuple(x) for x in row['episode_keys'])
        goal=np.asarray(windows[0]['goal']);distance=float(np.linalg.norm(goal));bearing=float(np.degrees(np.arctan2(goal[1],goal[0])))
        direction='forward' if abs(bearing)<=30 else 'left' if bearing>30 else 'right'
        distance_bin='near' if distance<.25 else 'middle' if distance<.5 else 'far'
        strata[(row['terrain_type'],direction,distance_bin)].append(len(means))
        details.append({'anchor_view_index':row['anchor_view_index'],'terrain_type':row['terrain_type'],
            'goal_distance_m':distance,'goal_bearing_deg':bearing,'goal_stratum':[direction,distance_bin],
            'follower_command_mode_labels':labels,'mode_count':len(means),
            'between_mode_quality_raw_gap':quality_gap,'episode_concentration_max_fraction':max(ep.values())/len(members),
            'residual_shape_pairwise_rms_m':shape.tolist(),'teacher_pairwise':metrics,
            'path_distinct_but_command_single':row['path_cluster_count']>1 and len(means)==1})
    summary={'status':'complete','split':'train','neighborhood_count':len(rows),'rows':details,
             'strata':[{'terrain_type':key[0],'goal_direction':key[1],'goal_distance_bin':key[2],
                        'count':len(v),'multimodal_fraction':float(np.mean(np.asarray(v)>=2))} for key,v in sorted(strata.items())],
             'path_distinct_command_single_fraction':float(np.mean([r['path_distinct_but_command_single'] for r in details])) if details else None,
             'median_between_mode_quality_raw_gap':float(np.median([r['between_mode_quality_raw_gap'] for r in details if r['mode_count']>=2])) if any(r['mode_count']>=2 for r in details) else None,
             'goal_bin_definition':{'direction_deg':[-30,30],'distance_m':[.25,.5]},'gate_uses_these_bins':False}
    atomic(args.output/'teacher-multimodality-details.json',summary)
    write_marker(args.output/'teacher-details.done.json',[args.output/'teacher-multimodality-details.json'],digest(args.output/'train-neighborhoods.jsonl.gz'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
