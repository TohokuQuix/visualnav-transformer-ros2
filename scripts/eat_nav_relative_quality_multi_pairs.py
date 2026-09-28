"""Train-frozen joint-vector matched pair control for Q-multi."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import yaml
from visualnav_transformer.eat_nav.quality_gate_evaluation import pair_cluster_ci
from visualnav_transformer.eat_nav.relative_quality_matching import geometric_candidates,label_thresholds,select_disjoint_pairs,episode_id_list
from eat_nav_relative_quality_analysis import pair_geometry

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
OUT=ROOT/'output'
SPEC=Path('/tmp/eatnav-train/relative-quality-config.yaml')
FIELDS=('z_v0','z_advantage_h','z_drawdown','z_recovery')


def write(name,payload):
    target=OUT/name;tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(target)


def values(data):
    return np.column_stack([data[k] for k in FIELDS])


def distance(x,i,j):
    return float(np.linalg.norm(x[i]-x[j])/np.sqrt(len(FIELDS)))


def main():
    s=yaml.safe_load(SPEC.read_text());train=np.load(ROOT/'splits/train.npz');test=np.load(ROOT/'splits/test.npz')
    kwargs={'goal_distance_m':s['match_goal_distance_m'],'goal_direction_deg':s['match_goal_direction_deg'],
            'min_mask_iou':s['match_min_mask_iou'],'max_dem_rmse_m':s['match_max_dem_rmse_m']}
    tr=geometric_candidates(train,seed=s['match_seed'],limit=s['match_train_candidate_cap'],**kwargs)
    x=values(train)
    threshold=label_thresholds([distance(x,i,j) for i,j,*_ in tr],
                               s['match_small_difference_train_quantile'],s['match_large_difference_train_quantile'])
    write('relative-quality-multi-match-threshold.json',{'source_split':'train','fields':FIELDS,
          'distance':'Euclidean train-standardized vector / sqrt(4)',
          'threshold':threshold,'geometry':kwargs,'train_candidate_count':len(tr),
          'heldout_used_for_threshold':False})
    te=geometric_candidates(test,seed=s['match_seed']+1,limit=s['match_heldout_candidate_cap'],**kwargs)
    x=values(test);episodes=[tuple(row) for row in test['episode']]
    candidates=[(i,j,distance(x,i,j),g,angle,iou,rmse) for i,j,g,angle,iou,rmse in te]
    cross=[r for r in candidates if r[2]>=threshold['large']]
    same=[r for r in candidates if r[2]<=threshold['small']]
    selected_cross=select_disjoint_pairs(cross,episodes,max_pairs=s['match_pairs_per_kind'],
                                         episode_cap=s['match_episode_cap'],descending=True)
    used={i for row in selected_cross for i in row[:2]}
    same=[r for r in same if r[0] not in used and r[1] not in used]
    selected_same=select_disjoint_pairs(same,episodes,max_pairs=s['match_pairs_per_kind'],
                                        episode_cap=s['match_episode_cap'],descending=False)
    result={}
    for name,selected in [('cross',selected_cross),('same',selected_same)]:
        result[name]={'count':len(selected),'pairs':[]}
        for i,j,d,g,angle,iou,rmse in selected:
            result[name]['pairs'].append({'left_index':int(test['index'][i]),'right_index':int(test['index'][j]),
                  'left_episode':episode_id_list(episodes[i]),'right_episode':episode_id_list(episodes[j]),
                  'left_terrain':str(test['terrain'][i]),'label_difference':d,
                  'goal_difference_m':g,'goal_angle_deg':angle,'mask_iou':iou,'dem_rmse_m':rmse,
                  **pair_geometry(test,i,j)})
    comparison={}
    for metric in ('trajectory_rms_m','endpoint_m','forward_progress_m','path_length_m','turning_rad'):
        a=result['cross']['pairs'];b=result['same']['pairs']
        if min(len(a),len(b))<10:
            comparison[metric]={'status':'insufficient support'}
        else:
            comparison[metric]=pair_cluster_ci([r[metric] for r in a],
                [(tuple(r['left_episode']),tuple(r['right_episode'])) for r in a],
                [r[metric] for r in b],
                [(tuple(r['left_episode']),tuple(r['right_episode'])) for r in b],
                seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
    write('relative-quality-multi-matched-pairs.json',{'source_split':'test','train_threshold_file':
          'relative-quality-multi-match-threshold.json','train_candidate_count':len(tr),
          'heldout_candidate_count':len(te),'metrics':{'multi':{**result,'cross_minus_same':comparison}}})
    print('RELATIVE_MULTI_PAIRS',len(selected_cross),len(selected_same),flush=True)


if __name__=='__main__':main()
