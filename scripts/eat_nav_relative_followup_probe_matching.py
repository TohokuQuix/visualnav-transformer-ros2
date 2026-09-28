"""Train-frozen matched trajectory controls for new Quality vectors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from visualnav_transformer.eat_nav.quality_gate_evaluation import pair_cluster_ci
from visualnav_transformer.eat_nav.relative_quality_matching import geometric_candidates,label_thresholds,select_disjoint_pairs,episode_id_list
from visualnav_transformer.eat_nav.relative_quality_followup_probe import FIELDS
from eat_nav_relative_quality_analysis import pair_geometry
from eat_nav_relative_followup_probe_train import write,OLD,ROOT,OUT,SPEC,sha

VECTOR=('probe_delta_drawdown','probe_relative_triplet','probe_multi_reward_free','probe_multi_no_v0')
SCALAR={'probe_delta':'delta_v','probe_recovery':'recovery','probe_return':'return_h'}
MEASURES=('trajectory_rms_m','endpoint_m','forward_progress_m','path_length_m','turning_rad')


def vector(data,kind):
    fields=FIELDS[kind]
    return np.column_stack([data[f] for f in fields])


def distance(x,i,j):return float(np.linalg.norm(x[i]-x[j])/np.sqrt(x.shape[1]))


def summarize(data,selected):
    rows=[];episodes=[tuple(e) for e in data['episode']]
    for i,j,d,g,angle,iou,rmse in selected:
        rows.append({'left_index':int(data['index'][i]),'right_index':int(data['index'][j]),
                     'left_episode':episode_id_list(episodes[i]),'right_episode':episode_id_list(episodes[j]),
                     'terrain':str(data['terrain'][i]),'label_difference':d,
                     'goal_difference_m':g,'goal_angle_deg':angle,'mask_iou':iou,'dem_rmse_m':rmse,
                     **pair_geometry(data,i,j)})
    return rows


def main():
    s=yaml.safe_load(SPEC.read_text());train=np.load(OLD/'splits/train.npz');test=np.load(OLD/'splits/test.npz')
    for split in ('train','test'):
        if sha(OLD/'splits'/f'{split}.npz')!=s['source_aligned_split_sha256'][split]:
            raise ValueError(f'{split} feature SHA mismatch')
    kwargs={'goal_distance_m':s['match_goal_distance_m'],'goal_direction_deg':s['match_goal_direction_deg'],
            'min_mask_iou':s['match_min_mask_iou'],'max_dem_rmse_m':s['match_max_dem_rmse_m']}
    tr=geometric_candidates(train,seed=s['match_seed'],limit=s['match_train_candidate_cap'],**kwargs)
    thresholds={}
    for kind in VECTOR:
        x=vector(train,kind)
        thresholds[kind]=label_thresholds([distance(x,i,j) for i,j,*_ in tr],
            s['match_small_difference_train_quantile'],s['match_large_difference_train_quantile'])
    write(OUT/'probe-ablation-match-thresholds.json',{'source_split':'train','source_sha256':s['source_aligned_split_sha256']['train'],
          'distance':'Euclidean train-standardized vector / sqrt(dimension)',
          'thresholds':thresholds,'geometry':kwargs,'candidate_count':len(tr),
          'heldout_used_for_threshold':False})
    te=geometric_candidates(test,seed=s['match_seed']+1,limit=s['match_heldout_candidate_cap'],**kwargs)
    episodes=[tuple(e) for e in test['episode']]
    result={}
    for kind in VECTOR:
        x=vector(test,kind);threshold=thresholds[kind]
        candidates=[(i,j,distance(x,i,j),g,a,iou,rmse) for i,j,g,a,iou,rmse in te]
        cross=[r for r in candidates if r[2]>=threshold['large']]
        same=[r for r in candidates if r[2]<=threshold['small']]
        selected_cross=select_disjoint_pairs(cross,episodes,max_pairs=s['match_pairs_per_kind'],
                                              episode_cap=s['match_episode_cap'],descending=True)
        used={i for row in selected_cross for i in row[:2]}
        same=[r for r in same if r[0] not in used and r[1] not in used]
        selected_same=select_disjoint_pairs(same,episodes,max_pairs=s['match_pairs_per_kind'],
                                             episode_cap=s['match_episode_cap'],descending=False)
        a=summarize(test,selected_cross);b=summarize(test,selected_same)
        measures={}
        for metric in MEASURES:
            if min(len(a),len(b))<10:measures[metric]={'status':'insufficient support'}
            else:measures[metric]=pair_cluster_ci([row[metric] for row in a],
                [(tuple(row['left_episode']),tuple(row['right_episode'])) for row in a],
                [row[metric] for row in b],
                [(tuple(row['left_episode']),tuple(row['right_episode'])) for row in b],
                seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
        result[kind]={'cross_count':len(a),'same_count':len(b),'cross_minus_same':measures,
                      'cross_episode_count':len({tuple(row['left_episode']) for row in a}|{tuple(row['right_episode']) for row in a})}
        print('FOLLOWUP_MATCHED',kind,len(a),len(b),flush=True)
    previous=json.loads((OLD/'output/relative-quality-matched-pairs.json').read_text())
    for kind,metric in SCALAR.items():
        item=previous['metrics'][metric]
        result[kind]={'cross_count':item['cross']['count'],'same_count':item['same']['count'],
                      'cross_minus_same':item['cross_minus_same'],'source':'frozen prior train-thresholded matched pairs',
                      'prior_artifact_sha256':sha(OLD/'output/relative-quality-matched-pairs.json')}
    result['probe_multi']={'source':'frozen prior Q-multi matched pairs',
       'prior_artifact_sha256':sha(OLD/'output/relative-quality-multi-matched-pairs.json')}
    write(OUT/'probe-ablation-matched-pairs.json',{'heldout_source_sha256':s['source_aligned_split_sha256']['test'],
          'train_threshold_file':'probe-ablation-match-thresholds.json',
          'train_candidate_count':len(tr),'heldout_candidate_count':len(te),'metrics':result})


if __name__=='__main__':main()
