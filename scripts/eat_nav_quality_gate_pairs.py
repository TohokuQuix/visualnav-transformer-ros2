"""Freeze train-only overfit pairs and held-out descriptive controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings
from visualnav_transformer.eat_nav.quality_gate_data import match_quality_pairs, select_overfit_pairs, build_canonical_pair


def serialize(row, windows):
    i,j,rms,direction,goal_delta,rmse,iou=row
    a,b=windows[i],windows[j]
    return dict(low_index=int(i),high_index=int(j),low_episode=list(a['episode_key']),high_episode=list(b['episode_key']),
                low_step=int(a['step']),high_step=int(b['step']),low_terrain_seed=int(a['terrain_seed']),
                high_terrain_seed=int(b['terrain_seed']),terrain_type=a['terrain_type'],
                q_low=float(a['quality']),q_high=float(b['quality']),trajectory_rms_m=float(rms),
                direction_metric=float(direction),goal_difference_m=float(goal_delta),
                dem_height_rmse_m=float(rmse),mask_iou=float(iou))


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','probe-config','overfit-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    config=load_eat_config(a.quality_config)
    settings=load_gate_settings(a.overfit_config)
    train=EATDataset(a.manifest,'train',config)
    windows=[train[i] for i in range(len(train))]
    candidates=match_quality_pairs(windows,q_mode='cross',limit=None,seed=settings.match_seed)
    if not candidates:raise ValueError('no train cross-q pairs')
    rms_cut=float(np.quantile([row[2] for row in candidates],settings.b_trajectory_rms_quantile))
    dir_cut=float(np.quantile([row[3] for row in candidates],settings.b_direction_abs_quantile))
    selected=select_overfit_pairs(windows,candidates,min_rms=rms_cut,min_direction=dir_cut,
                                  count=settings.max_b_pairs,max_episode_uses=settings.max_b_episode_uses)
    if len(selected)<settings.min_b_pairs:raise ValueError(f'only {len(selected)} overfit pairs')
    for row in selected:
        left,right=build_canonical_pair(windows[row[0]],windows[row[1]])
        if not np.array_equal(left['dem'],right['dem']) or not np.array_equal(left['goal'],right['goal']):
            raise AssertionError('canonical condition mismatch')
    result={'selection_seed':settings.match_seed,'source_split':'train','train_candidate_count':len(candidates),
            'trajectory_rms_threshold_m':rms_cut,'direction_metric_threshold':dir_cut,
            'threshold_quantiles':{'trajectory_rms':settings.b_trajectory_rms_quantile,
                                   'direction_metric':settings.b_direction_abs_quantile},
            'episode_use_cap':settings.max_b_episode_uses,'selected_count':len(selected),
            'selected_pairs':[serialize(row,windows) for row in selected]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('TRAIN_PAIRS_DONE',len(candidates),len(selected),rms_cut,dir_cut,flush=True)
    del candidates,selected,windows,train
    test=EATDataset(a.manifest,'test',config)
    windows=[test[i] for i in range(len(test))]
    cross=match_quality_pairs(windows,q_mode='cross',limit=1000,seed=settings.match_seed)
    same=match_quality_pairs(windows,q_mode='same',limit=1000,seed=settings.match_seed+1)
    controls={'source_split':'test','selection_seed':settings.match_seed,
              'cross_q_pairs':[serialize(row,windows) for row in cross],
              'same_q_pairs':[serialize(row,windows) for row in same]}
    (a.output.parent/'heldout-pair-controls.json').write_text(json.dumps(controls,indent=2,sort_keys=True)+'\n')
    print('HELDOUT_PAIRS_DONE',len(cross),len(same),flush=True)

if __name__=='__main__':main()
