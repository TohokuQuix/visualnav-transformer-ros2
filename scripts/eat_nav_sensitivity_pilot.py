"""Reproducible held-out, paired-noise Quality sensitivity pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.evaluation import diagnose_conditioning_sensitivity
from visualnav_transformer.eat_nav.model import EATNav


def _shape_metrics(window):
    xy=np.cumsum(window['action'],axis=0)
    steps=np.diff(np.vstack((np.zeros((1,2),dtype=np.float32),xy)),axis=0)
    path=float(np.linalg.norm(steps,axis=1).sum())
    angles=np.arctan2(steps[:,1],steps[:,0])
    curvature=float(np.abs(np.arctan2(np.sin(np.diff(angles)),np.cos(np.diff(angles)))).sum())
    return {'forward_progress_m':float(xy[-1,0]),'path_length_m':path,'curvature_rad':curvature}


def _matched_inputs(dataset,count=25):
    eligible=[i for i,w in enumerate(dataset.windows) if w['action_mask'].all()]
    low=[i for i in eligible if dataset.ecdf.transform(dataset.windows[i]['quality_raw'])<.25]
    high=[i for i in eligible if dataset.ecdf.transform(dataset.windows[i]['quality_raw'])>=.75]
    if not low or not high:return [],{'eligible':len(eligible),'low':len(low),'high':len(high)}
    tree=cKDTree(np.stack([dataset.windows[i]['goal'] for i in high]))
    selected=[];used=set();last_step={}
    rng=np.random.default_rng(123)
    for i in rng.permutation(low):
        a=dataset.windows[int(i)]
        if int(i) in used or a['step']-last_step.get(a['episode_key'],-10**9)<40:continue
        nearby=tree.query_ball_point(a['goal'],dataset.config.overlap_max_goal_distance_m)
        for j in nearby:
            h=high[j];b=dataset.windows[h]
            if h in used or a['episode_key']==b['episode_key'] or a['terrain_type']!=b['terrain_type']:continue
            if b['step']-last_step.get(b['episode_key'],-10**9)<40:continue
            mask_a=a['dem'][:,1]>.5;mask_b=b['dem'][:,1]>.5
            both=mask_a&mask_b
            iou=both.sum()/max((mask_a|mask_b).sum(),1)
            if iou<dataset.config.overlap_min_mask_iou:continue
            rmse=np.sqrt(np.square(a['dem'][:,0]-b['dem'][:,0])[both].mean())
            if rmse>dataset.config.overlap_max_height_rmse_m:continue
            selected.append((int(i),int(h),float(np.linalg.norm(a['goal']-b['goal'])),float(rmse)))
            used.update((int(i),int(h)))
            last_step[a['episode_key']]=a['step'];last_step[b['episode_key']]=b['step']
            break
        if len(selected)>=count:break
    return selected,{'eligible':len(eligible),'low':len(low),'high':len(high)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--quality-config',type=Path,required=True)
    parser.add_argument('--baseline-config',type=Path,required=True)
    parser.add_argument('--quality-checkpoint',type=Path,required=True)
    parser.add_argument('--baseline-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    config=load_eat_config(args.quality_config)
    dataset=EATDataset(args.manifest,'test',config)
    pairs,eligible_counts=_matched_inputs(dataset)
    if len(pairs)<25:
        report={'status':'inconclusive','reason':'fewer than 25 matched high/low Quality pairs for 50 fixed conditions',
                'matched_pairs':len(pairs),'eligible_counts':eligible_counts}
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('SENSITIVITY_DONE',json.dumps(report),flush=True)
        return
    indices=[i for pair in pairs for i in pair[:2]]
    windows=[dataset[i] for i in indices]
    q_values=[.1,.3,.5,.7,.9,.95]
    seeds=[11,29,47]
    quality_model,_,_=load_checkpoint(args.quality_checkpoint,config)
    baseline_model,_,_=load_checkpoint(args.baseline_checkpoint,load_eat_config(args.baseline_config))
    quality_model.cuda();baseline_model.cuda()
    print('SENSITIVITY_START',len(windows),'conditions',len(pairs),'matched_pairs',eligible_counts,flush=True)
    quality=diagnose_conditioning_sensitivity(quality_model,windows,q_values,seeds)
    print('QUALITY_DONE',quality['status'],quality['effect_size_ratio'],flush=True)
    baseline=diagnose_conditioning_sensitivity(baseline_model,windows,q_values,seeds)
    torch.manual_seed(123)
    untrained=EATNav(config).cuda().eval()
    untrained.set_action_statistics(quality_model.action_mean,quality_model.action_std)
    untrained_result=diagnose_conditioning_sensitivity(untrained,windows,q_values,seeds)
    observed=[]
    for lo,hi,goal_dist,dem_rmse in pairs:
        low=_shape_metrics(dataset[lo]);high=_shape_metrics(dataset[hi])
        observed.append({'goal_distance_m':goal_dist,'dem_rmse_m':dem_rmse,
                         'forward_progress_delta_m':high['forward_progress_m']-low['forward_progress_m'],
                         'path_length_delta_m':high['path_length_m']-low['path_length_m'],
                         'curvature_delta_rad':high['curvature_rad']-low['curvature_rad']})
    observed_summary={name:float(np.median([x[name] for x in observed])) for name in ('forward_progress_delta_m','path_length_delta_m','curvature_delta_rad')}
    direction={name:bool(np.sign(observed_summary[name])==np.sign(quality[model_name]))
               for name,model_name in (('forward_progress_delta_m','high_q_forward_progress_delta_m'),
                                       ('path_length_delta_m','path_length_delta_m'),
                                       ('curvature_delta_rad','curvature_delta_rad'))}
    report={'status':quality['status'],'selection':{'heldout_split':'test','input_count':len(windows),'matched_pairs':len(pairs),
            'eligible_counts':eligible_counts,'indices':indices,'q_bin_counts':dict(Counter(min(int(w['quality']*4),3) for w in windows)),
            'episode_count':len({w['episode_key'] for w in windows}),
            'terrain_types':dict(Counter(w['terrain_type'] for w in windows))},
            'quality_model':quality,'quality_disabled_baseline':baseline,'untrained_quality_model':untrained_result,
            'observed_matched_data_median_high_minus_low':observed_summary,'model_data_direction_agreement':direction,
            'paired_observed_data':observed}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('SENSITIVITY_DONE',json.dumps({k:report[k] for k in ('status','observed_matched_data_median_high_minus_low','model_data_direction_agreement')},sort_keys=True),flush=True)


if __name__=='__main__':main()
