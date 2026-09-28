"""Run scratch EAT overfit only on frozen train-split paired conditions."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings
from visualnav_transformer.eat_nav.quality_gate_overfit import prepare_overfit_pairs,train_overfit


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','overfit-config','matched-pairs','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('smoke','train'),required=True)
    a=p.parse_args();torch.set_num_threads(4)
    config=load_eat_config(a.quality_config)
    settings=load_gate_settings(a.overfit_config)
    pairs=json.loads(a.matched_pairs.read_text())
    if pairs['source_split']!='train' or pairs['selected_count']<settings.min_b_pairs:
        raise ValueError('invalid train-only pair manifest')
    if abs(pairs['trajectory_rms_threshold_m']-settings.b_trajectory_rms_threshold_m)>1e-12 or abs(pairs['direction_metric_threshold']-settings.b_direction_threshold)>1e-12:
        raise ValueError('frozen pair thresholds disagree with config')
    train=EATDataset(a.manifest,'train',config)
    if len(train.split_map)!=100:
        raise ValueError('unexpected split map')
    windows=[train[i] for i in range(len(train))]
    ids=[];episode_uses={}
    for pair in pairs['selected_pairs']:
        lo,hi=pair['low_index'],pair['high_index']
        ids.append((lo,hi))
        if windows[lo]['terrain_seed']!=pair['low_terrain_seed'] or windows[hi]['terrain_seed']!=pair['high_terrain_seed']:
            raise ValueError('pair source identity mismatch')
        for index in (lo,hi):
            key=windows[index]['episode_key'];episode_uses[key]=episode_uses.get(key,0)+1
    if max(episode_uses.values())>settings.max_b_episode_uses:
        raise ValueError('episode use cap exceeded')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    batch=prepare_overfit_pairs(windows,ids,device)
    if a.mode=='smoke':settings=replace(settings,overfit_epochs=2)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();results=[]
    seeds=settings.training_seeds[:1] if a.mode=='smoke' else settings.training_seeds
    for seed in seeds:
        result=train_overfit(config,train.stats,batch,seed=seed,settings=settings,output=a.output/f'seed-{seed}')
        results.append(result)
        print('OVERFIT_RUN_DONE',seed,result['final_fixed_bank_loss'],flush=True)
    summary={'mode':a.mode,'train_only':True,'pair_count':len(ids),'training_seeds':list(seeds),
             'scratch_start':True,'wall_seconds':time.monotonic()-start,
             'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
             'runs':results}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('OVERFIT_DONE',json.dumps({k:summary[k] for k in ('mode','pair_count','wall_seconds','peak_gpu_memory_bytes')}),flush=True)

if __name__=='__main__':main()
