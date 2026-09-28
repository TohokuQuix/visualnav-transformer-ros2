"""Train equal-architecture deterministic probes with aggregate validation selection."""
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
from visualnav_transformer.eat_nav.quality_gate_probe import tensorize_windows,train_probe


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','gate-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('smoke','train'),required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    config=load_eat_config(a.quality_config)
    settings=load_gate_settings(a.gate_config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=EATDataset(a.manifest,'train',config)
    val=EATDataset(a.manifest,'val',config)
    if train.split_map!=val.split_map or train.ecdf.values!=val.ecdf.values:
        raise ValueError('train/validation ECDF or split mismatch')
    train_indices=list(range(len(train)))
    val_indices=list(range(len(val)))
    if a.mode=='smoke':
        train_indices=train_indices[:512]
        val_indices=val_indices[:128]
        settings=replace(settings,max_probe_epochs=2,probe_patience=3)
    train_rows=[dict(train[i],window_index=i) for i in train_indices]
    val_rows=[dict(val[i],window_index=i) for i in val_indices]
    train_data=tensorize_windows(train_rows,train.stats,device)
    val_data=tensorize_windows(val_rows,train.stats,device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();runs=[]
    schedule=[('probe_0',seed,None) for seed in settings.training_seeds]
    schedule += [('probe_q',seed,None) for seed in settings.training_seeds]
    if a.mode=='smoke':schedule=schedule[:1]
    else:schedule += [('probe_q_shuffled',settings.training_seeds[i%len(settings.training_seeds)],i)
                    for i in range(settings.initial_permutations)]
    for kind,seed,shuffle in schedule:
        name=f'{kind}-seed{seed}' + (f'-shuffle{shuffle}' if shuffle is not None else '')
        result=train_probe(train_data,val_data,kind=kind,training_seed=seed,shuffle_seed=shuffle,
                           settings=settings,output=a.output/name)
        runs.append(result)
        print('PROBE_RUN_DONE',name,result['best_validation_loss'],flush=True)
    report={'mode':a.mode,'device':device,'train_windows':len(train_data['q']),
            'validation_windows':len(val_data['q']),'wall_seconds':time.monotonic()-start,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'runs':runs,'heldout_used':False}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('PROBE_TRAIN_DONE',json.dumps({k:report[k] for k in ('mode','train_windows','validation_windows','wall_seconds','peak_gpu_memory_bytes')}),flush=True)

if __name__=='__main__':main()
