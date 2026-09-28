"""Train seven missing relative-Quality Probes under the prior frozen recipe."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_probe import tensorize
from visualnav_transformer.eat_nav.relative_quality_followup_probe import followup_quality_tensor
from visualnav_transformer.eat_nav.relative_quality_followup_train import train_followup_probe

OLD=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup')
OUT=ROOT/'output'
SPEC=Path('/tmp/eatnav-train/relative-quality-followup-config.yaml')
CKPT=Path('/tmp/eatnav-train/pilot-quality-best.pt')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def main(smoke):
    if not (OUT/'relative-label-audit.json').exists():raise ValueError('audit must pass before training')
    audit=json.loads((OUT/'relative-label-audit.json').read_text())
    if audit['status']!='pass':raise ValueError('audit failed')
    s=yaml.safe_load(SPEC.read_text());torch.set_num_threads(4)
    checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    if checkpoint['split_hash']!=s['split_hash']:raise ValueError('action scale split mismatch')
    mean=np.asarray(checkpoint['action_statistics']['mean'],dtype=np.float32)
    std=np.asarray(checkpoint['action_statistics']['std'],dtype=np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def load(name,limit):
        raw=np.load(OLD/'splits'/f'{name}.npz')
        if sha(OLD/'splits'/f'{name}.npz')!=s['source_aligned_split_sha256'][name]:
            raise ValueError(f'{name} split SHA mismatch')
        raw={k:raw[k][:limit] if limit else raw[k] for k in raw.files}
        return raw,tensorize(raw,mean,std,device)
    train_raw,train=load('train',100 if smoke else None)
    val_raw,val=load('val',50 if smoke else None)
    reference=json.loads((OLD/'output/relative-quality-probe-training-summary.json').read_text())
    if not smoke and len(train['dem'])!=reference['train_windows'] or not smoke and len(val['dem'])!=reference['validation_windows']:
        raise ValueError('previous Probe sample count mismatch')
    fingerprint={name:sha(path) for name,path in [
        ('config',SPEC),('train',OLD/'splits/train.npz'),('val',OLD/'splits/val.npz'),
        ('model',Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/relative_quality_followup_probe.py')),
        ('trainer',Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/relative_quality_followup_train.py'))]}
    settings=dict(s)
    if smoke:settings.update(probe_max_epochs=2,probe_patience=3)
    schedule=[('probe_delta',s['probe_seeds'][0])] if smoke else [(kind,seed) for kind in s['probe_kinds'] for seed in s['probe_seeds']]
    target=ROOT/'smoke-probes' if smoke else ROOT/'probes';start=time.monotonic()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    result={'mode':'smoke' if smoke else 'train','device':device,'train_windows':len(train['dem']),
            'validation_windows':len(val['dem']),'runs':[],'completed':False,'heldout_used':False,
            'input_fingerprint':fingerprint,'prior_reference_fingerprint':reference['input_fingerprint'],
            'mean_trajectory_validation_mse':reference['mean_trajectory_validation_mse']}
    for kind,seed in schedule:
        tq=followup_quality_tensor(train_raw,kind,device);vq=followup_quality_tensor(val_raw,kind,device)
        run_dir=target/f'{kind}-seed{seed}'
        run=train_followup_probe(train,val,tq,vq,kind=kind,seed=seed,settings=settings,
                                 output=run_dir,fingerprint=fingerprint)
        run['checkpoint_sha256']=sha(run['checkpoint']);result['runs'].append(run)
        write(target/'training-summary.json',result)
        print('FOLLOWUP_PROBE_DONE',kind,seed,run['best_epoch'],round(run['best_validation_loss'],6),flush=True)
    result.update(completed=True,seconds=time.monotonic()-start,
                  peak_torch_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else None)
    write(target/'training-summary.json',result)
    if not smoke:write(OUT/'probe-ablation-training-summary.json',result)
    print('FOLLOWUP_TRAIN_COMPLETE',result['mode'],round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
