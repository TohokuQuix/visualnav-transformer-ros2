"""Reproduce prior aligned labels and verify frozen Probe/checkpoint provenance."""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_followup_audit import validate_aligned_record

OLD=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
BASE=Path('/tmp/eatnav-train')
ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup')
OUT=ROOT/'output'
SPEC=BASE/'relative-quality-followup-config.yaml'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(name,payload):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/name;temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def main():
    s=yaml.safe_load(SPEC.read_text());old=yaml.safe_load((BASE/'relative-quality-config.yaml').read_text())
    if sha(BASE/'relative-quality-config.yaml')!=s['source_relative_config_sha256']:
        raise ValueError('source config SHA mismatch')
    if sha(Path('/tmp/eatnav-extended/output/manifest.json'))!=s['source_manifest_sha256']:
        raise ValueError('source manifest SHA mismatch')
    if sha(OLD/'output/relative-quality-values.jsonl.gz')!=s['source_relative_values_sha256']:
        raise ValueError('source relative values SHA mismatch')
    if sha(BASE/'src/visualnav_transformer/eat_nav/relative_quality_probe.py')!=s['source_relative_probe_code_sha256']:
        raise ValueError('source Probe model SHA mismatch')
    if old['split_hash']!=s['split_hash'] or old['gamma']!=s['gamma'] or old['control_step_sec']!=s['control_step_sec']:
        raise ValueError('split or timing config mismatch')
    features={}
    for split in ('train','val','test'):
        p=OLD/'splits'/f'{split}.npz'
        if sha(p)!=s['source_aligned_split_sha256'][split]:raise ValueError(f'{split} feature SHA mismatch')
        raw=np.load(p);features[split]={k:raw[k] for k in raw.files}
        if len(features[split]['index'])!=s['complete_aligned'][split]:raise ValueError(f'{split} aligned count mismatch')
        if not np.isfinite(features[split]['action']).all():raise ValueError(f'{split} nonfinite action')
    summary=json.loads((OLD/'output/relative-quality-extraction.json').read_text())
    if summary['aligned_primary_counts']!=s['complete_aligned'] or summary['terminal_endpoint_counts']!=s['terminal_endpoint_excluded']:
        raise ValueError('relative extraction count mismatch')
    if summary['reward_steps']!=s['reward_steps'] or summary['gamma']!=s['gamma'] or not summary['original_ecdf_exact_match']:
        raise ValueError('label/ECDF provenance mismatch')
    positions=Counter();terminal=Counter();episode_sets={split:set() for split in features}
    with gzip.open(OLD/'output/relative-quality-values.jsonl.gz','rt') as source:
        for line in source:
            row=json.loads(line);split=row['split']
            if row['primary_available']:
                pos=positions[split]
                validate_aligned_record(row,features[split],pos,split)
                positions[split]+=1;episode_sets[split].add(tuple(row['episode_key']))
            else:terminal[split]+=1
    if dict(positions)!=s['complete_aligned'] or {x:int(terminal[x]) for x in features}!=s['terminal_endpoint_excluded']:
        raise ValueError('aligned or terminal count mismatch after record scan')
    if any(episode_sets[a]&episode_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('episode leakage')
    prior=json.loads((OLD/'output/relative-quality-probe-training-summary.json').read_text())
    if not prior['completed'] or prior['input_fingerprint']['train']!=s['source_aligned_split_sha256']['train'] or prior['input_fingerprint']['val']!=s['source_aligned_split_sha256']['val']:
        raise ValueError('prior Probe dataset mismatch')
    if prior['input_fingerprint']['probe_code']!=s['source_relative_probe_code_sha256'] or prior['input_fingerprint']['config']!=s['source_relative_config_sha256']:
        raise ValueError('prior Probe model/config mismatch')
    if sorted(set(r['seed'] for r in prior['runs']))!=s['probe_seeds']:
        raise ValueError('prior Probe seeds mismatch')
    bad=[]
    for run in prior['runs']:
        if sha(run['checkpoint'])!=run['checkpoint_sha256']:bad.append(run['checkpoint'])
    if bad:raise ValueError(f'prior Probe checkpoint SHA mismatch: {bad[:2]}')
    ckpt=torch.load(BASE/'pilot-quality-best.pt',map_location='cpu',weights_only=True)
    if ckpt['split_hash']!=s['split_hash'] or ckpt['action_statistics']['count']<=0:
        raise ValueError('train-only action scale mismatch')
    result={'status':'pass','aligned_primary_counts':dict(positions),'terminal_endpoint_excluded':dict(terminal),
            'episode_counts':{k:len(v) for k,v in episode_sets.items()},'episode_leakage':False,
            'previous_probe_checkpoints_verified':len(prior['runs']),
            'original_ecdf_exact_match':True,'raw_label_and_z_reproduced_count':sum(positions.values()),
            'train_only_action_statistics':ckpt['action_statistics'],
            'train_only_relative_scalers':summary['train_only_scalers'],
            'reward_steps':s['reward_steps'],'gamma':s['gamma'],
            'timing_semantics':'pre-action Critic; post-step pose/reward; anchor uses next-record Critic',
            'raw_record_sha256':s['source_relative_values_sha256']}
    write('relative-label-audit.json',result)
    print('FOLLOWUP_AUDIT',sum(positions.values()),len(prior['runs']),flush=True)


if __name__=='__main__':main()
