"""Docker-side stages for frozen, post-hoc relative EAT trajectory labels."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import _build_window,_records,assign_terrain_splits
from visualnav_transformer.eat_nav.quality import fit_train_statistics
from visualnav_transformer.eat_nav.quality_gate_data import trajectory_shape
from visualnav_transformer.eat_nav.relative_quality import aligned_endpoint,compute_relative_window,fit_train_scalers,normalize_labels

BASE=Path('/tmp/eatnav-train')
ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
OUT=ROOT/'output'
MANIFEST=Path('/tmp/eatnav-extended/output/manifest.json')
QUALITY=BASE/'quality.yaml'
SPEC=BASE/'relative-quality-config.yaml'
RL=Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
COMPLETE_IDS=BASE/'relative-complete-window-identities.jsonl.gz'
GATE_CHECKPOINT=BASE/'pilot-quality-best.pt'
METRICS=('v0','advantage_h','drawdown','recovery','delta_v','return_h','critic_slope_per_sec')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def inputs():
    s=yaml.safe_load(SPEC.read_text())
    for path,key in [(MANIFEST,'manifest_sha256'),(QUALITY,'quality_config_sha256'),(RL,'rl_checkpoint_sha256')]:
        if sha(path)!=s[key]:raise ValueError(f'{key} mismatch')
    checkpoint=torch.load(GATE_CHECKPOINT,map_location='cpu',weights_only=True)
    if checkpoint['split_hash']!=s['split_hash']:raise ValueError('split hash mismatch')
    saved=yaml.safe_load((RL.parent/'params'/'agent.yaml').read_text())
    if not math.isclose(float(saved['algorithm']['gamma']),s['gamma'],rel_tol=0,abs_tol=1e-12):raise ValueError('saved gamma mismatch')
    env=yaml.load((RL.parent/'params'/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    if not math.isclose(float(env['sim']['dt'])*int(env['decimation']),s['control_step_sec'],rel_tol=0,abs_tol=1e-12):
        raise ValueError('saved control step mismatch')
    if str(env['is_finite_horizon']).lower()!='false':raise ValueError('timeout semantics mismatch')
    return s,load_eat_config(QUALITY),checkpoint


def reference_identities():
    identities={}
    with gzip.open(COMPLETE_IDS,'rt') as f:
        for line in f:
            item=json.loads(line)
            key=(item['split'],int(item['window_index']))
            if key in identities:raise ValueError('duplicate frozen complete identity')
            identities[key]=(tuple(item['episode_key']),int(item['step']),int(item['terrain_seed']))
    return identities


def window_record(w,source,key,split,index,s,ecdf):
    i=int(w['local_step'])
    end=aligned_endpoint(source,i,s['horizon_sec'],.02)
    label=compute_relative_window(source,i,end,gamma=s['gamma'],step_sec=s['control_step_sec'])
    if label['reward_steps']!=40:raise ValueError('incorrect horizon reward count')
    if label['end_timestamp_ns']-label['start_timestamp_ns']!=2_000_000_000:raise ValueError('incorrect physical horizon')
    xy,progress,path,turn=trajectory_shape(w['action'])
    valid=w['dem'][:,1]>.5;height=w['dem'][:,0]
    hmean=float(height[valid].mean());hstd=float(height[valid].std())
    q=ecdf.transform(w['quality_raw'])
    return {'split':split,'window_index':index,'episode_key':list(key),'step':int(w['step']),
            'terrain_seed':int(w['terrain_seed']),'terrain_type':w['terrain_type'],
            'episode_terminal_reason':list(source[-1]['termination_reason']),
            'terminal_seconds':(source[-1]['timestamp_ns']-source[i]['timestamp_ns'])/1e9,
            'goal_distance_m':float(np.linalg.norm(w['goal'])),
            'dem_height_std_m':hstd,'dem_mask_fraction':float(valid.mean()),
            'forward_progress_m':progress,'path_length_m':path,'turning_rad':turn,
            'trajectory_endpoint_m':float(np.linalg.norm(xy[-1])),
            'quality_p10_raw':float(w['quality_raw']),'quality_abs':q,
            **label}


def smoke():
    start=time.monotonic();s,c,_=inputs();episodes=_records(MANIFEST);split_map=assign_terrain_splits([r[0]['terrain_seed'] for r in episodes.values()],123)
    rows=[];bad=Counter()
    for key,source in sorted(episodes.items()):
        for i,r in enumerate(source):
            if r['dem'] is None:continue
            w=_build_window(key,source,i,c)
            if w is None or not w['action_mask'].all():continue
            end=aligned_endpoint(source,i,s['horizon_sec'],.02)
            label=compute_relative_window(source,i,end,gamma=s['gamma'],step_sec=s['control_step_sec'])
            rows.append({'episode_key':list(key),'step':r['step'],'split':split_map[r['terrain_seed']],
                         'start_timestamp_ns':label['start_timestamp_ns'],'end_timestamp_ns':label['end_timestamp_ns'],
                         'reward_steps':label['reward_steps'],'primary_available':label['primary_available'],
                         'bootstrap_kind':label['bootstrap_kind'],'advantage_h':label['advantage_h']})
            bad[label['bootstrap_kind']]+=1
            if len(rows)==100:break
        if len(rows)==100:break
    if len(rows)!=100:raise ValueError('insufficient smoke windows')
    result={'windows':len(rows),'seconds':time.monotonic()-start,'bootstrap_counts':dict(bad),'sample':rows[:5],
            'host_memory_note':'raw episodes and DEM windows loaded in CPU; no GPU needed for extraction'}
    write(ROOT/'smoke.json',result)
    print('RELATIVE_SMOKE',json.dumps({k:result[k] for k in ('windows','seconds','bootstrap_counts')}),flush=True)


def extract():
    start=time.monotonic();s,c,checkpoint=inputs();episodes=_records(MANIFEST)
    split_map=assign_terrain_splits([r[0]['terrain_seed'] for r in episodes.values()],123)
    grouped=defaultdict(list);all_train=[];counters=Counter()
    for key,source in sorted(episodes.items()):
        split=split_map[source[0]['terrain_seed']]
        for i,r in enumerate(source):
            if r['dem'] is None:continue
            w=_build_window(key,source,i,c)
            if w is None:continue
            if split=='train':all_train.append(w)
            index=counters[(split,'all')];counters[(split,'all')]+=1
            if w['action_mask'].all():grouped[split].append((index,key,source,w))
    counts={split:len(grouped[split]) for split in ('train','val','test')}
    if counts!=s['expected_complete']:raise ValueError(f'complete window count mismatch: {counts}')
    stats,ecdf=fit_train_statistics([dict(w,critic_raw=w['critic_raw'][w['critic_mask']]) for w in all_train])
    if list(ecdf.values)!=list(checkpoint['quality_ecdf']):raise ValueError('original train ECDF mismatch')
    original=reference_identities()
    if len(original)!=sum(counts.values()):raise ValueError('frozen complete identity count mismatch')
    all_records=defaultdict(list);features=defaultdict(lambda:defaultdict(list));missing=Counter();q_mismatch=0
    for split in ('train','val','test'):
        for index,key,source,w in grouped[split]:
            if original[(split,index)]!=(key,int(w['step']),int(w['terrain_seed'])):
                raise ValueError('frozen window identity mismatch')
            record=window_record(w,source,key,split,index,s,ecdf)
            if not record['primary_available']:missing[(split,record['bootstrap_kind'])]+=1
            all_records[split].append(record)
            if record['primary_available']:
                f=features[split]
                for name,value in [('dem',w['dem']),('goal',w['goal']),('action',w['action']),
                                   ('index',index),('episode',key),('terrain',w['terrain_type']),
                                   ('quality_abs',record['quality_abs'])]:f[name].append(value)
                for name in METRICS:f[name].append(record[name])
    expected={'train':22,'val':7,'test':0}
    bysplit={name:sum(n for (split,_),n in missing.items() if split==name) for name in expected}
    if bysplit!=s['expected_terminal_endpoint']:raise ValueError(f'terminal endpoint count mismatch: {bysplit}')
    train=all_records['train'];usable=[row for row in train if row['primary_available']]
    scalers=fit_train_scalers(usable,METRICS)
    (ROOT/'splits').mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    with gzip.open(OUT/'relative-quality-values.jsonl.gz.tmp','wt') as f:
        for split in ('train','val','test'):
            fdict=features[split]
            normalized={name:[] for name in METRICS}
            for row in all_records[split]:
                row['relative_z']=normalize_labels(row,scalers) if row['primary_available'] else None
                f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                if row['primary_available']:
                    for name in METRICS:normalized[name].append(row['relative_z'][name])
            payload={name:np.asarray(value) for name,value in fdict.items()}
            for name,value in normalized.items():payload['z_'+name]=np.asarray(value,dtype=np.float32)
            np.savez_compressed(ROOT/'splits'/f'{split}.npz',**payload)
    (OUT/'relative-quality-values.jsonl.gz.tmp').replace(OUT/'relative-quality-values.jsonl.gz')
    result={'complete_counts':counts,'aligned_primary_counts':{name:len(features[name]['index']) for name in counts},
            'terminal_endpoint_counts':bysplit,'terminal_endpoint_kinds':{f'{k[0]}:{k[1]}':v for k,v in missing.items()},
            'reward_steps':40,'gamma':s['gamma'],'train_only_scalers':scalers,'original_ecdf_exact_match':True,
            'frozen_identity_match':len(original),'elapsed_seconds':time.monotonic()-start,
            'feature_npz_sha256':{name:sha(ROOT/'splits'/f'{name}.npz') for name in counts},
            'raw_values_sha256':sha(OUT/'relative-quality-values.jsonl.gz')}
    write(OUT/'relative-quality-extraction.json',result)
    print('RELATIVE_EXTRACT',json.dumps({'complete':counts,'primary':result['aligned_primary_counts'],
           'terminal':bysplit,'seconds':result['elapsed_seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','extract'],required=True)
    a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    {'smoke':smoke,'extract':extract}[a.stage]()

if __name__=='__main__':main()
