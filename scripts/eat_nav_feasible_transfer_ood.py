"""Frozen train-only normalized action distance for planned, command, and realized paths."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'feasible-transfer'
BASE=ROOT/'baseline-worktree-8e8f4fd'

def run():
    started=time.time()
    thresholds=json.loads((BASE/'docs/quality-identifiability-gate/phase-c-remediation/candidate-thresholds.json').read_text())
    mean=np.asarray(thresholds['train_action_mean'],dtype=np.float32)
    std=np.asarray(thresholds['train_action_std'],dtype=np.float32)
    limit=float(thresholds['train_ood_nearest_l2_per_dimension_p95'])
    records=json.loads((OUT/'feasibility-by-candidate.json').read_text())['rows']
    if len(records)!=91:raise ValueError('candidate count changed')
    cfg=load_eat_config(BASE/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    view=CompleteOnlyView(EATDataset(Path('/tmp/eatnav-extended/output/manifest.json'),'train',cfg,split_seed=123),8)
    train=np.stack([view[i]['action'] for i in range(len(view))]).astype(np.float32)
    if len(train)!=49598:raise ValueError('train count changed')
    train_tensor=torch.as_tensor(((train-mean)/std).reshape(len(train),-1),device='cuda')
    rows=[]
    for variant,key in [('P','planned_path'),('C','command_derived_path'),('R','realized_path')]:
        paths=np.asarray([r[key] for r in records],dtype=np.float32)
        increments=np.diff(np.concatenate((np.zeros((len(paths),1,2),dtype=np.float32),paths),axis=1),axis=1)
        query=torch.as_tensor(((increments-mean)/std).reshape(len(paths),-1),device='cuda')
        distance=(torch.cdist(query,train_tensor).min(dim=1).values/np.sqrt(query.shape[1])).cpu().numpy()
        for source,value in zip(records,distance):
            rows.append({'state_id':source['state_id'],'terrain':source['terrain'],
                'candidate_id':source['candidate_id'],'generator':source['generator'],'seed':source['seed'],
                'variant':variant,'nearest_train_action_distance':float(value),
                'over_frozen_train_p95':bool(value>limit)})
    summary={variant:{'median':float(np.median([r['nearest_train_action_distance'] for r in rows if r['variant']==variant])),
        'p90':float(np.quantile([r['nearest_train_action_distance'] for r in rows if r['variant']==variant],.9)),
        'over_p95_count':sum(r['over_frozen_train_p95'] for r in rows if r['variant']==variant)} for variant in ('P','C','R')}
    result={'source_split':'complete_only_train','train_window_count':len(train),
        'frozen_train_p95_limit':limit,'rows':rows,'summary':summary,
        'elapsed_sec':time.time()-started,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated())}
    output=OUT/'trajectory-ood-comparison.json';content=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if output.exists():
        if output.read_text()!=content:raise FileExistsError('OOD result differs')
    else:
        temp=output.with_name(output.name+'.tmp');temp.write_text(content);temp.replace(output)
    print('FEASIBLE_TRANSFER_OOD_COMPLETE',summary,'sec',round(time.time()-started,1),flush=True)

if __name__=='__main__':run()
