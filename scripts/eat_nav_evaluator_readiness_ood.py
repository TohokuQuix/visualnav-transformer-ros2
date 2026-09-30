"""Apply existing complete-only train nearest-action OOD screen to P/C/R."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

TRAIN_ACTION_SHA = 'b9df441243cb5b960d911fc3399852cb635ef846730e26c664a87b6c0a8d11fc'
LIMIT = .3605285808444023


def main():
    p=argparse.ArgumentParser()
    for name in ('paths','manifest','reference-config','reference-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    rows=json.loads(a.paths.read_text())
    if len(rows)!=98:raise ValueError('98 frozen path rows required')
    cfg=load_eat_config(a.reference_config)
    view=CompleteOnlyView(EATDataset(a.manifest,'train',cfg,split_seed=123),8)
    if len(view)!=49598:raise ValueError('complete train count changed')
    actions=np.stack([view[i]['action'] for i in range(len(view))]).astype(np.float32)
    if hashlib.sha256(actions.tobytes()).hexdigest()!=TRAIN_ACTION_SHA:
        raise ValueError('train action SHA changed')
    model,_,_=load_checkpoint(a.reference_checkpoint,cfg)
    mean=model.action_mean.detach().cpu().numpy();std=model.action_std.detach().cpu().numpy()
    variants=[]
    for row in rows:
        for variant in ('P','C','R'):
            path=np.asarray(row['paths'][variant],dtype=np.float32)
            if path.shape!=(8,2) or not np.isfinite(path).all():raise ValueError('candidate OOD path invalid')
            delta=np.diff(np.vstack((np.zeros((1,2),dtype=np.float32),path)),axis=0)
            variants.append((row,variant,delta))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=torch.as_tensor(((actions-mean)/std).reshape(len(actions),-1),device=device)
    query=torch.as_tensor(np.stack([((delta-mean)/std).reshape(-1) for _,_,delta in variants]),device=device)
    distances=[]
    for batch in query.split(32):
        distances.extend((torch.cdist(batch,train).min(dim=1).values/np.sqrt(16)).cpu().numpy().tolist())
    result={'source_split':'complete_only_train','train_window_count':49598,
            'train_action_sha256':TRAIN_ACTION_SHA,'frozen_train_p95_limit':LIMIT,
            'checkpoint_sha256':hashlib.sha256(a.reference_checkpoint.read_bytes()).hexdigest(),
            'paths_sha256':hashlib.sha256(a.paths.read_bytes()).hexdigest(),
            'rows':[{'run_id':row['run_id'],'state_id':row['state_id'],'candidate_id':row['candidate_id'],
                     'variant':variant,'nearest_train_action_distance':float(distance),
                     'over_frozen_train_p95':bool(distance>LIMIT)}
                    for (row,variant,_),distance in zip(variants,distances)]}
    if len(result['rows'])!=294:raise ValueError('OOD result count changed')
    content=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.output.exists():
        if a.output.read_text()!=content:raise FileExistsError('OOD output changed')
    else:
        tmp=a.output.with_suffix(a.output.suffix+'.tmp');tmp.write_text(content);tmp.replace(a.output)
    print('READINESS_OOD_COMPLETE',len(result['rows']),flush=True)


if __name__=='__main__':main()
