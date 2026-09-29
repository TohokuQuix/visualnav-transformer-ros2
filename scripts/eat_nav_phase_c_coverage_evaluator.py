"""Frozen Phase B GT ensemble predictions for diagnostic trajectories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from eat_nav_forward_evaluator import Evaluator, HEADS
from eat_nav_phase_c_coverage import digest

CHECKPOINT_ROOT=Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints')
SCALER_PATH=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/output/relative-quality-extraction.json')
SEEDS=(11,29,47)

class FrozenEvaluator:
    def __init__(self,device='cuda'):
        self.device=device
        self.models=[]
        self.checkpoint_sha256={}
        for seed in SEEDS:
            path=CHECKPOINT_ROOT/f'GT-{seed}'/'best.pt'
            saved=torch.load(path,map_location=device,weights_only=True)
            if saved['variant']!='GT' or saved['seed']!=seed:raise ValueError('GT checkpoint mismatch')
            model=Evaluator(True,False).to(device).eval()
            model.load_state_dict(saved['model'])
            self.models.append(model)
            self.checkpoint_sha256[str(seed)]=digest(path)
        self.scalers=json.loads(SCALER_PATH.read_text())['train_only_scalers']
        self.scaler_sha256=digest(SCALER_PATH)

    @torch.no_grad()
    def predict(self,dem,goal,paths):
        p=np.asarray(paths,dtype=np.float32)
        if p.ndim!=3 or p.shape[1:]!=(8,2):raise ValueError('Path shape mismatch')
        delta=np.diff(np.concatenate((np.zeros((len(p),1,2),dtype=np.float32),p),axis=1),axis=1)
        d=np.broadcast_to(np.asarray(dem,dtype=np.float32).reshape(1,8,20,20),(len(p),8,20,20)).copy()
        g=np.broadcast_to(np.asarray(goal,dtype=np.float32).reshape(1,2),(len(p),2)).copy()
        x=torch.from_numpy(d).to(self.device);y=torch.from_numpy(g).to(self.device)
        a=torch.from_numpy(delta.reshape(len(p),16)).to(self.device)
        zero=torch.zeros((len(p),1),device=self.device)
        z=np.stack([m(x,y,a,zero).cpu().numpy() for m in self.models])
        means=np.asarray([self.scalers[h]['mean'] for h in HEADS])
        stds=np.asarray([self.scalers[h]['std'] for h in HEADS])
        return z*stds+means


def score_spread(predictions):
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=3 or p.shape[0]!=3 or p.shape[2]!=4:raise ValueError('prediction shape mismatch')
    mean=p.mean(0)
    return {'heads':{h:{'candidate_range':float(np.ptp(mean[:,i])),
                       'ensemble_disagreement_median':float(np.median(np.std(p[:,:,i],axis=0)))} for i,h in enumerate(HEADS)},
            'all_same_score':bool(np.max(np.ptp(mean,axis=0))<1e-6),
            'candidate_over_uncertainty_fraction':float(np.mean(np.ptp(mean,axis=0)>np.median(np.std(p,axis=0),axis=0)))}
