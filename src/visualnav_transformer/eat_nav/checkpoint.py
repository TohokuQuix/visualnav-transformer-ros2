"""Strict EAT checkpoint metadata and train-only normalization artifacts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .config import EATConfig,assert_checkpoint_compatible
from .model import EATNav
from .quality import ActionStatistics,EmpiricalCDF


def save_checkpoint(model,optimizer,stats,ecdf,config:EATConfig,split_hash,path,*,epoch=0,best_metric=None,ema_model=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state={'metadata':config.checkpoint_metadata(), 'split_hash':str(split_hash),
           'model_state_dict':model.state_dict(),
           'ema_state_dict':(ema_model or model).state_dict(),
           'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
           'action_statistics':{'mean':np.asarray(stats.mean).tolist(),'std':np.asarray(stats.std).tolist(),'count':int(stats.count)},
           'quality_ecdf':list(ecdf.values),'epoch':int(epoch),'best_metric':best_metric}
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temp)
    temp.replace(path)


def load_checkpoint(path,config:EATConfig):
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert_checkpoint_compatible(config,data['metadata'])
    raw=data['action_statistics']
    stats=ActionStatistics(np.asarray(raw['mean'],dtype=np.float32),np.asarray(raw['std'],dtype=np.float32),int(raw['count']))
    ecdf=EmpiricalCDF(tuple(float(v) for v in data['quality_ecdf']))
    model=EATNav(config)
    which='ema_state_dict' if config.best_checkpoint_weights=='ema' else 'model_state_dict'
    model.load_state_dict(data[which],strict=True)
    model.set_action_statistics(stats.mean,stats.std)
    model.eval()
    return model,stats,ecdf
