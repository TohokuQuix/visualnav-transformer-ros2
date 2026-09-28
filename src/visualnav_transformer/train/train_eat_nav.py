"""Small or full EAT scratch training with validation-selected EMA checkpoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

from visualnav_transformer.eat_nav.checkpoint import save_checkpoint
from visualnav_transformer.eat_nav.config import assert_checkpoint_compatible,load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.evaluation import evaluate_batch
from visualnav_transformer.eat_nav.model import EATNav


@dataclass
class EarlyStopper:
    mode: str='min'
    patience: int=10
    min_delta: float=0.
    best: float|None=None
    best_epoch: int|None=None
    bad_epochs: int=0

    def update(self,value,epoch):
        improved=self.best is None or (value<self.best-self.min_delta if self.mode=='min' else value>self.best+self.min_delta)
        if improved:
            self.best=float(value)
            self.best_epoch=int(epoch)
            self.bad_epochs=0
        else:
            self.bad_epochs+=1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs>=self.patience


def train_batch(model,optimizer,batch,ema_model=None,ema_decay=.995):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dem,goal,quality,action,mask=(batch[k] for k in ('dem','goal','quality','action','action_mask'))
    context=model.encode(dem,goal,quality)
    loss=model.noise_loss(action,mask,context)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    if ema_model is not None:
        with torch.no_grad():
            for ema_param,param in zip(ema_model.parameters(),model.parameters()):
                ema_param.lerp_(param,1-ema_decay)
            for ema_buffer,buffer in zip(ema_model.buffers(),model.buffers()):
                ema_buffer.copy_(buffer)
    return float(loss.detach())


class _TrainingView(Dataset):
    def __init__(self,dataset:EATDataset,limit=None,seed=123):
        self.dataset=dataset
        self.indices=list(range(len(dataset)))
        if limit is not None and len(self.indices)>limit:
            groups={b:[] for b in range(4)}
            for i,w in enumerate(dataset.windows):
                b=min(int(dataset.ecdf.transform(w['quality_raw'])*4),3)
                groups[b].append(i)
            rng=np.random.default_rng(seed)
            selected=[]
            for group in groups.values():
                quota=min(len(group),max(1,limit//4))
                selected.extend(rng.choice(group,size=quota,replace=False).tolist())
            rest=sorted(set(self.indices)-set(selected))
            if len(selected)<limit:
                selected.extend(rng.choice(rest,size=limit-len(selected),replace=False).tolist())
            self.indices=sorted(selected[:limit])

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        w=self.dataset[self.indices[index]]
        return {'dem':torch.from_numpy(w['dem']),'goal':torch.from_numpy(w['goal']),
                'quality':torch.tensor([w['quality']],dtype=torch.float32),
                'action':torch.from_numpy(w['action']),
                'action_mask':torch.from_numpy(w['action_mask'])}


def _on_device(batch,device):
    return {k:v.to(device) for k,v in batch.items()}


def _validation_loss(model,loader,device):
    was_training=model.training
    model.eval()
    torch.manual_seed(41237)
    values=[]
    with torch.no_grad():
        for batch in loader:
            batch=_on_device(batch,device)
            context=model.encode(batch['dem'],batch['goal'],batch['quality'])
            values.append(float(model.noise_loss(batch['action'],batch['action_mask'],context)))
    model.train(was_training)
    return float(np.mean(values))


def train(config_path,manifest,output_dir,*,batch_size=64,learning_rate=1e-4,train_limit=None,val_limit=None,device=None,resume=None):
    config=load_eat_config(config_path)
    if config.validation_metric not in ('diffusion_loss','trajectory_error'):
        raise ValueError('validation_metric must be diffusion_loss or trajectory_error')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_data=EATDataset(Path(manifest),'train',config)
    val_data=EATDataset(Path(manifest),'val',config)
    train_view=_TrainingView(train_data,train_limit)
    val_view=_TrainingView(val_data,val_limit)
    if not len(train_view) or not len(val_view):raise ValueError('train/validation split is empty')
    train_loader=DataLoader(train_view,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_view,batch_size=batch_size,shuffle=False,num_workers=0)
    split_hash=hashlib.sha256(json.dumps(train_data.split_map,sort_keys=True).encode()).hexdigest()
    model=EATNav(config).to(device)
    model.set_action_statistics(train_data.stats.mean,train_data.stats.std)
    ema=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    stopper=EarlyStopper(config.best_checkpoint_metric_mode,config.early_stopping_patience,config.early_stopping_min_delta)
    start_epoch=1
    if resume is not None:
        payload=torch.load(resume,map_location='cpu',weights_only=True)
        assert_checkpoint_compatible(config,payload['metadata'])
        if payload['split_hash']!=split_hash:raise ValueError('checkpoint split hash mismatch')
        model.load_state_dict(payload['model_state_dict'])
        ema.load_state_dict(payload['ema_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        start_epoch=payload['epoch']+1
        stopper.best=payload['best_metric']
        stopper.best_epoch=payload['epoch']
    history=[]
    for epoch in range(start_epoch,config.max_epochs+1):
        losses=[]
        for batch in train_loader:
            batch=_on_device(batch,device)
            if not batch['action_mask'].any():continue
            losses.append(train_batch(model,optimizer,batch,ema))
        if not losses:raise ValueError('no valid training actions')
        if epoch%config.validation_interval_epochs:
            continue
        evaluated=ema if config.best_checkpoint_weights=='ema' else model
        if config.validation_metric=='diffusion_loss':
            val_metric=_validation_loss(evaluated,val_loader,device)
        else:
            val_metric=float(np.mean([evaluate_batch(evaluated,_on_device(b,device),seed=123)['mean_xy_error_m'] for b in val_loader]))
        improved=stopper.update(val_metric,epoch)
        item={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_metric':val_metric,'best':improved}
        history.append(item)
        if improved:
            save_checkpoint(model,optimizer,train_data.stats,train_data.ecdf,config,split_hash,output_dir/'best.pt',
                            epoch=epoch,best_metric=val_metric,ema_model=ema)
        (output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('EAT_EPOCH',json.dumps(item),flush=True)
        if stopper.should_stop:break
    return {'best_checkpoint':str(output_dir/'best.pt'),'best_metric':stopper.best,'best_epoch':stopper.best_epoch,
            'epochs_ran':len(history),'train_windows':len(train_view),'validation_windows':len(val_view),'split_hash':split_hash}
