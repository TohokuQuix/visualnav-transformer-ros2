"""Train-only transforms and the approved masked, seed-balanced risk model."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NORMALIZATION_SCHEMA='eatnav_fall_stall_normalization_v1'
MODEL_SCHEMA='eatnav_fall_stall_model_v1'


def indices_for(arrays,indices):
    selected=np.asarray(indices,dtype=np.int64)
    if selected.ndim!=1 or not len(selected) or len(set(selected.tolist()))!=len(selected) or \
            selected.min()<0 or selected.max()>=len(arrays['dem']):
        raise ValueError('nonempty unique in-range sample indices required')
    return selected


def fit_normalization(arrays,train_indices):
    selected=indices_for(arrays,train_indices)
    dem=arrays['dem'][selected];mask=dem[:,:,1]==1
    heights=dem[:,:,0][mask].astype(np.float64)
    states=np.concatenate((arrays['actor_state'][selected],arrays['body_state'][selected]),axis=1).astype(np.float64)
    active=arrays['candidate_mask'][selected]
    paths=arrays['paths'][selected][active].astype(np.float64).reshape(-1,2)
    if not len(heights) or not len(paths):
        raise ValueError('train normalization has no valid inputs')
    means,stds,counts=[],[],[]
    for h in range(4):
        valid=arrays['progress_mask'][selected,:,h]&active
        values=arrays['progress_targets'][selected,:,h][valid].astype(np.float64)
        counts.append(int(len(values)));means.append(float(values.mean()) if len(values) else 0.)
        stds.append(max(float(values.std()),.001) if len(values) else 1.)
    return {'schema':NORMALIZATION_SCHEMA,'source':'train_split_only','train_indices':selected.tolist(),
            'dem_mean':float(heights.mean()),'dem_std':max(float(heights.std()),.001),
            'state_mean':states.mean(0).tolist(),'state_std':np.maximum(states.std(0),.001).tolist(),
            'path_mean':paths.mean(0).tolist(),'path_std':np.maximum(paths.std(0),.001).tolist(),
            'progress_mean':means,'progress_std':stds,'progress_count':counts}


def prepare_inputs(arrays,indices,stats):
    if stats.get('schema')!=NORMALIZATION_SCHEMA or stats.get('source')!='train_split_only':
        raise ValueError('normalization schema/source mismatch')
    selected=indices_for(arrays,indices)
    dem=arrays['dem'][selected].copy();mask=dem[:,:,1]
    dem[:,:,0]=np.where(mask==1,(dem[:,:,0]-stats['dem_mean'])/stats['dem_std'],0)
    state=np.concatenate((arrays['actor_state'][selected],arrays['body_state'][selected]),axis=1)
    state=(state-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    paths=(arrays['paths'][selected]-np.asarray(stats['path_mean']))/np.asarray(stats['path_std'])
    result={'dem':torch.from_numpy(dem.reshape(len(selected),8,20,20)),
            'state':torch.from_numpy(state.astype(np.float32)),
            'paths':torch.from_numpy(paths.astype(np.float32)),
            'directions':torch.from_numpy(arrays['directions'][selected].copy()),
            'candidate_mask':torch.from_numpy(arrays['candidate_mask'][selected].copy())}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite normalized model inputs')
    return result


def training_support(arrays,train_indices):
    selected=indices_for(arrays,train_indices)
    labels=arrays['event_targets'][selected,:,3,:]
    valid=arrays['event_mask'][selected,:,3,:]&arrays['candidate_mask'][selected,:,None]
    counts=[];active=[];weights=[]
    for head in range(4):
        positive=int(((labels[:,:,head]==1)&valid[:,:,head]).sum())
        negative=int(((labels[:,:,head]==0)&valid[:,:,head]).sum())
        unknown=int((~valid[:,:,head]&arrays['candidate_mask'][selected]).sum())
        counts.append({'positive':positive,'negative':negative,'unknown':unknown})
        supported=positive>0 and negative>0;active.append(supported)
        weights.append(min(negative/positive,20.) if supported else 1.)
    return {'active':active,'pos_weight':weights,'counts':counts}


class RiskEvaluator(nn.Module):
    def __init__(self,use_dem=True):
        super().__init__()
        self.use_dem=bool(use_dem)
        self.dem_encoder=nn.Sequential(
            nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Linear(32,64),nn.ReLU()) if self.use_dem else None
        self.state_encoder=nn.Sequential(nn.Linear(31,32),nn.ReLU())
        self.path_encoder=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(128 if self.use_dem else 64,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.risk_head=nn.Linear(32,4);self.progress_head=nn.Linear(32,4)

    def forward(self,dem,state,paths,directions,candidate_mask):
        batch,candidates=paths.shape[:2]
        if candidates<1 or paths.shape!=(batch,candidates,8,2) or directions.shape!=(batch,candidates,1) or \
                state.shape!=(batch,31) or dem.shape!=(batch,8,20,20) or candidate_mask.shape!=(batch,candidates):
            raise ValueError('risk model input shape mismatch')
        if not candidate_mask.any(dim=1).all():
            raise ValueError('all candidates invalid in model input')
        shared=self.state_encoder(state)
        if self.use_dem:
            shared=torch.cat((self.dem_encoder(dem),shared),dim=-1)
        candidate=torch.cat((paths.reshape(batch,candidates,16),directions),dim=-1)
        path=self.path_encoder(candidate)
        features=self.fusion(torch.cat((shared[:,None].expand(-1,candidates,-1),path),dim=-1))
        return {'risk_logits':self.risk_head(features),'progress_z':self.progress_head(features)}


def target_batch(arrays,indices):
    selected=indices_for(arrays,indices)
    return {'events':torch.from_numpy(arrays['event_targets'][selected,:,3,:].astype(np.float32)),
            'event_mask':torch.from_numpy(arrays['event_mask'][selected,:,3,:].copy()),
            'progress':torch.from_numpy(arrays['progress_targets'][selected].copy()),
            'progress_mask':torch.from_numpy(arrays['progress_mask'][selected].copy()),
            'candidate_mask':torch.from_numpy(arrays['candidate_mask'][selected].copy())}


def balanced_groups(values,valid,seeds):
    """Average groups within seed, then valid seeds, then valid heads."""
    seed_tensor=torch.as_tensor(seeds,dtype=torch.int64,device=values.device)
    _,inverse=torch.unique(seed_tensor,sorted=True,return_inverse=True)
    seed_count=int(inverse.max().item())+1
    sums=values.new_zeros((seed_count,values.shape[1])).index_add(0,inverse,values*valid)
    counts=values.new_zeros((seed_count,values.shape[1])).index_add(0,inverse,valid.to(values.dtype))
    per_seed=sums/counts.clamp_min(1)
    per_head=per_seed.sum(0)/(counts>0).sum(0).clamp_min(1)
    available=(counts>0).any(0)
    return (per_head*available).sum()/available.sum().clamp_min(1)


def balanced_mean(values,mask,seeds):
    counts=mask.sum(1)
    groups=(values*mask).sum(1)/counts.clamp_min(1)
    return balanced_groups(groups,counts>0,seeds)


def event_bce(logits,batch,seeds,active,pos_weight=None):
    active=torch.as_tensor(active,dtype=torch.bool,device=logits.device)
    mask=batch['event_mask']&batch['candidate_mask'][:,:,None]&active
    labels=torch.where(mask,batch['events'],0.)
    weight=None if pos_weight is None else torch.as_tensor(pos_weight,dtype=logits.dtype,device=logits.device)
    values=F.binary_cross_entropy_with_logits(logits,labels,pos_weight=weight,reduction='none')
    return balanced_mean(values,mask,seeds)


def objective(logits,progress_z,batch,seeds,active,pos_weight,stats):
    active_tensor=torch.as_tensor(active,dtype=torch.bool,device=logits.device)
    event=event_bce(logits,batch,seeds,active,pos_weight)
    mask=batch['event_mask']&batch['candidate_mask'][:,:,None]&active_tensor
    labels=batch['events']
    pairs=(labels[:,:,None,:]==0)&(labels[:,None,:,:]==1)&mask[:,:,None,:]&mask[:,None,:,:]
    pair_values=F.softplus(logits[:,:,None,:]-logits[:,None,:,:])
    counts=pairs.sum((1,2))
    per_group=(pair_values*pairs).sum((1,2))/counts.clamp_min(1)
    # No-pair groups contribute zero rather than disappearing from the average.
    pair=balanced_groups(per_group,active_tensor[None].expand(len(logits),-1),seeds)
    mean=torch.as_tensor(stats['progress_mean'],dtype=progress_z.dtype,device=progress_z.device)
    std=torch.as_tensor(stats['progress_std'],dtype=progress_z.dtype,device=progress_z.device)
    supported=torch.as_tensor(stats['progress_count'],device=progress_z.device)>0
    p_mask=batch['progress_mask']&batch['candidate_mask'][:,:,None]&supported
    target=torch.where(p_mask,(batch['progress']-mean)/std,0.)
    progress=balanced_mean(F.smooth_l1_loss(progress_z,target,beta=1.,reduction='none'),p_mask,seeds)
    return {'event':event,'progress':progress,'pair':pair,'total':event+.1*progress+.1*pair}
