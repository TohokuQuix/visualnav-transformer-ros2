"""Separate, command-free two-second endpoint and event model.

No legacy output is reinterpreted: only named feature layers may be transferred.
All normalization and target support are fitted from explicitly supplied train ids.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .moving_pose2s_geometry import (sixd_to_matrix, matrix_to_quat, matrix_to_rpy, quat_to_matrix)

INPUT_SCHEMA='eatnav_moving_pose2s_input_v1'
MODEL_SCHEMA='eatnav_moving_pose2s_model_v1'
CHECKPOINT_SCHEMA='eatnav_moving_pose2s_checkpoint_v1'
NORMALIZATION_SCHEMA='eatnav_moving_pose2s_normalization_v1'
PREDICTION_SCHEMA='eatnav_moving_pose2s_prediction_v1'
KEEP=tuple(i for i in range(31) if i not in (6,7,8))
EVENTS=('tip_over','near_fall','linear_stall','turn_stall')
FEATURE_PREFIXES=('dem_encoder.','state_encoder.','path_encoder.','fusion.')


def _ids(arrays,indices):
    a=np.arange(len(arrays['dem'])) if indices is None else np.asarray(indices,dtype=np.int64)
    if a.ndim!=1 or not len(a) or len(set(a.tolist()))!=len(a) or a.min()<0 or a.max()>=len(arrays['dem']):
        raise ValueError('nonempty unique in-range indices required')
    return a


def fit_normalization(arrays,train_indices):
    ids=_ids(arrays,train_indices);dem=arrays['dem'][ids];heights=dem[:,:,0][dem[:,:,1]==1].astype(np.float64)
    state=np.concatenate([arrays['actor_state'][ids],arrays['body_state'][ids]],axis=1)[:,KEEP].astype(np.float64)
    cm=arrays['candidate_mask'][ids].astype(bool);paths=arrays['paths'][ids][cm].reshape(-1,2).astype(np.float64)
    pm=arrays['pose_mask'][ids].astype(bool)&cm;pos=arrays['position_targets'][ids][pm].astype(np.float64)
    if not len(heights) or not len(paths) or not len(pos):raise ValueError('training normalization lacks valid observations/pose')
    if not all(np.isfinite(v).all() for v in (heights,state,paths,pos)):raise ValueError('nonfinite train statistics')
    return dict(schema=NORMALIZATION_SCHEMA,source='train_split_only',train_indices=ids.tolist(),
        dem_mean=float(heights.mean()),dem_std=max(float(heights.std()),.001),
        state_mean=state.mean(0).tolist(),state_std=np.maximum(state.std(0),.001).tolist(),
        path_mean=paths.mean(0).tolist(),path_std=np.maximum(paths.std(0),.001).tolist(),
        position_mean=pos.mean(0).tolist(),position_std=np.maximum(pos.std(0),[.05,.05,.02]).tolist(),state_keep_indices=list(KEEP))


def validate_raw_inputs(arrays):
    """Validate the complete raw input contract before selection or lossy transforms."""
    shapes={'dem':(4,2,20,20),'actor_state':(22,),'body_state':(9,)}
    if 'dem' not in arrays or np.ndim(arrays['dem'])!=5:
        raise ValueError('DEM shape mismatch')
    b=len(arrays['dem'])
    if b<1 or 'paths' not in arrays or np.ndim(arrays['paths'])!=4:
        raise ValueError('raw input batch/path shape mismatch')
    k=np.shape(arrays['paths'])[1]
    if k<1:raise ValueError('candidate count must be positive')
    shapes.update(paths=(k,8,2),directions=(k,1),candidate_mask=(k,))
    for key,tail in shapes.items():
        if key not in arrays or np.shape(arrays[key])!=(b,*tail):
            raise ValueError('raw input shape mismatch: '+key)
        value=np.asarray(arrays[key])
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('nonfinite/nonnumeric raw input: '+key)
    if not np.isin(arrays['dem'][:,:,1],[0,1]).all() or not np.isin(arrays['candidate_mask'],[0,1]).all():
        raise ValueError('raw input masks must be binary')
    cm=np.asarray(arrays['candidate_mask'],bool)
    if not cm.any(1).all():raise ValueError('all candidates invalid')
    if not np.isin(arrays['directions'][cm],[-1,1]).all():
        raise ValueError('direction must be +1/-1')


def prepare_inputs(arrays,indices,stats,device='cpu'):
    if stats.get('schema')!=NORMALIZATION_SCHEMA or stats.get('source')!='train_split_only' or stats.get('state_keep_indices')!=list(KEEP):
        raise ValueError('normalization schema/source mismatch')
    validate_raw_inputs(arrays)
    ids=_ids(arrays,indices);dem=np.asarray(arrays['dem'][ids],np.float32).copy();mask=dem[:,:,1]
    if dem.shape[1:]!=(4,2,20,20) or not np.isin(mask,[0,1]).all():raise ValueError('DEM shape/mask mismatch')
    rawstate=np.concatenate([arrays['actor_state'][ids],arrays['body_state'][ids]],axis=1)[:,KEEP]
    state=(rawstate-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    dem[:,:,0]=np.where(mask==1,(dem[:,:,0]-stats['dem_mean'])/stats['dem_std'],0)
    paths=(arrays['paths'][ids]-np.asarray(stats['path_mean']))/np.asarray(stats['path_std'])
    direction=np.asarray(arrays['directions'][ids],np.float32);cm=np.asarray(arrays['candidate_mask'][ids],bool)
    if not np.isin(direction[cm],[-1,1]).all():raise ValueError('direction must be +1/-1')
    values=dict(dem=dem.reshape(len(ids),8,20,20),state=state.astype(np.float32),paths=paths.astype(np.float32),directions=direction,candidate_mask=cm)
    if not all(np.isfinite(x).all() for x in values.values()):raise ValueError('nonfinite model input')
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}


class MovingPose2sEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dem_encoder=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,64),nn.ReLU())
        self.state_encoder=nn.Sequential(nn.Linear(28,32),nn.ReLU())
        self.path_encoder=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.position_head=nn.Linear(32,3);self.rotation_head=nn.Linear(32,6);self.event_head=nn.Linear(32,4)
        nn.init.zeros_(self.rotation_head.weight)
        with torch.no_grad():self.rotation_head.bias.copy_(torch.tensor([1.,0,0,0,1,0]))

    def forward(self,dem,state,paths,directions,candidate_mask):
        b,k=paths.shape[:2]
        if k<1 or paths.shape!=(b,k,8,2) or directions.shape!=(b,k,1) or state.shape!=(b,28) or dem.shape!=(b,8,20,20) or candidate_mask.shape!=(b,k):raise ValueError('moving_pose2s input shape mismatch')
        if not candidate_mask.any(1).all():raise ValueError('all candidates invalid')
        shared=torch.cat([self.dem_encoder(dem),self.state_encoder(state)],-1)
        candidate=self.path_encoder(torch.cat([paths.reshape(b,k,16),directions],-1))
        fused=self.fusion(torch.cat([shared[:,None].expand(-1,k,-1),candidate],-1))
        return dict(position_z=self.position_head(fused),rotation_6d=self.rotation_head(fused),event_logits=self.event_head(fused))


def transfer_features(model,payload):
    if payload.get('schema')!='eatnav_fall_stall_checkpoint_v2_command_free':raise ValueError('feature transfer requires exact legacy command-free schema')
    current=model.state_dict();source=payload['state_dict'];selected={}
    for key,value in current.items():
        if key.startswith(FEATURE_PREFIXES):
            if key not in source or source[key].shape!=value.shape or not torch.isfinite(source[key]).all():raise ValueError('feature key/shape mismatch: '+key)
            selected[key]=source[key]
    current.update(selected);model.load_state_dict(current,strict=True)
    return dict(transferred_keys=sorted(selected),excluded_heads=['risk_head','progress_head'])


def training_support(arrays,train_indices):
    ids=_ids(arrays,train_indices);labels=arrays['event_targets'][ids];valid=arrays['event_mask'][ids].astype(bool)&arrays['candidate_mask'][ids,:,None]
    counts=[];trained=[];constants=[]
    for h in range(4):
        y=labels[...,h];m=valid[...,h];p=int(((y==1)&m).sum());n=int(((y==0)&m).sum())
        if not np.isin(y[m],[0,1]).all():raise ValueError('event labels must be binary under mask')
        counts.append(dict(positive=p,negative=n,unknown=int((~m).sum())));trained.append(p>0 and n>0);constants.append((p+.5)/(p+n+1))
    return dict(trained=trained,counts=counts,constant_probability=constants,pos_weight=[1.]*4)


def torch_rotation(sixd):
    a,b=sixd[...,:3],sixd[...,3:];an=torch.linalg.vector_norm(a,dim=-1,keepdim=True)
    u=a/an.clamp_min(1e-6);orth=b-(u*b).sum(-1,keepdim=True)*u;bn=torch.linalg.vector_norm(orth,dim=-1,keepdim=True)
    v=orth/bn.clamp_min(1e-6);matrix=torch.stack([u,v,torch.linalg.cross(u,v,dim=-1)],-1)
    return matrix,(an.squeeze(-1)>=1e-6)&(bn.squeeze(-1)>=1e-6),(an.squeeze(-1)-1).square()+(bn.squeeze(-1)-1).square()


def _masked_mean(values,mask,weights):
    weighted=mask.to(values.dtype)*weights
    return torch.where(mask,values,torch.zeros_like(values)).mul(weights).sum()/weighted.sum().clamp_min(1e-12)


def objective(outputs,arrays,indices,stats,support,weights=None):
    ids=_ids(arrays,indices);device=outputs['position_z'].device
    make=lambda x,dtype=None:torch.as_tensor(x,device=device,dtype=dtype)
    cm=make(arrays['candidate_mask'][ids],torch.bool);pm=make(arrays['pose_mask'][ids],torch.bool)&cm
    p=make(np.nan_to_num(arrays['position_targets'][ids]),torch.float32)
    p=(p-make(stats['position_mean']))/make(stats['position_std'])
    rot=make(np.nan_to_num(arrays['rotation_targets'][ids]),torch.float32)
    component=isinstance(weights,dict)
    w=make(np.ones(len(ids)) if weights is None or component else weights,torch.float32)[:,None]
    def reduce(values,mask,name,head=None):
        if not component:return _masked_mean(values,mask,w)
        selected=weights[name] if head is None else weights[name][...,head]
        cw=make(selected,torch.float32)
        return (torch.where(mask,values,torch.zeros_like(values))*cw).sum()*float(weights.get('global_scale',1.))
    pos=reduce(F.smooth_l1_loss(outputs['position_z'],p,reduction='none',beta=1.).mean(-1),pm,'position')
    predmat,rvalid,reg=torch_rotation(outputs['rotation_6d']);gtmat,_,_=torch_rotation(rot)
    rotation=reduce(.25*(predmat-gtmat).square().sum((-1,-2)),pm,'position')
    regularizer=reduce(reg,cm,'regularizer')
    em=make(arrays['event_mask'][ids],torch.bool)&cm[...,None]&make(support['trained'],torch.bool)
    y=make(np.nan_to_num(arrays['event_targets'][ids]),torch.float32)
    bce=F.binary_cross_entropy_with_logits(outputs['event_logits'],y,reduction='none')
    perhead=[reduce(bce[...,h],em[...,h],'event',h) for h in range(4) if support['trained'][h]]
    event=torch.stack(perhead).mean() if perhead else outputs['event_logits'].sum()*0
    return dict(total=pos+.5*rotation+.5*event+.001*regularizer,position=pos,rotation=rotation,event=event,rotation_regularizer=regularizer)


def save_checkpoint(path,model,normalization,support,metadata):
    payload=dict(schema=CHECKPOINT_SCHEMA,model_schema=MODEL_SCHEMA,input_schema=INPUT_SCHEMA,prediction_schema=PREDICTION_SCHEMA,horizon_sec=2.,event_names=list(EVENTS),pose_frame='full_initial_body_x_forward_y_left_z_up',quaternion_order='wxyz',rotation_6d='first_two_columns_concatenated',pose_supervision='complete_2s_only',state_keep_indices=list(KEEP),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},normalization=copy.deepcopy(normalization),support=copy.deepcopy(support),metadata=copy.deepcopy(metadata),torch_version=str(torch.__version__))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temporary);temporary.replace(path)


def load_checkpoint(path,device='cpu'):
    p=torch.load(path,map_location='cpu',weights_only=True)
    contract={'quaternion_order':'wxyz','pose_frame':'full_initial_body_x_forward_y_left_z_up','rotation_6d':'first_two_columns_concatenated','prediction_schema':PREDICTION_SCHEMA,'pose_supervision':'complete_2s_only'}
    if any(p.get(k)!=v for k,v in contract.items()):raise ValueError('moving_pose2s coordinate/prediction contract mismatch')
    if p.get('schema')!=CHECKPOINT_SCHEMA or p.get('model_schema')!=MODEL_SCHEMA or p.get('horizon_sec')!=2. or p.get('input_schema')!=INPUT_SCHEMA or p.get('state_keep_indices')!=list(KEEP):raise ValueError('moving_pose2s checkpoint schema/horizon mismatch')
    if p.get('event_names')!=list(EVENTS) or p.get('normalization',{}).get('schema')!=NORMALIZATION_SCHEMA:raise ValueError('moving_pose2s event/normalization schema mismatch')
    model=MovingPose2sEvaluator();model.load_state_dict(p['state_dict'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite checkpoint weights')
    model.to(device).eval();return model,p


def predict(model,normalization,arrays,indices=None,support=None):
    ids=_ids(arrays,indices);device=next(model.parameters()).device;model.eval()
    transformed=prepare_inputs(arrays,ids,normalization,device)
    with torch.inference_mode():o={k:v.detach().cpu().numpy() for k,v in model(**transformed).items()}
    large_state=(transformed['state'].detach().abs().amax(-1)>10).cpu().numpy()
    large_dem=(transformed['dem'][:,0::2].detach().abs().flatten(1).amax(-1)>10).cpu().numpy()
    large_path=(transformed['paths'].detach().abs().flatten(2).amax(-1)>10).cpu().numpy()
    if any(not np.isfinite(v).all() for v in o.values()):raise ValueError('nonfinite network output')
    position=o['position_z']*np.asarray(normalization['position_std'])+np.asarray(normalization['position_mean'])
    rotation,valid=sixd_to_matrix(o['rotation_6d']);q=matrix_to_quat(rotation);rpy,lock=matrix_to_rpy(rotation)
    g0=np.asarray(arrays['body_state'][ids,:3],np.float64);gnorm=np.linalg.norm(g0,axis=-1)
    if not np.isfinite(g0).all() or np.any(np.abs(gnorm-1)>.01):raise ValueError('current projected gravity must be unit vector')
    gf=np.einsum('bkji,bj->bki',rotation,g0);tilt=np.arccos(np.clip(-gf[...,2],-1,1))
    probabilities=1/(1+np.exp(-np.clip(o['event_logits'],-80,80)))
    if support:
        for h,trained in enumerate(support['trained']):
            if not trained:probabilities[...,h]=support['constant_probability'][h]
    futureq=futurerpy=futurelock=None
    if 'current_quat_wxyz' in arrays:
        current=quat_to_matrix(arrays['current_quat_wxyz'][ids])
        expected_gravity=np.einsum('bji,j->bi',current,np.array([0.,0.,-1.]))
        if not np.allclose(expected_gravity,g0,atol=.02,rtol=0):raise ValueError('current projected gravity and quaternion disagree')
        future=current[:,None]@rotation;futureq=matrix_to_quat(future);futurerpy,futurelock=matrix_to_rpy(future)
    if not all(np.isfinite(x).all() for x in (position,q,rpy,tilt,probabilities)):raise ValueError('nonfinite prediction')
    return dict(schema=PREDICTION_SCHEMA,horizon_sec=2.,event_names=list(EVENTS),event_probability=probabilities,delta_position_body_m=position,delta_rotation_6d=o['rotation_6d'],delta_quaternion_wxyz=q,relative_rpy_rad=rpy,future_projected_gravity=gf,future_tilt_rad=tilt,future_quaternion_wxyz=futureq,future_rpy_rad=futurerpy,future_orientation_missing_reason=None if futureq is not None else 'current_quat_wxyz_not_provided',validity=dict(candidate=np.asarray(arrays['candidate_mask'][ids],bool),large_standardized_state_input=large_state,large_standardized_dem_input=large_dem,large_standardized_path_input=large_path,standardized_warning_threshold=10.,rotation_valid=valid,relative_euler_gimbal_lock=lock,future_euler_gimbal_lock=futurelock),support=copy.deepcopy(support),pose_supervision='complete_2s_only',probability_calibration='uncalibrated_unweighted_BCE')
