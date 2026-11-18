"""Single-frame pose model; old schemas and unverified profiles are rejected."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .single_pose2s_contract import INPUT_SCHEMA,MODEL_SCHEMA,PROFILE_SHA,validate_input
from .moving_pose2s_geometry import sixd_to_matrix,matrix_to_quat,matrix_to_rpy,quat_to_matrix
from .moving_pose2s_model import torch_rotation


def fit_normalization(a,indices):
    ids=np.asarray(indices,np.int64)
    if ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(a['dem']):raise ValueError('nonempty unique train indices required')
    good=np.asarray(a['pose_mask'])[ids]&np.asarray(a['input_valid'])[ids]
    ids=ids[good]
    if not len(ids):raise ValueError('no valid train pose')
    state=np.asarray(a['state'])[ids];mask=np.asarray(a['state_mask'])[ids].astype(bool)
    means=[];stds=[]
    for i in range(27):
        x=state[:,i][mask[:,i]];means.append(float(x.mean()) if len(x) else 0.);stds.append(max(float(x.std()),.001) if len(x) else 1.)
    heights=np.asarray(a['dem'])[ids,0][np.asarray(a['dem'])[ids,1]>0]
    pos=np.asarray(a['position'])[ids]
    return dict(schema='eatnav_single_dem_pose2s_normalization_v1',source='train_only',train_indices=ids.tolist(),height_mean=float(heights.mean()),height_std=max(float(heights.std()),.001),state_mean=means,state_std=stds,position_mean=pos.mean(0).tolist(),position_std=np.maximum(pos.std(0),[.05,.05,.02]).tolist(),age_mean=np.asarray(a['state_age'])[ids].mean(0).tolist(),age_std=np.maximum(np.asarray(a['state_age'])[ids].std(0),.001).tolist(),context_mean=np.asarray(a['dem_context'])[ids].mean(0).tolist(),context_std=np.maximum(np.asarray(a['dem_context'])[ids].std(0),.001).tolist())


def prepare(a,norm,device='cpu'):
    dem=np.asarray(a['dem'],np.float32).copy();mask=np.asarray(a['state_mask'],np.float32)
    dem[:,0]=np.where(dem[:,1]>0,(dem[:,0]-norm['height_mean'])/norm['height_std'],0)
    state=np.asarray(a['state'],np.float32).copy();z=(state-np.asarray(norm['state_mean']))/np.asarray(norm['state_std'])
    # Preserve physical gravity and categorical mode, normalize the dimensional values only.
    z[:,:3]=state[:,:3];z[:,24:27]=state[:,24:27]
    ages=(np.asarray(a['state_age'])-np.asarray(norm['age_mean']))/np.asarray(norm['age_std'])
    context=np.asarray(a['dem_context']).copy();context_z=(context-np.asarray(norm['context_mean']))/np.asarray(norm['context_std']);context_z[:,[1,4,5]]=context[:,[1,4,5]]
    state_features=np.concatenate([np.where(mask>0,z,0),mask,np.where(mask>0,ages,0),context_z],1).astype(np.float32)
    return tuple(torch.as_tensor(x,dtype=torch.float32,device=device) for x in [dem,state_features,np.asarray(a['paths'],np.float32)/.75,np.asarray(a['directions'],np.float32)])


class SinglePose2s(nn.Module):
    def __init__(self):
        super().__init__()
        self.dem_encoder=nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(64*5*5,64),nn.ReLU())
        self.state_encoder=nn.Sequential(nn.Linear(87,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.path_encoder=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.position_head=nn.Linear(64,3);self.rotation_head=nn.Linear(64,6)
        nn.init.zeros_(self.rotation_head.weight)
        with torch.no_grad():self.rotation_head.bias.copy_(torch.tensor([1.,0,0,0,1,0]))

    def forward(self,dem,state,paths,direction):
        b,k=paths.shape[:2]
        if dem.shape!=(b,3,40,40) or state.shape!=(b,87) or paths.shape!=(b,k,8,2) or direction.shape!=(b,k,1) or k<1:raise ValueError('new model schema shape mismatch')
        shared=torch.cat([self.dem_encoder(dem),self.state_encoder(state)],-1)
        c=self.path_encoder(torch.cat([paths.reshape(b,k,16),direction],-1))
        f=self.fusion(torch.cat([shared[:,None].expand(-1,k,-1),c],-1))
        return dict(position_z=self.position_head(f),rotation_6d=self.rotation_head(f))


def pose_loss(outputs,position,rotation,normalization):
    device=outputs['position_z'].device
    std=torch.as_tensor(normalization['position_std'],device=device);mean=torch.as_tensor(normalization['position_mean'],device=device)
    p=F.smooth_l1_loss(outputs['position_z'][:,0],(position-mean)/std)
    pred,valid,penalty=torch_rotation(outputs['rotation_6d'][:,0]);truth,_,_=torch_rotation(rotation)
    r=(pred-truth).square().mean();loss=p+.5*r+.001*penalty.mean()
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    return loss


def predict(payload,a,device='cpu'):
    if payload.get('schema')!=MODEL_SCHEMA or payload.get('profile_sha256')!=PROFILE_SHA:raise ValueError('unsupported model schema/profile')
    if payload.get('input_schema')!=INPUT_SCHEMA:raise ValueError('unsupported input contract')
    if str(np.asarray(a.get('input_schema','')).item())!=INPUT_SCHEMA or str(np.asarray(a.get('profile_sha256','')).item())!=PROFILE_SHA:raise ValueError('input provenance does not match FAST-LIO contract')
    b=len(a['dem'])
    for i in range(b):
        sample={k:np.asarray(a[k])[i] for k in ['dem','state','state_mask','state_age','dem_context','paths','directions']}
        for k in ['t0_ns','dem_stamp_ns','dem_received_ns']:
            if k not in a:raise ValueError('inference requires acquisition timestamps')
            sample[k]=int(np.asarray(a[k])[i])
        sample['schema']=INPUT_SCHEMA;validate_input(sample)
    model=SinglePose2s().to(device);model.load_state_dict(payload['state_dict'],strict=True);model.eval();norm=payload['normalization']
    with torch.inference_mode():out=model(*prepare(a,norm,device))
    pos=out['position_z'].cpu().numpy()*np.asarray(norm['position_std'])+np.asarray(norm['position_mean'])
    six=out['rotation_6d'].cpu().numpy();R,valid=sixd_to_matrix(six)
    valid=valid&np.isfinite(pos).all(-1)
    q=matrix_to_quat(R);rpy,singular=matrix_to_rpy(R)
    result=dict(schema='eatnav_single_dem_pose2s_prediction_v1',horizon_seconds=2.,profile_sha256=PROFILE_SHA,position_body_m=pos,relative_rotation_6d=six,relative_quaternion_wxyz=q,relative_rpy_rad=rpy,rotation_valid=valid,euler_singular=singular,relative_rpy_deg=np.rad2deg(rpy),units=dict(position='m',rpy='rad',rpy_deg='deg',quaternion='wxyz',position_frame='full body at t0'))
    if 'q0_wxyz' in a:
        R0=quat_to_matrix(a['q0_wxyz']);future=R0[:,None]@R
        frpy,fsingular=matrix_to_rpy(future)
        result.update(future_quaternion_wxyz=matrix_to_quat(future),future_rpy_rad=frpy,future_rpy_deg=np.rad2deg(frpy),future_euler_singular=fsingular,future_gravity_body=np.einsum('...ji,j->...i',future,[0.,0.,-1.]))
    return result


def latency_summary(seconds):
    x=np.asarray(seconds,float)*1000
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('finite nonnegative latency samples required')
    return dict(repeats=len(x),mean_ms=float(x.mean()),median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),max_ms=float(x.max()))
