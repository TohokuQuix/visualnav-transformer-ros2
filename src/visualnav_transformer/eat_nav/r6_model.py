"""New 2-second event/trajectory contract; never reinterprets old checkpoints."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='eatnav_r6_two_second_pose_checkpoint_v1'
NORMALIZATION_SCHEMA='eatnav_r6_two_second_pose_normalization_v1'
CONTRACT=dict(schema='eatnav_r6_two_second_pose_contract_v1',event_horizon_sec=2,
 times_sec=[i/10 for i in range(1,21)],time_unit='seconds',events=['tip_over','near_fall','linear_stall','turn_stall'],
 dem_shape=[4,2,20,20],dem_extent_m=[2.,2.],dem_resolution_m=.1,dem_nominal_spacing_sec=.25,
 path_bank_schema='eatnav_forward_reverse_path_bank_v1',path_arc_length_m=.75,path_angles_deg=[0,-30,30,-60,60,-90,90],path_direction_order=[1]*7+[-1]*7,
 state_dim=28,actor_command_indices_excluded=[6,7,8],candidate_count=14,path_shape=[14,8,2],
 trajectory_shape=[14,20,9],trajectory_channels=['dx_m','dy_m','dz_m','sin_relative_yaw','cos_relative_yaw','sin_gravity_roll','cos_gravity_roll','sin_gravity_pitch','cos_gravity_pitch'],
 frame='initial_origin_inverse_initial_yaw_world_up',position_unit='metres',angle_unit='radians',progress='derived_from_two_second_predicted_position_only')
KEEP=[i for i in range(31) if i not in (6,7,8)]

def ids(arrays,indices):
 x=np.asarray(indices,dtype=np.int64)
 if x.ndim!=1 or not len(x) or len(set(x.tolist()))!=len(x) or x.min()<0 or x.max()>=len(arrays['dem']):raise ValueError('unique in-range sample indices required')
 return x

def _stats(values):
 values=np.asarray(values,np.float64)
 if not values.size or not np.isfinite(values).all():raise ValueError('no finite train-only normalization support')
 return values.mean(axis=0).tolist(),np.maximum(values.std(axis=0),.001).tolist()

def fit_normalization(arrays,train_indices):
 i=ids(arrays,train_indices);d=arrays['dem'][i];cm=arrays['candidate_mask'][i]
 heights=d[:,:,0][d[:,:,1]==1];states=np.concatenate((arrays['actor_state'][i],arrays['body_state'][i]),axis=1)[:,KEEP]
 paths=arrays['paths'][i][cm].reshape(-1,2);pm=arrays['trajectory_mask'][i]&cm[:,:,None]
 pos=arrays['trajectory'][i,...,:3][pm]
 dm,ds=_stats(heights);sm,ss=_stats(states);xm,xs=_stats(paths);pmu,ps=_stats(pos)
 return dict(schema=NORMALIZATION_SCHEMA,source='train_split_only',train_indices=i.tolist(),dem_mean=dm,dem_std=ds,state_mean=sm,state_std=ss,path_mean=xm,path_std=xs,position_mean=pmu,position_std=ps)

def prepare_inputs(arrays,indices,stats,device='cpu'):
 if stats.get('schema')!=NORMALIZATION_SCHEMA or stats.get('source')!='train_split_only':raise ValueError('normalization schema/source mismatch')
 i=ids(arrays,indices);dem=arrays['dem'][i].copy();mask=dem[:,:,1]
 if not np.isin(mask,[0,1]).all():raise ValueError('DEM validity mask must be binary')
 dem[:,:,0]=np.where(mask==1,(dem[:,:,0]-stats['dem_mean'])/stats['dem_std'],0)
 state=np.concatenate((arrays['actor_state'][i],arrays['body_state'][i]),axis=1)[:,KEEP]
 state=(state-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
 paths=(arrays['paths'][i]-np.asarray(stats['path_mean']))/np.asarray(stats['path_std'])
 values=dict(dem=dem.reshape(len(i),8,20,20),state=state,paths=paths,directions=arrays['directions'][i],candidate_mask=arrays['candidate_mask'][i])
 if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite normalized inputs')
 return {k:torch.as_tensor(v,dtype=torch.bool if k=='candidate_mask' else torch.float32,device=device) for k,v in values.items()}

def training_support(arrays,indices):
 i=ids(arrays,indices);y=arrays['event_targets'][i];m=arrays['event_mask'][i]&arrays['candidate_mask'][i,:,None]
 if y.shape!=(len(i),14,4):raise ValueError('2-second event targets must have no legacy horizon axis')
 counts=[]
 for j in range(4):
  valid=m[:,:,j];v=y[:,:,j][valid]
  if not np.isin(v,[0,1]).all():raise ValueError('known events must be binary')
  counts.append(dict(positive=int((v==1).sum()),negative=int((v==0).sum()),unknown=int((~valid&arrays['candidate_mask'][i]).sum())))
 return dict(active=[c['positive']>0 and c['negative']>0 for c in counts],counts=counts,pos_weight=[min(c['negative']/c['positive'],20.) if c['positive'] else 1. for c in counts])

class PoseRiskEvaluator(nn.Module):
 def __init__(self,use_dem=True):
  super().__init__();self.use_dem=bool(use_dem)
  self.dem_encoder=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),nn.Flatten(),nn.Linear(800,128),nn.ReLU())
  self.state_encoder=nn.Sequential(nn.Linear(28,64),nn.ReLU())
  self.path_encoder=nn.Sequential(nn.Linear(17,64),nn.ReLU())
  self.fusion=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,96),nn.ReLU())
  self.event_head=nn.Linear(96,4);self.trajectory_head=nn.Linear(96,180)
  with torch.no_grad():
   bias=self.trajectory_head.bias.reshape(20,9);bias.zero_();bias[:,[4,6,8]]=1.
 def forward(self,dem,state,paths,directions,candidate_mask):
  b=len(state)
  if dem.shape!=(b,8,20,20) or state.shape!=(b,28) or paths.shape!=(b,14,8,2) or directions.shape!=(b,14,1) or candidate_mask.shape!=(b,14):raise ValueError('R6 input shape mismatch: fixed fourteen-candidate contract')
  if not candidate_mask.any(1).all():raise ValueError('all candidates unknown')
  if any(not torch.isfinite(v).all() for v in (dem,state,paths,directions)):raise ValueError('nonfinite input')
  d=self.dem_encoder(dem) if self.use_dem else torch.zeros((b,128),device=state.device,dtype=state.dtype)
  shared=torch.cat((d,self.state_encoder(state)),dim=-1)
  path=self.path_encoder(torch.cat((paths.reshape(b,14,16),directions),dim=-1))
  h=self.fusion(torch.cat((shared[:,None].expand(-1,14,-1),path),-1))
  return dict(event_logits=self.event_head(h),trajectory_z=self.trajectory_head(h).reshape(b,14,20,9))

def _masked_mean(values,mask):return torch.where(mask,values,0.).sum()/mask.sum().clamp_min(1)

def objective(outputs,arrays,indices,stats,support):
 i=ids(arrays,indices);dev=outputs['event_logits'].device
 tensor=lambda v,dtype=torch.float32:torch.as_tensor(v,dtype=dtype,device=dev)
 cm=tensor(arrays['candidate_mask'][i],torch.bool);em=tensor(arrays['event_mask'][i],torch.bool)&cm[:,:,None]&tensor(support['active'],torch.bool)
 y=tensor(arrays['event_targets'][i]);y=torch.where(em,y,0.)
 event=_masked_mean(F.binary_cross_entropy_with_logits(outputs['event_logits'],y,pos_weight=tensor(support['pos_weight']),reduction='none'),em)
 pm=tensor(arrays['trajectory_mask'][i],torch.bool)&cm[:,:,None];target=tensor(arrays['trajectory'][i]);target=torch.where(pm[:,:,:,None],target,0.)
 predicted=outputs['trajectory_z'];position_target=(target[...,:3]-tensor(stats['position_mean']))/tensor(stats['position_std'])
 pos=_masked_mean(F.smooth_l1_loss(predicted[...,:3],position_target,reduction='none'),pm[:,:,:,None].expand(-1,-1,-1,3))
 raw=predicted[...,3:].reshape(len(i),14,20,3,2);angles=F.normalize(raw,dim=-1,eps=1e-6)
 truth=target[...,3:].reshape(len(i),14,20,3,2)
 orient=_masked_mean((angles-truth).square().sum(-1),pm[:,:,:,None].expand(-1,-1,-1,3))
 unit=_masked_mean((raw.square().sum(-1)-1).square(),pm[:,:,:,None].expand(-1,-1,-1,3))
 return dict(event=event,position=pos,orientation=orient,unit=unit,total=event+pos+.25*orient+.01*unit)

def save_checkpoint(path,model,stats,support,provenance):
 torch.save(dict(schema=SCHEMA,contract=copy.deepcopy(CONTRACT),normalization=copy.deepcopy(stats),support=copy.deepcopy(support),use_dem=model.use_dem,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},provenance=copy.deepcopy(provenance)),path)

def load_checkpoint(path,device='cpu'):
 p=torch.load(path,map_location='cpu',weights_only=True)
 if p.get('schema')!=SCHEMA or p.get('contract')!=CONTRACT or p.get('normalization',{}).get('schema')!=NORMALIZATION_SCHEMA:raise ValueError('R6 checkpoint schema/contract mismatch; legacy model is not two-second pose model')
 m=PoseRiskEvaluator(use_dem=p['use_dem']);m.load_state_dict(p['state_dict'],strict=True);m.to(device);m.eval();return m,p

def predict(model,stats,arrays,indices):
 dev=next(model.parameters()).device;model.eval()
 with torch.inference_mode():
  o=model(**prepare_inputs(arrays,indices,stats,dev));t=o['trajectory_z'].clone()
  t[...,:3]=t[...,:3]*torch.as_tensor(stats['position_std'],device=dev)+torch.as_tensor(stats['position_mean'],device=dev)
  pair=t[...,3:].reshape(len(indices),14,20,3,2);normalized=F.normalize(pair,dim=-1,eps=1e-6)
  fallback=torch.zeros_like(pair);fallback[...,1]=1.
  pair.copy_(torch.where(pair.norm(dim=-1,keepdim=True)>1e-6,normalized,fallback))
  result=dict(events=o['event_logits'].sigmoid().cpu().numpy(),trajectory=t.cpu().numpy())
 if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite prediction')
 return result
