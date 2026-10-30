"""Masked horizon-specific predictor metrics; absent classes stay unavailable."""
import numpy as np

def event_metrics(probabilities,labels,masks,threshold=.5):
 p=np.asarray(probabilities).reshape(-1,4);y=np.asarray(labels).reshape(-1,4);m=np.asarray(masks,bool).reshape(-1,4);out=[]
 for j in range(4):
  q=p[:,j][m[:,j]];t=y[:,j][m[:,j]];n=len(t);pos=int((t==1).sum());neg=n-pos
  if not np.isfinite(q).all() or not np.isin(t,[0,1]).all():raise ValueError('nonfinite known predictions or nonbinary labels')
  pred=q>=threshold;tp=int((pred&(t==1)).sum());fp=int((pred&(t==0)).sum());fn=pos-tp
  ap=None
  if pos and neg:
   order=np.argsort(-q,kind='stable');sortedq=q[order];sortedy=t[order];ends=np.r_[np.flatnonzero(sortedq[:-1]!=sortedq[1:]),n-1];totalpos=np.cumsum(sortedy)[ends];previous=np.r_[0,totalpos[:-1]]
   ap=float(np.sum((totalpos-previous)/pos*totalpos/(ends+1)))
  bins=[];ece=0.
  for k in range(10):
   b=(q>=k/10)&((q<(k+1)/10) if k<9 else (q<=1))
   if b.any():
    confidence=float(q[b].mean());frequency=float(t[b].mean());ece+=b.sum()/max(n,1)*abs(confidence-frequency);bins.append(dict(lower=k/10,upper=(k+1)/10,count=int(b.sum()),mean_probability=confidence,frequency=frequency))
  out.append(dict(count=n,positive=pos,negative=neg,average_precision=ap,recall=tp/pos if pos else None,precision=tp/(tp+fp) if tp+fp else None,false_positive=fp,false_negative=fn,false_positive_rate=fp/neg if neg else None,brier=float(((q-t)**2).mean()) if n else None,ece=float(ece) if n else None,calibration_bins=bins,threshold=threshold,event_horizon_sec=2))
 return out

def rotation_matrix(p):
 sy,cy,sr,cr,sp,cp=np.moveaxis(np.asarray(p)[...,3:],-1,0)
 return np.stack([cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr],axis=-1).reshape((*np.asarray(p).shape[:-1],3,3))

def trajectory_metrics(prediction,target,mask):
 p=np.asarray(prediction,float);y=np.asarray(target,float);m=np.asarray(mask,bool)
 if p.shape!=y.shape or p.shape[:-1]!=m.shape or p.shape[-2:]!=(20,9):raise ValueError('trajectory contract mismatch')
 if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():raise ValueError('nonfinite known pose')
 distances=np.linalg.norm(p[...,:3]-y[...,:3],axis=-1)
 complete=m[...,-1];count=int(m.sum());r1=rotation_matrix(np.where(m[...,None],p,0));r2=rotation_matrix(np.where(m[...,None],y,0))
 geo=np.degrees(np.arccos(np.clip((np.sum(r1*r2,axis=(-2,-1))-1)/2,-1,1)))
 angular=[]
 for j in [3,5,7]:
  delta=np.arctan2(p[...,j],p[...,j+1])-np.arctan2(y[...,j],y[...,j+1]);angular.append(float(np.degrees(np.abs(np.arctan2(np.sin(delta),np.cos(delta))))[m].mean()) if count else None)
 max_known=m.any(-1);tp=np.degrees(np.arccos(np.clip(p[...,6]*p[...,8],-1,1)));ty=np.degrees(np.arccos(np.clip(y[...,6]*y[...,8],-1,1)))
 pmax=np.max(np.where(m,tp,-np.inf),axis=-1);ymax=np.max(np.where(m,ty,-np.inf),axis=-1)
 return dict(trajectory_sample_count=count,complete_endpoint_count=int(complete.sum()),ade_3d_m=float(distances[m].mean()) if count else None,fde_3d_m=float(distances[...,-1][complete].mean()) if complete.any() else None,yaw_mae_deg=angular[0],roll_mae_deg=angular[1],pitch_mae_deg=angular[2],rotation_geodesic_deg=float(geo[m].mean()) if count else None,max_tilt_mae_deg=float(np.abs(pmax[max_known]-ymax[max_known]).mean()) if max_known.any() else None,position_horizon_sec=2)
