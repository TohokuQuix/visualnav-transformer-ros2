"""Censoring-aware endpoint diagnostics; no trajectory/Euler ADE is defined.

All angular errors use radians. Bootstrap sampling units are parent runs;
point estimates remain window-weighted, with cluster dependence in intervals.
"""
from __future__ import annotations
import numpy as np
from .moving_pose2s_geometry import quat_to_matrix, matrix_to_rpy, sixd_to_matrix, rotation_geodesic

EVENTS=('tip_over','near_fall','linear_stall','turn_stall')


def summary(values):
    v=np.asarray(values,dtype=float).ravel();v=v[np.isfinite(v)]
    if not len(v):return dict(count=0,mean=None,median=None,p95=None,mae=None,rmse=None)
    return dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.percentile(v,95)),mae=float(np.abs(v).mean()),rmse=float(np.sqrt(np.mean(v*v))))


def wrap(angle):return np.arctan2(np.sin(angle),np.cos(angle))


def constant_body_twist(velocity,angular_velocity,horizon=2.):
    """Exact SE(3) exponential in the initial full-body frame."""
    v,w=np.broadcast_arrays(np.asarray(velocity,float),np.asarray(angular_velocity,float))
    if v.shape[-1:]!=(3,) or not np.isfinite(v).all() or not np.isfinite(w).all():raise ValueError('finite body twist required')
    x,y,z=np.moveaxis(w,-1,0);zero=np.zeros_like(x)
    W=np.stack([zero,-z,y,z,zero,-x,-y,x,zero],axis=-1).reshape(w.shape[:-1]+(3,3));W2=W@W
    norm=np.linalg.norm(w,axis=-1);theta=norm*horizon
    # Taylor limits avoid cancellation when angular rates approach zero.
    a=horizon*np.sinc(theta/np.pi)
    b=horizon*horizon*.5*np.sinc(theta/(2*np.pi))**2
    c=np.empty_like(norm);small=np.abs(theta)<1e-4
    c[small]=horizon**3*(1/6-theta[small]**2/120+theta[small]**4/5040)
    c[~small]=(theta[~small]-np.sin(theta[~small]))/norm[~small]**3
    R=np.eye(3)+a[...,None,None]*W+b[...,None,None]*W2
    J=horizon*np.eye(3)+b[...,None,None]*W+c[...,None,None]*W2
    return np.einsum('...ij,...j->...i',J,v),R


def candidate_arc(paths,directions,speed=.3,horizon=2.):
    """Circular candidate geometry at upper speed, in initial yaw XY only.

    The frozen bank has analytic 0.75m circular arcs, including reverse paths.
    Recover signed arc angle from its endpoint, avoiding polyline chord bias.
    Degenerate paths are returned as unavailable (NaN), never as zero error.
    """
    path=np.asarray(paths,float);direction=np.asarray(directions,float).reshape(path.shape[:-2])
    local=path*direction[...,None,None];end=local[...,-1,:]
    theta=2*np.arctan2(end[...,1],end[...,0]);distance=np.linalg.norm(end,axis=-1)
    sinc=np.sinc(theta/(2*np.pi));length=distance/np.maximum(sinc,1e-12)
    travel=np.minimum(speed*horizon,length);phase=np.divide(theta*travel,length,out=np.zeros_like(theta),where=length>0)
    px=travel*np.sinc(phase/np.pi);py=travel*.5*phase*np.sinc(phase/(2*np.pi))**2
    xy=np.stack([px,py],-1)*direction[...,None]
    valid=np.isfinite(path).all(axis=(-2,-1))&(length>1e-8)&np.isin(direction,[-1,1])
    return np.where(valid[...,None],xy,np.nan),np.where(valid,phase,np.nan)


def body_to_initial_yaw(position,current_quat):
    R=quat_to_matrix(current_quat);yaw=matrix_to_rpy(R)[0][...,2];c,s=np.cos(yaw),np.sin(yaw)
    Z=np.stack([c,s,np.zeros_like(c),-s,c,np.zeros_like(c),np.zeros_like(c),np.zeros_like(c),np.ones_like(c)],-1).reshape(R.shape)
    T=Z@R;p=np.asarray(position,float)
    while T.ndim<p.ndim+1:T=np.expand_dims(T,-3)
    return np.einsum('...ij,...j->...i',T,p)


def event_metrics(target,probability,mask=None,parents=None):
    y=np.asarray(target,float).ravel();p=np.asarray(probability,float).ravel()
    if y.shape!=p.shape:raise ValueError('event shape mismatch')
    known=np.isfinite(y) if mask is None else np.asarray(mask,bool).ravel()&np.isfinite(y)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('finite probabilities in [0,1] required')
    if not np.isin(y[known],[0,1]).all():raise ValueError('binary known event labels required')
    parent=np.asarray(parents if parents is not None else np.arange(len(y))).ravel()
    if parent.shape!=y.shape:raise ValueError('parent shape mismatch')
    positive=int(np.sum(y[known]==1));negative=int(np.sum(y[known]==0));total=positive+negative
    result=dict(positive=positive,negative=negative,unknown=int(len(y)-total),count=total,independent_positive_parents=len(set(parent[known&(y==1)].tolist())),independent_parents=len(set(parent[known].tolist())),ap=None,auroc=None,na_reason=None if positive and negative else 'both classes required')
    y=y[known];p=p[known]
    result['brier']=float(np.mean((p-y)**2)) if total else None
    clipped=np.clip(p,1e-12,1-1e-12);result['bce']=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))) if total else None
    if positive and negative:
        order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
        tp=np.cumsum(ys)[ends];fp=(ends+1)-tp;recall=tp/positive;precision=tp/(ends+1)
        result['ap']=float(np.sum(np.diff(np.r_[0,recall])*precision))
        result['auroc']=float(np.trapz(np.r_[0,recall],np.r_[0,fp/negative]))
    tp=int(np.sum((p>=.5)&(y==1)));fp=int(np.sum((p>=.5)&(y==0)));fn=positive-tp;tn=negative-fp
    result['threshold_0.5']=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/positive if positive else None,fpr=fp/negative if negative else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    result['reliability_10_bins']=[]
    bins=np.minimum((p*10).astype(int),9)
    for b in range(10):
        take=bins==b;n=int(take.sum());result['reliability_10_bins'].append(dict(lower=b/10,upper=(b+1)/10,count=n,mean_probability=float(p[take].mean()) if n else None,observed_frequency=float(y[take].mean()) if n else None))
    return result


def pose_errors(position,target_position,rotation,target_rotation,gravity,mask,rotation_valid=None):
    p=np.asarray(position,float);gt=np.asarray(target_position,float);shape=p.shape[:-1];mask=np.asarray(mask,bool).reshape(shape)
    pe=p-gt;valid=mask&np.isfinite(pe).all(-1);rotvalid=valid.copy()
    if rotation_valid is not None:rotvalid&=np.asarray(rotation_valid,bool).reshape(shape)
    pr=np.asarray(rotation,float);tr=np.asarray(target_rotation,float)
    # Target matrices are identity placeholders only for unobserved endpoints.
    pr=np.where(rotvalid[...,None,None],pr,np.eye(3));tr=np.where(rotvalid[...,None,None],tr,np.eye(3))
    rpy,plock=matrix_to_rpy(pr);tpy,tlock=matrix_to_rpy(tr)
    gravity=np.broadcast_to(np.asarray(gravity,float),shape+(3,))
    pg=np.einsum('...ji,...j->...i',pr,gravity);tg=np.einsum('...ji,...j->...i',tr,gravity)
    tilt=wrap(np.arccos(np.clip(-pg[...,2],-1,1))-np.arccos(np.clip(-tg[...,2],-1,1)))
    return dict(position_axis_m=np.where(valid[...,None],pe,np.nan),position_xy_m=np.where(valid,np.linalg.norm(pe[...,:2],axis=-1),np.nan),position_3d_m=np.where(valid,np.linalg.norm(pe,axis=-1),np.nan),so3_rad=np.where(rotvalid,rotation_geodesic(pr,tr),np.nan),rpy_rad=np.where(rotvalid[...,None],wrap(rpy-tpy),np.nan),gravity_tilt_rad=np.where(rotvalid,tilt,np.nan),gravity_angle_rad=np.where(rotvalid,np.arccos(np.clip(np.sum(pg*tg,axis=-1),-1,1)),np.nan),gimbal_lock=(plock|tlock)&rotvalid,predicted_displacement_m=np.where(valid,np.linalg.norm(p,axis=-1),np.nan),target_displacement_m=np.where(valid,np.linalg.norm(gt,axis=-1),np.nan),predicted_rotation_rad=np.where(rotvalid,rotation_geodesic(np.broadcast_to(np.eye(3),pr.shape),pr),np.nan),target_rotation_rad=np.where(rotvalid,rotation_geodesic(np.broadcast_to(np.eye(3),tr.shape),tr),np.nan))


def pose_metrics(errors):
    return dict(position_axis_m={axis:summary(errors['position_axis_m'][...,i]) for i,axis in enumerate('xyz')},position_xy_m=summary(errors['position_xy_m']),position_3d_m=summary(errors['position_3d_m']),rotation_so3_rad=summary(errors['so3_rad']),wrapped_rpy_rad={axis:summary(np.abs(errors['rpy_rad'][...,i])) for i,axis in enumerate(('roll','pitch','yaw'))},gravity_tilt_error_rad=summary(np.abs(errors['gravity_tilt_rad'])),gravity_direction_error_rad=summary(errors['gravity_angle_rad']),euler_gimbal_lock_count=int(errors['gimbal_lock'].sum()),predicted_displacement_m=summary(errors['predicted_displacement_m']),target_displacement_m=summary(errors['target_displacement_m']),predicted_rotation_rad=summary(errors['predicted_rotation_rad']),target_rotation_rad=summary(errors['target_rotation_rad']))


def parent_bootstrap(values,parents,*,reference=None,resamples=200,seed=123):
    v=np.asarray(values,float).ravel();p=np.asarray(parents).ravel()
    if reference is not None:v=v-np.asarray(reference,float).ravel()
    good=np.isfinite(v);v=v[good];p=p[good];unique=np.unique(p)
    result=dict(estimate=float(v.mean()) if len(v) else None,low=None,high=None,count=len(v),independent_parents=len(unique),resamples=resamples,seed=seed,confidence=.95,unit='parent',statistic='window_mean_difference' if reference is not None else 'window_mean')
    if not len(unique):return result
    sums=np.array([v[p==u].sum() for u in unique]);counts=np.array([np.sum(p==u) for u in unique]);rng=np.random.default_rng(seed);samples=rng.integers(0,len(unique),size=(resamples,len(unique)))
    means=sums[samples].sum(1)/counts[samples].sum(1);result['low'],result['high']=map(float,np.percentile(means,[2.5,97.5]));return result


def state_groups(arrays,index):
    body=np.asarray(arrays['body_state']);speed=np.linalg.norm(body[:,3:6],axis=-1);wz=np.abs(body[:,8]);n=len(body)
    initial=np.array([r.get('is_initial',False) for r in index],bool);pose=np.asarray(arrays['pose_mask'],bool).reshape(n,-1).any(1)
    result={'all':np.ones(n,bool),'initial':initial,'moving':~initial,'speed:<0.05':speed<.05,'speed:0.05-0.2':(speed>=.05)&(speed<.2),'speed:>=0.2':speed>=.2,'abs_wz:<0.2':wz<.2,'abs_wz:>=0.2':wz>=.2,'pose:complete':pose,'pose:early_terminal':~pose}
    for key in ('field','level'):
        values=[str(r.get(key,'unknown')) for r in index]
        for value in sorted(set(values)):result[key+':'+value]=np.array([x==value for x in values])
    if 'current_quat_wxyz' in arrays:
        rpy=matrix_to_rpy(quat_to_matrix(arrays['current_quat_wxyz']))[0];pitch=rpy[:,1]
        result.update({'pitch:<-10deg':pitch<np.deg2rad(-10),'pitch:within10deg':np.abs(pitch)<=np.deg2rad(10),'pitch:>10deg':pitch>np.deg2rad(10)})
    tilt=np.arccos(np.clip(-body[:,2],-1,1));result.update({'current_tilt:>=60deg':tilt>=np.deg2rad(60),'current_tilt:<60deg':tilt<np.deg2rad(60)})
    if 'directions' in arrays:
        d=np.asarray(arrays['directions']).reshape(n,-1)[:,0];result['direction:forward']=d>0;result['direction:reverse']=d<0
    for key in ('qualified_state','state_name','collection_state'):
        values=[r.get(key) for r in index]
        for value in sorted(set(x for x in values if isinstance(x,str))):result[key+':'+value]=np.array([x==value for x in values])
    return result


def validate_prediction(prediction,shape):
    if prediction.get('schema')!='eatnav_moving_pose2s_prediction_v1' or prediction.get('horizon_sec')!=2.:raise ValueError('prediction schema/horizon mismatch')
    for key,last in (('delta_position_body_m',3),('delta_quaternion_wxyz',4),('event_probability',4)):
        value=np.asarray(prediction[key]);
        if value.shape!=tuple(shape)+(last,) or not np.isfinite(value).all():raise ValueError('invalid/nonfinite '+key)
    if np.any((prediction['event_probability']<0)|(prediction['event_probability']>1)):raise ValueError('invalid event probability')


def evaluate_predictions(arrays,index,indices,prediction,*,train_indices,bootstrap_resamples=200,seed=123):
    """Pure evaluation; train labels used solely for the prevalence baseline."""
    ids=np.asarray(indices,int);train=np.asarray(train_indices,int);a={k:np.asarray(v)[ids] for k,v in arrays.items()};rows=[index[i] for i in ids];shape=a['pose_mask'].shape
    validate_prediction(prediction,shape)
    candidate=np.asarray(a['candidate_mask'],bool);pose=np.asarray(a['pose_mask'],bool)&candidate;gtR,gtvalid=sixd_to_matrix(a['rotation_targets']);pose&=gtvalid
    prR=quat_to_matrix(prediction['delta_quaternion_wxyz']);rotvalid=np.asarray(prediction.get('validity',{}).get('rotation_valid',np.ones(shape,bool)),bool)
    parents=np.array([r.get('parent_id',str((r.get('field'),r.get('seed')))) for r in rows]);cp=np.broadcast_to(parents[:,None],shape).ravel()
    gravity=np.broadcast_to(a['body_state'][:,None,:3],shape+(3,));events=np.asarray(a['event_targets'],float);emask=np.asarray(a['event_mask'],bool)&candidate[...,None]
    prevalence=[]
    for h in range(4):
        y=arrays['event_targets'][train,...,h];known=np.asarray(arrays['event_mask'][train,...,h],bool)&np.isfinite(y)&np.asarray(arrays['candidate_mask'][train],bool);prevalence.append(float(y[known].mean()) if known.any() else None)
    baseline_prob=np.broadcast_to(np.array([x if x is not None else 0. for x in prevalence]),shape+(4,))
    twistp,twistr=constant_body_twist(a['body_state'][:,3:6],a['body_state'][:,6:9]);twistp=np.broadcast_to(twistp[:,None],shape+(3,));twistr=np.broadcast_to(twistr[:,None],shape+(3,3))
    all_errors={};result=dict(scope='executed_candidates_only; pose_complete_2s_only',window_count=len(ids),candidate_count=int(candidate.sum()),independent_parents=len(set(parents.tolist())),pose_missing_count=int((candidate&~pose).sum()),invalid_rotation_count=int((candidate&~rotvalid).sum()),baseline_event_train_prevalence=prevalence,probability_calibration='uncalibrated_unweighted_BCE',threshold=.5,units=dict(position='m',angles='rad'),qualified_terrain_state_note='only explicit qualified_state/state_name/collection_state metadata is used; pitch alone is not ascent/descent qualification')
    variants={'model':(prediction['delta_position_body_m'],prR,prediction['event_probability'],rotvalid),'no_motion':(np.zeros(shape+(3,)),np.broadcast_to(np.eye(3),shape+(3,3)),baseline_prob,np.ones(shape,bool)),'constant_body_twist':(twistp,twistr,baseline_prob,np.ones(shape,bool))}
    for name,(p,R,prob,rv) in variants.items():
        e=pose_errors(p,a['position_targets'],R,gtR,gravity,pose,rv);all_errors[name]=e
        event_report={head:event_metrics(events[...,h],prob[...,h],emask[...,h],cp) for h,head in enumerate(EVENTS)}
        ci={'position_3d_mean_m':parent_bootstrap(e['position_3d_m'],cp,resamples=bootstrap_resamples,seed=seed),'so3_mean_rad':parent_bootstrap(e['so3_rad'],cp,resamples=bootstrap_resamples,seed=seed)}
        for h,head in enumerate(EVENTS):ci[head+'_brier']=parent_bootstrap(np.where(emask[...,h]&np.isfinite(events[...,h]),(prob[...,h]-events[...,h])**2,np.nan),cp,resamples=bootstrap_resamples,seed=seed)
        result[name]=dict(pose=pose_metrics(e),events=event_report,parent_bootstrap_95ci=ci)
    result['paired_model_minus_baseline_95ci']={}
    for name in ('no_motion','constant_body_twist'):
        result['paired_model_minus_baseline_95ci'][name]={key:parent_bootstrap(all_errors['model'][key],cp,reference=all_errors[name][key],resamples=bootstrap_resamples,seed=seed) for key in ('position_3d_m','so3_rad')}
        for h,head in enumerate(EVENTS):
            valid=emask[...,h]&np.isfinite(events[...,h]);value=np.where(valid,(prediction['event_probability'][...,h]-events[...,h])**2,np.nan);ref=np.where(valid,(baseline_prob[...,h]-events[...,h])**2,np.nan)
            result['paired_model_minus_baseline_95ci'][name][head+'_brier']=parent_bootstrap(value,cp,reference=ref,resamples=bootstrap_resamples,seed=seed)
    result['state_stratified']={}
    for group,take in state_groups(a,rows).items():
        if not take.any():continue
        result['state_stratified'][group]=dict(window_count=int(take.sum()),independent_parents=len(set(parents[take].tolist())),models={name:dict(pose=pose_metrics({k:v[take] for k,v in e.items()}),events={head:event_metrics(events[take,...,h],variants[name][2][take,...,h],emask[take,...,h],np.broadcast_to(parents[take,None],(int(take.sum()),shape[1])).ravel()) for h,head in enumerate(EVENTS)}) for name,e in all_errors.items()})
    result['candidate_arc']=dict(scope='initial_yaw_horizontal_XY_and_world_yaw_only',speed_upper_mps=.3,horizon_sec=2.)
    if 'current_quat_wxyz' in a:
        xy,yaw=candidate_arc(a['paths'],a['directions']);truth=body_to_initial_yaw(a['position_targets'],a['current_quat_wxyz']);pred=body_to_initial_yaw(prediction['delta_position_body_m'],a['current_quat_wxyz']);q0=quat_to_matrix(a['current_quat_wxyz']);initialyaw=matrix_to_rpy(q0)[0][:,2,None];future_gt=matrix_to_rpy(q0[:,None]@gtR)[0][...,2];future_pred=matrix_to_rpy(q0[:,None]@prR)[0][...,2]
        result['candidate_arc']['available_count']=int((pose&np.isfinite(yaw)).sum())
        for name,p,y in (('candidate_arc',xy,initialyaw+yaw),('model',pred[...,:2],future_pred)):
            valid=pose&np.isfinite(yaw);result['candidate_arc'][name]=dict(xy_error_m=summary(np.where(valid,np.linalg.norm(p-truth[...,:2],axis=-1),np.nan)),wrapped_yaw_error_rad=summary(np.where(valid&(rotvalid if name=='model' else True),np.abs(wrap(y-future_gt)),np.nan)))
    else:result['candidate_arc']['unavailable_reason']='current_quat_wxyz required for common yaw frame'
    return result
