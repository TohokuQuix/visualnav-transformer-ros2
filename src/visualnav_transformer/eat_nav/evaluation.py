"""Trajectory evaluation and paired Quality-conditioning sensitivity."""

from __future__ import annotations

from itertools import combinations
import numpy as np
import torch


def _features(xy):
    points=np.concatenate((np.zeros((*xy.shape[:-2],1,2),dtype=xy.dtype),xy),axis=-2)
    steps=np.diff(points,axis=-2)
    path=np.linalg.norm(steps,axis=-1).sum(axis=-1)
    heading=np.arctan2(steps[...,1],steps[...,0])
    curvature=np.abs(np.arctan2(np.sin(np.diff(heading,axis=-1)),np.cos(np.diff(heading,axis=-1)))).sum(axis=-1)
    return {'path_length':path,'curvature':curvature,'forward_progress':xy[...,-1,0]}


def evaluate_batch(model,batch,seed=0):
    was_training=model.training
    model.eval()
    device=next(model.parameters()).device
    dem=batch['dem'].to(device)
    goal=batch['goal'].to(device)
    quality=batch['quality'].to(device)
    with torch.no_grad():
        predicted=model.sample(dem,goal,quality,seed=seed)
    target=torch.cumsum(batch['action'].to(device),dim=1)
    valid=batch['action_mask'].to(device).bool()
    errors=torch.linalg.vector_norm(predicted-target,dim=-1)
    pointwise=[]
    for index in range(errors.shape[1]):
        selected=errors[:,index][valid[:,index]]
        pointwise.append(float(selected.mean()) if len(selected) else None)
    model.train(was_training)
    return {'pointwise_xy_error_m':pointwise,'mean_xy_error_m':float(errors[valid].mean()) if valid.any() else None}


def _rank_correlation(x,y):
    if np.allclose(y,y[0]):return 0.
    return float(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1])


def diagnose_conditioning_sensitivity(model,heldout_windows,q_values,seeds):
    windows=list(heldout_windows)
    q_values=[float(q) for q in q_values]
    seeds=[int(s) for s in seeds]
    if len(windows)<50:
        return {'status':'inconclusive','input_count':len(windows),'reason':'fewer than 50 held-out fixed inputs'}
    if len(q_values)<2 or len(set(q_values))!=len(q_values) or min(q_values)<0 or max(q_values)>1 or len(seeds)<2:
        raise ValueError('need distinct q values in [0,1] and at least two seeds')
    was_training=model.training
    model.eval()
    device=next(model.parameters()).device if hasattr(model,'parameters') else torch.device('cpu')
    dem=torch.as_tensor(np.stack([w['dem'] for w in windows]),dtype=torch.float32,device=device)
    goal=torch.as_tensor(np.stack([w['goal'] for w in windows]),dtype=torch.float32,device=device)
    outputs=[]
    with torch.no_grad():
        for seed in seeds:
            by_q=[]
            for q in q_values:
                quality=torch.full((len(windows),1),q,device=device)
                result=model.sample(dem,goal,quality,seed=seed)
                by_q.append(result.detach().cpu().numpy())
            outputs.append(np.stack(by_q,axis=0))
    model.train(was_training)
    # [seed, q, input, point, xy], with identical diffusion RNG state across q.
    generated=np.stack(outputs,axis=0)
    paired=np.linalg.norm(generated[:,-1]-generated[:,0],axis=-1)
    paired_sample=np.sqrt(np.mean(paired**2,axis=-1))
    middle=min(range(len(q_values)),key=lambda i:abs(q_values[i]-.5))
    stochastic=[]
    for a,b in combinations(range(len(seeds)),2):
        delta=np.linalg.norm(generated[a,middle]-generated[b,middle],axis=-1)
        stochastic.append(np.sqrt(np.mean(delta**2,axis=-1)))
    effect=float(np.median(paired_sample))
    sampling=float(np.median(np.stack(stochastic)))
    feature=_features(generated)
    progress=feature['forward_progress']
    rho=[_rank_correlation(np.asarray(q_values),progress[s,:,i]) for s in range(len(seeds)) for i in range(len(windows))]
    high_low=progress[:,-1]-progress[:,0]
    sign=np.sign(np.median(high_low))
    consistency=float(np.mean(np.sign(high_low)==sign)) if sign else 0.
    observed_high=[];observed_low=[]
    for w in windows:
        distance=float(np.asarray(w['action'])[:,0].sum())
        (observed_high if float(w.get('quality',.5))>=.75 else observed_low if float(w.get('quality',.5))<.25 else []).append(distance)
    data_delta=float(np.mean(observed_high)-np.mean(observed_low)) if observed_high and observed_low else None
    model_delta=float(np.median(high_low))
    direction=(bool(np.sign(data_delta)==np.sign(model_delta)) if data_delta is not None and model_delta!=0 else None)
    report={'status':'pass' if effect>max(.001,sampling) else 'fail',
            'input_count':len(windows),'q_values':q_values,'seeds':seeds,
            'paired_trajectory_difference_m':effect,
            'sampling_trajectory_difference_m':sampling,
            'effect_size_ratio':effect/max(sampling,1e-12),
            'pointwise_difference_m':np.median(paired,axis=(0,1)).tolist(),
            'endpoint_difference_m':float(np.median(np.linalg.norm(generated[:,-1,:,:]-generated[:,0,:,:],axis=-1)[:,:,-1])),
            'path_length_delta_m':float(np.median(feature['path_length'][:,-1]-feature['path_length'][:,0])),
            'curvature_delta_rad':float(np.median(feature['curvature'][:,-1]-feature['curvature'][:,0])),
            'high_q_forward_progress_delta_m':model_delta,
            'data_high_quality_forward_progress_delta_m':data_delta,
            'direction_agreement_forward_progress':direction,
            'q_progress_rank_correlation_median':float(np.median(rho)),
            'q_progress_sign_consistency_fraction':consistency,
            'q_pair_trajectory_difference_m':{f'{q_values[i]}->{q_values[j]}':float(np.median(np.sqrt(np.mean(np.linalg.norm(generated[:,j]-generated[:,i],axis=-1)**2,axis=-1)))) for i,j in combinations(range(len(q_values)),2)}}
    return report
