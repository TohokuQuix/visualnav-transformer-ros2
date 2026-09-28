"""Episode-cluster uncertainty and support-aware Quality gate evaluation."""
from __future__ import annotations
from collections import Counter
import numpy as np


def episode_cluster_ci(values, episodes, *, seed=613, repeats=2000):
    values=np.asarray(values,dtype=float)
    episodes=[tuple(e) if isinstance(e,(tuple,list,np.ndarray)) else e for e in episodes]
    unique=list(dict.fromkeys(episodes))
    if len(values)!=len(episodes) or not unique:
        raise ValueError('cluster bootstrap input mismatch')
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repeats):
        draw=rng.choice(len(unique),size=len(unique),replace=True)
        weights=Counter(unique[int(i)] for i in draw)
        row_weights=np.asarray([weights.get(e,0) for e in episodes],dtype=float)
        samples.append(float(np.average(values,weights=row_weights)))
    return np.quantile(samples,[.025,.975]).tolist(),samples


def terrain_support(distributions,min_bin_count=30):
    common=sorted(set.intersection(*(set(terrain) for terrain in distributions.values())))
    by_split={}
    for split,terrains in distributions.items():
        by_split[split]={}
        for terrain,bins in terrains.items():
            supported=all(bins.get(b,0)>=min_bin_count for b in range(4))
            by_split[split][terrain]={'status':'supported' if supported else 'insufficient support',
                                       'bin_counts':{str(b):int(bins.get(b,0)) for b in range(4)},
                                       'metric':None}
    return {'common_terrain_types':common,'by_split':by_split,'minimum_per_bin':min_bin_count}


def pair_cluster_ci(cross_values,cross_episodes,same_values,same_episodes,*,seed=613,repeats=2000):
    """Two-episode pair weights preserve dependence when either source repeats."""
    cross_values=np.asarray(cross_values,dtype=float)
    same_values=np.asarray(same_values,dtype=float)
    if len(cross_values)!=len(cross_episodes) or len(same_values)!=len(same_episodes):
        raise ValueError('pair value and episode lengths differ')
    keys=list(dict.fromkeys(e for pair in list(cross_episodes)+list(same_episodes) for e in pair))
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repeats):
        drawn=rng.choice(len(keys),size=len(keys),replace=True)
        counts=Counter(keys[int(i)] for i in drawn)
        cw=np.asarray([counts.get(a,0)*counts.get(b,0) for a,b in cross_episodes],dtype=float)
        sw=np.asarray([counts.get(a,0)*counts.get(b,0) for a,b in same_episodes],dtype=float)
        if cw.sum() and sw.sum():
            samples.append(float(np.average(cross_values,weights=cw)-np.average(same_values,weights=sw)))
    if len(samples)<repeats//2:
        raise ValueError('too few valid two-episode bootstrap replicates')
    return {'observed_mean_difference':float(cross_values.mean()-same_values.mean()),
            'ci_95':np.quantile(samples,[.025,.975]).tolist(),'valid_replicates':len(samples)}


def trajectory_errors(prediction, target, mask):
    prediction=np.asarray(prediction,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if prediction.shape!=target.shape or mask.shape!=prediction.shape[:2]:
        raise ValueError('trajectory error shape mismatch')
    pxy=np.cumsum(prediction,axis=1)
    txy=np.cumsum(target,axis=1)
    delta=np.square(prediction-target).sum(axis=2)
    point=np.linalg.norm(pxy-txy,axis=2)
    count=np.maximum(mask.sum(axis=1),1)
    mse=(delta*mask).sum(axis=1)/(2*count)
    rms=np.sqrt((np.square(point)*mask).sum(axis=1)/count)
    last=np.maximum(mask.sum(axis=1)-1,0)
    endpoint=point[np.arange(len(point)),last]
    forward=np.abs(pxy[np.arange(len(point)),last,0]-txy[np.arange(len(point)),last,0])
    path=np.abs((np.linalg.norm(prediction,axis=2)*mask).sum(axis=1)-(np.linalg.norm(target,axis=2)*mask).sum(axis=1))
    def turning(x):
        angles=np.arctan2(x[:,:,1],x[:,:,0])
        return (np.abs(np.arctan2(np.sin(np.diff(angles,axis=1)),np.cos(np.diff(angles,axis=1))))
                *(mask[:,1:] & mask[:,:-1])).sum(axis=1)
    turn=np.abs(turning(prediction)-turning(target))
    return {'mse':mse,'rms_m':rms,'endpoint_m':endpoint,
            'forward_progress_error_m':forward,'path_length_error_m':path,
            'total_turning_error_rad':turn,'point_error_m':point}
