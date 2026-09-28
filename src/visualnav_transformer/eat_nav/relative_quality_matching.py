"""Train-frozen geometry and non-repeating episode-pair selection."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from scipy.spatial import cKDTree


def goal_direction_delta_deg(a,b,zero_threshold_m:float=.01)->float:
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if na<zero_threshold_m and nb<zero_threshold_m:return 0.
    if na<zero_threshold_m or nb<zero_threshold_m:return 180.
    return float(math.degrees(math.acos(np.clip(np.dot(a,b)/(na*nb),-1.,1.))))


def label_thresholds(train_candidate_differences,small_quantile:float,large_quantile:float)->dict:
    values=np.asarray(train_candidate_differences,dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0<=small_quantile<large_quantile<=1:
        raise ValueError('invalid train-only threshold source')
    return {'small':float(np.quantile(values,small_quantile)),
            'large':float(np.quantile(values,large_quantile)),
            'train_candidates':len(values),'small_quantile':small_quantile,'large_quantile':large_quantile}


def select_disjoint_pairs(pairs,episodes,*,max_pairs:int,episode_cap:int,descending:bool):
    """Select scored cross-episode pairs with no reused windows."""
    if max_pairs<1 or episode_cap<1:raise ValueError('invalid pair selection limits')
    used=set();epuse=Counter();out=[]
    order=sorted(pairs,key=lambda r:((-r[2] if descending else r[2]),int(r[0]),int(r[1])))
    for item in order:
        i,j=int(item[0]),int(item[1]);ei=tuple(episodes[i]) if isinstance(episodes[i],(list,tuple,np.ndarray)) else episodes[i]
        ej=tuple(episodes[j]) if isinstance(episodes[j],(list,tuple,np.ndarray)) else episodes[j]
        if i==j or i in used or j in used or ei==ej or epuse[ei]>=episode_cap or epuse[ej]>=episode_cap:continue
        out.append(item);used.update((i,j));epuse[ei]+=1;epuse[ej]+=1
        if len(out)>=max_pairs:break
    return out


def geometric_candidates(data,*,seed:int,limit:int,goal_distance_m:float,
                         goal_direction_deg:float,min_mask_iou:float,max_dem_rmse_m:float):
    """Geometry-only candidate pairs, independent of any quality label or trajectory."""
    goals=np.asarray(data['goal'],dtype=np.float32);terrains=np.asarray(data['terrain'])
    dems=np.asarray(data['dem'],dtype=np.float32);episodes=np.asarray(data['episode'])
    rng=np.random.default_rng(seed);candidates=[]
    terrain_types=sorted(set(terrains))
    per_terrain=max(1,limit//len(terrain_types))
    for terrain in terrain_types:
        terrain_count=0
        ids=np.flatnonzero(terrains==terrain);tree=cKDTree(goals[ids])
        for i in rng.permutation(ids):
            neighbors=tree.query_ball_point(goals[i],goal_distance_m)
            neighbors=sorted((int(ids[j]) for j in neighbors if int(ids[j])>i),
                             key=lambda j:(float(np.linalg.norm(goals[i]-goals[j])),j))[:12]
            for j in neighbors:
                if np.array_equal(episodes[i],episodes[j]):continue
                angle=goal_direction_delta_deg(goals[i],goals[j])
                if angle>goal_direction_deg:continue
                ma=dems[i,:,1]>.5;mb=dems[j,:,1]>.5
                common=ma&mb;union=ma|mb
                iou=float(common.sum()/max(int(union.sum()),1))
                if iou<min_mask_iou or not common.any():continue
                rmse=float(np.sqrt(np.mean((dems[i,:,0][common]-dems[j,:,0][common])**2)))
                if not np.isfinite(rmse) or rmse>max_dem_rmse_m:continue
                candidates.append((int(i),int(j),float(np.linalg.norm(goals[i]-goals[j])),angle,iou,rmse))
                terrain_count+=1
                if terrain_count>=per_terrain:break
            if terrain_count>=per_terrain:break
    return candidates


def episode_id_list(value) -> list[int]:
    """Return stable plain integers for JSON pair identities."""
    return [int(x) for x in value]
