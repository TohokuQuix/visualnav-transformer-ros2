"""Fixed-threshold, cross-episode Quality neighborhood diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import EATConfig


@dataclass(frozen=True)
class OverlapReport:
    windows: int
    bin_counts: dict[int,int]
    matching_pairs: int
    multi_bin_anchors: int
    distinct_anchors: int
    distinct_anchor_fraction: float
    by_terrain: dict[str,dict]
    by_goal_distance: dict[str,dict]
    gate_pass: bool


def diagnose_quality_overlap(windows,config:EATConfig):
    windows=list(windows)
    n=len(windows)
    if n==0:
        return OverlapReport(0,{},0,0,0,0.,{},{},False)
    goals=np.asarray([w['goal'] for w in windows],dtype=np.float32)
    dems=np.asarray([w['dem'] for w in windows],dtype=np.float32)
    if dems.shape[1:]!=(config.dem_frame_count,2,config.dem_height_cells,config.dem_width_cells):
        raise ValueError('DEM history shape mismatch')
    masks=dems[:,:,1]>0.5
    heights=dems[:,:,0]
    bins=np.minimum((np.asarray([w['quality'] for w in windows])*4).astype(np.int8),3)
    if (bins<0).any():raise ValueError('quality must be in [0,1]')
    found=[{int(b)} for b in bins]
    pairs=0
    tree=cKDTree(goals)
    for i,nearby in enumerate(tree.query_ball_point(goals,config.overlap_max_goal_distance_m)):
        candidates=np.array([j for j in nearby if j>i and windows[j]['episode_key']!=windows[i]['episode_key']],dtype=np.int32)
        if not len(candidates):continue
        common=np.logical_and(masks[candidates],masks[i])
        union=np.logical_or(masks[candidates],masks[i]).sum(axis=(1,2,3))
        iou=common.sum(axis=(1,2,3))/np.maximum(union,1)
        keep=iou>=config.overlap_min_mask_iou
        candidates=candidates[keep]
        common=common[keep]
        if not len(candidates):continue
        rmse=np.sqrt(np.sum(np.square(heights[candidates]-heights[i])*common,axis=(1,2,3))/np.maximum(common.sum(axis=(1,2,3)),1))
        candidates=candidates[rmse<=config.overlap_max_height_rmse_m]
        pairs+=len(candidates)
        for j in candidates:
            found[i].add(int(bins[j]));found[j].add(int(bins[i]))
    multi=[i for i,values in enumerate(found) if len(values)>1]
    selected=[]
    last={}
    for i in sorted(multi,key=lambda i:(windows[i]['episode_key'],windows[i]['step'])):
        key=windows[i]['episode_key']
        if windows[i]['step']-last.get(key,-10**9)>=40:
            selected.append(i);last[key]=windows[i]['step']
    multi_set=set(multi);selected_set=set(selected)
    def counts(indices):
        indices=list(indices)
        return {'windows':len(indices),'multi_bin_anchors':sum(i in multi_set for i in indices),
                'distinct_anchors':sum(i in selected_set for i in indices)}
    by_terrain={t:counts(i for i,w in enumerate(windows) if w['terrain_type']==t) for t in sorted({w['terrain_type'] for w in windows})}
    distances=np.linalg.norm(goals,axis=1)
    bands=[('0-.2',0,.2),('.2-.4',.2,.4),('.4-.6',.4,.6),('.6-.8',.6,float('inf'))]
    by_goal={name:counts(i for i,v in enumerate(distances) if lo<=v<hi) for name,lo,hi in bands}
    fraction=len(selected)/n
    return OverlapReport(n,dict(Counter(map(int,bins))),pairs,len(multi),len(selected),fraction,by_terrain,by_goal,
                         len(selected)>=config.overlap_min_distinct_anchors and fraction>=config.overlap_min_anchor_fraction)
