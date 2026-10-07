"""Descriptive DEM coverage of frozen USD collision-bound corner proxies."""
from __future__ import annotations

import math

import numpy as np

CATEGORIES=('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')


def footprint_diagnostic(mask, local_points: dict[str,list[tuple[float,float]]], *,
                         radius_m: float, theta_rad: float, tile_half_extent_m: float = 4.0) -> dict:
    known=np.asarray(mask,dtype=bool)
    if known.shape!=(20,20) or not math.isfinite(radius_m) or not math.isfinite(theta_rad):
        raise ValueError('frozen DEM shape or pose invalid')
    result={}
    for category in CATEGORIES:
        points=np.asarray(local_points.get(category,[]),dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)==0 or not np.isfinite(points).all():
            raise ValueError('missing or nonfinite collision proxy: '+category)
        xindex=np.rint(points[:,0]/.1+9.5).astype(int)
        yindex=np.rint(points[:,1]/.1+9.5).astype(int)
        inside=(xindex>=0)&(xindex<20)&(yindex>=0)&(yindex<20)
        observed=np.zeros(len(points),dtype=bool)
        observed[inside]=known[yindex[inside],xindex[inside]]
        result[category]={'sample_count':len(points),'inside_dem_fraction':float(inside.mean()),
            'outside_dem_fraction':float(1-inside.mean()),
            'known_fraction':float(observed.mean()),'unknown_or_outside_fraction':float(1-observed.mean()),
            'local_bbox_xy_m':[points.min(axis=0).tolist(),points.max(axis=0).tolist()]}
    full=np.asarray(local_points['full_collision_union'],dtype=float)
    c,s=math.cos(theta_rad),math.sin(theta_rad)
    world_local_x=radius_m*c+c*full[:,0]-s*full[:,1]
    world_local_y=radius_m*s+s*full[:,0]+c*full[:,1]
    result['articulated_reach_m']=float(max(math.hypot(x,y) for x,y in local_points['articulated_reach']))
    result['tile_boundary_margin_m']=float(min(tile_half_extent_m-np.max(np.abs(world_local_x)),
                                           tile_half_extent_m-np.max(np.abs(world_local_y))))
    result['proxy_semantics']='USD collision world AABB corners transformed to initial root-local XY; diagnostic only'
    return result
