"""Prospective outcome-blind geometry classification of an anchor-time DEM."""
from __future__ import annotations

import numpy as np

HEIGHT_CHANGE_M = .04
STEP_CHANGE_M = .05
GRADE_SLOPE = .12
DIFFERENTIAL_M = .04
MIN_CORRIDOR_VALID = .70
MIN_BAND_VALID = .50
ASSUMED_TWO_SECOND_REACH_M = .60
ASSUMED_SPEED_MPS = .30


def classify_dem(height, mask, resolution_m: float) -> dict:
    height=np.asarray(height,dtype=float);valid=np.asarray(mask,dtype=float)>.5
    if height.shape!=(20,20) or valid.shape!=(20,20) or not np.isfinite(height).all() or resolution_m!=.1:
        raise ValueError('frozen 20x20 DEM, 0.1 m resolution, finite height required')
    x=(np.arange(20)-9.5)*resolution_m
    y=(np.arange(20)-9.5)*resolution_m
    cols=np.where((x>=0)&(x<=ASSUMED_TWO_SECOND_REACH_M))[0]
    bands={'left':np.where((y>=.15)&(y<=.45))[0],
           'center':np.where(np.abs(y)<=.15)[0],
           'right':np.where((y<=-.15)&(y>=-.45))[0]}
    corridor_fraction=float(valid[:,cols].mean())
    band_coverage={name:float(valid[rows[:,None],cols].mean()) for name,rows in bands.items()}
    result={'schema':'eatnav_radial_dem_interaction_v1','class':'unknown',
            'interaction_start_m':None,'interaction_within_two_second_proxy':False,
            'supply_candidate':False,'corridor_valid_fraction':corridor_fraction,
            'band_valid_fraction':band_coverage,'assumed_speed_mps':ASSUMED_SPEED_MPS,
            'assumed_two_second_reach_m':ASSUMED_TWO_SECOND_REACH_M,
            'max_forward_slope':None,'max_step_m':None,'forward_height_change_m':None,
            'forward_roughness_m':None,'left_center_right_difference_m':None,
            'unknown_fraction':1-corridor_fraction}
    if corridor_fraction<MIN_CORRIDOR_VALID or min(band_coverage.values())<MIN_BAND_VALID:
        return result
    profiles={}
    for name,rows in bands.items():
        values=[]
        for col in cols:
            selected=height[rows,col][valid[rows,col]]
            values.append(float(np.median(selected)) if len(selected) else np.nan)
        profiles[name]=np.asarray(values)
    if any(not np.isfinite(profile).all() for profile in profiles.values()):
        return result
    center=profiles['center'];all_profiles=np.stack(list(profiles.values()))
    differences=np.ptp(all_profiles,axis=0)
    step=float(max(np.max(np.abs(np.diff(profile))) for profile in profiles.values()))
    baseline=float(np.median(center[:2]))
    delta=float(center[-1]-baseline)
    slope=float(max(np.max(np.abs(np.diff(profile)))/resolution_m for profile in profiles.values()))
    roughness=float(np.std(np.diff(center)))
    lateral=float(np.max(differences))
    result.update({'max_forward_slope':slope,'max_step_m':step,
                   'forward_height_change_m':delta,'forward_roughness_m':roughness,
                   'left_center_right_difference_m':lateral,
                   'left_center_right_profiles_m':{k:v.tolist() for k,v in profiles.items()}})
    if lateral>=DIFFERENTIAL_M:
        cls='differential'
    elif step>=STEP_CHANGE_M and abs(float(center[1]-center[0]))<HEIGHT_CHANGE_M:
        cls='transition'
    elif abs(delta)>=HEIGHT_CHANGE_M or slope>=GRADE_SLOPE:
        cls='graded'
    else:
        cls='flat-only'
    start=None
    for index,col in enumerate(cols):
        change=max(abs(profiles[name][index]-profiles[name][0]) for name in profiles)
        if change>=HEIGHT_CHANGE_M or differences[index]>=DIFFERENTIAL_M or (index>0 and
            max(abs(profiles[name][index]-profiles[name][index-1]) for name in profiles)>=STEP_CHANGE_M):
            start=float(x[col]);break
    result.update({'class':cls,'interaction_start_m':start,
                   'interaction_within_two_second_proxy':start is not None and start<=ASSUMED_TWO_SECOND_REACH_M,
                   'supply_candidate':cls in ('transition','graded','differential') and start is not None})
    return result
