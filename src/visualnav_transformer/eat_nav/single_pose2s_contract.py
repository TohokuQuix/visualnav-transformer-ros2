"""Versioned single-observation contract; no goal or future measurements."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from .demo_terrain import TERRAIN_TYPES
from .bidirectional_bank import bank

INPUT_SCHEMA='eatnav_single_dem_pose2s_input_v1'
MODEL_SCHEMA='eatnav_single_dem_pose2s_model_v1'
ACTOR_SHA='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d'
CONTROL_DT=.05
PHYSICS_DT=.005
HORIZON_STEPS=40
TRIAL_STEPS=200
REQUIRED=tuple(list(range(20))+[23,24,25,26])
PROFILE=dict(actor_sha256=ACTOR_SHA,fold_mode=0,control_dt=.05,physics_dt=.005,dem_hz=4,dem_cells=40,dem_resolution_m=.1,lookahead_m=.5,linear_speed_mps=.3,angular_gain=.5,max_angular_rps=1.6,goal_tolerance_m=.2,recovery=False,angular_boost=False,measured_tilt_stop=False)
PROFILE_SHA=hashlib.sha256(json.dumps(PROFILE,sort_keys=True).encode()).hexdigest()


def make_manifest(seed:int=123)->dict:
    if not isinstance(seed,int) or seed<0:raise ValueError('nonnegative seed required')
    parents=[];used=set()
    for role_id,(role,count) in enumerate([('train',10),('validation',14),('final_test',14)]):
        for fi,field in enumerate(TERRAIN_TYPES):
            for level in range(5):
                permutation=np.random.default_rng(np.random.SeedSequence([seed,role_id,fi,level,91])).permutation(14)
                for r in range(count):
                    stream=np.random.SeedSequence([seed,role_id,fi,level,r])
                    pseed=int(stream.generate_state(1)[0])
                    if pseed in used:raise ValueError('seed collision')
                    used.add(pseed)
                    parents.append(dict(parent_id=f'{role}-{field}-l{level}-{r:03d}',role=role,field=field,level=level,trial_index=r,seed=pseed,initial_seed=int(np.random.SeedSequence([pseed,1]).generate_state(1)[0]),dem_phase_steps=r%5,candidates=[int(permutation[(r+3*j)%14]) for j in range(5)]))
    return dict(schema='eatnav_single_dem_pose2s_manifest_v1',seed=seed,fields=list(TERRAIN_TYPES),parents=parents,profile=PROFILE,profile_sha256=PROFILE_SHA,expected_groups=490,planned_windows=len(parents)*5)


def validate_input(sample:dict)->None:
    if sample.get('schema')!=INPUT_SCHEMA:raise ValueError('wrong input schema')
    tails=dict(dem=(3,40,40),state=(27,),state_mask=(27,),state_age=(27,),dem_context=(6,))
    for key,shape in tails.items():
        a=np.asarray(sample[key])
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid '+key)
    dem=np.asarray(sample['dem']);mask=np.asarray(sample['state_mask']);age=np.asarray(sample['state_age'])
    if np.any((dem[1:]<0)|(dem[1:]>1)) or not np.isin(mask,[0,1]).all():raise ValueError('invalid mask')
    if not np.all(mask[list(REQUIRED)]==1):raise ValueError('missing required state')
    if np.any(age[mask==1]<0) or np.any(age[mask==1]>.20+1e-9):raise ValueError('stale/future state')
    t=int(sample['t0_ns']);stamp=int(sample['dem_stamp_ns']);received=int(sample['dem_received_ns'])
    if stamp>t or received>t or t-stamp>500000000:raise ValueError('DEM unavailable/future/stale')
    if abs(float(sample['dem_context'][0])-(t-stamp)*1e-9)>1e-6:raise ValueError('DEM age inconsistent')
    paths=np.asarray(sample['paths']);directions=np.asarray(sample['directions'])
    if paths.ndim!=3 or paths.shape[1:]!=(8,2) or not len(paths) or not np.isfinite(paths).all():raise ValueError('invalid paths')
    if directions.shape!=(len(paths),1) or not np.isin(directions,[-1,1]).all():raise ValueError('invalid direction')
    if not np.any(dem[1]):raise ValueError('DEM entirely unavailable')
