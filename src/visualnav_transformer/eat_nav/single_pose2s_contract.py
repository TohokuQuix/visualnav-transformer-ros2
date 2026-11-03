"""Versioned single-observation contract; no goal or future measurements."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from .demo_terrain import TERRAIN_TYPES
from .bidirectional_bank import bank

INPUT_SCHEMA='eatnav_single_dem_pose2s_fastlio_input_v2'
MODEL_SCHEMA='eatnav_single_dem_pose2s_fastlio_model_v2'
ACTOR_SHA='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d'
CONTROL_DT=.05
PHYSICS_DT=.005
HORIZON_STEPS=40
TRIAL_STEPS=200
REQUIRED=tuple(range(27))
PROFILE=dict(actor_sha256=ACTOR_SHA,fold_mode=0,control_dt=.05,physics_dt=.005,dem_hz=4,dem_cells=40,dem_resolution_m=.1,lookahead_m=.5,linear_speed_mps=.3,angular_gain=.5,max_angular_rps=1.6,goal_tolerance_m=.2,recovery=False,angular_boost=False,measured_tilt_stop=False,follower_revision='legacy_nominal_end_tangent_v1',initialization='terrain_support_clearance_v1',observation_profile='single_grid_actual_fastlio_pose_velocity_v2',settling_seconds=2,lio_initialization_seconds=2,lidar_profile='terrain_only_16x512_10hz_colocated_root_mesh_placement_v2',imu_hz=20,imu_profile='link_origin_past_velocity_specific_force_v1',fastlio_binary_sha256='d91e9e2806280a56070e55fec2490f754bdbd8e7c85f362911c887bb2c7e026e',fastlio_config_sha256='46e0f4a9aaa8d6542665c988d586b7090fea352751a9f3d285d8adf632071ea8')
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
    if abs(np.linalg.norm(np.asarray(sample['state'])[:3])-1.)>.01:raise ValueError('gravity must be a unit body vector')
    mode=np.asarray(sample['state'])[24:27]
    if not np.isin(mode,[0,1]).all() or mode.sum()!=1:raise ValueError('unknown fold mode')
    t=int(sample['t0_ns']);stamp=int(sample['dem_stamp_ns']);received=int(sample['dem_received_ns'])
    if stamp>t or received>t or t-stamp>500000000:raise ValueError('DEM unavailable/future/stale')
    if abs(float(sample['dem_context'][0])-(t-stamp)*1e-9)>1e-6:raise ValueError('DEM age inconsistent')
    paths=np.asarray(sample['paths']);directions=np.asarray(sample['directions'])
    if paths.ndim!=3 or paths.shape[1:]!=(8,2) or not len(paths) or not np.isfinite(paths).all():raise ValueError('invalid paths')
    if directions.shape!=(len(paths),1) or not np.isin(directions,[-1,1]).all():raise ValueError('invalid direction')
    matched=[]
    for path,direction in zip(paths,directions[:,0]):
        candidates=[i for i,c in enumerate(bank()) if c.direction==direction and np.allclose(path,c.waypoints,rtol=0,atol=1e-6)]
        if len(candidates)!=1:raise ValueError('candidate geometry is not the fixed bank')
        matched.append(candidates[0])
    if len(paths) not in [1,14] or len(set(matched))!=len(paths):raise ValueError('expected one executed candidate or all14 unique candidates')
    if not np.any(dem[1]):raise ValueError('DEM entirely unavailable')
