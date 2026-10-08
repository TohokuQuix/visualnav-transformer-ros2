"""Pure offline geometry and contract checks for the 0.75 m goal review."""

import math
import itertools

import numpy as np

from .curvature_attainable import solve_symmetric_target
from .goal_bank_offline import audit_geometry, corridor_audit, follower_diagnostic


BEARINGS_DEG = (-20, -10, 0, 10, 20)
GOAL_DISTANCE_M = .75
FOLLOWER_SCREEN_PER_M = 1.236
INITIAL_TANGENT_TOLERANCE_RAD = 1e-6


def goal075(bearing_deg):
    if bearing_deg not in BEARINGS_DEG:
        raise ValueError('bearing outside frozen contract')
    angle=math.radians(bearing_deg)
    return np.array([GOAL_DISTANCE_M*math.cos(angle),GOAL_DISTANCE_M*math.sin(angle)])


def contract075():
    return {'schema':'eatnav_common_goal075_offline_v1',
            'goal_distance_m':GOAL_DISTANCE_M,'bearings_deg':list(BEARINGS_DEG),
            'future_path_points':8,'path_dimensions':2,'path_frame':'robot-local-base',
            'path_endpoint':'common_global_goal','initial_point_xy_m':[0.,0.],
            'initial_tangent_xy':[1.,0.],
            'initial_tangent_tolerance_rad':INITIAL_TANGENT_TOLERANCE_RAD,
            'endpoint_tolerance_m':.001,'curvature_screen_per_m':FOLLOWER_SCREEN_PER_M,
            'reverse_estimate_allowed':False,'turn_in_place_estimate_allowed':False,
            'teacher_steps':40,'teacher_step_s':.05,'teacher_duration_s':2.,
            'primary_terrain_arc_end_m':.6,'diagnostic_path_arc_end_m':.75,
            'nominal_speed_mps':.3,'nominal_time_to_endpoint_s':2.5,
            'candidate_count_primary':5,'candidate_sides':['direct','left','left','right','right'],
            'source_anchor_count':8,'state_group_count':40,
            'separation_frame':'goal-ray-relative',
            'no_rollout':True}


def initial_forward_tangent_valid(tangent):
    vector=np.asarray(tangent,dtype=float)
    if vector.shape!=(2,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<=1e-12:
        return False
    angle=math.atan2(vector[1],vector[0])
    return abs(angle)<=INITIAL_TANGENT_TOLERANCE_RAD


def side_relative_to_direct(curve,direct,goal):
    goal=np.asarray(goal,dtype=float)
    normal=np.array([-goal[1],goal[0]])/np.linalg.norm(goal)
    common=min(.6,float(curve['stations_m'][-1]),float(direct['stations_m'][-1]))
    stations=np.linspace(0,common,101)
    def sampled(row):
        return np.stack([np.interp(stations,row['stations_m'],row['dense'][:,axis])
                         for axis in range(2)],axis=1)
    signed=float(np.mean((sampled(curve)-sampled(direct))@normal))
    return ('left' if signed>1e-6 else 'right' if signed< -1e-6 else 'direct',signed)


def shared_symmetric_bank(ids,sides,compatible_by_group,conflicts_by_group,direct_id,*,pair_count):
    if len(ids)!=len(sides) or len(set(ids))!=len(ids):
        raise ValueError('unique aligned candidate IDs required')
    common=set(ids)
    for supplied in compatible_by_group.values():
        common.intersection_update(supplied)
    subset=[(identifier,side) for identifier,side in zip(ids,sides) if identifier in common]
    if direct_id not in common:
        return {'feasible':False,'selected_ids':[],'reason':'direct_missing'}
    local={identifier:index for index,(identifier,_) in enumerate(subset)}
    conflicts=set()
    for edges in conflicts_by_group.values():
        for first,second in edges:
            if first in local and second in local:
                i,j=sorted((local[first],local[second]))
                conflicts.add((i,j))
    solved=solve_symmetric_target([side for _,side in subset],sorted(conflicts),
                                   pair_count=pair_count,direct_index=local[direct_id])
    selected=[subset[index][0] for index in solved['selected_indices']] if solved['feasible'] else []
    return {'feasible':solved['feasible'],'selected_ids':selected,
            'reason':None if solved['feasible'] else solved.get('solver_message',solved.get('solver_status')),
            'solver_status':solved.get('solver_status')}


def audit_candidate(curve,goal,anchor,follower_config):
    goal=np.asarray(goal,dtype=float)
    path=np.asarray(curve['waypoints'],dtype=float)
    row={'bank':'A','dense':curve['dense'],'stations_m':curve['stations_m'],
         'waypoints':path,'endpoint':path[-1],'global_goal':goal,
         'arc_length_m':curve['arc_length_m'],
         'maximum_curvature_per_m':curve['analytic_or_dense_max_curvature_per_m']}
    geometry=audit_geometry(row,goal,anchor['settled_map_xy_yaw'])
    hard_reasons=list(geometry['invalid_reasons'])
    if not initial_forward_tangent_valid(curve['initial_tangent_xy']):
        hard_reasons.append('initial_tangent_error')
    endpoint_error=float(np.linalg.norm(path[-1]-goal))
    if endpoint_error>.001 or curve['endpoint_error_m']>.001:
        hard_reasons.append('common_goal_endpoint_error')
    if len(np.unique(path,axis=0))!=8:
        hard_reasons.append('duplicate_waypoint')
    dense=np.asarray(curve['dense'])
    cols=np.rint(dense[:,0]/.1+9.5).astype(int)
    rows=np.rint(dense[:,1]/.1+9.5).astype(int)
    inside=(np.abs(dense[:,0])<=1.)&(np.abs(dense[:,1])<=1.)&(
        (cols>=0)&(cols<20)&(rows>=0)&(rows<20))
    mask=np.asarray(anchor['dem'][1],dtype=float)
    known=np.zeros(len(dense),dtype=bool)
    known[inside]=mask[rows[inside],cols[inside]]>.5
    if not known.any():hard_reasons.append('unknown_only_centerline')
    command=follower_diagnostic(row,follower_config)
    follower_reasons=[]
    if curve['analytic_or_dense_max_curvature_per_m']>FOLLOWER_SCREEN_PER_M+1e-9:
        follower_reasons.append('dense_curvature_screen')
    if curve['eight_point_max_curvature_per_m']>FOLLOWER_SCREEN_PER_M+1e-9:
        follower_reasons.append('eight_point_curvature_screen')
    if command['command_mode']!='tracking':
        follower_reasons.append('source_first_command_'+command['command_mode'])
    if command['reverse_estimate']:
        follower_reasons.append('reverse_estimate')
    if command['turn_in_place_entry']:
        follower_reasons.append('turn_in_place_estimate')
    first=command['initial_command_estimate']
    velocity=first.get('linear_x');yaw_rate=first.get('angular_z')
    if (velocity is None or yaw_rate is None or
        not math.isfinite(velocity) or not math.isfinite(yaw_rate) or
        velocity<0 or velocity>follower_config['linear_speed']+1e-9 or
        abs(yaw_rate)>follower_config['max_angular_speed']+1e-9):
        follower_reasons.append('nonfinite_or_out_of_bounds_first_command')
    hard_valid=not hard_reasons
    compatible=hard_valid and not follower_reasons
    dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes']) if compatible else None
    return {'family':curve['family'],'variant_id':curve['variant_id'],
            'candidate_id':curve['family']+':'+curve['variant_id'],
            'hard_valid':hard_valid,'compatible':compatible,
            'hard_reasons':hard_reasons,'follower_reasons':follower_reasons,
            'reasons':hard_reasons+follower_reasons,
            'endpoint_error_m':endpoint_error,
            'initial_tangent_error_rad':abs(math.atan2(float(curve['initial_tangent_xy'][1]),
                                                       float(curve['initial_tangent_xy'][0]))),
            'dense_max_curvature_per_m':curve['analytic_or_dense_max_curvature_per_m'],
            'eight_point_max_curvature_per_m':curve['eight_point_max_curvature_per_m'],
            'centerline_known_fraction':float(known.mean()),
            'centerline_unknown_fraction':float((inside&~known).mean()),
            'centerline_outside_fraction':float((~inside).mean()),
            'geometry':geometry,'follower':command,'dem_full':dem}


def prefix_row(row,end_m):
    stations=np.asarray(row['stations_m'],dtype=float)
    dense=np.asarray(row['dense'],dtype=float)
    if (stations.ndim!=1 or dense.shape!=(len(stations),2) or
        not np.isfinite(stations).all() or not np.isfinite(dense).all() or
        stations[0]!=0 or np.any(np.diff(stations)<=0) or
        end_m<=0 or end_m>stations[-1]+1e-9):
        raise ValueError('valid increasing dense Path and in-range prefix required')
    keep=stations<end_m-1e-9
    clipped_stations=np.r_[stations[keep],float(end_m)]
    clipped_dense=np.stack([np.interp(clipped_stations,stations,dense[:,axis])
                            for axis in range(2)],axis=1)
    return {**row,'stations_m':clipped_stations,'dense':clipped_dense,
            'arc_length_m':float(end_m)}


def coverage_record(row,height,mask,proxies):
    expected={'main_track_sprocket','initial_flipper_support',
              'articulated_reach','full_collision_union'}
    if set(proxies)!=expected:
        raise ValueError('all four frozen collision proxies required')
    primary=prefix_row(row,.6)
    dem_primary=corridor_audit(primary,height,mask,proxies)
    dem_full=corridor_audit(row,height,mask,proxies)
    curve=np.asarray(row['dense'],dtype=float)
    tangent=np.gradient(curve,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normal=np.stack((-tangent[:,1],tangent[:,0]),axis=1)
    margins={}
    for name,bounds in proxies.items():
        low,high=np.asarray(bounds,dtype=float)
        corners=np.array(list(itertools.product((low[0],high[0]),(low[1],high[1]))))
        touched=(curve[:,None,:]+corners[None,:,0,None]*tangent[:,None,:]
                 +corners[None,:,1,None]*normal[:,None,:])
        margins[name]=float(np.min(1-np.max(np.abs(touched),axis=2)))
    return {'primary_centerline_known_fraction':dem_primary['centerline_known_fraction'],
            'primary_centerline_unknown_fraction':dem_primary['centerline_unknown_fraction'],
            'primary_centerline_outside_fraction':dem_primary['centerline_outside_fraction'],
            'full_centerline_known_fraction':dem_full['centerline_known_fraction'],
            'full_centerline_unknown_fraction':dem_full['centerline_unknown_fraction'],
            'full_centerline_outside_fraction':dem_full['centerline_outside_fraction'],
            'minimum_full_tile_margin_m':float(np.min(1-np.max(np.abs(curve),axis=1))),
            'proxies':{name:{'primary':{key:dem_primary['corridors'][name][key] for key in
                            ('known_fraction','unknown_fraction','outside_fraction')},
                             'full':{key:dem_full['corridors'][name][key] for key in
                            ('known_fraction','unknown_fraction','outside_fraction')},
                             'minimum_full_tile_margin_m':margins[name]}
                       for name in proxies},
            '_dem_primary':dem_primary,'_dem_full':dem_full}


def variable_count_mask(candidate_ids,direct_id):
    if len(candidate_ids)!=len(set(candidate_ids)):
        raise ValueError('duplicate candidate ID is forbidden')
    if not 3<=len(candidate_ids)<=5 or direct_id not in candidate_ids:
        raise ValueError('three to five real candidates including direct required')
    return {'candidate_ids':list(candidate_ids),'candidate_count':len(candidate_ids),
            'valid_mask':[True]*len(candidate_ids),
            'padding_candidate_count':0,
            'loss_reduction':'sum(per_candidate_loss * valid_mask) / sum(valid_mask)',
            'diagnostic_only':True}
