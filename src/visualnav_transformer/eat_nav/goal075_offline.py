"""Pure offline geometry and contract checks for the 0.75 m goal review."""

import math
import itertools

import numpy as np
from scipy.optimize import least_squares

from .curvature_attainable import solve_symmetric_target
from .curvature_attainable import (
    generate_goal_library, cubic_controls, build_curve, integrate_profile,
    profile_curve, piecewise_arc_line_arc, piecewise_curve,
)
from .goal_bank_offline import audit_geometry, corridor_audit, follower_diagnostic, _sample_grid


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
            'supplemental_G1_forward_handles_m':[.24,.28],
            'supplemental_G1_terminal_handles_m':[.12,.20,.28],
            'supplemental_G1_lateral_m':[-.12,-.08,0.,.08,.12],
            'supplemental_G2_G3_lengths_m':[.75,.78,.80,.84,.88,.92],
            'supplemental_G2_signchange_a_per_m':[-1.,-.5,0.,.5,1.],
            'supplemental_G3_fractions':[[.5,0.,.5],[.25,.5,.25],[.2,.5,.3],[.3,.5,.2]],
            'no_rollout':True}


def generate_goal075_library(goal):
    """Extend the SHA-frozen G0–G3 generator with 0.75 m-relevant lengths."""
    goal=np.asarray(goal,dtype=float)
    curves,failures=generate_goal_library(goal)
    failures=[{**row,'source_parameters':{'frozen_variant_id':row['variant_id'],
              'parameter_detail':'see SHA-frozen G0–G3 generator source'},
              'first_failed_constraint':row['reason']} for row in failures]
    for forward in (.24,.28):
        for terminal in (.12,.20,.28):
            for lateral in (-.12,-.08,0.,.08,.12):
                variant=f'075-a{forward:.2f}-b{terminal:.2f}-c{lateral:+.2f}'
                params={'forward_handle_m':forward,'terminal_handle_m':terminal,
                        'terminal_lateral_m':lateral}
                try:curves.append(build_curve('G1',cubic_controls(goal,forward,terminal,lateral),
                                              goal,variant=variant))
                except ValueError as exc:
                    failures.append({'family':'G1','variant_id':variant,'source_parameters':params,
                                     'reason':str(exc),'first_failed_constraint':str(exc)})
    for length in (.75,.78,.80,.84,.88,.92):
        for a in (-1.,-.5,0.,.5,1.):
            variant=f'075-signchange-L{length:.2f}-a{a:+.1f}'
            def profile(b,c):
                return lambda t:a*np.sin(2*np.pi*t)+b*np.sin(4*np.pi*t)+c
            def residual(values):return integrate_profile(length,profile(*values))[-1]-goal
            solved=least_squares(residual,[0.,0.],bounds=([-1.236,-1.236],[1.236,1.236]),
                                 max_nfev=90,xtol=1e-10,ftol=1e-10,gtol=1e-10)
            b,c=map(float,solved.x)
            params={'profile':'sign_change','length_m':length,'a_per_m':a,
                    'b_per_m':b,'c_per_m':c}
            error=float(np.linalg.norm(solved.fun))
            if error>.001:
                failures.append({'family':'G2','variant_id':variant,'source_parameters':params,
                                 'reason':'endpoint_solver_residual','first_failed_constraint':'endpoint_tolerance_m',
                                 'endpoint_error_m':error})
            else:
                try:curves.append(profile_curve('G2',variant,goal,length,profile(b,c),params))
                except ValueError as exc:
                    failures.append({'family':'G2','variant_id':variant,'source_parameters':params,
                                     'reason':str(exc),'first_failed_constraint':str(exc)})
    fractions_list=((.5,0.,.5),(.25,.5,.25),(.2,.5,.3),(.3,.5,.2))
    for length in (.75,.78,.80,.84,.88,.92):
        for fractions in fractions_list:
            for seed_index,initial in enumerate(((0.,0.),(.7,-.7),(-.7,.7))):
                variant=f'075-L{length:.2f}-f{fractions[0]:.2f}-{seed_index}'
                def residual(curvatures):
                    return piecewise_arc_line_arc(length,*curvatures,fractions)[-1]-goal
                solved=least_squares(residual,initial,bounds=([-1.236,-1.236],[1.236,1.236]),
                                     max_nfev=90,xtol=1e-10,ftol=1e-10,gtol=1e-10)
                values=tuple(map(float,solved.x))
                params={'length_m':length,'curvatures_per_m':values,
                        'fractions':fractions,'solver_seed_index':seed_index}
                error=float(np.linalg.norm(solved.fun))
                if error>.001:
                    failures.append({'family':'G3','variant_id':variant,'source_parameters':params,
                                     'reason':'endpoint_solver_residual','first_failed_constraint':'endpoint_tolerance_m',
                                     'endpoint_error_m':error})
                else:
                    try:curves.append(piecewise_curve(variant,goal,length,values,fractions))
                    except ValueError as exc:
                        failures.append({'family':'G3','variant_id':variant,'source_parameters':params,
                                         'reason':str(exc),'first_failed_constraint':str(exc)})
    return curves,failures


def class_proxy_interval(row,height,mask,end_m):
    """Path-aligned three-band diagnostic, evaluated at a named arc endpoint."""
    from .radial_interaction import DIFFERENTIAL_M,GRADE_SLOPE,HEIGHT_CHANGE_M,STEP_CHANGE_M
    clipped=prefix_row(row,end_m)
    stations=np.linspace(0,end_m,round(end_m/.1)+1)
    center=np.stack([np.interp(stations,clipped['stations_m'],clipped['dense'][:,axis])
                     for axis in range(2)],axis=1)
    tangent=np.gradient(center,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normal=np.stack((-tangent[:,1],tangent[:,0]),axis=1)
    profiles={};coverage={}
    for name,offset in (('left',.3),('center',0.),('right',-.3)):
        _,known,values,_,_=_sample_grid(center+offset*normal,np.asarray(height),np.asarray(mask))
        coverage[name]=float(known.mean())
        profiles[name]=values if known.all() else None
    result={'class_proxy':'unknown','band_known_fraction':coverage,
            'interval_arc_m':[0.,float(end_m)],
            'method':'path_aligned_3_band_diagnostic_not_frozen_anchor_classifier'}
    if min(coverage.values())<.5 or any(value is None for value in profiles.values()):
        return result
    all_profiles=np.stack([profiles[name] for name in ('left','center','right')])
    lateral=float(np.max(np.ptp(all_profiles,axis=0)))
    step=float(np.max(np.abs(np.diff(all_profiles,axis=1))))
    central=profiles['center'];delta=float(central[-1]-np.median(central[:2]))
    slope=step/max(float(stations[1]-stations[0]),1e-12)
    if lateral>=DIFFERENTIAL_M:cls='differential'
    elif step>=STEP_CHANGE_M and abs(float(central[1]-central[0]))<HEIGHT_CHANGE_M:cls='transition'
    elif abs(delta)>=HEIGHT_CHANGE_M or slope>=GRADE_SLOPE:cls='graded'
    else:cls='flat-only'
    return {**result,'class_proxy':cls,'max_lateral_difference_m':lateral,
            'maximum_step_m':step,'forward_height_change_m':delta,'maximum_slope':slope}


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
    full=prefix_row(row,.75)
    dem_primary=corridor_audit(primary,height,mask,proxies)
    dem_full=corridor_audit(full,height,mask,proxies)
    curve=np.asarray(full['dense'],dtype=float)
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
