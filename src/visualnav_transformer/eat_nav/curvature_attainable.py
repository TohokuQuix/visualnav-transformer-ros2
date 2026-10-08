"""Deterministic offline curve constructions and finite-library mode solver.

No simulator, follower process, learned model, reward or Outcome is used here.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, least_squares
from scipy.sparse import lil_matrix

from .candidate_bank_geometry import arc_resample
from .goal_bank_offline import bezier_candidate, path_sha


CURVATURE_LIMIT_PER_M = 1.236
ENDPOINT_TOLERANCE_M = .001
DENSE_SPACING_MAX_M = .02


def cubic_controls(goal, forward_handle_m: float, terminal_handle_m: float,
                   terminal_lateral_m: float) -> np.ndarray:
    goal=np.asarray(goal,dtype=float)
    normal=np.array([-goal[1],goal[0]])/np.linalg.norm(goal)
    unit=goal/np.linalg.norm(goal)
    return np.stack(([0.,0.],[forward_handle_m,0.],
                     goal-terminal_handle_m*unit+terminal_lateral_m*normal,goal))


def _cubic(controls: np.ndarray,t: np.ndarray) -> np.ndarray:
    u=1-t
    return (u[:,None]**3*controls[0]+3*(u*u*t)[:,None]*controls[1]
            +3*(u*t*t)[:,None]*controls[2]+t[:,None]**3*controls[3])


def _cubic_curvature(controls: np.ndarray,t: np.ndarray) -> np.ndarray:
    u=1-t
    d=3*(u[:,None]**2*(controls[1]-controls[0])
         +2*(u*t)[:,None]*(controls[2]-controls[1])
         +t[:,None]**2*(controls[3]-controls[2]))
    dd=6*(u[:,None]*(controls[2]-2*controls[1]+controls[0])
          +t[:,None]*(controls[3]-2*controls[2]+controls[1]))
    speed=np.linalg.norm(d,axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return (d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/speed**3


def eight_point_curvature(points) -> float:
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3:
        raise ValueError('at least three XY samples required')
    a=points[1:-1]-points[:-2]
    b=points[2:]-points[1:-1]
    c=points[2:]-points[:-2]
    denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(c,axis=1)
    if np.any(denominator<=1e-12):
        return math.inf
    signed=2*(a[:,0]*b[:,1]-a[:,1]*b[:,0])/denominator
    return float(np.max(np.abs(signed)))


def _pack(family: str, variant: str, source: np.ndarray,
          signed_curvature: np.ndarray, initial_tangent: np.ndarray,
          endpoint: np.ndarray, *, source_parameters: dict) -> dict:
    source=np.asarray(source,dtype=float)
    curvature=np.asarray(signed_curvature,dtype=float)
    if source.shape!=(len(curvature),2) or not np.isfinite(source).all() or not np.isfinite(curvature).all():
        raise ValueError('finite matched points and curvature required')
    if np.linalg.norm(source[0])>1e-10:
        raise ValueError('start is not robot origin')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(source,axis=0),axis=1))]
    if np.any(np.diff(arc)<=1e-12):
        raise ValueError('nonmonotone sampled curve')
    stations=np.linspace(0,float(arc[-1]),math.ceil(float(arc[-1])/.015)+1)
    dense=np.stack([np.interp(stations,arc,source[:,axis]) for axis in range(2)],axis=1)
    profile=np.interp(stations,arc,curvature)
    waypoints=arc_resample(source,8)
    terminal=(source[-1]-source[-2])/np.linalg.norm(source[-1]-source[-2])
    path_hash=hashlib.sha256(json.dumps(waypoints.tolist(),separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return {'family':family,'variant_id':variant,'source_parameters':source_parameters,
            'dense':dense,'stations_m':stations,'waypoints':waypoints,
            'signed_curvature_profile_per_m':profile,
            'analytic_or_dense_max_curvature_per_m':float(np.max(np.abs(curvature))),
            'eight_point_max_curvature_per_m':eight_point_curvature(np.vstack(([0.,0.],waypoints))),
            'arc_length_m':float(arc[-1]),'endpoint':np.asarray(endpoint,dtype=float),
            'endpoint_error_m':float(np.linalg.norm(source[-1]-endpoint)),
            'initial_tangent_xy':np.asarray(initial_tangent,dtype=float),
            'terminal_tangent_xy':terminal,
            'path_sha256':path_hash,
            'maximum_dense_step_m':float(np.max(np.linalg.norm(np.diff(dense,axis=0),axis=1)))}


def build_curve(family: str, controls, goal, *, variant: str='cubic') -> dict:
    controls=np.asarray(controls,dtype=float)
    if controls.shape!=(4,2):
        raise ValueError('cubic controls must be 4x2')
    t=np.linspace(0,1,2049)
    source=_cubic(controls,t)
    curvature=_cubic_curvature(controls,t)
    initial=(controls[1]-controls[0])/np.linalg.norm(controls[1]-controls[0])
    result=_pack(family,variant,source,curvature,initial,np.asarray(goal,dtype=float),
                 source_parameters={'controls_xy_m':controls.tolist()})
    return result


def integrate_profile(length_m: float, curvature_profile) -> np.ndarray:
    if length_m<=0 or not math.isfinite(length_m):
        raise ValueError('positive finite length required')
    t=np.linspace(0,1,513)
    s=t*length_m
    k=np.asarray(curvature_profile(t),dtype=float)
    if k.shape!=t.shape or not np.isfinite(k).all():
        raise ValueError('finite profile required')
    ds=np.diff(s)
    heading=np.r_[0,np.cumsum((k[:-1]+k[1:])*.5*ds)]
    x=np.r_[0,np.cumsum((np.cos(heading[:-1])+np.cos(heading[1:]))*.5*ds)]
    y=np.r_[0,np.cumsum((np.sin(heading[:-1])+np.sin(heading[1:]))*.5*ds)]
    return np.stack((x,y),axis=1)


def profile_curve(family: str, variant: str, goal, length_m: float,
                  curvature_profile, parameters: dict) -> dict:
    source=integrate_profile(length_m,curvature_profile)
    values=np.asarray(curvature_profile(np.linspace(0,1,len(source))),dtype=float)
    return _pack(family,variant,source,values,np.array([1.,0.]),np.asarray(goal),
                 source_parameters=parameters)


def piecewise_arc_line_arc(length_m: float, first_curvature: float,
                           last_curvature: float,
                           fractions: tuple[float,float,float]) -> np.ndarray:
    if len(fractions)!=3 or fractions[0]<=0 or fractions[1]<0 or fractions[2]<=0 or abs(sum(fractions)-1)>1e-9:
        raise ValueError('positive arc and nonnegative line fractions summing to one required')
    lengths=np.asarray(fractions)*length_m
    pieces=[]
    x=y=theta=0.
    for segment,(length,k) in enumerate(zip(lengths,(first_curvature,0.,last_curvature))):
        if length<=1e-12:
            continue
        ds=np.linspace(0,float(length),171)
        if abs(k)<1e-12:
            px=x+ds*np.cos(theta)
            py=y+ds*np.sin(theta)
        else:
            angle=theta+k*ds
            px=x+(np.sin(angle)-math.sin(theta))/k
            py=y+(-np.cos(angle)+math.cos(theta))/k
        points=np.stack((px,py),axis=1)
        pieces.append(points if not pieces else points[1:])
        x,y=float(px[-1]),float(py[-1])
        theta+=k*float(length)
    return np.concatenate(pieces)


def piecewise_curve(variant: str, goal, length_m: float,
                    curvatures: tuple[float,float],fractions: tuple[float,float,float]) -> dict:
    source=piecewise_arc_line_arc(length_m,*curvatures,fractions)
    pieces=[]
    for length,k in zip(np.asarray(fractions)*length_m,(curvatures[0],0.,curvatures[1])):
        if length>1e-12:
            pieces.append(np.full(171 if not pieces else 170,k))
    values=np.concatenate(pieces)
    return _pack('G3',variant,source,values,np.array([1.,0.]),np.asarray(goal),
                 source_parameters={'length_m':length_m,'curvatures_per_m':curvatures,
                                    'fractions':fractions})


def _baseline_curve(bank: str, goal, magnitude: float, side: str) -> dict:
    mode='common' if bank=='A' else 'goal_connected'
    row=bezier_candidate(bank,goal,magnitude,side,mode)
    controls=np.asarray(row['controls'])
    t_source=np.linspace(0,1,513)
    sampled=_cubic(controls,t_source)
    sampled_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(sampled,axis=0),axis=1))]
    t=np.interp(row['stations_m'],sampled_arc,t_source)
    signed=_cubic_curvature(controls,t)
    return {'family':'G0A' if bank=='A' else 'G0B',
            'variant_id':f'{bank}-{side}-{magnitude:.2f}',
            'source_parameters':{'bank':bank,'side':side,'magnitude_m':magnitude,'terminal_mode':mode},
            'dense':row['dense'],'stations_m':row['stations_m'],
            'waypoints':row['waypoints'],'signed_curvature_profile_per_m':signed,
            'analytic_or_dense_max_curvature_per_m':row['maximum_curvature_per_m'],
            'eight_point_max_curvature_per_m':eight_point_curvature(np.vstack(([0.,0.],row['waypoints']))),
            'arc_length_m':row['arc_length_m'],'endpoint':row['endpoint'],
            'endpoint_error_m':float(np.linalg.norm(row['dense'][-1]-row['endpoint'])),
            'initial_tangent_xy':row['initial_tangent_xy'],
            'terminal_tangent_xy':row['terminal_tangent_xy'],
            'path_sha256':path_sha(row['waypoints']),
            'maximum_dense_step_m':float(np.max(np.linalg.norm(np.diff(row['dense'],axis=0),axis=1)))}


def generate_goal_library(goal) -> tuple[list[dict],list[dict]]:
    """Deterministic dense search; return curves and explicit construction failures."""
    goal=np.asarray(goal,dtype=float)
    curves=[];failures=[]
    for bank,values in (('A',(.04,.08,.12,.16,.20)),('B',(.04,.06,.08,.10,.12))):
        curves.append(_baseline_curve(bank,goal,0.,'direct'))
        for side in ('left','right'):
            for magnitude in values:
                curves.append(_baseline_curve(bank,goal,magnitude,side))
    for forward in (.08,.12,.16,.20):
        for terminal in (.08,.14,.20):
            for lateral in (-.08,-.06,-.04,-.02,0.,.02,.04,.06,.08):
                variant=f'a{forward:.2f}-b{terminal:.2f}-c{lateral:+.2f}'
                controls=cubic_controls(goal,forward,terminal,lateral)
                try:
                    curves.append(build_curve('G1',controls,goal,variant=variant))
                except ValueError as exc:
                    failures.append({'family':'G1','variant_id':variant,'reason':str(exc)})
    templates={
        'constant':lambda t:np.ones_like(t),
        'linear_ramp':lambda t:t,
        'symmetric_ramp':lambda t:np.sin(np.pi*t),
    }
    for name,shape in templates.items():
        scale=float(np.max(np.abs(shape(np.linspace(0,1,513)))))
        limit=CURVATURE_LIMIT_PER_M/scale
        beta=math.atan2(goal[1],goal[0])
        initial=np.array([.62,float(np.clip(2*beta/.62,-limit*.8,limit*.8))])
        def residual(variables):
            length,k=variables
            return integrate_profile(length,lambda t:k*shape(t))[-1]-goal
        solved=least_squares(residual,initial,bounds=([.6,-limit],[1.,limit]),
                             max_nfev=90,xtol=1e-11,ftol=1e-11,gtol=1e-11)
        length,k=map(float,solved.x)
        variant=name
        if np.linalg.norm(solved.fun)>ENDPOINT_TOLERANCE_M:
            failures.append({'family':'G2','variant_id':variant,'reason':'endpoint_solver_residual',
                             'endpoint_error_m':float(np.linalg.norm(solved.fun))})
        else:
            curves.append(profile_curve('G2',variant,goal,length,lambda t:k*shape(t),
                                        {'profile':name,'length_m':length,'amplitude_per_m':k}))
    for length in (.60,.66,.72,.78):
        for a in (-.8,-.4,0.,.4,.8):
            variant=f'signchange-L{length:.2f}-a{a:+.1f}'
            def make_profile(b,c):
                return lambda t:a*np.sin(2*np.pi*t)+b*np.sin(4*np.pi*t)+c
            def residual(values):
                return integrate_profile(length,make_profile(*values))[-1]-goal
            solved=least_squares(residual,[0.,0.],bounds=([-1.236,-1.236],[1.236,1.236]),
                                 max_nfev=90,xtol=1e-10,ftol=1e-10,gtol=1e-10)
            b,c=map(float,solved.x)
            profile=make_profile(b,c)
            if np.linalg.norm(solved.fun)>ENDPOINT_TOLERANCE_M:
                failures.append({'family':'G2','variant_id':variant,'reason':'endpoint_solver_residual',
                                 'endpoint_error_m':float(np.linalg.norm(solved.fun))})
            else:
                curves.append(profile_curve('G2',variant,goal,length,profile,
                                            {'profile':'sign_change','length_m':length,
                                             'a_per_m':a,'b_per_m':b,'c_per_m':c}))
    fractions_list=((.5,0.,.5),(.495,.01,.495),(.25,.5,.25),(.2,.5,.3),(.3,.5,.2))
    for length in (.60,.612,.62,.68,.74,.80):
        for fractions in fractions_list:
            for seed_index,initial in enumerate(((0.,0.),(.7,-.7),(-.7,.7))):
                variant=f'L{length:.2f}-f{fractions[0]:.2f}-{seed_index}'
                def residual(curvatures):
                    return piecewise_arc_line_arc(length,*curvatures,fractions)[-1]-goal
                solved=least_squares(residual,initial,
                                     bounds=([-1.236,-1.236],[1.236,1.236]),
                                     max_nfev=90,xtol=1e-10,ftol=1e-10,gtol=1e-10)
                values=tuple(map(float,solved.x))
                if np.linalg.norm(solved.fun)>ENDPOINT_TOLERANCE_M:
                    failures.append({'family':'G3','variant_id':variant,'reason':'endpoint_solver_residual',
                                     'endpoint_error_m':float(np.linalg.norm(solved.fun))})
                else:
                    curves.append(piecewise_curve(variant,goal,length,values,fractions))
    return curves,failures


def solve_distinct_modes(candidate_count: int, conflict_pairs: list[tuple[int,int]],
                         *, time_limit_s: float=20.) -> dict:
    if candidate_count<0:
        raise ValueError('candidate_count must be nonnegative')
    if candidate_count==0:
        return {'optimal':True,'maximum_mode_count':0,'selected_indices':[],'solver_status':0}
    if not conflict_pairs:
        return {'optimal':True,'maximum_mode_count':candidate_count,
                'selected_indices':list(range(candidate_count)),'solver_status':0}
    matrix=lil_matrix((len(conflict_pairs),candidate_count),dtype=float)
    for row,(i,j) in enumerate(conflict_pairs):
        if not (0<=i<j<candidate_count):
            raise ValueError('invalid conflict edge')
        matrix[row,i]=matrix[row,j]=1.
    result=milp(c=-np.ones(candidate_count),integrality=np.ones(candidate_count),
                bounds=Bounds(np.zeros(candidate_count),np.ones(candidate_count)),
                constraints=LinearConstraint(matrix.tocsr(),
                                             -np.inf*np.ones(len(conflict_pairs)),
                                             np.ones(len(conflict_pairs))),
                options={'time_limit':time_limit_s,'mip_rel_gap':0.})
    selected=np.flatnonzero(result.x>.5).tolist() if result.x is not None else []
    return {'optimal':result.status==0,'maximum_mode_count':len(selected),
            'selected_indices':selected,'solver_status':int(result.status),
            'solver_message':str(result.message),
            'dual_bound':float(-result.mip_dual_bound) if hasattr(result,'mip_dual_bound') and result.mip_dual_bound is not None else None}


def solve_symmetric_target(sides: list[str], conflict_pairs: list[tuple[int,int]],
                           *, pair_count: int, direct_index: int,
                           time_limit_s: float=20.) -> dict:
    """Find direct plus equal left/right counts in one mutually distinct set."""
    count=len(sides)
    target=1+2*pair_count
    if sides[direct_index]!='direct' or any(side not in ('direct','left','right') for side in sides):
        raise ValueError('invalid direct or side label')
    if sides.count('left')<pair_count or sides.count('right')<pair_count:
        return {'feasible':False,'selected_indices':[],'solver_status':'insufficient_side_supply'}
    matrix=lil_matrix((len(conflict_pairs)+4,count),dtype=float)
    lower=np.full(len(conflict_pairs)+4,-np.inf)
    upper=np.ones(len(conflict_pairs)+4)
    for row,(i,j) in enumerate(conflict_pairs):
        matrix[row,i]=matrix[row,j]=1.
    row=len(conflict_pairs)
    matrix[row,direct_index]=1.;lower[row]=upper[row]=1.
    matrix[row+1,:]=1.;lower[row+1]=upper[row+1]=target
    for offset,side in enumerate(('left','right'),start=2):
        matrix[row+offset,[i for i,value in enumerate(sides) if value==side]]=1.
        lower[row+offset]=upper[row+offset]=pair_count
    result=milp(c=np.linspace(0,1e-6,count),integrality=np.ones(count),
                bounds=Bounds(np.zeros(count),np.ones(count)),
                constraints=LinearConstraint(matrix.tocsr(),lower,upper),
                options={'time_limit':time_limit_s,'mip_rel_gap':0.})
    selected=np.flatnonzero(result.x>.5).tolist() if result.x is not None else []
    feasible=(len(selected)==target and direct_index in selected
              and sum(sides[i]=='left' for i in selected)==pair_count
              and sum(sides[i]=='right' for i in selected)==pair_count
              and all(not(i in selected and j in selected) for i,j in conflict_pairs))
    return {'feasible':feasible,'selected_indices':selected if feasible else [],
            'solver_status':int(result.status),'solver_message':str(result.message)}
