"""Numeric contracts for the approved seven-field continuation experiment."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
from eat_nav_risk_navigation import candidate_paths
from visualnav_transformer.eat_nav.fall_stall_targets import path_arc

FIELDS=('k_rail_field','k_rail_step_field','pyramid_stairs','pyramid_stairs_inv','random_rough','boxes','hf_pyramid_slope')
ANGLES=(0.,-math.pi/6,math.pi/6,0.)
WAYPOINTS=np.array([[0,-2],[2,-2],[2,2],[-2,2],[-2,-2],[0,0]],dtype=float)

def rotation(a):
    return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])

def extend_path(points,terminal_heading,candidate):
    points=np.asarray(points,dtype=float)
    tail=points[-1]+candidate_paths()[candidate].astype(float)@rotation(terminal_heading).T
    return np.concatenate((points,tail[1:])),terminal_heading+ANGLES[candidate]

def needs_extension(points,pose):
    points=np.asarray(points);position=np.asarray(pose)[:2]
    total=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    return total-path_arc(position.tolist(),points.tolist())<=.30 or np.linalg.norm(position-points[-1])<=.30

def geometry_choice(goal_body):
    endpoints=candidate_paths()[:,-1].astype(float)
    proxy=.75*endpoints/np.linalg.norm(endpoints,axis=1,keepdims=True)
    return int(np.argmin(np.linalg.norm(np.asarray(goal_body)[None]-proxy,axis=1)))

def field_bounds(field,size,level,*,rows=5,cols=1,col=0,pallet=(2.4,1.2),counts=(0,0)):
    # Isaac TerrainGenerator._get_terrain_mesh centers the local mesh;
    # _add_sub_terrain places it at (row+.5)*size; constructor centers full grid.
    size=np.asarray(size,dtype=float)
    center=np.array([(level+.5-rows/2)*size[0],(col+.5-cols/2)*size[1]])
    half=size/2
    placed=half.copy()
    if field in ('k_rail_field','k_rail_step_field'):
        n=np.asarray([counts[i] if counts[i]>0 else max(1,int(size[i]//pallet[i])) for i in range(2)])
        placed=np.minimum(half,n*np.asarray(pallet)/2)
    safe=placed-2.
    if np.any(safe<=0) or np.any(np.max(np.abs(WAYPOINTS),axis=0)>safe):raise ValueError('route outside inset field')
    return dict(tile_center=center.tolist(),tile_bounds=[*(center-half),*(center+half)],
                placement_bounds=[*(center-placed),*(center+placed)],
                root_bounds=[*(center-safe),*(center+safe)],margin_m=2.,generator_rows=rows,generator_cols=cols)

def inside(xy,bounds):
    lo=np.asarray(bounds[:2]);hi=np.asarray(bounds[2:])
    return bool(np.all(np.asarray(xy)>=lo)&np.all(np.asarray(xy)<=hi))

class Route:
    def __init__(self,center,spawn):
        self.points=WAYPOINTS+np.asarray(center);self.index=0;self.laps=0;self.count=0
        self.start=np.asarray(spawn,dtype=float);self.completed_m=0.;self.entries=[]
    @property
    def goal(self):return self.points[self.index]
    def progress(self,xy):
        d=self.goal-self.start;length=float(np.linalg.norm(d))
        return self.completed_m+(float(np.clip(np.dot(np.asarray(xy)-self.start,d)/(length**2),0,1))*length if length>1e-12 else 0.)
    def observe(self,xy,tilt,stopped,failed):
        entered=bool(np.linalg.norm(np.asarray(xy)-self.goal)<=.25)
        normal=entered and tilt<45 and not stopped and not failed
        result=dict(entered=entered,normal=normal,index=self.index,lap=self.laps)
        if normal:
            self.completed_m+=float(np.linalg.norm(self.goal-self.start));self.start=self.goal.copy()
            self.index=(self.index+1)%len(self.points);self.count+=1
            if self.index==0:self.laps+=1
            self.entries.append(result.copy())
        return result

def compare_anchor(actual,reference):
    errors=[]
    if set(actual)!=set(reference):return ['keys']
    for k in sorted(actual):
        a,b=np.asarray(actual[k]),np.asarray(reference[k])
        if a.dtype.hasobject or b.dtype.hasobject:raise ValueError('object anchor forbidden')
        if a.shape!=b.shape or a.dtype!=b.dtype:errors.append(k);continue
        same=np.allclose(a,b,atol=1e-6,rtol=0,equal_nan=False) if a.dtype.kind in 'fc' else np.array_equal(a,b)
        if not same:errors.append(k)
    return errors

def make_manifest(registry,previous_seeds):
    excluded={int(g['terrain_seed']) for g in registry['groups']}
    excluded.update(int(s) for row in registry['known_exclusions'] for s in row['base_seeds'])
    excluded.update(previous_seeds)
    def subs(seed):return {int(np.random.SeedSequence([seed,stream,i]).generate_state(1)[0]) for stream in range(3) for i in range(20)}
    used=set().union(*(subs(s) for s in excluded));groups=[];collect=[];evaluate=[]
    for field in FIELDS:
        for slot in range(11):
            token=f'eatnav-all-fields-v0.1-20261023|{field}|{slot}'
            seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'big')
            derived=subs(seed)
            if not seed or seed in excluded or derived&used:raise ValueError('seed collision: '+token)
            excluded.add(seed);used.update(derived)
            role='train' if slot<6 else 'validation' if slot<8 else 'development_evaluation'
            groups.append(dict(field=field,slot=slot,seed=seed,role=role,token=token,sub_seeds=sorted(derived)))
            if slot<8:
                for anchor in (0,10):
                    for candidate in range(4):collect.append(dict(field=field,seed=seed,role=role,slot=slot,anchor=anchor,candidate=candidate,phase='collect'))
            else:
                offset=slot-8
                for condition in (['R0','R1','R2','R3'][offset:]+['R0','R1','R2','R3'][:offset]):
                    evaluate.append(dict(field=field,seed=seed,role=role,slot=slot,condition=condition,phase='evaluate'))
    return dict(schema='eatnav_all_fields_suite_v1',fields=list(FIELDS),groups=groups,collect=collect,evaluate=evaluate,
                reserved_test_opened=False,tile_size_m=[16.,16.],level=4,control_hz=20,replan_steps=40,
                branch_steps=200,evaluation_steps=1200,teacher='eatnav_fall_stall_targets_v2',max_attempts_per_run=2)

def delay_snapshot(env):
    """Numeric, read-only snapshot of active Onix latency and history rings."""
    result={}
    def save(key,value):
        if value is None:result[key]=np.empty(0,dtype=np.float32)
        else:result[key]=np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value).copy()
    for name in ('_latency_substeps','_onix_latency_seconds','_latency_policy_obs_prev','_latency_policy_obs_reset_mask'):
        save('latency/'+name,getattr(env,name,None))
    for name,delay in getattr(env,'_action_latency_buffers',{}).items():
        ring=delay._circular_buffer;prefix='delay/'+name+'/'
        save(prefix+'lags',delay._time_lags);save(prefix+'buffer',ring._buffer)
        save(prefix+'num_pushes',ring._num_pushes);save(prefix+'pointer',ring._pointer)
    manager=getattr(env,'observation_manager',None)
    for group,terms in getattr(manager,'_group_obs_term_history_buffer',{}).items():
        for name,ring in terms.items():
            prefix='obs_history/'+group+'/'+name+'/'
            save(prefix+'buffer',ring._buffer);save(prefix+'num_pushes',ring._num_pushes);save(prefix+'pointer',ring._pointer)
    return result
