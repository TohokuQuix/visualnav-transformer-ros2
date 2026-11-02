"""Frozen moving-state collection protocol, CPU gates and resumable Isaac host.

Planning/importing never starts Isaac. Output records distinguish planned,
unavailable, failed and observed branches. PhysX caches are not serializable.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
SOURCE=Path('/home/ubuntu/visualnav-transformer-ros2')
DEFAULT_ROOT=Path('/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/collection')
STATES=('moving_flat','turning','uphill_before','uphill_mid','step_before','descent','slow')
RECIPES=dict(zip(STATES,[('random_rough',1),('boxes',1),('hf_pyramid_slope',4),('hf_pyramid_slope',4),('pyramid_stairs',4),('pyramid_stairs_inv',4),('k_rail_field',4)]))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.tolist())+'\n');tmp.replace(p)


def prepare_collection(parents_per_state=10,excluded_seeds=()):
    if parents_per_state not in (10,20):raise ValueError('approved cohort sizes are 10 or 20')
    used=set(map(int,excluded_seeds));parents=[]
    for state in STATES:
        field,level=RECIPES[state]
        for index in range(parents_per_state):
            nonce=0
            while True:
                seed=int.from_bytes(hashlib.sha256(f'moving-pose2s-20261103/{state}/{index}/{nonce}'.encode()).digest()[:4],'big')
                if seed and seed not in used:break
                nonce+=1
            used.add(seed)
            role=('train' if index<parents_per_state*.6 else 'validation' if index<parents_per_state*.8 else 'audit-eval')
            parents.append(dict(group_id=f'{state}/seed{seed}',state=state,field=field,level=level,seed=seed,terrain_seed=seed,role=role,
                parent_prefix_id=f'moving-pose2s-{seed}',tile_index=index,terrain_cols=parents_per_state,scene='field',index=index))
    return dict(schema='eatnav_moving_pose2s_collection_v1',parents=parents,parents_per_state=parents_per_state,
        planned_branches=len(parents)*14,candidate_count=14,horizon_seconds=2.,horizon_steps=40,control_hz=20,
        prefix_search_limit_sec=20,persistence_samples=4,physics_cache_restorable=False,
        restoration_description='observable snapshot branch; PhysX contact warm-start caches not serialized',
        interventions=dict(append=False,internal_replan=False,protective_stop=False,recovery=False),
        ruling='deadline980 selected before outcomes to reduce implementation and collection risk' if parents_per_state==10 else 'basic1960',
        split_fixed_before_outcomes=True,anchor_qualification='measured causal DEM and physical state; no future observations',
        terrain_predicate=dict(profile_rows=[8,9,10,11],forward_boundary_range_m=[.2,.5],boundary_rise_m=.02,step_rise_m=.06,
                               local_slope_threshold=.08,flat_slope_threshold=.17632698,valid_fraction=.7),
        disk_free_min_gib=50,collection_budget_sec=10800,raw_budget_gib=20)


def jobs_for(plan,num_envs):
    if num_envs not in (1,2,5,10):raise ValueError('capacity must be 1/2/5/10')
    jobs=[]
    for state in STATES:
        parents=[p for p in plan['parents'] if p['state']==state]
        for start in range(0,len(parents),num_envs):
            subset=parents[start:start+num_envs]
            jobs.append(dict(job_id=state+'-tiles'+'-'.join(str(p['tile_index']) for p in subset),
                parents=subset,all_parents=parents,terrain_cols=len(parents)))
    return jobs


def amendment_jobs(plan,selected_capacity,amendment):
    jobs=[]
    for state in amendment['states']:
        capacity=amendment.get('capacity_by_state',{}).get(state,selected_capacity)
        jobs.extend(j for j in jobs_for(plan,capacity) if j['parents'][0]['state']==state)
    return jobs


def anchor_inside_guard(position,origin,halfwidth=7.95):
    relative=np.asarray(position,dtype=float)[:2]-np.asarray(origin,dtype=float)[:2]
    return bool(np.isfinite(relative).all() and np.all(abs(relative)<=halfwidth))


def reset_recipe(state):
    x={'moving_flat':0.,'turning':0.,'uphill_before':7.95,'uphill_mid':-6.,'step_before':-6.,'descent':-6.,'slow':0.}[state]
    return dict(x_m=x,yaw_rad=math.pi if state=='uphill_before' else 0.)


def fresh_schedule(policy):
    if policy=='full_reverse':return fresh_sequence(True)
    if policy=='forward_postzero':return [0,0]+list(range(1,14))+[0]
    raise ValueError('unknown diagnostic policy')


def persistent_budget(output,plan,now=None):
    output=Path(output);path=output/'persistent-budget-binding.json'
    if path.exists():return read(path)
    start=time.time() if now is None else now
    # Immutable first creation, never plan mtime or a resume-local monotonic clock.
    result=dict(original_start_unix=start,deadline_unix=start+plan['collection_budget_sec'],
        budget_seconds=plan['collection_budget_sec'],raw_limit_bytes=plan['raw_budget_gib']*2**30,
        raw_roots=[str(output),str(output.parent/'fresh-process-runtime')])
    write(path,result);return result


def budget_status(output,binding,now=None,free_bytes=None):
    now=time.time() if now is None else now;size=0
    for root in binding['raw_roots']:
        for path in Path(root).rglob('*'):
            try:
                if path.is_file():size+=path.stat().st_size
            except FileNotFoundError:pass
    free=shutil.disk_usage(output).free if free_bytes is None else free_bytes
    reason='persistent_deadline' if now>=binding['deadline_unix'] else 'raw_limit' if size>=binding['raw_limit_bytes'] else 'disk_free' if free<50*2**30 else None
    return dict(allowed=reason is None,reason=reason,raw_bytes=size,free_bytes=free,now_unix=now,deadline_unix=binding['deadline_unix'])


def prefix_command(state,seconds):
    if state not in STATES:raise ValueError('unknown prefix state')
    if state=='turning':return [.22,0.,.55]
    if state=='slow':return [0.,0.,0.] if seconds<5 else [.3,0.,0.]
    return [.3,0.,0.]


def dem_metrics(frame,resolution=.1):
    """Causal yaw DEM: rows are y, columns x, forward is increasing column.

    Boundary = nearest forward centerline-profile positive rise >=2cm from
    local center height; step qualifier separately requires >=6cm. This is a
    known observed terrain boundary, never a physical contact guarantee.
    """
    frame=np.asarray(frame);height=frame[0];valid=frame[1]>.5
    profile=[]
    for col in range(20):
        values=height[8:12,col][valid[8:12,col]]
        profile.append(float(np.median(values)) if len(values)>=2 else np.nan)
    p=np.asarray(profile);xs=(np.arange(20)-9.5)*resolution
    center=float(np.nanmedian(p[9:11])) if np.isfinite(p[9:11]).any() else 0.
    ahead=(xs>=.15)&(xs<=.75)&np.isfinite(p)
    boundaries=np.flatnonzero(ahead & (p-center>=.02))
    distance=float(xs[boundaries[0]]) if len(boundaries) else None
    rise=float(np.max(p[ahead])-center) if ahead.any() else 0.
    local=np.flatnonzero((abs(xs)<=.25)&np.isfinite(p))
    slope=float(np.polyfit(xs[local],p[local],1)[0]) if len(local)>=3 else 0.
    yy,xx=np.meshgrid(xs,xs,indexing='ij')
    patch=valid & (abs(xx)<=.25) & (abs(yy)<=.25)
    gradient=float('inf')
    if patch.sum()>=16:
        coefficients=np.linalg.lstsq(np.column_stack([xx[patch],yy[patch],np.ones(patch.sum())]),height[patch],rcond=None)[0]
        gradient=float(np.linalg.norm(coefficients[:2]))
    return dict(valid_dem_fraction=float(valid.mean()),front_rise_m=rise,forward_boundary_distance_m=distance,
                terrain_local_slope=slope,terrain_plane_gradient_norm=gradient if np.isfinite(gradient) else None,low_gradient_dem=bool(gradient<math.tan(math.radians(10))),
                terrain_profile_m=[None if not np.isfinite(x) else float(x) for x in p])


def fresh_sequence(reverse=True):
    return [0,0]+list(range(1,14))+(list(reversed(range(14)))+[0] if reverse else [])

def parent_visible(snapshot,index,num_envs):
    return {key:(np.asarray(value)[index].copy() if np.asarray(value).ndim and np.shape(value)[0]==num_envs else np.asarray(value).copy()) for key,value in snapshot.items()}


def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def gate_contract(plan,binding):
    return dict(schema='eatnav_moving_pose2s_gate_binding_v1',plan_sha256=digest_json(plan),
        runtime_sha256=digest_json(dict(image_id=binding['image_id'],source_sha256=binding['source_sha256'])))

def validate_gate_contract(contract,plan,binding):
    if contract!=gate_contract(plan,binding):raise ValueError('frozen plan/runtime gate binding differs')
    return True

class CommonWorldAnchor:
    """Choose one full-world control step; never combine asynchronous clocks."""
    def __init__(self,num_envs):self.mask=[False]*num_envs;self.step=None
    def consider(self,mask,step):
        mask=list(map(bool,mask))
        if len(mask)!=len(self.mask):raise ValueError('one qualification per environment required')
        if sum(mask)>sum(self.mask):self.mask=mask;self.step=step;return 'capture'
        return 'keep'
    @property
    def complete(self):return all(self.mask)


def capture_lazy_caches(obj):
    """Preserve Isaac TimestampedBuffer pairs when the owner clock rewinds."""
    return {name:dict(timestamp=copy.deepcopy(value.timestamp),data=copy.deepcopy(value.data))
        for name,value in vars(obj).items() if hasattr(value,'timestamp') and hasattr(value,'data')}


def restore_lazy_caches(obj,saved):
    for name,state in saved.items():
        cache=getattr(obj,name);cache.timestamp=copy.deepcopy(state['timestamp']);cache.data=copy.deepcopy(state['data'])


def physical_metrics(root,vz_history):
    from visualnav_transformer.eat_nav.fall_stall_targets import body_vector
    root=np.asarray(root).tolist()
    return dict(speed_mps=float(np.linalg.norm(root[7:10])),
        body_wz_radps=float(body_vector(root,root[10:13])[2]),
        mean_world_vz_mps=float(np.mean(vz_history)),
        tilt_rad=float(math.acos(np.clip(-body_vector(root,[0.,0.,-1.])[2],-1,1))))


def policy_input(obs):
    return obs['policy'] if hasattr(obs,'keys') and 'policy' in obs.keys() else obs


def mesh_height(vertices,faces,xy):
    """Vertical ray against actual triangles, without optional trimesh rtree."""
    tri=np.asarray(vertices)[np.asarray(faces)];xy=np.asarray(xy)
    a=tri[:,0,:2];u=tri[:,1,:2]-a;v=tri[:,2,:2]-a;p=xy-a
    det=u[:,0]*v[:,1]-u[:,1]*v[:,0];valid=abs(det)>1e-12
    den=np.where(valid,det,1.)
    b=(p[:,0]*v[:,1]-p[:,1]*v[:,0])/den
    c=(u[:,0]*p[:,1]-u[:,1]*p[:,0])/den
    valid &= (b>=-1e-8)&(c>=-1e-8)&(b+c<=1+1e-8)
    if not valid.any():raise ValueError('initial XY outside actual terrain mesh')
    height=tri[:,0,2]+b*(tri[:,1,2]-tri[:,0,2])+c*(tri[:,2,2]-tri[:,0,2])
    return float(height[valid].max())


def qualify_state(state,metrics):
    if state not in STATES:raise ValueError('unknown state')
    m=metrics;s=m['speed_mps'];d=m.get('forward_boundary_distance_m')
    common=m.get('valid_dem_fraction',0)>=.7 and m.get('dem_history_seconds',0)>=.75
    near=d is not None and .2<=d<=.5
    condition={
        'moving_flat':s>=.1 and m['tilt_rad']<math.radians(10) and m['low_gradient_dem'],
        'turning':abs(m['body_wz_radps'])>=.2,
        'uphill_before':near and m['front_rise_m']>=.02 and s>=.05 and m['low_gradient_dem'],
        'uphill_mid':m['terrain_local_slope']>=.08 and s>=.05 and m['mean_world_vz_mps']>=.02,
        'step_before':near and m['front_rise_m']>=.06 and s>=.05,
        'descent':m['terrain_local_slope']<=-.08 and s>=.05 and m['mean_world_vz_mps']<=-.02,
        'slow':s<.05}[state]
    return dict(qualified=bool(common and condition),state=state,metrics=m,
                slow_kind=('stopped' if np.linalg.norm(m.get('command',[0,0,0]))<1e-8 else 'commanded_slow') if state=='slow' else None)


class AnchorQualification:
    def __init__(self,state):self.state=state;self.count=0;self.last=None
    def observe(self,metrics,stamp_ns):
        result=qualify_state(self.state,metrics)
        self.count=(self.count+1 if self.last is not None and stamp_ns-self.last==50_000_000 else 1) if result['qualified'] else 0
        self.last=stamp_ns
        return dict(result,consecutive_samples=self.count,ready=self.count>=4)


def quaternion_angle(q,r):
    q=np.asarray(q,float);r=np.asarray(r,float)
    q=q/np.linalg.norm(q);r=r/np.linalg.norm(r)
    return float(2*np.arccos(np.clip(abs(np.dot(q,r)),0,1)))


def compare_visible_state(reference,current):
    errors={}
    for key,expected in reference.items():
        expected=np.asarray(expected,dtype=np.float64);value=np.asarray(current.get(key,[]),dtype=np.float64)
        if expected.shape!=value.shape:errors[key]='shape';continue
        if key=='root':
            if not np.allclose(expected[:3],value[:3],atol=1e-5,rtol=0):errors[key+'.position']=float(np.max(abs(expected[:3]-value[:3])))
            if quaternion_angle(expected[3:7],value[3:7])>1e-4:errors[key+'.rotation']='quaternion_angle'
            if not np.allclose(expected[7:],value[7:],atol=1e-5,rtol=0):errors[key+'.velocity']='velocity'
        elif not np.allclose(expected,value,atol=1e-5,rtol=1e-5,equal_nan=False):errors[key]=float(np.max(abs(expected-value)))
    return dict(passed=not errors,errors=errors,atol=1e-5,rtol=1e-5)


def duplicate_gate(root_a,root_b,event_a,event_b):
    position=float(np.linalg.norm(np.asarray(root_a)[:3]-np.asarray(root_b)[:3]));angle=quaternion_angle(root_a[3:7],root_b[3:7])
    events=bool(np.array_equal(event_a,event_b))
    return dict(passed=position<=.02 and angle<=math.radians(2) and events,position_difference_m=position,rotation_difference_rad=angle,event_equal=events)


def stage_runtime(container,root):
    stage=Path(tempfile.mkdtemp(prefix='eatnav-moving-pose2s-'))
    shutil.copytree(REPO/'src/visualnav_transformer',stage/'src/visualnav_transformer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for source in (REPO/'scripts').glob('eat_nav*.py'):shutil.copy2(source,stage/source.name)
    inputs={'actor.pt':SOURCE/'checkpoints/eat_nav/rl_actor/model_10000.pt','env.yaml':SOURCE/'checkpoints/eat_nav/rl_actor/env.yaml','eat_nav.yaml':SOURCE/'config/eat_nav.yaml','collection-protocol.json':REPO/'config/eat_nav/moving_pose2s/collection.json'}
    for name,source in inputs.items():shutil.copy2(source,stage/name)
    hashes={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    subprocess.run(['docker','cp',str(stage),container+':/tmp/'],check=True)
    actual=subprocess.check_output(['docker','exec',container,'sha256sum',*[str(stage/name) for name in hashes]],text=True)
    observed={line.split(maxsplit=1)[1].strip():line.split()[0] for line in actual.splitlines()}
    if any(observed.get(str(stage/name))!=digest for name,digest in hashes.items()):raise ValueError('staging hash mismatch')
    binding=dict(stage=str(stage),container=container,source_sha256=hashes,input_sources={k:str(v) for k,v in inputs.items()},
                 image_id=subprocess.check_output(['docker','inspect','--format','{{.Image}}',container],text=True).strip())
    write(Path(root)/'runtime-binding.json',binding);return binding


def current_runtime_hashes():
    files={str(p.relative_to(REPO)):p for p in (REPO/'src/visualnav_transformer').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    files.update({p.name:p for p in (REPO/'scripts').glob('eat_nav*.py')})
    files.update({'actor.pt':SOURCE/'checkpoints/eat_nav/rl_actor/model_10000.pt',
        'env.yaml':SOURCE/'checkpoints/eat_nav/rl_actor/env.yaml','eat_nav.yaml':SOURCE/'config/eat_nav.yaml','collection-protocol.json':REPO/'config/eat_nav/moving_pose2s/collection.json'})
    return {name:sha(path) for name,path in files.items()}

def verify_runtime_binding(binding,*,immutable_only=False):
    if not immutable_only and current_runtime_hashes()!=binding['source_sha256']:
        raise ValueError('current runtime source bytes differ from frozen benchmark/gate binding; rerun gates with fresh staging')
    stage=Path(binding['stage']);container=binding['container']
    actual=subprocess.check_output(['docker','exec',container,'sha256sum',*[str(stage/name) for name in binding['source_sha256']]],text=True)
    observed={line.split(maxsplit=1)[1].strip():line.split()[0] for line in actual.splitlines()}
    if any(observed.get(str(stage/name))!=digest for name,digest in binding['source_sha256'].items()):
        raise ValueError('immutable staged runtime bytes changed after gate')
    image=subprocess.check_output(['docker','inspect','--format','{{.Image}}',container],text=True).strip()
    if image!=binding['image_id']:raise ValueError('container image differs from frozen runtime binding')
    return True


def sim_command(binding,simout,parent,groupfile=None,num_envs=1,smoke=False,gate=False,method='snapshot'):
    stage=Path(binding['stage']);envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
    command=['docker','exec','-e','PYTHONPATH='+envpath,binding['container'],'/isaac-sim/python.sh',str(stage/'eat_nav_moving_pose2s_collect_sim.py'),
        '--headless','--output',str(simout),'--terrain-type',parent['field'],'--level',str(parent['level']),'--seed',str(parent['seed']),
        '--num-envs',str(num_envs),'--terrain-cols',str(num_envs),'--prefix-state',parent['state'],'--restoration-method',method,
        '--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
    if groupfile:command+=['--groups',str(groupfile)]
    if smoke:command+=['--smoke-steps','40']
    if gate:command+=['--duplicate-gate']
    return command


def execute(binding,directory,parent,num_envs=1,smoke=False,gate=False,timeout=900,method='snapshot',fresh=False,candidate_index=None,prefix_steps=None):
    directory=Path(directory);directory.mkdir(parents=True)
    stage=Path(binding['stage'])
    if 'parents' in parent:
        job=parent;groups=[dict(p,env_index=i) for i,p in enumerate(job['parents'])];num_envs=len(groups)
        all_parents=job.get('all_parents',job['parents']);terrain_cols=job.get('terrain_cols',len(all_parents));parent=groups[0]
    else:
        groups=[dict(parent,env_index=i,tile_index=i,group_id=parent['group_id']+f'/env{i}') for i in range(num_envs)]
        all_parents=groups;terrain_cols=num_envs
    groupfile=stage/(directory.name+'-groups.json');write(groupfile,dict(groups=groups))
    parentfile=stage/(directory.name+'-parents.json');write(parentfile,all_parents)
    for source in (groupfile,parentfile):subprocess.run(['docker','cp',str(source),binding['container']+':'+str(source)],check=True)
    simout=stage/'outputs'/directory.name
    command=sim_command(binding,simout,parent,groupfile,num_envs,smoke,gate,method)
    if fresh:
        command+=['--fresh-process','--candidate-indices',str(candidate_index)]
        if prefix_steps is not None:command+=['--prefix-steps',','.join(map(str,prefix_steps))]
    command+=['--parent-manifest',str(parentfile),'--tile-indices',','.join(str(g['tile_index']) for g in groups)]
    command[command.index('--terrain-cols')+1]=str(terrain_cols)
    command[command.index('--seed')+1]=str(all_parents[0]['seed'])
    write(directory/'command.json',command);started=time.time()
    try:
        with (directory/'simulator.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        code=result.returncode
    except subprocess.TimeoutExpired:
        return dict(state='failed',reason='host_timeout_runtime_may_still_run',started_unix=started,completed_unix=time.time())
    copied=subprocess.run(['docker','cp',binding['container']+':'+str(simout),str(directory/'sim-output')],capture_output=True,text=True)
    complete=directory/'sim-output/complete.json';failure=directory/'sim-output/failure.json'
    success=code==0 and copied.returncode==0 and complete.exists() and not failure.exists()
    return dict(state='completed' if success else 'failed',returncode=code,copy_returncode=copied.returncode,started_unix=started,completed_unix=time.time(),complete=read(complete) if complete.exists() else None)


def benchmark(root,container='isaac-lab-onix'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);binding=stage_runtime(container,root);records=[]
    parent=dict(group_id='capacity-benchmark',state='moving_flat',field='random_rough',level=1,seed=310035521,role='benchmark')
    for n in (2,5,10):
        rec=execute(binding,root/f'capacity-{n}',parent,n,smoke=True,timeout=900,fresh=True,candidate_index=0);records.append(dict(num_envs=n,**rec));write(root/'benchmark-progress.json',records)
        if rec.get('reason')=='host_timeout_runtime_may_still_run':break
    passed=[r for r in records if r['state']=='completed' and r['complete'].get('recorded_rows',0)>0]
    result=dict(passed=bool(passed),records=records,selected_num_envs=max(passed,key=lambda r:r['complete']['rows_per_second'])['num_envs'] if passed else None,
                production_capacity_reason='largest stable measured capacity; frozen terrain width and per-parent terrain/root seeds retain geometry identity',prior_r6_rates_are_estimates_only=True)
    write(root/'benchmark.json',result);return result


def should_replay(result,directory):
    if result['state']=='completed':
        available=result.get('complete',{}).get('anchor_available_by_env',[])
        return bool(available) and all(available) and not result.get('restoration_gate_passed',False)
    failure=Path(directory)/'sim-output/failure.json'
    if result.get('reason')=='host_timeout_runtime_may_still_run' or not failure.exists():return False
    evidence=read(failure)
    return evidence.get('type')=='ValueError' and any(token in evidence.get('message','') for token in
        ('visible state restoration failed','branch anchor restoration failed','qualified full-world visible state restoration failed','qualified prefix state changed during per-env snapshot stitching'))


def fresh_probe_passed(checks,reference,current):
    return Path(reference)!=Path(current) and len(checks)==2 and all(x['passed'] for x in checks)


def outcome_observed(record):
    return record.get('control_steps',0)>0 and record.get('stop_reason') in ('horizon_complete','natural_termination')


def branch_outcome(directory,candidate):
    from visualnav_transformer.eat_nav.moving_pose2s_targets import build_window
    directory=Path(directory);rows=[json.loads(line) for line in (directory/f'candidate-{candidate:02d}/trace.jsonl').read_text().splitlines()]
    if not rows:return None
    with np.load(directory/f'candidate-{candidate:02d}/input.npz') as data:
        path=data['paths'][0,candidate];direction=float(data['directions'][0,candidate,0])
    target=build_window(rows,0,dict(root_state=rows[0]['root_state'],t0_ns=rows[0]['logical_sim_time_ns'],candidate_id=rows[0]['candidate_id'],path_body_xy=path,direction=direction))
    return dict(root=rows[-1]['post_root_state'],events=np.where(target['event_mask4'],target['events4'],-1).tolist(),event_mask=target['event_mask4'].tolist(),count=len(rows),pose_mask=target['pose_mask'],first_action=rows[0]['executed_action'])


def compare_fresh_parent(reference,current,candidate,*,compare_outcome=False):
    reference=Path(reference);current=Path(current)
    if not (reference/'anchor-visible-state.npz').exists() or not (current/'anchor-visible-state.npz').exists():return dict(passed=False,reason='anchor_unavailable')
    with np.load(reference/'anchor-visible-state.npz') as data:a={k:data[k] for k in data.files}
    with np.load(current/'anchor-visible-state.npz') as data:b={k:data[k] for k in data.files}
    result=compare_visible_state(a,b)
    ma=read(reference/'group.json');mb=read(current/'group.json')
    same=ma['prefix_control_steps']==mb['prefix_control_steps'] and ma['actual_tile_metadata']['geometry_sha256']==mb['actual_tile_metadata']['geometry_sha256']
    result['same_prefix_geometry']=same;result['passed'] &= same
    if compare_outcome:
        x=branch_outcome(reference,candidate);y=branch_outcome(current,candidate)
        if x is None or y is None:result.update(passed=False,outcome_reason='missing_actual_trace')
        else:
            check=duplicate_gate(x['root'],y['root'],x['events'],y['events'])
            check['passed'] &= x['count']==y['count'] and x['event_mask']==y['event_mask']
            check['first_action_difference']=float(np.max(abs(np.asarray(x['first_action'])-np.asarray(y['first_action']))))
            result['outcome']=check;result['passed'] &= check['passed']
    return result


def run_fresh_process_collection(plan,output,container='isaac-lab-onix',max_jobs=None,amendment=None):
    """First branches from fresh physics worlds; preserve all failed comparisons.

    Candidate results never choose parents. Per-parent prefix times are frozen
    from the first process's causal qualification and reused by every child.
    Gate event comparisons inspect train/validation parents only.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists() and read(output/'plan.json')!=plan:raise ValueError('frozen parent plan changed')
    if not (output/'plan.json').exists():write(output/'plan.json',plan)
    persistent=persistent_budget(output,plan)
    if amendment is not None:
        if amendment['plan_sha256']!=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest():raise ValueError('amendment plan mismatch')
        if amendment['driver_sha256']!=sha(__file__):raise ValueError('amendment driver mismatch')
    benchmarks=[p for p in output.parent.glob('benchmark*/benchmark.json') if read(p).get('passed') and not (p.parent/'invalidated.json').exists()]
    if not benchmarks:raise ValueError('fresh runtime benchmark required')
    benchmark_file=max(benchmarks,key=lambda p:p.stat().st_mtime);bench=read(benchmark_file)
    binding=read(benchmark_file.parent/'runtime-binding.json');verify_runtime_binding(binding,immutable_only=amendment is not None)
    if binding['container']!=container:raise ValueError('frozen container differs')
    contract=gate_contract(plan,binding)
    if amendment is None:
        write(output/'runtime-binding.json',binding);write(output/'gate-binding.json',contract)
    runtime=output.parent/'fresh-process-runtime';runtime.mkdir(exist_ok=True)
    if not (runtime/'runtime-binding.json').exists():write(runtime/'runtime-binding.json',binding)
    ledger_file=output/'progress.json';ledger=read(ledger_file) if ledger_file.exists() else {}
    capacity=bench['selected_num_envs'];jobs=jobs_for(plan,capacity)
    if amendment is not None:
        jobs=amendment_jobs(plan,capacity,amendment)
    if max_jobs:jobs=jobs[:max_jobs]
    start=time.monotonic();budget=False
    for job in jobs:
        state=job['parents'][0]['state'];policy='full_reverse'
        if amendment is not None:
            policy=amendment['diagnostic_policy']
            binding=read(amendment['runtime_binding_by_state'][state]);verify_runtime_binding(binding,immutable_only=True)
            contract=gate_contract(plan,binding)
            if contract!=amendment['runtime_contract_by_state'][state]:raise ValueError('amendment runtime binding mismatch')
            if binding['container']!=container:raise ValueError('amended frozen container differs')
            if state in amendment.get('job_suffix_by_state',{}):job['job_id']+='-'+amendment['job_suffix_by_state'][state]
        key=job['job_id'];record=ledger.setdefault(key,dict(state='pending',parents=job['parents'],rounds=[],gate_contract=contract))
        validate_gate_contract(record['gate_contract'],plan,binding)
        if record['state'] in ('completed','unavailable'):continue
        if amendment is not None:
            record['execution_amendment']=amendment
            record['diagnostic_policy']=policy
        prefix_steps=record.get('prefix_steps');gate_indices=record.get('gate_indices',[]);forward={};reference_root=None
        for prior in record['rounds']:
            if prior['state']=='running':prior.update(state='failed',reason='interrupted_attempt_preserved')
            if prior['state']=='completed':
                directory=Path(prior['directory'])/'sim-output';reference_root=reference_root or directory
                forward.setdefault(prior['candidate'],directory)
        sequence=fresh_schedule(policy);record['state']='running';write(ledger_file,ledger)
        for ri,candidate in enumerate(sequence):
            if ri<len(record['rounds']):continue
            if policy=='full_reverse' and ri>=15 and not record.get('probe_passed',False):break
            status=budget_status(output,persistent);write(output/'budget-status.json',status)
            if not status['allowed']:
                budget=True;record['state']='partial';record['reason']=status['reason'];break
            directory=runtime/f'fresh-{key}-r{ri:02d}-c{candidate:02d}'
            attempt=dict(state='running',candidate=candidate,round=ri,directory=str(directory));record['rounds'].append(attempt);write(ledger_file,ledger)
            result=execute(binding,directory,job,fresh=True,candidate_index=candidate,prefix_steps=prefix_steps,timeout=900)
            attempt.update(result);write(ledger_file,ledger)
            if result.get('reason')=='host_timeout_runtime_may_still_run':budget=True;record['state']='partial';break
            if result['state']!='completed':continue
            sim=directory/'sim-output'
            if reference_root is None:
                reference_root=sim
                metas=[read(sim/f'group-{i:03d}/group.json') for i in range(len(job['parents']))]
                prefix_steps=[m['prefix_control_steps'] or 0 for m in metas];record['prefix_steps']=prefix_steps
                gate_indices=[next((i for i,m in enumerate(metas) if m['role']==role and m['anchor_available']),None) for role in ('train','validation')]
                record['gate_indices']=gate_indices
                write(runtime/f'{key}-frozen-prefix.json',dict(prefix_steps=prefix_steps,parents=job['parents'],gate_contract=contract))
                if not any(prefix_steps):record.update(state='unavailable',reason='no_qualified_prefix_within_20s');break
            checks=[]
            for i in range(len(job['parents'])):
                check=compare_fresh_parent(reference_root/f'group-{i:03d}',sim/f'group-{i:03d}',candidate)
                checks.append(dict(env_index=i,**check))
            attempt['start_checks']=checks
            if ri==1:
                probe=[]
                for i in gate_indices:
                    if i is None:probe.append(dict(passed=False,reason='train_or_validation_anchor_unavailable'));continue
                    probe.append(dict(env_index=i,**compare_fresh_parent(reference_root/f'group-{i:03d}',sim/f'group-{i:03d}',0,compare_outcome=True)))
                record['probe_checks']=probe;record['probe_passed']=fresh_probe_passed(probe,reference_root,sim)
            if candidate in forward and ri>=15:
                order=[]
                for i in gate_indices:
                    if i is None:order.append(dict(passed=False,reason='gate_anchor_unavailable'));continue
                    order.append(dict(env_index=i,**compare_fresh_parent(forward[candidate]/f'group-{i:03d}',sim/f'group-{i:03d}',candidate,compare_outcome=True)))
                attempt['order_checks']=order
                if policy=='forward_postzero':record['post_forward_zero_checks']=order
            forward.setdefault(candidate,sim);write(ledger_file,ledger)
        if reference_root is not None:
            assembly=output/key/'sim-output';assembly.mkdir(parents=True,exist_ok=True)
            for name in ('terrain.json','manifest.json','actor-audit.json','actual-physics-properties.npz','physics-audit.json'):
                if (reference_root/name).exists():shutil.copy2(reference_root/name,assembly/name)
            order_checks=[c for r in record['rounds'] for c in r.get('order_checks',[])]
            gate_passed=policy=='full_reverse' and record.get('probe_passed',False) and len(record['rounds'])==30 and len(order_checks)==30 and all(c['passed'] for c in order_checks)
            record['restoration_gate_passed']=gate_passed;record['order_reverse_attempted']=policy=='full_reverse' and len(record['rounds'])>15
            record['reduced_diagnostics']=policy=='forward_postzero'
            for i,parent in enumerate(job['parents']):
                source=reference_root/f'group-{i:03d}';dest=assembly/f'group-{i:03d}';dest.mkdir(exist_ok=True)
                meta=read(source/'group.json')
                for name in ('initial-input.npz','anchor-state.npz','anchor-visible-state.npz','anchor-body-world.npz','reset-state.npz','prefix-trace.jsonl'):
                    if (source/name).exists():shutil.copy2(source/name,dest/name)
                completions=[]
                for candidate in range(14):
                    if candidate not in forward:continue
                    branch=forward[candidate]/f'group-{i:03d}'/f'candidate-{candidate:02d}'
                    if branch.exists():
                        shutil.copytree(branch,dest/branch.name,dirs_exist_ok=True);completions.append(read(branch/'complete.json'))
                starts=all(r['start_checks'][i]['passed'] for r in record['rounds'] if r.get('start_checks'))
                all14=len(completions)==14;outcomes=all14 and all(outcome_observed(c) for c in completions);fullpose=outcomes and all(c['control_steps']==40 for c in completions)
                meta.update(restoration_method='fresh_process_prefix',restoration_gate_passed=gate_passed,start_buffers_matched=starts,
                    counterfactual_complete=bool(gate_passed and starts and outcomes),all14_started=all14,all14_outcomes_observed=bool(outcomes),all14_full2s_pose=bool(fullpose),
                    diagnostic_policy=policy,execution_amendment=amendment,gate_contract=contract,gate_record=str(ledger_file),actual_branch_sources={str(c):str(forward[c]/f'group-{i:03d}') for c in forward})
                write(dest/'group.json',meta)
            if record['state'] not in ('partial','unavailable'):record['state']='completed' if len(forward)==14 else 'partial'
            write(output/key/'complete.json',dict(state=record['state'],restoration_gate_passed=gate_passed,observed_candidate_ids=sorted(forward),gate_contract=contract))
        write(ledger_file,ledger)
        if budget:break
    inventory=collection_inventory(output,plan)
    summary=dict(method='fresh_process_prefix',planned_parents=len(plan['parents']),planned_branches=plan['planned_branches'],
        actual_observed_branches=inventory['observed_branches'],actual_qualified_parents=inventory['observed_parents'],counterfactual_complete_groups=inventory['counterfactual_complete_groups'],
        elapsed_seconds=time.monotonic()-start,job_states={k:v['state'] for k,v in ledger.items()},restoration_gate_by_job={k:v.get('restoration_gate_passed',False) for k,v in ledger.items()},budget_stop=budget,ledger=str(ledger_file))
    write(output/'summary.json',summary);return summary


def collection_inventory(root,plan):
    root=Path(root);groups=[];branches=[]
    for path in sorted(root.glob('*/sim-output/group-*/group.json')):
        meta=read(path);groups.append(dict(path=str(path.resolve()),**meta))
        for complete in sorted(path.parent.glob('candidate-*/complete.json')):
            record=read(complete);trace=complete.parent/'trace.jsonl'
            branches.append(dict(group_id=meta['group_id'],path=str(complete.parent.resolve()),record=record,trace_sha256=sha(trace),observed=record.get('control_steps',0)>0))
    files={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.glob('*/sim-output/**/*') if p.is_file()}
    result=dict(schema='eatnav_moving_pose2s_raw_collection_manifest_v1',planned_parents=len(plan['parents']),planned_branches=plan['planned_branches'],
        observed_parents=sum(g.get('anchor_available',False) for g in groups),unavailable_parents=sum(not g.get('anchor_available',False) for g in groups),
        counterfactual_complete_groups=sum(g.get('counterfactual_complete',False) for g in groups),observed_branches=sum(b['observed'] for b in branches),groups=groups,branches=branches,raw_files=files)
    write(root/'collection-manifest.json',result);return result


def run_collection(plan,output,container='isaac-lab-onix',max_jobs=None,gate_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=output/'plan.json'
    if frozen.exists() and read(frozen)!=plan:raise ValueError('frozen manifest differs')
    write(frozen,plan)
    ledger=read(output/'progress.json') if (output/'progress.json').exists() else {}
    for rec in ledger.values():
        if rec['state']=='running':rec.update(state='failed',reason='interrupted_preserved')
    write(output/'progress.json',ledger)
    started=time.monotonic()
    parents=plan['parents']
    candidates=[(p,read(p)) for p in output.parent.glob('benchmark*/benchmark.json')]
    passed=[(p,d) for p,d in candidates if d.get('passed') and d.get('selected_num_envs') in (2,5,10) and not (p.parent/'invalidated.json').exists()]
    if not passed:raise ValueError('successful actual vector benchmark required')
    benchfile,bench=max(passed,key=lambda x:x[0].stat().st_mtime)
    capacity=bench['selected_num_envs']
    write(output/'capacity.json',dict(num_envs=capacity,benchmark_path=str(benchfile),benchmark_sha256=sha(benchfile)))
    gate_record=None
    if not gate_only:
        gate_records=[p for p in output.parent.glob('gates*/summary.json') if not (p.parent/'invalidated.json').exists()]
        if not gate_records:raise ValueError('representative restoration gates required before production')
        gate_record=max(gate_records,key=lambda p:p.stat().st_mtime)
    binding=read((benchfile.parent if gate_only else gate_record.parent)/'runtime-binding.json')
    verify_runtime_binding(binding)
    contract=gate_contract(plan,binding)
    if gate_only:
        contract_file=output/'gate-binding.json'
        if contract_file.exists():validate_gate_contract(read(contract_file),plan,binding)
        else:write(contract_file,contract)
    else:validate_gate_contract(read(gate_record.parent/'gate-binding.json'),plan,binding)
    write(output/'runtime-binding.json',binding)
    jobs=jobs_for(plan,capacity)
    if gate_only:
        jobs=[]
        for state in STATES:
            all_parents=[p for p in parents if p['state']==state]
            subset=[next(p for p in all_parents if p['role']==r) for r in ('train','validation')]
            jobs.append(dict(job_id='gate-'+state,parents=subset,all_parents=all_parents,terrain_cols=len(all_parents)))
    if max_jobs:jobs=jobs[:max_jobs]
    for job in jobs:
        key=job['job_id']
        if key in ledger:
            previous=ledger[key]
            if gate_only and 'matched_prefix_fallback' not in previous and should_replay(previous,output/key):
                fallback_dir=output/(key+'-matched-prefix')
                if not fallback_dir.exists():
                    fallback=execute(binding,fallback_dir,job,gate=True,method='matched_prefix')
                    fallback_gate=fallback_dir/'sim-output/duplicate-gate.json'
                    fallback['restoration_gate_passed']=fallback_gate.exists() and read(fallback_gate)['passed']
                    fallback['restoration_method']='matched_prefix';fallback['gate_contract']=contract
                    if fallback_gate.exists():fallback['gate_evidence_sha256']=sha(fallback_gate)
                    previous['matched_prefix_fallback']=fallback;write(output/'progress.json',ledger)
            continue
        raw_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if time.monotonic()-started>plan['collection_budget_sec'] or shutil.disk_usage(output).free<50*2**30 or raw_bytes>20*2**30:
            write(output/'budget-stop.json',dict(elapsed_seconds=time.monotonic()-started,raw_bytes=raw_bytes,free_bytes=shutil.disk_usage(output).free));break
        parent=job;directory=output/key
        method='snapshot'
        if not gate_only:
            gates=read(gate_record.parent/'progress.json');gate=gates.get('gate-'+job['parents'][0]['state'],{})
            if gate:validate_gate_contract(gate.get('gate_contract'),plan,binding)
            if not gate.get('restoration_gate_passed'):
                fallback=gate.get('matched_prefix_fallback',{})
                if fallback.get('restoration_gate_passed'):
                    validate_gate_contract(fallback.get('gate_contract'),plan,binding);method='matched_prefix'
                else:
                    ledger[key]=dict(state='unavailable',reason='representative_restoration_gate_failed_or_anchor_unavailable',parents=job['parents']);write(output/'progress.json',ledger);continue
        if not gate_only:
            evidence_parent=gate_record.parent/('gate-'+job['parents'][0]['state']+('-matched-prefix' if method=='matched_prefix' else ''))
            evidence=evidence_parent/'sim-output/duplicate-gate.json'
            chosen=gate['matched_prefix_fallback'] if method=='matched_prefix' else gate
            if not evidence.exists() or sha(evidence)!=chosen.get('gate_evidence_sha256'):
                raise ValueError('restoration gate evidence changed or missing')
        ledger[key]=dict(state='running',parent=parent,started_unix=time.time());write(output/'progress.json',ledger)
        result=execute(binding,directory,parent,gate=gate_only,method=method)
        result['gate_contract']=contract
        if gate_only and result['state']=='completed':
            gatefile=directory/'sim-output/duplicate-gate.json'
            result['restoration_gate_passed']=gatefile.exists() and read(gatefile)['passed']
            if gatefile.exists():result['gate_evidence_sha256']=sha(gatefile)
        if gate_only and should_replay(result,directory):
            fallback_dir=output/(key+'-matched-prefix')
            fallback=execute(binding,fallback_dir,parent,gate=True,method='matched_prefix')
            fallback_gate=fallback_dir/'sim-output/duplicate-gate.json'
            fallback['restoration_gate_passed']=fallback_gate.exists() and read(fallback_gate)['passed']
            fallback['restoration_method']='matched_prefix';fallback['gate_contract']=contract
            if fallback_gate.exists():fallback['gate_evidence_sha256']=sha(fallback_gate)
            result['matched_prefix_fallback']=fallback
        elif gate_only:result['restoration_method']='snapshot'
        if not gate_only and result['state']=='completed':
            for path in (directory/'sim-output').glob('group-*/group.json'):
                meta=read(path);meta.update(restoration_gate_passed=True,restoration_method=method,representative_gate_path=str(gate_record.parent/'progress.json'),gate_contract=contract,representative_gate_evidence_sha256=chosen['gate_evidence_sha256']);write(path,meta)
        ledger[key].update(result);write(output/'progress.json',ledger)
        if result.get('reason')=='host_timeout_runtime_may_still_run':break
    summary=dict(planned_parents=len(parents),unavailable=sum(r['state']=='unavailable' for r in ledger.values()),completed=sum(r['state']=='completed' for r in ledger.values()),failed=sum(r['state']=='failed' for r in ledger.values()),not_attempted=sum(j['job_id'] not in ledger for j in jobs),ledger=str(output/'progress.json'))
    inventory=collection_inventory(output,plan);summary.update(actual_observed_branches=inventory['observed_branches'],actual_qualified_parents=inventory['observed_parents'],counterfactual_complete_groups=inventory['counterfactual_complete_groups']);write(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-plan',type=Path);p.add_argument('--parents-per-state',type=int,default=10);p.add_argument('--plan',type=Path)
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT);p.add_argument('--benchmark',action='store_true');p.add_argument('--gates',action='store_true');p.add_argument('--fresh-process',action='store_true');p.add_argument('--container',default='isaac-lab-onix');p.add_argument('--max-jobs',type=int);p.add_argument('--execution-amendment',type=Path)
    a=p.parse_args()
    if a.prepare_plan:
        from eat_nav_r6_run import public_exclusions
        excluded,sources=public_exclusions(SOURCE);plan=prepare_collection(a.parents_per_state,excluded);plan['exclusion_sources']=sources
        if a.prepare_plan.exists() and read(a.prepare_plan)!=plan:raise ValueError('existing plan differs')
        write(a.prepare_plan,plan);print(json.dumps(dict(planned_branches=plan['planned_branches'],path=str(a.prepare_plan))));return
    if a.benchmark:result=benchmark(a.output,a.container)
    elif a.plan:result=run_fresh_process_collection(read(a.plan),a.output,a.container,a.max_jobs,read(a.execution_amendment) if a.execution_amendment else None) if a.fresh_process else run_collection(read(a.plan),a.output,a.container,a.max_jobs,a.gates)
    else:p.error('--prepare-plan, --benchmark or --plan required')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
