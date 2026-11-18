import numpy as np
from scipy.spatial.transform import Rotation
FIELDS=('k_rail_step_field','pyramid_stairs','pyramid_stairs_inv','random_rough','boxes','hf_pyramid_slope','k_rail_field')
def manifest():
 out=[]
 for fi,f in enumerate(FIELDS):
  r=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261111,fi,301])));first,yaw=r.uniform(-np.pi,np.pi,2);angles=[first,*r.uniform(-np.pi,np.pi,9)]
  for lev in range(5):
   seed=int(np.random.SeedSequence([20261111,fi,lev,807]).generate_state(1)[0])
   for gi,theta in enumerate(angles):out.append(dict(trial_id=f'{f}-l{lev}-g{gi:02d}',field=f,field_index=fi,level=lev,goal_index=gi,goal_id=f'g{gi:02d}',goal_xy=(3*np.array([np.cos(theta),np.sin(theta)])).tolist(),yaw=float(yaw),seed=seed,initial_seed=seed,parent_id=f'{f}-l{lev}-g{gi:02d}'))
 return out

def state_class(v,w):
 if np.linalg.norm(v)>=.03:return 'moving'
 return 'stopped' if np.linalg.norm(w)<.1 else 'turning_in_place_or_other_rotation'
def result(tip,goal,step):return 'tip_over' if tip else 'goal_reached' if goal else 'timeout' if step>=600 else None
def window_valid(start,last,cid):return cid is not None and last>=start+40

def pose_errors(p,R,t,T,R0,path):
 world_delta=R0@(p-t);yaw=np.arctan2(R0[1,0],R0[0,0]);yawR=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);actualxy=yawR.T@(R0@t)[:2]
 d=np.diff(path,axis=0);u=np.clip(np.sum((actualxy-path[:-1])*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-12),0,1);k=np.linalg.norm(path[:-1]+u[:,None]*d-actualxy,axis=1).argmin();tangent=d[k]/np.linalg.norm(d[k]);normal=np.array([-tangent[1],tangent[0]]);diffxy=yawR.T@world_delta[:2]
 rp=Rotation.from_matrix(R0@R).as_euler('xyz');rt=Rotation.from_matrix(R0@T).as_euler('xyz');angles=np.rad2deg(np.arctan2(np.sin(rp-rt),np.cos(rp-rt)))
 signed=dict(forward_signed_cm=float(tangent@diffxy*100),left_signed_cm=float(normal@diffxy*100),height_signed_cm=float(world_delta[2]*100),body_z_signed_cm=float((p-t)[2]*100),roll_signed_deg=float(angles[0]),pitch_signed_deg=float(angles[1]),yaw_signed_deg=float(angles[2]))
 for k,v in list(signed.items()):signed[k.replace('_signed_','_abs_')]=abs(v)
 signed.update(position3d_cm=float(np.linalg.norm(p-t)*100),so3_deg=float(np.rad2deg(Rotation.from_matrix(R.T@T).magnitude())),euler_singular=bool(abs(np.cos(rp[1]))<1e-6 or abs(np.cos(rt[1]))<1e-6))
 return signed


def reference_properties(reference,n):
    out={}
    for k in ['masses','inertias','materials']:
        a=np.asarray(reference[k]);assert np.isfinite(a).all()
        out[k]=np.repeat(a[None],n,axis=0)
    return out

def verify_trial_set(expected,actual):
    e={x['trial_id']:x for x in expected};a={x['trial_id']:x for x in actual}
    if len(a)!=len(actual) or set(e)!=set(a):raise ValueError('incomplete or duplicate trial IDs')
    for tid,x in e.items():
        for k,v in x.items():
            if a[tid].get(k)!=v:raise ValueError('manifest identity mismatch '+tid+' '+k)
    return True

def stable_benchmark(row,n):
    return row.get('collector_exit')==0 and row.get('parents')==n and row.get('technical_failures',0)==0 and row.get('physics_valid') is True
