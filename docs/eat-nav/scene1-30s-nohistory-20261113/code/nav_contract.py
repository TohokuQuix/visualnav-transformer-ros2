"""Pure numerical scene trial contract; diagnostics do not change control."""
import numpy as np
from scipy.spatial.transform import Rotation
BOUNDS=(0.,4.9,0.,2.48)
def inside_course(xy,bounds=BOUNDS):
 p=np.asarray(xy,float);lo=np.array([bounds[0],bounds[2]]);hi=np.array([bounds[1],bounds[3]])
 return np.isfinite(p).all(-1)&(p>=lo-1e-8).all(-1)&(p<=hi+1e-8).all(-1)
def course_mask(world_paths,predicted_world_positions):
 return inside_course(world_paths).all(-1)&inside_course(np.asarray(predicted_world_positions)[...,:2])
def outcome(elapsed,position,tilt_deg,bypass,goal=(4.3,.5),tolerance=.2,limit=300.):
 p=np.asarray(position,float)
 if not np.isfinite(p).all() or not np.isfinite(tilt_deg):return 'technical_failure'
 if tilt_deg>85.:return 'tip_over'
 if not bypass and np.linalg.norm(p[:2]-goal)<=tolerance+1e-10:return 'goal_reached'
 if elapsed>=limit-1e-9:return 'timeout'
 return None
def choose_best(rows):
 if not rows:raise ValueError('no completed trials')
 return min(rows,key=lambda x:(x['minimum_course_valid_goal_distance_m'],x['trial_id']))
def needs_walls(rows):return len(rows)==10 and all(x.get('bypass_observed',False) for x in rows)
def scene_location(xy):
 x,y=map(float,xy)
 if not inside_course([x,y]):name='outer_flat_or_outside_course'
 elif 1.25<=x<=1.6 and y>=1.25:name='central_upper_dividing_wall'
 elif 3.4<=x<=3.75 and y<=1.25:name='central_lower_dividing_wall'
 elif x<1.3:name='left_V_section'
 elif x>3.6:name='right_V_section'
 else:name='central_upper_ramps' if y>=1.25 else 'central_lower_ramps'
 return dict(region=name,x_m=x,y_m=y)
def diagnose_motion(times,root,commands):
 t=np.asarray(times);p=np.asarray(root);c=np.asarray(commands);rows=[]
 for s in range(0,len(c)-39,40):
  e=s+40;duration=float(t[e]-t[s]);distance=float(np.linalg.norm(np.diff(p[s:e+1,:3],axis=0),axis=1).sum());yaw=Rotation.from_quat(p[s:e+1,[4,5,6,3]]).as_euler('xyz')[:,2];turn=float(np.abs(np.diff(np.unwrap(yaw))).sum());v=float(np.mean(np.abs(c[s:e,0])));w=float(np.mean(np.abs(c[s:e,2])))
  kind='linear_motion_blocked' if v>=.1 and distance<.04 else ('turn_motion_blocked' if w>=.2 and turn<.1 else ('commanded_stop' if np.max(np.abs(c[s:e]))<1e-8 else None))
  if kind:rows.append(dict(kind=kind,t0_s=float(t[s]),t1_s=float(t[e]),duration_s=duration,travel_distance_m=distance,actual_yaw_travel_rad=turn,mean_abs_v_command=v,mean_abs_w_command=w,**scene_location(p[s,:2])))
 return rows

def initialization_ok(xy_error,yaw_error,warm_termination,xy_tolerance=.05,yaw_tolerance_deg=5.):
 return bool(not warm_termination and np.isfinite([xy_error,yaw_error]).all() and xy_error<=xy_tolerance+1e-10 and abs(yaw_error)<=np.deg2rad(yaw_tolerance_deg)+1e-10)
def bank_paths():
 result=[];dist=np.linspace(0,.75,8)
 for direction in [1,-1]:
  for angle in [0,-30,30,-60,60,-90,90]:
   theta=np.deg2rad(angle);path=np.c_[dist,np.zeros(8)] if angle==0 else np.c_[.75/theta*np.sin(theta*dist/.75),.75/theta*(1-np.cos(theta*dist/.75))]
   result.append(path*direction)
 return np.asarray(result,np.float32)
def audit_decision(d,root_world,state,origin,goal):
 root=np.asarray(root_world,float);R0=Rotation.from_quat(root[[4,5,6,3]]).as_matrix();pp=np.asarray(d['predicted_positions']);RR=np.asarray(d['predicted_rotations']);origin=np.asarray(origin)
 def check(condition,message):
  if not condition:raise ValueError(message)
 check(np.allclose(d['root'],root,rtol=0,atol=1e-7),'decision root differs from trace t0');check(np.allclose(d['state'],state,rtol=0,atol=1e-7),'decision state differs from trace t0')
 check(d['dem_stamp_ns']<=d['t0_ns'] and d['dem_received_ns']<=d['t0_ns'],'future DEM');check(d['t0_ns']==(40+d['step'])*50000000,'t0 step stamp mismatch')
 yaw=np.arctan2(R0[1,0],R0[0,0]);yawmat=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
 paths=bank_paths()@yawmat+root[:2];future=root[:3]+pp@R0.T;course=course_mask(paths-origin[:2],future-origin);tilt=np.rad2deg(np.arccos(np.clip((R0[None]@RR)[:,2,2],-1,1)));original=tilt<=60.+1e-10
 if d.get('input_failure',False):original=np.zeros(14,bool)
 allowed=original&course
 for key,value in [('course_allowed',course),('original_tilt_allowed',original),('allowed',allowed)]:check(np.array_equal(d[key],value),'independent mask mismatch '+key)
 ids=np.flatnonzero(allowed);score=np.linalg.norm(root[:2]-goal)-np.linalg.norm(future[:,:2]-goal,axis=1);chosen=None if not len(ids) else int(ids[np.argmax(score[ids])]);check(d['candidate_id']==chosen,'selected candidate differs from recomputed score/mask')
 if 'score_m' in d:check(np.allclose(d['score_m'],score,rtol=0,atol=1e-7),'score mismatch')
 return True

def settled_spawn_root(observed_root,start_xy,start_yaw,height_delta=0.):
 root=np.asarray(observed_root,dtype=float).copy()
 if root.shape!=(13,) or not np.isfinite(root).all():raise ValueError('finite root13 required')
 angles=Rotation.from_quat(root[[4,5,6,3]]).as_euler('xyz');angles[2]=start_yaw;q=Rotation.from_euler('xyz',angles).as_quat()
 root[:2]=start_xy;root[2]+=height_delta;root[3:7]=q[[3,0,1,2]];root[7:]=0
 return root
