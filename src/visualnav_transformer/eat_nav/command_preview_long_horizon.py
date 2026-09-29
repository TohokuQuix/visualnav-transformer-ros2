"""Frozen 120-step duplicate and 40/120 outcome diagnostics."""
from __future__ import annotations
import math
import numpy as np

ANCHOR_KEYS=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
             'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
             'critic_observation','critic_value','action_manager_action','latency_substeps',
             'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
TRACE_KEYS=('command','actor_input','actor_latent','actor_action','critic_observation',
            'critic_value','reward','contact_force')

def post_root(row):
 t=row.get('terminal')
 return t['root_state'] if t and 'root_state' in t else row['post_root_state']

def duplicate_120(summaries,traces,limit):
 if len(summaries)!=3 or len(traces)!=3 or any(not t or len(t)>120 or len(t)!=len(traces[0]) for t in traces):
  raise ValueError('three equal-length nominal repeats of at most 120 steps required')
 if len({s['fixed_map_path_sha256'] for s in summaries})!=1:raise ValueError('duplicate Path SHA mismatch')
 anchors=[s.get('anchor',s) for s in summaries]
 for key in ANCHOR_KEYS:
  if key in anchors[0] and any(a.get(key)!=anchors[0][key] for a in anchors[1:]):
   raise ValueError(f'duplicate anchor differs: {key}')
 xy=[];differences={key:[] for key in TRACE_KEYS if key not in ('contact_force',)}
 n=len(traces[0])
 for t in traces[1:]:
  for k in range(n):
   base=np.asarray(post_root(traces[0][k])[:2],dtype=float)-np.asarray(traces[0][0]['root_state'][:2],dtype=float)
   test=np.asarray(post_root(t[k])[:2],dtype=float)-np.asarray(t[0]['root_state'][:2],dtype=float)
   xy.extend((base-test).tolist())
   for key in differences:
    if key in traces[0][k] and key in t[k]:
     a=np.asarray(traces[0][k][key],dtype=float);b=np.asarray(t[k][key],dtype=float)
     if a.shape==b.shape:differences[key].extend((a-b).ravel().tolist())
 rms=math.sqrt(float(np.mean(np.square(xy))))
 if rms>limit:raise ValueError(f'duplicate XY RMS {rms:.6f} > {limit:.6f}')
 return {'pass':True,'xy_rms_m':rms,'xy_rms_limit_m':limit,'step_count':n,
         'signal_rms':{k:math.sqrt(float(np.mean(np.square(v)))) if v else None for k,v in differences.items()},
         'dem_sha_equal':all([r['dem']['transformed_sha256'] for r in t]==[r['dem']['transformed_sha256'] for r in traces[0]] for t in traces[1:])}

def _tilt(root):
 w,x,y,z=root[3:7]
 roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
 pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
 return max(abs(roll),abs(pitch))

def endpoint_metrics(trace,steps,end_value,anchor_map,goal_map,planned_path=None):
 if steps not in (40,120) or len(trace)<steps:raise ValueError('invalid 40/120 routing')
 t=trace[:steps];values=np.asarray([r['critic_value'] for r in t],dtype=float)
 rewards=np.asarray([r['reward'] for r in t],dtype=float)
 series=np.r_[values,float(end_value)]
 initial=np.asarray(t[0]['root_state'][:2],dtype=float)
 root=np.asarray([post_root(r)[:2] for r in t],dtype=float)
 local=root-initial
 goal=np.asarray(goal_map,dtype=float)-np.asarray(anchor_map,dtype=float)
 progress=float(np.linalg.norm(goal)-np.linalg.norm(goal-local[-1]))
 tilt=max(_tilt(post_root(r)) for r in t)
 contact=max(float(np.max(np.linalg.norm(np.asarray(r['contact_force'],dtype=float),axis=-1))) for r in t)
 length=float(np.linalg.norm(np.diff(np.vstack([np.zeros(2),local]),axis=0),axis=1).sum())
 tracking=None
 if planned_path is not None:
  p=np.asarray(planned_path,dtype=float)-np.asarray(anchor_map,dtype=float)
  tracking=float(np.sqrt(np.mean(np.min(np.sum((local[:,None,:]-p[None,:,:])**2,axis=2),axis=1))))
 return {'step_count':steps,'delta_v':float(end_value-values[0]),
         'advantage_h':float(np.dot(.99**np.arange(steps),rewards)+.99**steps*end_value-values[0]),
         'drawdown':float(np.min(series)-values[0]),'recovery':float(end_value-np.min(series)),
         'forward_goal_progress_m':progress,'path_length_m':length,'tracking_error_rms_m':tracking,
         'maximum_tilt_rad':tilt,'maximum_contact_n':contact,
         'stack':bool(np.linalg.norm(local[-1])<.02),'termination':t[-1].get('terminal')}
