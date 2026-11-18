"""Validation-only T015 gates; never change controller commands/masks."""
import numpy as np
def verify_initial(expected,actual):
 keys=['root','state','joint_position','joint_velocity','effort'];fields={}
 if set(actual)!=set(keys):raise ValueError('initial keys')
 for k in keys:
  a=np.asarray(expected[k])[0];b=np.asarray(actual[k])
  if a.shape!=b.shape or a.shape[0]!=10 or not np.isfinite(b).all() or not np.array_equal(a,b):raise ValueError('initial physics mismatch '+k)
  fields[k]=dict(shape=list(b.shape),maxdiff=0.)
 return dict(state='passed',trials=10,exact=True,time_s=0.,fields=fields,closed_loop_prefix_not_compared=True)
def verify_wall_samples(rays,boxes):
 r=np.asarray(rays)
 if r.ndim!=3 or r.shape[0]!=10 or r.shape[2]!=3 or not np.isfinite(r).all():raise ValueError('finite10environment rayXYZ required')
 lookup={b['name']:b for b in boxes};rows=[]
 for i in range(10):
  counts={}
  for name in ['left','inner_vertical']:
   if name not in lookup:raise ValueError('missing side wall '+name)
   x0,x1,y0,y1=lookup[name]['xy_bounds'];p=r[i];sel=(p[:,0]>=x0-1e-6)&(p[:,0]<=x1+1e-6)&(p[:,1]>=y0-1e-6)&(p[:,1]<=y1+1e-6)&(abs(p[:,2]-.5)<1e-5);counts[name]=int(sel.sum())
   if not counts[name]:raise ValueError('initial side wall absent '+str(i)+' '+name)
  rows.append(dict(environment=i,wall_top_samples=counts))
 return dict(state='passed',rows=rows,actual_ideal_sim_raycast=True,all_pose_visibility_not_guaranteed=True)
