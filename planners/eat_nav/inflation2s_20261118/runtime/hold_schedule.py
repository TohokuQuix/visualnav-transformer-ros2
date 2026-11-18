"""All14 candidates and STOP hold exactly2s; predicts2s; no completion-triggered early replan."""
import numbers
import numpy as np

def hold_steps(candidate):
 if candidate is None:return 40
 if isinstance(candidate,bool) or not isinstance(candidate,numbers.Integral) or not 0<=candidate<14:raise ValueError('candidate must be0..13 or STOP')
 return 40

def early_replan_allowed(candidate,completed):
 return candidate is not None and hold_steps(candidate)>40 and bool(completed)

def due_mask(step,next_steps,active,completed_long=None):
 early=np.zeros_like(np.asarray(active,bool)) if completed_long is None else np.asarray(completed_long,bool)
 return np.asarray(active,bool)&((step>=np.asarray(next_steps))|early)

def verify_prefix2(expected,actual):
 sizes={**dict.fromkeys(['root','state','joint_position','joint_velocity','effort'],41),**dict.fromkeys(['command','action','actor_input'],40),'dem':9}
 if set(expected)!=set(actual) or set(expected)!=set(sizes):raise ValueError('prefix keys differ')
 fields={}
 for k in sizes:
  a=np.asarray(expected[k]);b=np.asarray(actual[k])
  if a.shape!=b.shape or a.ndim<2 or a.shape[:2]!=(sizes[k],10):raise ValueError('prefix shape '+k)
  if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a,b):raise ValueError('prefix mismatch '+k+' maxdiff='+str(float(np.max(np.abs(a-b)))))
  fields[k]=dict(shape=list(a.shape),maxdiff=0.)
 return dict(state='passed',exact=True,through_time_s=2.,trials=10,fields=fields)
