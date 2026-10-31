"""Measured link geometry transformed with current body poses (Fabric compatible)."""
import itertools
import numpy as np


def rotation(q):
 q=np.asarray(q,float);q=q/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def live_footprint(lower,upper,body_positions,body_quaternions,root,margin=.06):
 lo=np.asarray(lower,float);hi=np.asarray(upper,float);p=np.asarray(body_positions,float);q=np.asarray(body_quaternions,float);r=np.asarray(root,float)
 if lo.shape!=hi.shape or lo.shape!=p.shape or lo.ndim!=2 or lo.shape[1]!=3 or q.shape!=(len(lo),4) or np.any(lo>hi):raise ValueError('measured per-body local bounds/poses required')
 if not all(np.isfinite(v).all() for v in (lo,hi,p,q,r)) or np.any(np.linalg.norm(q,axis=1)<1e-8):raise ValueError('finite live body transforms required')
 points=[]
 for a,b,position,quat in zip(lo,hi,p,q):
  corners=np.asarray(list(itertools.product(*zip(a,b))))
  points.extend(corners@rotation(quat).T+position)
 points=np.asarray(points)
 mat=rotation(r[3:7]);yaw=np.arctan2(mat[1,0],mat[0,0]);c,s=np.cos(yaw),np.sin(yaw)
 local=(points-r[:3])@np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
 dimensions=dict(front_m=float(max(.001,local[:,0].max())),rear_m=float(max(.001,-local[:,0].min())),half_width_m=float(max(.001,np.abs(local[:,1]).max())),stopping_margin_m=float(margin))
 return dimensions,points


def axis_rotation(axis,angle):
 a=np.asarray(axis,float);a=a/np.linalg.norm(a);x,y,z=a
 skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def calibrated_footprint(calibration,flipper_angles_rad,gravity,margin=.06):
 """Current JointState angles with audited simulation CAD/FK, no mean replacement.

 Real encoder calibration and actual manufactured dimensions remain distinct
 assumptions; this is an offline bag geometry estimate, not a real safety proof.
 """
 c=calibration
 if c.get('schema')!='eatnav_r6_sim_flipper_geometry_v1' or not c.get('numeric_sim_reference_verified'):raise ValueError('verified simulation FK calibration required')
 if set(flipper_angles_rad)!=set(c['axes']) or not np.isfinite(list(flipper_angles_rad.values())).all():raise ValueError('all raw flipper angles required')
 g=np.asarray(gravity,float)
 if g.shape!=(3,) or not np.isfinite(g).all() or abs(np.linalg.norm(g)-1)>.01:raise ValueError('finite unit projected gravity required')
 lower=np.asarray(c['local_bounds_min']);upper=np.asarray(c['local_bounds_max']);poses=c['poses']['0']
 positions=np.asarray(poses['body_positions_chassis']).copy();rotations=np.asarray(poses['body_rotations_chassis']).copy()
 for side,arm in c['axes'].items():
  angle=float(flipper_angles_rad[side])-arm['zero_angle_rad'];A=axis_rotation(arm['axis_chassis'],angle)
  ids=arm['body_indices'];hinge=np.asarray(arm['hinge_chassis'])
  positions[ids]=(positions[ids]-hinge)@A.T+hinge
  rotations[ids]=A@rotations[ids]
 roll=np.arctan2(-g[1],-g[2]);pitch=np.arcsin(np.clip(g[0],-1,1))
 R=axis_rotation([0,1,0],pitch)@axis_rotation([1,0,0],roll)
 points=[]
 for lo,hi,p,mat in zip(lower,upper,positions,rotations):
  corners=np.asarray(list(itertools.product(*zip(lo,hi))))
  points.extend((corners@mat.T+p)@R.T)
 points=np.asarray(points)
 if not np.isfinite(points).all():raise ValueError('nonfinite calibrated footprint')
 return dict(front_m=float(points[:,0].max()),rear_m=float(-points[:,0].min()),half_width_m=float(np.abs(points[:,1]).max()),stopping_margin_m=float(margin))
