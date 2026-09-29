"""Offline command preview through the unmodified production ROS follower.

Only a frozen Path and anchor pose enter the control loop. The bridge returns
commands from the actual simple_trajectory_follower; a unicycle updates its
predicted pose, without consulting recorded motion or future simulator state.
"""
from __future__ import annotations
import math
import numpy as np


def pose_from_root_and_map_path(root, map_path):
    q=np.asarray(root[3:7],dtype=float)
    w,x,y,z=q
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    xy=np.asarray(map_path[0],dtype=float)
    return [float(xy[0]),float(xy[1]),yaw]


def map_to_body(points,pose):
    p=np.asarray(points,dtype=float);x,y,yaw=pose
    dx=p-np.asarray([x,y]);c,s=math.cos(yaw),math.sin(yaw)
    return np.stack([c*dx[...,0]+s*dx[...,1],-s*dx[...,0]+c*dx[...,1]],axis=-1)


def body_to_map(points,pose):
    p=np.asarray(points,dtype=float);x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
    return np.stack([x+c*p[...,0]-s*p[...,1],y+s*p[...,0]+c*p[...,1]],axis=-1)


def unicycle_step(pose,command,dt):
    x,y,yaw=map(float,pose);v=float(command[0]);w=float(command[2])
    if not all(math.isfinite(v) for v in (x,y,yaw,v,w,dt)) or dt<=0:raise ValueError('nonfinite pose/command or invalid dt')
    if abs(w)<1e-9:
        x+=v*dt*math.cos(yaw);y+=v*dt*math.sin(yaw)
    else:
        x+=v/w*(math.sin(yaw+w*dt)-math.sin(yaw))
        y+=v/w*(math.cos(yaw)-math.cos(yaw+w*dt))
    yaw=math.atan2(math.sin(yaw+w*dt),math.cos(yaw+w*dt))
    return [x,y,yaw]


def commands_to_waypoints(commands,*,dt=.05,stride=5,initial_yaw=0.):
    c=np.asarray(commands,dtype=float)
    if c.ndim!=2 or c.shape[1]!=3 or len(c)%stride:raise ValueError('command shape/stride mismatch')
    pose=[0.,0.,float(initial_yaw)];out=[]
    for i,row in enumerate(c):
        pose=unicycle_step(pose,row,dt)
        if (i+1)%stride==0:out.append(pose[:2])
    return np.asarray(out,dtype=float)


def command_error(preview,recorded,*,linear_threshold,angular_threshold):
    a=np.asarray(preview,dtype=float);b=np.asarray(recorded,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ValueError('command shape mismatch')
    dv=a[:,0]-b[:,0];dw=a[:,2]-b[:,2]
    bad=np.flatnonzero((np.abs(dv)>linear_threshold)|(np.abs(dw)>angular_threshold))
    lr=float(np.sqrt(np.mean(dv*dv)));ar=float(np.sqrt(np.mean(dw*dw)))
    return {'linear_rmse_mps':lr,'linear_mae_mps':float(np.mean(np.abs(dv))),
            'linear_max_mps':float(np.max(np.abs(dv))),
            'angular_rmse_radps':ar,'angular_mae_radps':float(np.mean(np.abs(dw))),
            'angular_max_radps':float(np.max(np.abs(dw))),
            'first_threshold_step':int(bad[0]) if len(bad) else None,
            'same_cluster':bool(lr<=linear_threshold and ar<=angular_threshold)}


def macro_mean(by_state):
    return float(np.mean([np.mean(v) for v in by_state.values()]))


def preview_candidate(client,map_path,initial_pose,steps=40,dt=.05):
    """Request actual follower commands with predicted branch-0 pose each step."""
    if len(map_path)!=9 or steps<=0:raise ValueError('Path or horizon mismatch')
    start=list(map(float,initial_pose));pose=start.copy()
    reset=client.request('reset',poses=[start]*8,paths=[map_path]*8)
    closed=[];opened=[];poses=[]
    for _ in range(steps):
        response=client.request('step',poses=[pose]+[start]*7,timeout_sec=2.)
        if not response['fresh']:raise RuntimeError('follower command timeout')
        c=np.asarray(response['commands'],dtype=float)
        if c.shape!=(8,3) or not np.isfinite(c).all():raise ValueError('follower command invalid')
        closed.append(c[0].tolist());opened.append(c[1].tolist())
        pose=unicycle_step(pose,c[0],dt);poses.append(pose)
    return {'closed_commands':closed,'open_commands':opened,'predicted_poses':poses,
            'reset_readiness':reset['readiness']}
