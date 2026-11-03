"""Causal FAST-LIO pose conversion and simulated IMU measurements.

No wheel fallback and no simulated root velocity is used by odom_velocity_at.
ROS pose order is converted explicitly by the caller to xyz,wxyz.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .moving_pose2s_geometry import quat_to_matrix


def odom_velocity_at(t0_ns,stamps_ns,received_ns,poses):
    stamps=np.asarray(stamps_ns,np.int64);received=np.asarray(received_ns,np.int64);poses=np.asarray(poses,float)
    if stamps.shape!=received.shape or poses.shape!=(len(stamps),7):raise ValueError('invalid odometry arrays')
    ids=np.flatnonzero((stamps<=t0_ns)&(received<=t0_ns))
    if len(ids)<2:raise ValueError('need two causal odometry poses')
    ids=ids[np.argsort(stamps[ids],kind='stable')]
    b=int(ids[-1]);earlier=[int(i) for i in ids[:-1] if stamps[i]<stamps[b]]
    if not earlier:raise ValueError('need two causal distinct odometry poses')
    a=earlier[-1];dt=(int(stamps[b])-int(stamps[a]))*1e-9;age=(int(t0_ns)-int(stamps[b]))*1e-9
    if age>.2+1e-9 or dt>.25+1e-9:raise ValueError('stale odometry')
    if not np.isfinite(poses[[a,b]]).all():raise ValueError('nonfinite odometry')
    Ra=quat_to_matrix(poses[a,3:]);Rb=quat_to_matrix(poses[b,3:])
    v=Rb.T@(poses[b,:3]-poses[a,:3])/dt
    w=Rotation.from_matrix(Rb.T@Ra).as_rotvec()*(-1/dt)
    return v,w,age,[a,b]


def imu_specific_force(previous_root,current_root,dt):
    """Requires root_link_state_w: pose and velocity at the same actor origin.

    Isaac root_state_w mixes link pose with COM velocity and must not be used.
    """
    if dt<=0:raise ValueError('positive sample interval required')
    r=np.asarray(current_root);a=(r[7:10]-np.asarray(previous_root)[7:10])/dt
    return quat_to_matrix(r[3:7]).T@(a-np.array([0.,0.,-9.81]))


def mesh_placement_translation(env_origins,local_origins):
    """Terrain origin includes support z; mesh placement is a separate transform."""
    env=np.asarray(env_origins,float);local=np.asarray(local_origins,float)
    if env.shape!=local.shape or env.shape[-1]!=3 or not np.isfinite(env).all() or not np.isfinite(local).all():raise ValueError('invalid mesh origin transforms')
    return env-local
