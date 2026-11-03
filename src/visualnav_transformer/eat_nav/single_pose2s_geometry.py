"""Body-relative teachers and common GT-projected path coordinates."""
import numpy as np
from .bidirectional_bank import bank, LENGTH_M
from .moving_pose2s_geometry import quat_to_matrix, matrix_to_sixd, matrix_to_rpy, rotation_geodesic


def endpoint_target(root0,root2):
    a=np.asarray(root0,float);b=np.asarray(root2,float)
    if a.shape not in [(7,),(13,)] or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite consistent root required')
    R0=quat_to_matrix(a[3:7]);R2=quat_to_matrix(b[3:7]);dR=R0.T@R2
    return dict(position=R0.T@(b[:3]-a[:3]),rotation=matrix_to_sixd(dR))


def arc_position(s,theta,direction):
    if abs(theta)<1e-12:return direction*np.array([s,0.])
    return direction*LENGTH_M/theta*np.array([np.sin(theta*s/LENGTH_M),1-np.cos(theta*s/LENGTH_M)])


def path_errors(pred,truth,q0,candidate_id):
    R=quat_to_matrix(q0);yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    Ryaw=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    p=Ryaw.T@R@np.asarray(pred);g=Ryaw.T@R@np.asarray(truth)
    candidate=bank()[int(candidate_id)];theta=np.deg2rad(candidate.arc_angle_deg);d=candidate.direction
    if abs(theta)<1e-12:along=np.clip(d*g[0],0,LENGTH_M)
    else:
        # Analytic circle projection; compare constrained endpoints and all phases in range.
        radius=LENGTH_M/theta;scaled=g[:2]*d
        phase=np.arctan2(scaled[0]/radius,1-scaled[1]/radius)
        options=[0.,LENGTH_M]
        for k in [-1,0,1]:
            candidate_s=(phase+2*np.pi*k)*LENGTH_M/theta
            if 0<=candidate_s<=LENGTH_M:options.append(float(candidate_s))
        along=min(options,key=lambda x:(float(np.sum((arc_position(x,theta,d)-g[:2])**2)),x))
    T=d*np.array([np.cos(theta*along/LENGTH_M),np.sin(theta*along/LENGTH_M)]);N=np.array([-T[1],T[0]])
    residual=p-g
    return dict(longitudinal_m=float(T@residual[:2]),lateral_m=float(N@residual[:2]),height_m=float(residual[2]),projection_s_m=float(along),projection_clamped=bool(along<1e-9 or along>LENGTH_M-1e-9),distance_3d_m=float(np.linalg.norm(residual)))


def orientation_errors(pred_rotation,truth_rotation,q0):
    R0=quat_to_matrix(q0);pred=R0@pred_rotation;truth=R0@truth_rotation
    p,pl=matrix_to_rpy(pred);g,gl=matrix_to_rpy(truth)
    errors=np.rad2deg(np.abs(np.arctan2(np.sin(p-g),np.cos(p-g))))
    return dict(rpy_abs_deg=errors,euler_singular=bool(pl or gl),rotation_deg=float(np.rad2deg(rotation_geodesic(pred,truth))))
