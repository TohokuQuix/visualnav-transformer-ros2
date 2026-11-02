"""NumPy geometry for full-body, wxyz moving-state pose targets.

Rotations map body coordinates to world coordinates. The six-dimensional
representation concatenates the first and second *columns*, in that order.
"""
from __future__ import annotations
import numpy as np


def normalize_quaternion(quaternion):
    q=np.asarray(quaternion,dtype=np.float64)
    if q.shape[-1:]!=(4,) or not np.isfinite(q).all():
        raise ValueError('finite wxyz quaternion with last dimension 4 required')
    scale=np.max(np.abs(q),axis=-1,keepdims=True)
    safe=np.where(scale>0,scale,1.)
    norm_scaled=np.linalg.norm(q/safe,axis=-1,keepdims=True)
    if np.any(scale<=1e-8/np.maximum(norm_scaled,1e-300)):
        raise ValueError('quaternion norm must exceed 1e-8')
    return (q/safe)/norm_scaled


def quat_to_matrix(quaternion):
    q=normalize_quaternion(quaternion)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                     2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                     2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))


def _rotation_matrix(matrix):
    R=np.asarray(matrix,dtype=np.float64)
    if R.shape[-2:]!=(3,3) or not np.isfinite(R).all():
        raise ValueError('finite rotation matrix with last dimensions 3,3 required')
    if not np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-6,rtol=1e-6) or np.any(np.linalg.det(R)<0):
        raise ValueError('matrix must belong to SO(3)')
    return R


def matrix_to_quat(matrix):
    R=_rotation_matrix(matrix)
    result=[]
    for m in R.reshape(-1,3,3):
        # Select the best-conditioned quaternion component, including pi rotations.
        terms=np.maximum(0.,[1+np.trace(m),1+m[0,0]-m[1,1]-m[2,2],
                             1-m[0,0]+m[1,1]-m[2,2],1-m[0,0]-m[1,1]+m[2,2]])
        i=int(np.argmax(terms));s=2*np.sqrt(terms[i])
        if i==0:q=[s/4,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s]
        elif i==1:q=[(m[2,1]-m[1,2])/s,s/4,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s]
        elif i==2:q=[(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,s/4,(m[1,2]+m[2,1])/s]
        else:q=[(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,s/4]
        q=normalize_quaternion(q)
        # w is the primary sign convention; at numerical zero use the first nonzero component.
        nonzero=np.flatnonzero(np.abs(q)>1e-12)
        if len(nonzero) and q[nonzero[0]]<0:q=-q
        result.append(q)
    return np.asarray(result).reshape(R.shape[:-2]+(4,))


def matrix_to_sixd(matrix):
    R=_rotation_matrix(matrix)
    return np.concatenate([R[..., :,0],R[..., :,1]],axis=-1)


def sixd_to_matrix(sixd,epsilon=1e-6):
    x=np.asarray(sixd,dtype=np.float64)
    if x.shape[-1:]!=(6,):raise ValueError('last dimension must be 6')
    finite=np.isfinite(x).all(axis=-1)
    clean=np.where(np.isfinite(x),x,0.)
    a,b=clean[...,:3],clean[...,3:]
    an=np.linalg.norm(a,axis=-1,keepdims=True)
    u=a/np.maximum(an,epsilon)
    orth=b-np.sum(u*b,axis=-1,keepdims=True)*u
    bn=np.linalg.norm(orth,axis=-1,keepdims=True)
    v=orth/np.maximum(bn,epsilon)
    valid=finite&np.isfinite(an[...,0])&np.isfinite(bn[...,0])&(an[...,0]>=epsilon)&(bn[...,0]>=epsilon)
    R=np.stack([u,v,np.cross(u,v)],axis=-1)
    return np.where(valid[...,None,None],R,np.eye(3)),valid


def matrix_to_rpy(matrix,epsilon=1e-6):
    R=_rotation_matrix(matrix)
    pitch=np.arcsin(np.clip(-R[...,2,0],-1,1))
    lock=np.abs(np.cos(pitch))<epsilon
    roll=np.where(lock,0.,np.arctan2(R[...,2,1],R[...,2,2]))
    yaw=np.where(lock,np.arctan2(-R[...,0,1],R[...,1,1]),np.arctan2(R[...,1,0],R[...,0,0]))
    return np.stack([roll,pitch,yaw],axis=-1),lock


def rotation_geodesic(first,second):
    relative=np.swapaxes(_rotation_matrix(first),-1,-2)@_rotation_matrix(second)
    return np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))


def relative_endpoint(root0,root2):
    first=np.asarray(root0,dtype=np.float64);second=np.asarray(root2,dtype=np.float64)
    if first.shape[-1:]!=(13,) or second.shape[-1:]!=(13,) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('finite root state13 required')
    R0=quat_to_matrix(first[...,3:7]);R2=quat_to_matrix(second[...,3:7])
    relative=np.swapaxes(R0,-1,-2)@R2
    delta=np.einsum('...ji,...j->...i',R0,second[...,:3]-first[...,:3])
    return dict(delta_position_body_m=delta,delta_rotation_6d=matrix_to_sixd(relative),
                delta_quaternion_wxyz=matrix_to_quat(relative))
