"""Ideal terrain-only spinning-pattern LiDAR used ONLY to feed real FAST-LIO.

No hardware-equivalence claim: 16x512 rays, -40..20deg, 12m, no noise,
instantaneous scan end, terrain-only visibility, LiDAR and IMU collocated
at body root. These simulated points are NOT neural network inputs.
"""
import numpy as np
import torch
from isaaclab.utils.warp import convert_to_warp_mesh,raycast_mesh
from .moving_pose2s_geometry import quat_to_matrix
from .single_pose2s_fastlio import imu_specific_force


class SimLioSensors:
    def __init__(self, meshes, origins, device, count):
        self.device=device;self.count=count;self.origins=np.asarray(origins)
        self.mesh=[convert_to_warp_mesh(*meshes[i],device=device) for i in range(count)]
        az,el=np.meshgrid(np.arange(512)*2*np.pi/512,np.deg2rad(np.linspace(-40,20,16)))
        self.directions=np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)],-1).reshape(-1,3).astype(np.float32)
        self.imu_stamps=[];self.acc=[];self.gyro=[];self.lidar_stamps=[];self.points=[];self.previous=None
    def capture(self, stamp_ns, roots):
        roots=np.asarray(roots);prior=roots if self.previous is None else self.previous
        dt=.05 if not self.imu_stamps else (stamp_ns-self.imu_stamps[-1])*1e-9
        self.acc.append(np.stack([imu_specific_force(prior[i],roots[i],dt) for i in range(self.count)]).astype(np.float32))
        self.gyro.append(np.stack([quat_to_matrix(r[3:7]).T@r[10:13] for r in roots]).astype(np.float32));self.imu_stamps.append(stamp_ns)
        if stamp_ns%100000000==0:
            points=[]
            for i,r in enumerate(roots):
                R=quat_to_matrix(r[3:7]);start=np.broadcast_to((r[:3]-self.origins[i]).astype(np.float32),self.directions.shape).copy()
                directions=(self.directions@R.T).astype(np.float32)
                hit,_,_,_=raycast_mesh(torch.as_tensor(start,device=self.device),torch.as_tensor(directions,device=self.device),self.mesh[i],max_dist=12.)
                hits=hit.cpu().numpy();valid=np.isfinite(hits).all(1)
                local=np.full_like(hits,np.nan);local[valid]=(hits[valid]-start[valid])@R
                points.append(local.astype(np.float32))
            self.points.append(np.stack(points));self.lidar_stamps.append(stamp_ns)
        self.previous=roots.copy()
    def save(self,path,index):
        np.savez_compressed(path,imu_stamp_ns=np.asarray(self.imu_stamps,np.int64),imu_acc=np.asarray(self.acc)[:,index],imu_gyro=np.asarray(self.gyro)[:,index],lidar_end_stamp_ns=np.asarray(self.lidar_stamps,np.int64),lidar_points=np.asarray(self.points)[:,index])
