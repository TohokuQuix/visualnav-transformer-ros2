"""ROS-independent sensor geometry. Quaternion arguments use xyzw order."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

FLIPPERS = tuple('chassis/'+s+'_flipper_joint' for s in ('rr','rl','fr','fl'))
TRACKS = tuple('chassis/'+s+'_sprocket_axle' for s in ('fr','rr','fl','rl'))


def finite(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'nonfinite or wrong shape: expected {shape}')
    return a


def fresh(stamp, now, max_age):
    # Sensor/TF samples may lead or lag the reference clock within the same bound.
    return stamp > 0 and abs(now-stamp) <= max_age


def stamp_ns(stamp):
    return int(stamp.sec)*1_000_000_000+int(stamp.nanosec)


def rotation(q):
    q=finite(q,(4,));norm=np.linalg.norm(q)
    if norm < 1e-8 or abs(norm-1) > .01:
        raise ValueError('invalid quaternion norm')
    x,y,z,w=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def quaternion(q):
    return [q.x,q.y,q.z,q.w]


def xyz(v):
    return [v.x,v.y,v.z]


def imu_state(orientation, angular, body_from_sensor):
    r=finite(body_from_sensor,(3,3))
    return r @ rotation(orientation).T @ np.array([0.,0.,-1.]), r @ finite(angular,(3,))


def transform_twist(linear, angular, target_from_source, source_origin_in_target):
    r=finite(target_from_source,(3,3));w=r @ finite(angular,(3,))
    return r @ finite(linear,(3,))+np.cross(finite(source_origin_in_target,(3,)),w),w


@dataclass
class GridRaster:
    height: np.ndarray  # logical rows +y, columns +x; already unwrapped
    valid: np.ndarray
    center: np.ndarray
    resolution: float
    yaw: float
    frame_id: str
    stamp: int


def _matrix(array, nx, ny):
    dims=array.layout.dim
    if len(dims)!=2:
        raise ValueError('GridMap must have two layout dimensions')
    labels=[d.label for d in dims]
    if labels==['column_index','row_index']:
        sizes=(ny,nx);order='F'
    elif labels==['row_index','column_index']:
        sizes=(nx,ny);order='C'
    else:
        raise ValueError(f'unsupported GridMap layout labels: {labels}')
    if tuple(d.size for d in dims)!=sizes or tuple(d.stride for d in dims)!=(nx*ny,sizes[1]):
        raise ValueError('GridMap layout/geometry mismatch')
    offset=int(array.layout.data_offset)
    data=np.asarray(array.data,dtype=np.float64)
    if offset<0 or len(data)!=offset+nx*ny:
        raise ValueError('GridMap data length mismatch')
    return data[offset:].reshape((nx,ny),order=order)


def decode_grid_map(message, layer):
    info=message.info;r=float(info.resolution)
    if not math.isfinite(r) or r<=0:
        raise ValueError('invalid GridMap resolution')
    lengths=finite([info.length_x,info.length_y],(2,))
    if (lengths<=0).any():raise ValueError('invalid GridMap lengths')
    nx,ny=np.rint(lengths/r).astype(int)
    if min(nx,ny)<2 or not np.allclose(lengths,[nx*r,ny*r],atol=1e-6,rtol=0):
        raise ValueError('GridMap length is not an integral cell count')
    if len(message.layers)!=len(message.data) or len(set(message.layers))!=len(message.layers):
        raise ValueError('GridMap layer mismatch')
    if layer not in message.layers:
        raise ValueError(f'height layer {layer!r} absent; available: {list(message.layers)}')
    starts=(int(message.outer_start_index),int(message.inner_start_index))
    if not (0<=starts[0]<nx and 0<=starts[1]<ny):raise ValueError('invalid GridMap start index')
    def unpack(name):
        if name not in message.layers:raise ValueError('missing GridMap basic layer: '+name)
        raw=_matrix(message.data[message.layers.index(name)],nx,ny)
        return np.roll(raw,(-starts[0],-starts[1]),axis=(0,1))[::-1,::-1].T
    height=unpack(layer);valid=np.isfinite(height)
    for name in message.basic_layers:valid &= np.isfinite(unpack(name))
    rot=rotation(quaternion(info.pose.orientation))
    if not np.allclose(rot[2],[0,0,1],atol=1e-6):
        raise ValueError('tilted GridMap geometry is unsupported')
    header=getattr(message,'header',None) or info.header
    center=finite(xyz(info.pose.position),(3,))
    # GridMap elevations are already expressed in header.frame_id, not relative to pose.z.
    return GridRaster(height,valid,center[:2],r,math.atan2(rot[1,0],rot[0,0]),
                      header.frame_id,stamp_ns(header.stamp))


def sample_dem(grid, pose):
    px,py,yaw=finite(pose,(3,));axis=(np.arange(20)-9.5)*.1
    bx,by=np.meshgrid(axis,axis);c,s=math.cos(yaw),math.sin(yaw)
    dx=px+c*bx-s*by-grid.center[0];dy=py+s*bx+c*by-grid.center[1]
    c,s=math.cos(grid.yaw),math.sin(grid.yaw)
    rows,cols=grid.height.shape
    gx=(c*dx+s*dy)/grid.resolution+(cols-1)/2
    gy=(-s*dx+c*dy)/grid.resolution+(rows-1)/2
    inside=(gx>=-1e-7)&(gx<=cols-1+1e-7)&(gy>=-1e-7)&(gy<=rows-1+1e-7)
    gx=np.clip(gx,0,cols-1);gy=np.clip(gy,0,rows-1)
    gx=np.where(abs(gx-np.rint(gx))<1e-7,np.rint(gx),gx)
    gy=np.where(abs(gy-np.rint(gy))<1e-7,np.rint(gy),gy)
    x0=np.floor(gx).astype(int);y0=np.floor(gy).astype(int)
    x1=np.minimum(x0+1,cols-1);y1=np.minimum(y0+1,rows-1)
    wx=gx-x0;wy=gy-y0;height=np.zeros((20,20));valid=inside.copy()
    for x,y,w in ((x0,y0,(1-wx)*(1-wy)),(x1,y0,wx*(1-wy)),
                  (x0,y1,(1-wx)*wy),(x1,y1,wx*wy)):
        ok=grid.valid[y,x]&np.isfinite(grid.height[y,x]);active=w>1e-12
        valid &= ~active|ok
        height+=w*np.where(ok,grid.height[y,x],0.)
    if valid.mean()<.25:raise ValueError('DEM coverage below 25%')
    radius=np.hypot(bx,by);reference=valid&(radius>=.3)&(radius<=.8)
    if not reference.any():raise ValueError('DEM missing reference-height annulus')
    height-=np.median(height[reference]);height[~valid]=0
    return np.stack([height,valid]).astype(np.float32)


def actor_state(joints, gravity, angular, fold):
    if fold not in (-1,0,1):raise ValueError('invalid flipper fold mode')
    if len(set(joints.name))!=len(joints.name):raise ValueError('duplicate joint names')
    lookup={n:i for i,n in enumerate(joints.name)}
    if any(n not in lookup for n in FLIPPERS+TRACKS):raise ValueError('missing required joint')
    def values(field,names):
        a=getattr(joints,field)
        if len(a)!=len(joints.name):raise ValueError(f'missing or truncated JointState.{field}')
        return finite([a[lookup[n]] for n in names],(len(names),))
    pos=values('position',FLIPPERS);vel=values('velocity',TRACKS);effort=values('effort',FLIPPERS+TRACKS)
    speed=np.array([vel[:2].mean(),vel[2:].mean()])
    efforts=np.r_[effort[:4],effort[4:6].sum(),effort[6:8].sum()]
    return np.r_[np.clip(finite(angular,(3,)),-100,100)*.25,
                 np.clip(finite(gravity,(3,)),-1,1),[0.,0.,0.],fold,
                 np.clip(pos,-100,100),np.clip(speed,-100,100)*.05,
                 np.clip(efforts,-100,100)*.01].astype(np.float32)


def body_state(gravity, linear, angular):
    g=finite(gravity,(3,))
    if abs(np.linalg.norm(g)-1)>.01:raise ValueError('invalid projected gravity')
    return np.concatenate([g,finite(linear,(3,)),finite(angular,(3,))]).astype(np.float32)


class DemHistory:
    def __init__(self,tolerance_ns=50_000_000,interval_ns=250_000_000,max_age_ns=250_000_000):
        if tolerance_ns<0 or interval_ns<=0 or max_age_ns<=0:raise ValueError('invalid history timing')
        self.tolerance_ns=tolerance_ns;self.interval_ns=interval_ns;self.max_age_ns=max_age_ns
        self.frames=deque(maxlen=200)

    def clear(self):self.frames.clear()

    def push(self,stamp,frame):
        if stamp<=0:raise ValueError('missing DEM stamp')
        if self.frames and stamp<=self.frames[-1][0]:
            return False  # late/duplicate packets must not erase usable history
        frame=finite(frame,(2,20,20)).astype(np.float32)
        self.frames.append((stamp,frame.copy()))
        return True

    def select(self,now_ns):
        if len(self.frames)<4:
            raise ValueError(f'insufficient DEM history: {len(self.frames)}/4 distinct frames')
        if not fresh(self.frames[-1][0],now_ns,self.max_age_ns):
            age=(now_ns-self.frames[-1][0])/1e9
            raise ValueError(f'stale DEM history: offset={age:.3f}s, limit=+/-{self.max_age_ns/1e9:.3f}s')
        stamps=np.array([s for s,_ in self.frames],dtype=np.int64)
        targets=stamps[-1]-np.arange(3,-1,-1)*self.interval_ns
        errors=np.abs(targets[:,None]-stamps[None,:])
        n=len(stamps)
        costs=np.full((4,n),np.inf)
        parents=np.full((4,n),-1,dtype=int)
        costs[0]=np.where(errors[0]<=self.tolerance_ns,errors[0],np.inf)
        # Order-preserving minimum-error assignment. A greedy nearest neighbour
        # can pick the same frame twice when timing windows overlap.
        for row in range(1,4):
            best_cost=float('inf');best_index=-1
            for col in range(n):
                if col>0 and costs[row-1,col-1]<best_cost:
                    best_cost=costs[row-1,col-1];best_index=col-1
                if errors[row,col]<=self.tolerance_ns and best_index>=0:
                    costs[row,col]=best_cost+errors[row,col]
                    parents[row,col]=best_index
        if not np.isfinite(costs[3,-1]):
            raise ValueError('DEM rate/spacing cannot provide four distinct frames '
                             f'within +/-{self.tolerance_ns/1e9:.3f}s')
        ids=[n-1]
        for row in range(3,0,-1):ids.append(int(parents[row,ids[-1]]))
        ids.reverse()
        return np.stack([self.frames[i][1] for i in ids]),[int(stamps[i]) for i in ids]
