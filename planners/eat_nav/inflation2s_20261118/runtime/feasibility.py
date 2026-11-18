"""Frozen-articulation candidate geometry, not a physical rollout or probability."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation,Slerp
from kinematics import matrix

WALL_HEIGHT_M=.2
CELL_M=.1

def causal_frame(stamp,received,t,pose_times):
 stamp=np.asarray(stamp);received=np.asarray(received);times=np.asarray(pose_times)
 if stamp.shape!=received.shape or not np.isfinite(stamp).all() or not np.isfinite(received).all():raise ValueError('DEM clocks')
 good=np.flatnonzero((stamp<=t+1e-9)&(received<=t+1e-9))
 if not len(good):raise ValueError('no causal DEM')
 i=int(good[-1])
 if not np.any(np.isclose(times,stamp[i],rtol=0,atol=1e-7)):raise ValueError('missing acquisition pose')
 return i

def hull2(points):
 p=np.asarray(points,float)
 if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('finite XY polygon required')
 return p[ConvexHull(p).vertices]

class DEMGrid:
 def __init__(self,dem,acquisition_pose):
  self.dem=np.asarray(dem,float);self.pose=matrix(acquisition_pose);self.yaw=np.arctan2(self.pose[1,0],self.pose[0,0]);c,s=np.cos(self.yaw),np.sin(self.yaw);self.rot=np.array([[c,-s],[s,c]])
  if self.dem.shape!=(3,40,40) or not np.isfinite(self.dem).all() or np.any((self.dem[2]<0)|(self.dem[2]>1)):raise ValueError('DEM3x40x40 with rawsupportmask required')
  self.known=self.dem[2]>.5;self.wall=self.known & (self.dem[0]+self.pose[2,3]>WALL_HEIGHT_M)
 def local(self,points):return (np.asarray(points)-self.pose[:2,3])@self.rot
 def cells(self,poly):
  p=self.local(hull2(poly));h=ConvexHull(p);lo=np.floor((p.min(0)+2)/CELL_M).astype(int)-1;hi=np.floor((p.max(0)+2)/CELL_M).astype(int)+1
  if np.max(hi-lo)>200:raise ValueError('unbounded footprint')
  xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));ij=np.c_[xx.ravel(),yy.ravel()];centres=(ij+.5)*CELL_M-2
  # SAT: convex hull face normals plus box axes. Intersection includes grid quantization, not clearance penalty.
  a=h.equations[:,:2];b=h.equations[:,2];inside=np.all(centres@a.T+b<=CELL_M/2*np.abs(a).sum(1)+1e-10,axis=1)
  inside &= np.all(centres>=p.min(0)-CELL_M/2-1e-10,axis=1)&np.all(centres<=p.max(0)+CELL_M/2+1e-10,axis=1)
  return {tuple(x) for x in ij[inside]}
 def classify(self,cells):
  if not cells:return dict(known_wall_fraction=0.,known_free_fraction=0.,unknown_fraction=1.,cells=0)
  walls=free=0
  for x,y in cells:
   if 0<=x<40 and 0<=y<40 and self.known[y,x]:
    if self.wall[y,x]:walls+=1
    else:free+=1
  n=len(cells);return dict(known_wall_fraction=walls/n,known_free_fraction=free/n,unknown_fraction=(n-walls-free)/n,cells=n)
 def coverage(self,poly):return self.classify(self.cells(poly))

def evaluate_polygons(grid,polygons,turning):
 cells=[grid.cells(p) for p in polygons];start=cells[0];cover=[grid.classify(x) for x in cells];new=[grid.classify(x-start) if x-start else dict(known_wall_fraction=0.,known_free_fraction=0.,unknown_fraction=0.,cells=0) for x in cells]
 obstruction=float(max(x['known_wall_fraction'] for x in new))
 return dict(obstruction_fraction=obstruction,initial_known_wall_fraction=cover[0]['known_wall_fraction'],opening_known_free_fraction=cover[-1]['known_free_fraction'] if turning and obstruction==0 and cover[-1]['known_free_fraction']==1 else 0.,unknown_fraction=grid.classify(set().union(*cells))['unknown_fraction'],endpoint_known_wall_fraction=cover[-1]['known_wall_fraction'],endpoint_known_free_fraction=cover[-1]['known_free_fraction'],coverage=cover,polygons=[p.tolist() for p in polygons])

def world_shape_points(shapes,frames):
 return [np.asarray(s['points'])@frames[s['body_name']][:3,:3].T+frames[s['body_name']][:3,3] for s in shapes]

def candidates(root,world_points,angles,directions,positions=None,rotations=None):
 t=matrix(root);r0=t[:3,:3];local=(np.concatenate(world_points)-t[:3,3])@r0;local=local[ConvexHull(local).vertices];result=[]
 for i,(angle,direction) in enumerate(zip(angles,directions)):
  full=[]
  for s in np.linspace(0,.75,31):
   a=np.deg2rad(angle)*s/.75
   pos=np.array([s,0.]) if angle==0 else .75/np.deg2rad(angle)*np.array([np.sin(a),1-np.cos(a)])
   c,si=np.cos(a),np.sin(a);R=np.array([[c,-si],[si,c]])
   R3=np.eye(3);R3[:2,:2]=R;yaw=np.arctan2(r0[1,0],r0[0,0]);cy,sy=np.cos(yaw),np.sin(yaw);yawR=np.array([[cy,-sy],[sy,cy]]);cloud=local@(R3@r0).T+t[:3,3]+np.r_[yawR@(direction*pos),0.];full.append(hull2(cloud[:,:2]))
  item=dict(full=full)
  if positions is not None:
   endR=np.asarray(rotations[i]);slerp=Slerp([0,1],Rotation.from_matrix(np.stack([np.eye(3),endR])));prefix=[]
   for f,R in zip(np.linspace(0,1,21),slerp(np.linspace(0,1,21)).as_matrix()):
    cloud=local@(r0@R).T+t[:3,3]+r0@np.asarray(positions[i])*f;prefix.append(hull2(cloud[:,:2]))
   item['proxy2s']=prefix
  result.append(item)
 return result
