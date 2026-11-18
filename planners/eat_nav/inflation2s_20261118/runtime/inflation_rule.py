"""Known-wall inflation evaluated on NN2s endpoint whole-body projection."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from feasibility import hull2,CELL_M
from kinematics import matrix
MARGIN_M=.2

def make_costmap(grid):
 if not grid.wall.any():return np.zeros((40,40),float)
 return np.clip(1-distance_transform_edt(~grid.wall)*CELL_M/MARGIN_M,0,1)

def footprint_cost(grid,polygon,costmap=None):
 c=make_costmap(grid) if costmap is None else costmap;cells=grid.cells(polygon);values=[c[y,x] for x,y in cells if 0<=x<40 and 0<=y<40];coverage=grid.classify(cells)
 return dict(C=float(max(values,default=0.)),unknown_fraction=coverage['unknown_fraction'],known_wall_fraction=coverage['known_wall_fraction'],polygon=np.asarray(polygon).tolist())

def predicted_costs(grid,root,world_points,positions,rotations):
 root=np.asarray(root,float);Q=matrix(root)[:3,:3];local=(np.concatenate(world_points)-root[:3])@Q
 # Rotation can change the XY silhouette, so retain a 3D convex support hull.
 from scipy.spatial import ConvexHull
 if len(local)>4 and np.linalg.matrix_rank(local-local[0])==3:local=local[ConvexHull(local).vertices]
 P=np.asarray(positions,float);R=np.asarray(rotations,float)
 if P.ndim!=2 or P.shape[1]!=3 or R.shape!=(len(P),3,3) or not np.isfinite(P).all() or not np.isfinite(R).all():raise ValueError('finite NN endpoint poses required')
 c=make_costmap(grid);out=[]
 for p,r in zip(P,R):
  world=local@(Q@r).T+root[:3]+Q@p
  out.append(footprint_cost(grid,hull2(world[:,:2]),c))
 return out
