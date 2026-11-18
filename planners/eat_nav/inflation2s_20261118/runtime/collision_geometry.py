"""Read-only collision support geometry, reused from reviewed collector; no data/weights imported."""
from pathlib import Path
import numpy as np,json,re
from scipy.spatial import ConvexHull

def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def convex_distance(a,b):
 """GJK distance between convex point sets; closest simplex via allfaces KKT."""
 import itertools
 a=np.asarray(a,float);b=np.asarray(b,float)
 def support(d):return a[np.argmax(a@d)]-b[np.argmin(b@d)]
 def nearest(points):
  best=None
  for n in range(1,min(len(points),4)+1):
   for ids in itertools.combinations(range(len(points)),n):
    p=np.asarray([points[i] for i in ids]);g=p@p.T;k=np.block([[g,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    w=np.linalg.lstsq(k,np.r_[np.zeros(n),1.],rcond=None)[0][:n]
    if (w < -1e-9).any():continue
    q=w@p;norm=float(q@q)
    if best is None or norm<best[0]:best=(norm,q,[points[i] for i,v in zip(ids,w) if v>1e-10])
  return best[1],best[2]
 d=a.mean(0)-b.mean(0)
 if np.linalg.norm(d)<1e-10:d=np.array([1.,0,0])
 simplex=[support(-d)];p=simplex[0]
 for _ in range(128):
  if p@p<1e-20:return 0.
  w=support(-p)
  if p@p-p@w<=1e-12:return float(np.linalg.norm(p))
  q,simplex=nearest(simplex+[w])
  if np.linalg.norm(q-p)<1e-12:return float(np.linalg.norm(q))
  p=q
 raise ValueError('GJK did not converge')

def cylinder_points(radius,height,axis,segments=512):
 if axis not in 'XYZ' or radius<=0 or height<=0 or segments<16:raise ValueError('cylinder geometry invalid')
 axis_id='XYZ'.index(axis);others=[x for x in range(3) if x!=axis_id];theta=np.arange(segments)*2*np.pi/segments;ring=np.zeros((segments,3));ring[:,others[0]]=radius*np.cos(theta);ring[:,others[1]]=radius*np.sin(theta);a=ring.copy();b=ring.copy();a[:,axis_id]=-height/2;b[:,axis_id]=height/2
 return np.concatenate([a,b]),float(radius*(1-np.cos(np.pi/segments)))

def collision_shapes(robot,output):
 import omni.usd
 from pxr import Usd,UsdGeom,UsdPhysics
 import isaaclab.sim as sim_utils
 stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache();roots=sorted([str(p.GetPath()) for p in sim_utils.find_matching_prims(robot.cfg.prim_path)],key=lambda p:int(re.search(r'/env_(\d+)/',p).group(1)));robot_root=roots[0];base=stage.GetPrimAtPath(robot_root);out=[];unsupported=[]
 write(output/'robot-paths.json',dict(cfg=robot.cfg.prim_path,asset_roots=roots,articulation_roots=robot.root_physx_view.prim_paths))
 for prim in Usd.PrimRange(base,Usd.TraverseInstanceProxies()):
  path=str(prim.GetPath())
  if not path.startswith(robot_root+'/') or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
  body=prim
  while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
  if not body or body.GetName() not in robot.body_names:continue
  approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
  error_bound=0.
  if prim.IsA(UsdGeom.Mesh):pts=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
  elif prim.IsA(UsdGeom.Cube):size=float(UsdGeom.Cube(prim).GetSizeAttr().Get());pts=np.array([[x,y,z] for x in [-size/2,size/2] for y in [-size/2,size/2] for z in [-size/2,size/2]])
  elif prim.IsA(UsdGeom.Cylinder):
   cyl=UsdGeom.Cylinder(prim);pts,error_bound=cylinder_points(float(cyl.GetRadiusAttr().Get()),float(cyl.GetHeightAttr().Get()),str(cyl.GetAxisAttr().Get()),512)
  else:unsupported.append(dict(path=path,type=prim.GetTypeName(),approximation=approx));continue
  matrix=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse());pts=(np.c_[pts,np.ones(len(pts))]@matrix)[:,:3];pts=np.unique(pts,axis=0)
  if len(pts)>4 and not prim.IsA(UsdGeom.Cylinder):pts=pts[ConvexHull(pts).vertices]
  error_bound*=float(np.linalg.norm(matrix[:3,:3],ord=2))
  out.append(dict(path=path,body_name=body.GetName(),body_index=robot.body_names.index(body.GetName()),approximation=approx,primitive_type=prim.GetTypeName(),distance_error_bound_m=error_bound,points=pts.tolist()))
 write(output/'collision-shapes.json',dict(shapes=out,unsupported=unsupported,method='body-local USD collision support points; convex support used for distances'))
 if unsupported or not out:raise ValueError('unsupported/missing collision shapes '+str(unsupported))
 return out
