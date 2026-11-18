"""CPU verification against actual MuJoCo3.3.6 compiled original XML, not physics equivalence."""
from pathlib import Path
import numpy as np,mujoco,json,argparse
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(a.assets/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);z=np.load(a.mesh);h=z['heights'];x=z['x'];y=z['y'];raw=np.fromfile(a.assets/'heightfield.bin',dtype='<f4',offset=8).reshape(h.shape)
assert np.array_equal(m.hfield_data.reshape(h.shape),raw)
assert np.max(abs(m.hfield_data.reshape(h.shape)*m.hfield_size[0,2]-h))<1e-7
rng=np.random.default_rng(20261111);errs=[];samples=[]
for i in range(1500):
 c=int(rng.integers(0,len(x)-1));r=int(rng.integers(0,len(y)-1));u,v=rng.random(2);px=x[c]+u*.02;py=y[r]+v*.02
 # diagonal(r,c)->(r+1,c+1), exact triangular interpolation.
 if u>=v:expected=(1-u)*h[r,c]+(u-v)*h[r,c+1]+v*h[r+1,c+1]
 else:expected=(1-v)*h[r,c]+u*h[r+1,c+1]+(v-u)*h[r+1,c]
 gid=np.array([-1],np.int32);dist=mujoco.mj_ray(m,d,np.array([px,py,3.]),np.array([0.,0.,-1.]),None,1,-1,gid)
 actual=3.-dist;err=abs(actual-expected);errs.append(err)
 if i<12:samples.append(dict(x=px,y=py,expected=expected,mujoco_ray_z=actual,error=err,geom=int(gid[0])))
assert max(errs)<1e-7,max(errs)
report=dict(mujoco_version=mujoco.__version__,compiled_binary_equal=True,number_random_interior_rays=len(errs),max_triangle_ray_error_m=max(errs),mean_triangle_ray_error_m=float(np.mean(errs)),triangulation='confirmed SW->NE diagonal by actual mj_ray',samples=samples,note='CPU geometry validation only; no claim of MuJoCo/PhysX contact equivalence')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
