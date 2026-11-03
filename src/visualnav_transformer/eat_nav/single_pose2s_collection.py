"""CPU collection/teacher primitives shared with the simulator."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .moving_pose2s_geometry import quat_to_matrix


def window_endpoints(stamps,alive):
    stamps=np.asarray(stamps,np.int64);alive=np.asarray(alive,bool)
    if stamps.shape!=(201,) or alive.shape!=(201,) or not np.array_equal(stamps,np.arange(201,dtype=np.int64)*50000000):raise ValueError('exact 20Hz complete time index required')
    return [(s,s+40,bool(alive[s:s+40].all())) for s in range(0,200,40)]


def acquisition_steps(phase):
    if phase not in range(5):raise ValueError('phase must be0..4')
    return sorted(set([0]+list(range(phase,201,5))))


def inpaint_frame(heights,raw_mask,z_ref):
    h=np.asarray(heights,float);raw=np.asarray(raw_mask,bool)&np.isfinite(h)
    if h.shape!=(40,40) or raw.shape!=h.shape or not raw.any() or not np.isfinite(z_ref):raise ValueError('valid raw support required')
    indices=distance_transform_edt(~raw,return_distances=False,return_indices=True)
    filled=h[tuple(indices)]-z_ref
    return np.stack([filled,np.isfinite(filled).astype(float),raw.astype(float)]).astype(np.float32)


def causal_dem_context(frame_root,current_root,stamp_ns,t0_ns):
    if stamp_ns>t0_ns:raise ValueError('future DEM')
    a=np.asarray(frame_root);b=np.asarray(current_root);Ra=quat_to_matrix(a[3:7]);Rb=quat_to_matrix(b[3:7])
    ya=np.arctan2(Ra[1,0],Ra[0,0]);yb=np.arctan2(Rb[1,0],Rb[0,0]);c,s=np.cos(yb),np.sin(yb)
    delta=np.array([[c,s],[-s,c]])@(a[:2]-b[:2]);angle=ya-yb
    return np.array([(t0_ns-stamp_ns)*1e-9,1.,*delta,np.sin(angle),np.cos(angle)],np.float32)


def terrain_clearance_z(vertices,faces,xy,clearance):
    """Place initial body above its local ground footprint; never changes a trial."""
    triangles=np.asarray(vertices,float)[np.asarray(faces)];a=triangles[:,0,:2];u=triangles[:,1,:2]-a;v=triangles[:,2,:2]-a
    det=u[:,0]*v[:,1]-u[:,1]*v[:,0];nonvertical=np.abs(det)>1e-12;safe=np.where(nonvertical,det,1.)
    heights=[]
    for dx in np.linspace(-.7,.7,7):
        for dy in np.linspace(-.7,.7,7):
            p=np.asarray(xy)+[dx,dy]-a
            s=(p[:,0]*v[:,1]-p[:,1]*v[:,0])/safe;t=(u[:,0]*p[:,1]-u[:,1]*p[:,0])/safe
            hit=nonvertical&(s>=-1e-8)&(t>=-1e-8)&(s+t<=1+1e-8)
            if hit.any():heights.append(float(np.max(triangles[hit,0,2]+s[hit]*(triangles[hit,1,2]-triangles[hit,0,2])+t[hit]*(triangles[hit,2,2]-triangles[hit,0,2]))))
    if not heights:raise ValueError('no terrain beneath initial footprint')
    return max(heights)+float(clearance)


def pending_parent_batches(manifest, completed_parent_ids, num_envs):
    """Group distinct pending parents by terrain cell, preserving their split roles."""
    if not isinstance(num_envs,int) or num_envs<1:raise ValueError('positive environment count required')
    parents=manifest['parents'];known={p['parent_id'] for p in parents}
    if not set(completed_parent_ids)<=known:raise ValueError('unknown completed parent')
    batches=[]
    for field in manifest['fields']:
        for level in range(5):
            rows=[p for p in parents if p['field']==field and p['level']==level and p['parent_id'] not in completed_parent_ids]
            batches.extend(rows[i:i+num_envs] for i in range(0,len(rows),num_envs))
    return batches
