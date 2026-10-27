"""Validate initial tensor/time/frame evidence independently of saved gate flags."""
from .config import load_eat_config
from .dem import SourceGrid, robot_dem
from .fall_stall_targets import root_yaw


def validate_initial(sim,seed,repo,read,sha,require,cache):
    import numpy as np
    stamps=[250_000_000,500_000_000,750_000_000,1_000_000_000]
    identity=read(sim/'prepath-identity.json');metadata=read(sim/'raw-dem-metadata.json')
    require(identity.get('seed')==seed,'initial seed mismatch')
    require(identity.get('dem_frame')=='robot_base' and identity.get('dem_resolution_m')==.1,'initial DEM frame metadata mismatch')
    require(identity.get('dem_stamps_ns')==metadata.get('logical_stamp_ns')==stamps,'initial DEM timestamp mismatch')
    require(metadata.get('source')=='height_scanner_eat native world-frame rays','raw DEM world-frame provenance mismatch')
    key=(sha(sim/'prepath.npz'),sha(sim/'raw-dem.npz'))
    if key in cache:
        return cache[key]
    with np.load(sim/'prepath.npz',allow_pickle=False) as initial:
        expected={'root':(4,13),'root_tile_local':(4,13),'dem':(4,4,2,20,20),
                  'actor_pre_noise':(4,22),'actor_input':(4,22),'actor_post_noise':(4,22)}
        for name,shape in expected.items():
            require(name in initial.files,f'missing required prepath tensor: {name}')
            require(initial[name].shape==shape,f'initial {name} shape mismatch')
        arrays={name:initial[name].copy() for name in expected}
    for name in expected:
        if name!='dem':require(np.isfinite(arrays[name]).all(),f'initial {name} is nonfinite')
    dem=arrays['dem'];mask=dem[:,:,1];height=dem[:,:,0]
    require(np.isfinite(mask).all() and np.logical_or(mask==0,mask==1).all(),'initial DEM mask is not binary/finite')
    require(np.isfinite(height[mask==1]).all() and (mask.mean(axis=(-2,-1))>=.25).all(),'initial DEM height/coverage invalid')
    with np.load(sim/'raw-dem.npz',allow_pickle=False) as raw:
        shapes={'ray_hits_w':(4,4,400,3),'ray_starts_w':(4,4,400,3),'ray_valid':(4,4,400),
                'root_state_w':(4,4,13),'env_origins_w':(4,4,3),'logical_stamp_ns':(4,)}
        for name,shape in shapes.items():
            require(name in raw.files and raw[name].shape==shape,f'raw DEM {name} shape/missing contract')
        native={name:raw[name].copy() for name in shapes}
    require(native['logical_stamp_ns'].tolist()==stamps,'raw DEM array timestamp mismatch')
    for name in ('ray_starts_w','root_state_w','env_origins_w'):
        require(np.isfinite(native[name]).all(),f'raw DEM {name} nonfinite')
    valid=np.isfinite(native['ray_hits_w'][...,2])
    require(np.array_equal(valid,native['ray_valid']),'raw DEM validity mask mismatch')
    require(np.isfinite(native['ray_hits_w'][valid]).all(),'valid raw DEM XYZ is nonfinite')
    require(np.allclose(native['root_state_w'][-1],arrays['root'],atol=1e-6,rtol=0),'raw/initial root state mismatch')
    require(np.allclose(arrays['root'][:,:3]-arrays['root_tile_local'][:,:3],native['env_origins_w'][-1],
                        atol=2e-6,rtol=0),'tile/world origin mismatch')
    config=load_eat_config(repo/'config/eat_nav.yaml')
    maximum_error=0.
    for frame in range(4):
        for env in range(4):
            starts=native['ray_starts_w'][frame,env].reshape(20,20,3)
            rows,cols=np.meshgrid(np.arange(20)*.1,np.arange(20)*.1,indexing='ij')
            require(np.allclose(starts[:,:,0],starts[0,0,0]+cols,atol=1e-5,rtol=0) and
                    np.allclose(starts[:,:,1],starts[0,0,1]+rows,atol=1e-5,rtol=0),'native DEM raster axes/frame mismatch')
            root=native['root_state_w'][frame,env].tolist();angle=root_yaw(root)
            require(angle is not None,'invalid raw DEM root quaternion')
            hits=native['ray_hits_w'][frame,env,:,2].reshape(20,20)
            present=np.isfinite(hits)
            grid=SourceGrid(np.where(present,hits,0).astype(np.float32),present,
                            (float(starts[0,0,0]),float(starts[0,0,1])),.1,(0,0),config.reference_frame,stamps[frame])
            reconstructed=robot_dem(grid,(root[0],root[1],angle),config)
            require(np.array_equal(reconstructed[1],mask[env,frame]),'raw-to-initial DEM mask reconstruction mismatch')
            error=float(np.max(np.abs(reconstructed[0][mask[env,frame]==1]-height[env,frame][mask[env,frame]==1])))
            require(error<=2e-5,'raw-to-initial DEM height/frame reconstruction mismatch')
            maximum_error=max(maximum_error,error)
    result={'root':arrays['root'].tolist(),'root_tile_local':arrays['root_tile_local'].tolist(),
            'dem_reconstruction_max_abs_m':maximum_error,'dem_frames_verified':16}
    cache[key]=result
    return result
