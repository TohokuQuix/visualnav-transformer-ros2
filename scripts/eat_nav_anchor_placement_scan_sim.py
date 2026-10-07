"""Candidate-free four-radius RayCaster scan; no env.step or settling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--parity-report',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
import omni.usd
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid,robot_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER,terrain_seed
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.anchor_placement_geometry import footprint_diagnostic
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator,make_collection_cfg


class RadialTerrainGenerator(EATTerrainGenerator):
    """Same generated tile seed and difficulty down each terrain column."""

    def _generate_random_terrains(self):
        cfgs = [self.cfg.sub_terrains[name] for name in TERRAIN_ORDER]
        if (self.cfg.num_rows, self.cfg.num_cols) != (16, 7):
            raise ValueError('frozen scene must be 16x7')
        for row in range(16):
            for col, cfg in enumerate(cfgs):
                mesh, origin = self._get_terrain_mesh(.5, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)

    def _get_terrain_mesh(self, difficulty, cfg):
        col = self._tile_index % 7
        seed = terrain_seed(int(self.cfg.seed), col)
        old_seed, old_np = self.cfg.seed, np.random.get_state()
        try:
            self.cfg.seed = seed
            np.random.seed(seed)
            mesh, origin = TerrainGenerator._get_terrain_mesh(self, difficulty, cfg)
            name = TERRAIN_ORDER[col]
            self.tile_metadata.append({'tile_index': self._tile_index, 'row': self._tile_index // 7,
                                       'column': col, 'seed': seed, 'terrain_type': name,
                                       'difficulty': float(difficulty)})
            self._tile_index += 1
            return mesh, origin
        finally:
            self.cfg.seed = old_seed
            np.random.set_state(old_np)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def collision_proxy(stage,robot):
    root_prim=stage.GetPrimAtPath('/World/envs/env_0/onix_chassis')
    if not root_prim.IsValid():raise ValueError('robot root prim missing')
    # URDF-generated collision shapes have USD purpose=guide. Excluding guide
    # yields an empty bound even when CollisionAPI is composed in the stage.
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.guide])
    # IsaacLab's cloned Actor state is already at the terrain origin, while
    # the USD collision template remains at its authoring transform. Rebase
    # bounds in the USD root's own frame before using them at virtual poses.
    usd_root_to_world=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    world_to_usd_root=usd_root_to_world.GetInverse()
    actor_root=robot.data.root_pos_w[0].detach().cpu().numpy()
    groups={'main_track_sprocket':[],'initial_flipper_support':[],
            'articulated_reach':[],'full_collision_union':[]}
    counts={'collision_api_prim_count':0,'usable_collision_bound_count':0,
            'usd_template_root_xyz':list(usd_root_to_world.ExtractTranslation()),
            'sim_actor_root_xyz':actor_root.tolist(),
            'coordinate_rule':'USD world bounds transformed by inverse USD root transform'}
    for prim in Usd.PrimRange(root_prim,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        counts['collision_api_prim_count']+=1
        try:
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower,upper=np.asarray(bounds.GetMin(),dtype=float),np.asarray(bounds.GetMax(),dtype=float)
        except Exception:
            continue
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper<lower):
            continue
        path=str(prim.GetPath()).lower()
        corners=[]
        for x in (lower[0],upper[0]):
            for y in (lower[1],upper[1]):
                local=world_to_usd_root.Transform(Gf.Vec3d(float(x),float(y),float((lower[2]+upper[2])/2)))
                corners.append((float(local[0]),float(local[1])))
        groups['full_collision_union'].extend(corners)
        if any(f'/chassis_{prefix}' in path for prefix in ('fl','fr','rl','rr')):
            groups['initial_flipper_support'].extend(corners)
            groups['articulated_reach'].extend(corners)
        elif 'track' in path or 'sprocket' in path:
            groups['main_track_sprocket'].extend(corners)
        counts['usable_collision_bound_count']+=1
    if any(not groups[key] for key in groups):
        raise ValueError('incomplete USD collision proxy categories: '+repr({k:len(v) for k,v in groups.items()}))
    return groups,counts


def outside_corridor_difference(height,mask):
    x=(np.arange(20)-9.5)*.1;y=(np.arange(20)-9.5)*.1
    cols=np.where((x>.6)&(x<=.95))[0]
    bands=[np.where((y>=.15)&(y<=.45))[0],np.where(np.abs(y)<=.15)[0],
           np.where((y<=-.15)&(y>=-.45))[0]]
    differences=[]
    for col in cols:
        values=[]
        for rows in bands:
            known=height[rows,col][mask[rows,col]]
            if not len(known):break
            values.append(float(np.median(known)))
        if len(values)==3:differences.append(max(values)-min(values))
    return max(differences) if differences else None


def scan_radius(bare,rows,radius,config,proxy):
    robot=bare.scene['robot'];scanner=bare.scene['height_scanner_eat']
    origins=bare.scene.env_origins.detach().cpu().numpy()
    metadata=bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata
    theta=torch.tensor([r['row']*math.pi/8 for r in rows],device=bare.device,dtype=robot.data.root_state_w.dtype)
    states=torch.zeros((112,7),device=bare.device,dtype=robot.data.root_state_w.dtype)
    origin_tensor=bare.scene.env_origins
    states[:,:3]=origin_tensor
    states[:,0]+=radius*torch.cos(theta);states[:,1]+=radius*torch.sin(theta);states[:,2]+=.35
    states[:,3:7]=math_utils.quat_from_euler_xyz(theta*0,theta*0,theta)
    ids=torch.arange(112,device=bare.device,dtype=torch.long)
    robot.write_root_pose_to_sim(states,env_ids=ids)
    bare.scene.write_data_to_sim();bare.sim.forward()
    scanner.reset(ids);scanner.update(0.0,force_recompute=True)
    candidates=[]
    for row in rows:
        i=row['env_id'];m=metadata[i];angle=row['row'];t=float(theta[i].item())
        if (m['seed']!=row['terrain_seed'] or m['terrain_type']!=row['terrain'] or
            m['row']!=row['row'] or m['column']!=row['column'] or float(m['difficulty'])!=.5):
            raise ValueError('terrain identity mismatch: '+row['pose_id'])
        pos=robot.data.root_pos_w[i].detach().cpu().numpy()
        hits=scanner.data.ray_hits_w[i,:,2].detach().cpu().numpy().reshape(20,20)
        valid=np.isfinite(hits)
        start=scanner._ray_starts_w[i,0,:2].detach().cpu().numpy()
        source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
                          origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
                          start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=0)
        pose_id=f"b20261007-{row['terrain']}-r{radius:.2f}-a{angle:02d}"
        base={'pose_id':pose_id,'terrain':row['terrain'],'terrain_seed':row['terrain_seed'],
              'source_family':f"20261007:{row['terrain']}",'env_id':i,
              'direction_index':angle,'theta_rad':t,'radius_m':radius,
              'requested_xy_yaw':[float(pos[0]),float(pos[1]),t],
              'root_z_rule':'scene.env_origins.z + 0.35 m',
              'requested_root_z_m':float(origins[i,2]+.35),
              'env_origin_xyz':origins[i].tolist(),
              'terrain_difficulty':.5,'geometry_valid':True,
              'frame_id':config.reference_frame,'axis':'robot-relative-grid-xy',
              'resolution_m':config.dem_resolution_m,'source_origin_xy':list(source.origin_xy),
              'source_known_fraction':float(valid.mean()),
              'physical_stability':'unmeasured'}
        try:
            dem=np.asarray(robot_dem(source,(float(pos[0]),float(pos[1]),t),config),dtype=np.float32)
        except ValueError as exc:
            base.update({'dem_complete':False,'classification':'insufficient',
                         'interaction_onset_m':None,'corridor_known_fraction':None,
                         'centerline_known':False,'unknown_only_difference':True,
                         'profile_vector_m':[],'dem_error':repr(exc),
                         'footprint_known_fraction':None,'dem_valid_fraction':None,
                         'dem_sha256':None})
        else:
            height=dem[0];mask=dem[1]>.5
            interaction=classify_dem(height,mask,config.dem_resolution_m)
            x=(np.arange(20)-9.5)*.1;y=(np.arange(20)-9.5)*.1
            cols=np.where((x>=0)&(x<=.6))[0];center=np.where(np.abs(y)<=.15)[0]
            corridor=height[:,cols][mask[:,cols]]
            center_known=bool(np.all(np.any(mask[center[:,None],cols],axis=0)))
            profile=interaction.get('left_center_right_profiles_m',{})
            vector=[v for band in ('left','center','right') for v in profile.get(band,[])]
            foot=footprint_diagnostic(mask,proxy,radius_m=radius,theta_rad=t)
            base.update({'dem_complete':True,'classification':interaction['class'],
                         'interaction_onset_m':interaction['interaction_start_m'],
                         'interaction':interaction,
                         'corridor_known_fraction':interaction['corridor_valid_fraction'],
                         'centerline_known':center_known,
                         'unknown_only_difference':False if interaction['class']!='unknown' else True,
                         'profile_vector_m':vector,
                         'centerline_height_profile_m':profile.get('center',[]),
                         'left_center_right_profiles_m':profile,
                         'corridor_height_range_m':float(np.ptp(corridor)) if len(corridor) else None,
                         'corridor_outside_lateral_difference_m':outside_corridor_difference(height,mask),
                         'footprint_known_fraction':foot['full_collision_union']['known_fraction'],
                         'footprint_diagnostic':foot,
                         'dem_valid_fraction':float(mask.mean()),
                         'dem_sha256':hashlib.sha256(dem.tobytes()).hexdigest(),
                         'dem_shape':[2,20,20],
                         'dem':dem.tolist()})
        candidates.append(base)
    return candidates


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    env=None;started=time.monotonic()
    try:
        inputs=json.loads(args.inputs.read_text());parity=json.loads(args.parity_report.read_text())
        if parity['decision']!='parity_pass' or parity['pass_count']!=112 or parity['sim_step_counter_after']!=parity['sim_step_counter_before']:
            raise ValueError('112/112 zero-step DEM parity gate not passed')
        for label,path in inputs['frozen_source_paths'].items():
            if sha(path)!=inputs['frozen_source_sha256'][label]:
                raise ValueError('frozen source SHA drift: '+label)
        rows=sorted(inputs['pose_frames'],key=lambda x:x['env_id'])
        if [r['env_id'] for r in rows]!=list(range(112)):
            raise ValueError('frozen env identity drift')
        config=load_eat_config(Path(inputs['frozen_source_paths']['eat_config']))
        saved=yaml.load(Path(inputs['frozen_source_paths']['checkpoint_env']).read_text(),Loader=yaml.BaseLoader)
        toggles={name:saved[name].lower()=='true' for name in
                 ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,20261007,112)
        gen=cfg.scene.terrain.terrain_generator
        gen.class_type=RadialTerrainGenerator;gen.num_rows,gen.num_cols=16,7
        gen.curriculum=False;gen.difficulty_range=(.5,.5);gen.use_cache=False;gen.seed=20261007
        gen.sub_terrains={name:gen.sub_terrains[name] for name in TERRAIN_ORDER}
        cfg.scene.terrain.max_init_terrain_level=15
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped;stage=omni.usd.get_context().get_stage()
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/World/ground/terrain/mesh'))
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64)
        mesh_sha=hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()
        if mesh_sha!=parity['terrain_mesh_sha256']:
            raise ValueError('terrain mesh SHA mismatch from parity')
        proxy,proxy_counts=collision_proxy(stage,bare.scene['robot'])
        write(args.output/'collision-proxy.json',{'counts':proxy_counts,
             'point_count_by_category':{k:len(v) for k,v in proxy.items()},
             'proxy_sha256':hashlib.sha256(json.dumps(proxy,sort_keys=True).encode()).hexdigest(),
             'semantics':'initial USD collision world AABB corner proxy; diagnostic only'})
        before=int(bare._sim_step_counter)
        all_candidates=[]
        for radius in (.30,.40,.50,.60):
            items=scan_radius(bare,rows,radius,config,proxy)
            write(args.output/f'radius-{radius:.2f}.json',{'radius_m':radius,'candidates':items,
                  'terrain_mesh_sha256':mesh_sha,'physics_steps_after_scene_init':int(bare._sim_step_counter)-before})
            all_candidates.extend(items)
        after=int(bare._sim_step_counter)
        if after!=before or len(all_candidates)!=448 or len({x['pose_id'] for x in all_candidates})!=448:
            raise RuntimeError('physics advanced or 448-pose grid incomplete')
        write(args.output/'scan-summary.json',{'decision':'offline_scan_complete','candidate_count':448,
            'dem_complete':sum(x['dem_complete'] for x in all_candidates),
            'terrain_mesh_sha256':mesh_sha,'collision_proxy':proxy_counts,
            'physics_steps_after_scene_init':after-before,'elapsed_sec':time.monotonic()-started,
            'parity_report_sha256':sha(args.parity_report),'sim_source_sha256':sha(__file__)})
    except BaseException as exc:
        write(args.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'elapsed_sec':time.monotonic()-started})
        raise
    finally:
        if env is not None:env.close()
        simulation_app.close()


if __name__=='__main__':main()
