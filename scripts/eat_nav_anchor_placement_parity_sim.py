"""Recast the saved 112 DEM frames from frozen terrain without env.step()."""
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
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
import omni.usd
from pxr import UsdGeom
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from visualnav_transformer.eat_nav.anchor_placement_parity import compare_dem_frame, PARITY_TOLERANCE
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, terrain_seed
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator,make_collection_cfg


class RadialTerrainGenerator(EATTerrainGenerator):
    """Same frozen generator as the radial inventory; source parity is audited."""

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
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def recast(bare, rows, config):
    robot=bare.scene['robot'];scanner=bare.scene['height_scanner_eat']
    states=torch.tensor([r['root_state_w'][:7] for r in rows],device=bare.device,dtype=robot.data.root_state_w.dtype)
    ids=torch.tensor([r['env_id'] for r in rows],device=bare.device,dtype=torch.long)
    robot.write_root_pose_to_sim(states,env_ids=ids)
    bare.scene.write_data_to_sim()
    bare.sim.forward()
    scanner.reset(ids)
    scanner.update(0.0,force_recompute=True)
    result=[]
    origins=bare.scene.env_origins.detach().cpu().numpy()
    metadata=bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata
    for row in rows:
        i=row['env_id'];m=metadata[i]
        if (m['seed']!=row['terrain_seed'] or m['terrain_type']!=row['terrain'] or
            m['row']!=row['row'] or m['column']!=row['column'] or float(m['difficulty'])!=.5):
            raise ValueError('terrain mesh identity mismatch: '+row['pose_id'])
        pos=robot.data.root_pos_w[i].detach().cpu().numpy()
        _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
        hits=scanner.data.ray_hits_w[i,:,2].detach().cpu().numpy().reshape(20,20)
        valid=np.isfinite(hits)
        start=scanner._ray_starts_w[i,0,:2].detach().cpu().numpy()
        source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
                          origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
                          start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=0)
        transformed=np.asarray(robot_dem(source,(float(pos[0]),float(pos[1]),float(yaw.item())),config),dtype=np.float32)
        result.append({'pose_id':row['pose_id'],'env_id':i,'terrain':row['terrain'],
            'terrain_seed':m['seed'],'frame_id':config.reference_frame,
            'axis':'robot-relative-grid-xy','resolution_m':config.dem_resolution_m,
            'shape':[2,20,20],'physics_step':row['physics_step'],
            'source_origin_xy':list(source.origin_xy),'source_height':source.height.tolist(),
            'source_mask':valid.astype(int).tolist(),'transformed':transformed.tolist(),
            'yaw_rad':float(yaw.item()),'env_origin_xyz':origins[i].tolist()})
    return result


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    env=None;started=time.monotonic()
    try:
        inputs=json.loads(args.inputs.read_text())
        rows=inputs['pose_frames']
        if len(rows)!=112 or len({r['pose_id'] for r in rows})!=112 or len({r['env_id'] for r in rows})!=112:
            raise ValueError('112 unique pose/frame identities required')
        for label,path in inputs['frozen_source_paths'].items():
            if sha(path)!=inputs['frozen_source_sha256'][label]:
                raise ValueError('frozen source SHA drift: '+label)
        config=load_eat_config(Path(inputs['frozen_source_paths']['eat_config']))
        saved=yaml.load(Path(inputs['frozen_source_paths']['checkpoint_env']).read_text(),Loader=yaml.BaseLoader)
        toggles={name:saved[name].lower()=='true' for name in
                 ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,20261007,112)
        gen=cfg.scene.terrain.terrain_generator
        gen.class_type=RadialTerrainGenerator
        gen.num_rows,gen.num_cols=16,7
        gen.curriculum=False;gen.difficulty_range=(.5,.5);gen.use_cache=False;gen.seed=20261007
        gen.sub_terrains={name:gen.sub_terrains[name] for name in TERRAIN_ORDER}
        cfg.scene.terrain.max_init_terrain_level=15
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        mesh=UsdGeom.Mesh(omni.usd.get_context().get_stage().GetPrimAtPath('/World/ground/terrain/mesh'))
        if not mesh.GetPrim().IsValid():
            raise ValueError('frozen terrain mesh prim absent')
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64)
        mesh_sha=hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()
        before=int(bare._sim_step_counter)
        actual=recast(bare,rows,config)
        after=int(bare._sim_step_counter)
        if before!=after:
            raise RuntimeError('physics step counter advanced during parity recast')
        comparisons=[{'pose_id':e['pose_id'],**compare_dem_frame(e,a)} for e,a in zip(rows,actual)]
        write(args.output/'reconstructed-frames.json',{'frames':actual})
        write(args.output/'per-pose-parity.json',{'comparisons':comparisons})
        failed=[r for r in comparisons if not r['pass']]
        write(args.output/'parity-report.json',{'decision':'parity_pass' if not failed else 'dem_reconstruction_parity_fail_stop',
            'pose_count':len(rows),'pass_count':len(rows)-len(failed),'failed_count':len(failed),
            'first_failure':failed[0] if failed else None,'tolerance':PARITY_TOLERANCE,
            'terrain_mesh_sha256':mesh_sha,'terrain_mesh_vertex_count':len(points),'terrain_mesh_face_index_count':len(faces),
            'sim_step_counter_before':before,'sim_step_counter_after':after,
            'elapsed_sec':time.monotonic()-started})
        write(args.output/'complete.json',{'native_logic_complete':True,'parity_pass':not failed,
            'input_sha256':sha(args.inputs),'sim_source_sha256':sha(__file__),
            'physics_steps_after_scene_init':after-before})
    except BaseException as exc:
        write(args.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
                                         'elapsed_sec':time.monotonic()-started})
        raise
    finally:
        if env is not None:env.close()
        simulation_app.close()


if __name__=='__main__':main()
