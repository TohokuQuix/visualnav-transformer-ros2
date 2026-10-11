"""Candidate-free read-only G3 material diagnostic. Publishes no Path or command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--anchor-id',required=True)
parser.add_argument('--batch-index',type=int,required=True)
parser.add_argument('--gate1-probe',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from pxr import Usd, UsdPhysics, UsdShade
import omni.usd
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.p1_statistical_runtime import CachedTerrainGeometry
from visualnav_transformer.eat_nav.p1_path_crossover_v2 import fingerprint_material, read_usd_material_values
from visualnav_transformer.eat_nav.p1_g3_material_diagnostic import map_effective_materials


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


contract=json.loads((args.input/'material-diagnostic-contract.json').read_text())
if contract['schema']!='eatnav_p1_g3_material_diagnostic_v1' or \
        contract['Path_publications']!=0 or not args.gate1_probe or \
        args.anchor_id!='first-differential' or args.batch_index!=0 or \
        sha(__file__)!=contract['simulator_source_sha256']:
    raise ValueError('material diagnostic source/scope drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py')!= \
        contract['material_module_sha256']:
    raise ValueError('material diagnostic module SHA drift')
for name,expected in contract['input_sha256'].items():
    if sha(args.input/name)!=expected:raise ValueError('input SHA drift: '+name)
source=json.loads((args.input/'source-contract.json').read_text())
assignment=json.loads((args.input/'assignment.json').read_text())
config=json.loads((args.input/'statistical-config.json').read_text())
if sha(config['rl_checkpoint'])!=contract['rl_checkpoint_sha256']:
    raise ValueError('frozen RL checkpoint SHA drift')
anchor=next(row for row in source['anchors'] if row['anchor_id']==args.anchor_id)
batch_seed=assignment['batch_seeds'][args.anchor_id][args.batch_index]


class FourSameTerrain(EATTerrainGenerator):
    last_fingerprints=None

    def _generate_random_terrains(self):
        if (self.cfg.num_rows,self.cfg.num_cols)!=(1,4):
            raise ValueError('frozen four-tile terrain layout drift')
        terrain_cfg=self.cfg.sub_terrains[anchor['terrain']]
        old_seed,old_numpy=self.cfg.seed,np.random.get_state()
        try:
            self.cfg.seed=int(anchor['terrain_seed'])
            np.random.seed(int(anchor['terrain_seed']))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(int(anchor['terrain_seed']))
                torch.cuda.manual_seed_all(int(anchor['terrain_seed']))
                cached=CachedTerrainGeometry(lambda: TerrainGenerator._get_terrain_mesh(self,.5,terrain_cfg))
                first_mesh,first_origin=cached.copy_for_slot()
            fingerprints=[]
            for col in range(4):
                mesh,origin=(first_mesh,first_origin) if col==0 else cached.copy_for_slot()
                fingerprints.append({'env_id':col,'terrain_type':anchor['terrain'],
                                     'terrain_seed':int(anchor['terrain_seed']),
                                     'canonical_mesh_sha256':cached.canonical_sha256})
                self.tile_metadata.append({'tile_index':col,'row':0,'column':col,
                                           'seed':int(anchor['terrain_seed']),
                                           'terrain_type':anchor['terrain'],'difficulty':.5})
                self._tile_index+=1
                self._add_sub_terrain(mesh,origin,0,col,terrain_cfg)
            FourSameTerrain.last_fingerprints=fingerprints
        finally:
            self.cfg.seed=old_seed
            np.random.set_state(old_numpy)


def material_row(prim, env_id=None):
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    row={'env_id':env_id,'path':str(prim.GetPath()),'shape_type':prim.GetTypeName(),
         'binding_status':'bound' if material and material.GetPrim().IsValid() else 'unbound',
         'material_prim':None,'composed_attributes':None,'attribute_error':None,
         'collision_api':True}
    if row['binding_status']=='bound':
        row['material_prim']=str(material.GetPrim().GetPath())
        try:
            row['composed_attributes']=read_usd_material_values(material.GetPrim())
            row['composed_canonical_sha256']=fingerprint_material(row['composed_attributes'])
        except Exception as exc:
            row['attribute_error']=type(exc).__name__+':'+str(exc)
    return row


def collect_materials(stage, robot, cfg):
    collision=[prim for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies())
               if prim.HasAPI(UsdPhysics.CollisionAPI)]
    terrain=[material_row(prim) for prim in collision
             if str(prim.GetPath()).startswith('/World/ground')]
    robot_rows=[]
    for prim in collision:
        match=re.search(r'/env_([0-3])/',str(prim.GetPath()))
        if match and '/World/ground/' not in str(prim.GetPath()):
            robot_rows.append(material_row(prim,int(match.group(1))))
    view=robot.root_physx_view
    link_paths=[[str(path) for path in env] for env in view.link_paths]
    if len(link_paths)!=4:
        raise ValueError('PhysX link_paths not four-env')
    link_rows=[]
    for env_paths in link_paths:
        rows=[]
        for path in env_paths:
            count=int(robot._physics_sim_view.create_rigid_body_view(path).max_shapes)
            rows.append({'path':path,'physx_shape_count':count})
        link_rows.append(rows)
    tensor=view.get_material_properties().detach().cpu().tolist()
    startup=cfg.events.physics_material.params
    ranges=[list(startup[name]) for name in
            ('static_friction_range','dynamic_friction_range','restitution_range')]
    declared_terrain_sha=fingerprint_material(cfg.scene.terrain.physics_material)
    terrain_pass=bool(terrain) and all(row['binding_status']=='bound' and
        row['attribute_error'] is None and
        row.get('composed_canonical_sha256')==declared_terrain_sha for row in terrain)
    return {'schema':'eatnav_p1_g3_material_diagnostic_v1',
            'terrain':{'collision_prim_count':len(terrain),'rows':terrain,
                       'declared_canonical_sha256':declared_terrain_sha,
                       'live_binding_and_attributes_pass':terrain_pass},
            'robot':{'collision_prims':robot_rows,'link_paths_and_shape_counts':link_rows,
                     'root_physx_view_max_shapes':int(view.max_shapes),
                     'startup_randomization':{'static_friction_range':ranges[0],
                        'dynamic_friction_range':ranges[1],
                        'restitution_range':ranges[2],
                        'num_buckets':int(startup['num_buckets']),
                        'make_consistent':bool(startup.get('make_consistent',False))},
                     'effective':map_effective_materials(link_rows,robot_rows,tensor,ranges)},
            'Path_publications':0}


def main():
    parent=Path('/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1')
    if not args.output.is_relative_to(parent):
        raise ValueError('material diagnostic must use persistent root')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    durable(args.output/'status.json',{'stage':'startup','Path_publications':0})
    env=None
    try:
        eat_cfg=load_eat_config(args.input/'eat_nav.yaml')
        saved=yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),Loader=yaml.BaseLoader)
        toggles={name:saved[name].lower()=='true' for name in
                 ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,4)
        cfg.episode_length_s=40.0
        gen=cfg.scene.terrain.terrain_generator
        gen.class_type=FourSameTerrain
        gen.num_rows,gen.num_cols=1,4
        gen.curriculum=False;gen.difficulty_range=(.5,.5);gen.use_cache=False;gen.seed=batch_seed
        gen.sub_terrains={anchor['terrain']:gen.sub_terrains[anchor['terrain']]}
        cfg.scene.terrain.max_init_terrain_level=0
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        if abs(float(bare.step_dt)-.05)>1e-9 or int(bare.max_episode_length)!=800:
            raise ValueError('frozen physics/control/episode contract drift')
        env.reset()
        if len(FourSameTerrain.last_fingerprints or [])!=4:
            raise ValueError('terrain geometry provenance incomplete')
        evidence=collect_materials(omni.usd.get_context().get_stage(),bare.scene['robot'],cfg)
        evidence['terrain']['fingerprints']=FourSameTerrain.last_fingerprints
        evidence['checkpoint_sha256']=sha(config['rl_checkpoint'])
        evidence['source_sha256']=sha(__file__)
        evidence['input_contract_sha256']=sha(args.input/'material-diagnostic-contract.json')
        durable(args.output/'material-diagnostic.json',evidence)
        durable(args.output/'material-diagnostic-complete.json',{
            'stage':'candidate_free_material_measurement_complete',
            'Path_publications':0,'physics_control_steps':0,
            'material_diagnostic_sha256':sha(args.output/'material-diagnostic.json'),
            'elapsed_s':time.monotonic()-started})
    except BaseException as exc:
        durable(args.output/'diagnostic-error.json',{'type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'Path_publications':0})
        raise
    finally:
        if env is not None:env.close()


try:
    main()
finally:
    simulation_app.close()
