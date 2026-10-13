"""ROS-free G3 crossover: four independent C++ follower cores in Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--anchor-id', required=True)
parser.add_argument('--batch-index', type=int, choices=(0, 1, 2), required=True)
parser.add_argument('--execution-index',type=int,choices=(0,1,2),required=True)
parser.add_argument('--gate1-probe',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils.math import euler_xyz_from_quat
from isaacsim.core.simulation_manager import SimulationManager
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.p1_dem_capture import (persist_dem_evidence,
    compare_direct_frames, _durable_json, _durable_npz)
from visualnav_transformer.eat_nav.p1_g3_shared_dem import (
    build_shared_history, validate_shared_history, persist_shared_history, diagnose_branch_inputs)
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, clone_internal_branch_state
from visualnav_transformer.eat_nav.phase_c_sim import ObservationCapture, capture_actor_critic
from visualnav_transformer.eat_nav.p1_statistical_pilot import (
    local_pose_from_root, path_in_map,
)
from visualnav_transformer.eat_nav.p1_follower_core import compile_core
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import (
    RosFreeG3Control, RosFreeSameEpisodeGate, reconcile_command_observation,
)
from visualnav_transformer.eat_nav.p1_teacher_path_contract import (
    validate_simulator_paths, mount_points_from_proc,
)
from rsl_rl.runners import OnPolicyRunner
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.p1_statistical_runtime import CachedTerrainGeometry, require_prepath_gate, jsonable_values
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from visualnav_transformer.eat_nav.p1_paired_noise import PairedPolicyNoise, key_seed
from visualnav_transformer.eat_nav.p1_paired_integrity import assess_step
from visualnav_transformer.eat_nav.p1_path_crossover_v2 import fingerprint_material, read_usd_material_values
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import SCHEMA, assess_material_gate, prepare_shared_output_dir
from visualnav_transformer.eat_nav.p1_longhorizon_protocol import DurableTrace, G3Exposure, STEP_NS
from visualnav_transformer.eat_nav.p1_teacher_capture import RewardTermCapture, joint_body_maps
from visualnav_transformer.eat_nav.p1_teacher_runtime import PhysicsSubstepTap, validate_teacher_step
from visualnav_transformer.eat_nav.p1_teacher_chunks import ChunkWriter, verify_chunks

ROSFREE_SCHEMA = 'eatnav_p1_g3_rosfree_crossover_v1'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def values(tensor):
    return jsonable_values(tensor)


manifest=json.loads((args.input/'assignment.json').read_text())
freeze=json.loads((args.input/'freeze.json').read_text())
config_json=json.loads((args.input/'statistical-config.json').read_text())
contract=json.loads((args.input/'source-contract.json').read_text())
fixture=json.loads((args.input/'anchor-fixtures.json').read_text())
if sha(args.input/'assignment.json') != freeze['assignment_sha256']:
    raise ValueError('assignment SHA drift')
if sha(args.input/'statistical-config.json') != freeze['config_sha256']:
    raise ValueError('config SHA drift')
if sha(args.input/'source-contract.json') != freeze['source_contract_sha256']:
    raise ValueError('source contract SHA drift')
g3_contract=json.loads((args.input/'g3-contract.json').read_text())
g3_assignment=json.loads((args.input/'g3-assignment.json').read_text())
if sha(args.input/'g3-assignment.json') != g3_contract['assignment_sha256']:
    raise ValueError('G3 assignment SHA drift')
if g3_contract.get('schema') != ROSFREE_SCHEMA or g3_contract.get('material_schema') != SCHEMA:
    raise ValueError('ROS-free G3 protocol/material version drift')
if g3_contract.get('teacher_capture_schema') != 'eatnav_p1_teacher_capture_v1':
    raise ValueError('teacher capture protocol missing')
if g3_contract.get('teacher_runtime_authorized') is not True:
    raise ValueError('teacher capture runtime not authorized')
for name, key in (('p1_teacher_capture.py','teacher_capture_source_sha256'),
                  ('p1_teacher_runtime.py','teacher_runtime_source_sha256'),
                  ('p1_teacher_chunks.py','teacher_chunks_source_sha256')):
    if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav'/name) != g3_contract[key]:
        raise ValueError('teacher source SHA drift: '+name)
if args.gate1_probe or g3_contract.get('execution_mode') != 'path_crossover_four_env_rosfree':
    raise ValueError('ROS-free G3 diagnostic mode/manifest mismatch')
if sha(__file__) != g3_contract['g3_simulator_source_sha256']:
    raise ValueError('G3 simulator source SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py') != g3_contract['shared_dem_module_sha256']:
    raise ValueError('G3 simulator source SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_direct_repeat.py') != g3_contract['direct_repeat_module_sha256']:
    raise ValueError('G3 direct-repeat module SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_crossover_runtime.py') != g3_contract['crossover_runtime_module_sha256']:
    raise ValueError('G3 crossover runtime module SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_material_gate_v2.py') != g3_contract['material_gate_module_sha256']:
    raise ValueError('G3 material gate v2 source SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py') != g3_contract['material_diagnostic_module_sha256']:
    raise ValueError('G3 material coverage source SHA drift')
if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav/p1_path_crossover_v2.py') != g3_contract['path_crossover_module_sha256']:
    raise ValueError('G3 crossover material module SHA drift')
for name,key in (('p1_follower_core.cpp','cpp_core_source_sha256'),
                 ('p1_follower_core.py','python_core_source_sha256'),
                 ('p1_g3_rosfree_control.py','rosfree_control_source_sha256'),
                 ('p1_teacher_path_contract.py','rosfree_path_contract_source_sha256')):
    if sha(Path(__file__).resolve().parent/'src/visualnav_transformer/eat_nav'/name) != g3_contract[key]:
        raise ValueError('ROS-free core source SHA drift: '+name)
if (g3_contract['unreached_control_step_cap'] != 600 or g3_contract['post_arrival_hold_steps'] != 40
        or g3_contract['max_control_steps'] != 640 or g3_contract['episode_length_s'] != 40.0):
    raise ValueError('G3 contract drift')
anchor=next(a for a in contract['anchors'] if a['anchor_id']==args.anchor_id)
batch_seed=manifest['batch_seeds'][args.anchor_id][args.batch_index]
matched=[row for row in g3_assignment['processes'] if row['anchor_id']==args.anchor_id and row['batch_index']==args.batch_index and row['execution_index']==args.execution_index]
if len(matched)!=1 or matched[0]['batch_seed']!=batch_seed:
    raise ValueError('direct repeat process assignment mismatch')
assigned=sorted(matched[0]['trials'],key=lambda row:row['slot'])
conditions=[row['condition'] for row in sorted(assigned,key=lambda row:row['slot'])]
if len(assigned)!=4 or {row['slot'] for row in assigned}!=set(range(4)):
    raise ValueError('G3 assignment does not cover four slots')
if config_json['policy_action_mode']!='act_inference' or not config_json['paired_policy_noise']:
    raise ValueError('inference contract drift')
if config_json['primary_control_steps']!=40 or config_json['diagnostic_control_steps']!=120:
    raise ValueError('P1 primary/diagnostic window contract')
if conditions != ['A']*4:
    raise ValueError('G3 requires unmodified follower commands')
if [row['candidate_slot'] for row in assigned] != [f'crossover_{i}' for i in range(4)]:
    raise ValueError('crossover candidate slots invalid')
active_assigned=assigned
if [row['path_role'] for row in assigned] != [
        next(p for p in g3_contract['processes'] if p['anchor_id']==args.anchor_id and
             p['batch_index']==args.batch_index and p['execution_index']==args.execution_index)['trials'][i]['path_role']
        for i in range(4)]:
    raise ValueError('crossover Path role drift')


def paths_for_slots(rows, path_lookup, poses):
    if len(rows)!=4 or len(poses)!=4 or [row['slot'] for row in rows]!=list(range(4)):
        raise ValueError('crossover branch assignment required')
    result=[]
    for row,pose in zip(rows,poses):
        frozen=path_lookup[row['path_sha256']]
        result.append(path_in_map(frozen['waypoints'],pose))
    return result


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


def read_live_materials(stage, robot, cfg, terrain_declared_sha, frozen_startup):
    """Record all composed USD bindings and startup-effective PhysX shape values."""
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
    startup=cfg.events.physics_material.params
    resolved={name:list(startup[name]) for name in
              ('static_friction_range','dynamic_friction_range','restitution_range')}
    resolved.update({'num_buckets':int(startup['num_buckets']),
                     'make_consistent':bool(startup.get('make_consistent',False))})
    result=assess_material_gate(terrain,robot_rows,link_rows,
           view.get_material_properties().detach().cpu().tolist(),
           resolved,frozen_startup,terrain_declared_sha)
    result['robot_link_paths_and_shape_counts']=link_rows
    result['root_physx_view_max_shapes']=int(view.max_shapes)
    return result


def await_host_prepath_approval(output, snapshot_sha, started):
    """Keep physics paused and Path unpublished while host compares periods."""
    deadline=time.monotonic()+75.
    while time.monotonic()<deadline:
        if (output/'prepath-rejection.json').exists():
            raise ValueError('cross-period pre-Path gate rejected by host')
        marker=output/'prepath-approval.json'
        if marker.exists():
            approval=json.loads(marker.read_text())
            if approval.get('snapshot_sha256')!=snapshot_sha or approval.get('pass') is not True:
                raise ValueError('cross-period pre-Path approval invalid')
            return approval
        atomic(output/'heartbeat.json',{'stage':'awaiting_cross_period_gate',
               'elapsed_s':time.monotonic()-started,'Path_activations':0})
        time.sleep(.2)
    raise TimeoutError('cross-period pre-Path host approval timeout')


class SevenSameTerrain(EATTerrainGenerator):
    last_fingerprints=None

    def _generate_random_terrains(self):
        if (self.cfg.num_rows,self.cfg.num_cols)!=(1,4):
            raise ValueError('seven-slot tile layout drift')
        terrain_name=anchor['terrain']
        terrain_cfg=self.cfg.sub_terrains[terrain_name]
        old_seed,old_numpy=self.cfg.seed,np.random.get_state()
        try:
            self.cfg.seed=int(anchor['terrain_seed'])
            np.random.seed(int(anchor['terrain_seed']))
            # The boxes generator draws from CUDA PyTorch. Isolate its stream
            # and restore the global CPU/CUDA state before any policy or
            # observation randomization. Generate only once per StateGroup.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(int(anchor['terrain_seed']))
                torch.cuda.manual_seed_all(int(anchor['terrain_seed']))
                cached=CachedTerrainGeometry(lambda: TerrainGenerator._get_terrain_mesh(self,.5,terrain_cfg))
                first_mesh,first_origin=cached.copy_for_slot()
            fingerprints=[]
            for col in range(4):
                mesh,origin=(first_mesh,first_origin) if col==0 else cached.copy_for_slot()
                fingerprints.append({'slot':col,'terrain_type':terrain_name,
                                     'terrain_seed':int(anchor['terrain_seed']),
                                     'canonical_mesh_sha256':cached.canonical_sha256})
                self.tile_metadata.append({'tile_index':col,'row':0,'column':col,
                                           'seed':int(anchor['terrain_seed']),
                                           'terrain_type':terrain_name,'difficulty':.5})
                self._tile_index += 1
                self._add_sub_terrain(mesh,origin,0,col,terrain_cfg)
            SevenSameTerrain.last_fingerprints=fingerprints
        finally:
            self.cfg.seed=old_seed
            np.random.set_state(old_numpy)


def row_value(tensor,index):
    if not torch.is_tensor(tensor):
        return None
    if tensor.ndim==0:
        return tensor.item()
    if tensor.shape[0]==4:
        return values(tensor[index])
    return None


def ring_snapshot(ring,index):
    if ring is None:
        return None
    data=getattr(ring,'_buffer',None)
    return {'num_pushes':row_value(getattr(ring,'_num_pushes',None),index),
            'pointer':getattr(ring,'_pointer',None),
            'stored':values(data[:,index]) if torch.is_tensor(data) else None}


def snapshot(bare,index,tile_origin,mesh_sha,collision_sha,material_sha):
    robot=bare.scene['robot']; actions=bare.action_manager
    flipper=actions._terms['flipper_pos']
    delay={}
    for name,buffer in bare._action_latency_buffers.items():
        delay[name]={'lag':row_value(buffer._time_lags,index),
                     'ring':ring_snapshot(buffer._circular_buffer,index)}
    history={}
    for group,terms in getattr(bare.observation_manager,'_group_obs_term_history_buffer',{}).items():
        history[group]={name:ring_snapshot(ring,index) for name,ring in terms.items()}
    targets={name:row_value(getattr(robot.data,name,None),index) for name in
             ('joint_pos_target','joint_vel_target','joint_effort_target')}
    return {'slot':index,'tile_origin_xy':values(tile_origin[:2]),
            'root_state_w':values(robot.data.root_state_w[index]),
            'joint_position':values(robot.data.joint_pos[index]),
            'joint_velocity':values(robot.data.joint_vel[index]),
            'flipper_latch':row_value(bare._onix_flipper_fold_mode,index),
            'flipper_previous_targets':row_value(flipper._previous_targets,index),
            'flipper_preclip_targets':row_value(flipper._pre_clip_processed_actions,index),
            'action_current':row_value(actions._action,index),
            'action_previous':row_value(actions._prev_action,index),
            'actuator_targets':targets,
            'latency_substeps':row_value(bare._latency_substeps,index),
            'latency_buffers':delay,'observation_history':history,
            'canonical_mesh_sha256':mesh_sha,'collision_sha256':collision_sha,
            'material_sha256':material_sha}


def composed_collision_fingerprint():
    stage=omni.usd.get_context().get_stage()
    rows=[];ground=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            path=str(prim.GetPath())
            rows.append(path)
            if path.startswith('/World/ground'):
                mesh=UsdGeom.Mesh(prim)
                points=mesh.GetPointsAttr().Get() if mesh else None
                faces=mesh.GetFaceVertexIndicesAttr().Get() if mesh else None
                ground.append({'path':path,'points_sha256':hashlib.sha256(str(points).encode()).hexdigest()
                               if points is not None else None,
                               'face_indices_sha256':hashlib.sha256(str(faces).encode()).hexdigest()
                               if faces is not None else None})
    if not rows or not ground:
        raise ValueError('composed terrain CollisionAPI prim missing')
    return {'count':len(rows),'ground_prims':ground,
            'sha256':hashlib.sha256(json.dumps(ground,sort_keys=True).encode()).hexdigest()}


def dem_frame(bare,index,config,settling_step):
    robot,scanner=bare.scene['robot'],bare.scene['height_scanner_eat']
    pos=robot.data.root_pos_w[index].detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[index:index+1])
    ray_hits_w=scanner.data.ray_hits_w[index].detach().cpu().numpy().copy()
    hits=ray_hits_w[:,2].reshape(20,20)
    valid=np.isfinite(hits)
    ray_starts_w=scanner._ray_starts_w[index].detach().cpu().numpy().copy()
    start=ray_starts_w[0,:2]
    source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
        origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
        start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=(settling_step+1)*STEP_NS)
    transformed=np.asarray(robot_dem(source,(float(pos[0]),float(pos[1]),float(yaw.item())),config),dtype=np.float32)
    if transformed.shape!=(2,20,20) or not np.isfinite(transformed).all():
        raise ValueError('pre-Path DEM shape/nonfinite')
    return {'identity':f'settling_step_{settling_step}','timestamp_ns':(settling_step+1)*STEP_NS,
        'shape':[2,20,20],'frame':'robot_base','resolution_m':float(config.dem_resolution_m),
        'raw_sha256':hashlib.sha256(hits.tobytes()).hexdigest(),
        'transformed':transformed,'ray_hits_w':ray_hits_w,'ray_starts_w':ray_starts_w,
        'source_height':hits.copy(),'source_valid':valid.copy(),
        'env_origin_w':bare.scene.env_origins[index].detach().cpu().numpy().copy(),
        'robot_pose_w':robot.data.root_state_w[index].detach().cpu().numpy().copy(),
        'root_state_w':robot.data.root_state_w[index].detach().cpu().numpy().copy(),
        'joint_pos':robot.data.joint_pos[index].detach().cpu().numpy().copy(),
        'joint_vel':robot.data.joint_vel[index].detach().cpu().numpy().copy(),
        'sensor_timestamp_s':float(scanner._timestamp[index].item()),
        'sensor_update_timestamp_s':float(scanner._timestamp_last_update[index].item()),
        'sensor_is_outdated':bool(scanner._is_outdated[index].item()),
        'sensor_update_period_s':float(scanner.cfg.update_period)}


def direct_dem_gate(frames):
    if len(frames)!=4 or any(len(item)!=4 for item in frames):
        raise ValueError('pre-Path four-frame DEM history incomplete')
    diagnostics=[]
    for first,second in zip(frames[0],frames[1]):
        for key in ('identity','timestamp_ns','shape','frame','resolution_m'):
            if first[key]!=second[key]:raise ValueError('direct DEM metadata mismatch:'+key)
        a,b=np.asarray(first['transformed']),np.asarray(second['transformed'])
        mask_a,mask_b=a[1]>.5,b[1]>.5
        union=np.logical_or(mask_a,mask_b).sum()
        iou=float(np.logical_and(mask_a,mask_b).sum()/union) if union else 1.0
        rms=float(np.sqrt(np.mean((a[0]-b[0])**2)))
        maximum=float(np.max(np.abs(a[0]-b[0])))
        diagnostics.append({'height_rms_m':rms,'height_max_abs_m':maximum,'mask_iou':iou,
            'raw_sha_equal':first['raw_sha256']==second['raw_sha256']})
        if rms>1e-5 or maximum>1e-4 or iou<1.0:
            raise ValueError('direct DEM numeric gate')
    return diagnostics


def main():
    path_contract=validate_simulator_paths(
        g3_contract,args.input,args.output,Path(__file__),
        mount_points_from_proc(Path('/proc/self/mountinfo').read_text()))
    prepare_shared_output_dir(args.output)
    atomic(args.output/'path-contract.json',path_contract)
    started=time.monotonic()
    pid=os.getpid()
    tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    atomic(args.output/'owned-process.json',{'pid':pid,'start_tick':tick,'Path_activations':0})
    atomic(args.output/'status.json',{'stage':'startup','Path_activations':0})
    atomic(args.output/'process-provenance.json',{
        'protocol_version':ROSFREE_SCHEMA,
        'anchor_id':args.anchor_id,'batch_index':args.batch_index,
        'execution_index':args.execution_index,'assignment_period':matched[0]['assignment_period'],
        'batch_seed':batch_seed,
        'terrain_seed':int(anchor['terrain_seed']),
        'source_sha256':sha(__file__),
        'g3_assignment_sha256':sha(args.input/'g3-assignment.json'),
        'statistical_assignment_sha256':sha(args.input/'assignment.json'),
        'config_sha256':sha(args.input/'statistical-config.json'),
        'source_contract_sha256':sha(args.input/'source-contract.json'),
        'fixture_sha256':sha(args.input/'anchor-fixtures.json'),
        'checkpoint_sha256':config_json['rl_checkpoint_sha256'],
        'path_sha256_by_env':[row['path_sha256'] for row in active_assigned],
        'env_ids':[0,1,2,3]})
    eat_cfg=load_eat_config(args.input/'eat_nav.yaml')
    saved=yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    toggles={name:saved[name].lower()=='true' for name in
             ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,4)
    # Diagnostic-only timeout. Physics, physical terminations, policy and follower are unchanged.
    cfg.episode_length_s=g3_contract['episode_length_s']
    gen=cfg.scene.terrain.terrain_generator
    gen.class_type=SevenSameTerrain
    gen.num_rows,gen.num_cols=1,4
    gen.curriculum=False
    gen.difficulty_range=(.5,.5)
    gen.use_cache=False
    gen.seed=batch_seed
    gen.sub_terrains={anchor['terrain']:gen.sub_terrains[anchor['terrain']]}
    cfg.scene.terrain.max_init_terrain_level=0
    env=None;override=None;capture=None;pairer=None;control=None
    try:
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        if abs(float(bare.step_dt)-0.05)>1e-9 or int(bare.max_episode_length)!=800:
            raise ValueError('G3 50ms/40s environment contract')
        terrain=bare.scene.terrain
        fingerprints=SevenSameTerrain.last_fingerprints
        if not fingerprints or len(fingerprints)!=4 or len({r['canonical_mesh_sha256'] for r in fingerprints})!=1:
            raise ValueError('terrain geometry not reused')
        collision=composed_collision_fingerprint()
        material=fingerprint_material(cfg.scene.terrain.physics_material)
        if material!=g3_contract['declared_terrain_material_sha256']:
            raise ValueError('frozen declared terrain material SHA drift')
        origins=bare.scene.env_origins.detach().cpu().numpy()
        atomic(args.output/'terrain.json',{'fingerprints':fingerprints,
                'world_tile_origins':origins.tolist(),'collision':collision,
                'material_sha256':material,'scene_num_envs':4,
                'sim_dt':float(bare.physics_dt),'control_dt':float(bare.step_dt)})
        env.reset()
        robot=bare.scene['robot']
        live_materials=read_live_materials(omni.usd.get_context().get_stage(),robot,cfg,material,
                                          g3_contract['live_material']['resolved_startup_randomization'])
        _durable_json(args.output/'live-material.json',live_materials)
        if not live_materials['pass']:
            raise ValueError('live material gate v2 failed: '+repr(live_materials['failures']))
        contact_sensor=bare.scene['contact_forces']
        contact_body_names=list(contact_sensor.body_names)
        if len(contact_body_names)!=contact_sensor.data.net_forces_w.shape[1]:
            raise ValueError('contact body identity/count mismatch')
        atomic(args.output/'contact-body-names.json',{'body_names':contact_body_names,
            'count':len(contact_body_names),'force_frame':'world'})
        joint_names=list(robot.data.joint_names)
        actuator_names={}
        for term_name, term in bare.action_manager._terms.items():
            cfg_term=term.cfg
            names=list(getattr(cfg_term,'broadcast_joint_names',()) or ())
            names += list(getattr(cfg_term,'joint_names',()) or ())
            actuator_names[term_name]=list(dict.fromkeys(name for name in names if name in joint_names))
        mapping=joint_body_maps(joint_names,contact_body_names,actuator_names)
        atomic(args.output/'teacher-index-mapping.json',{
            'schema':'eatnav_p1_teacher_capture_v1',**mapping,
            'torque_meaning':'simulator applied joint effort, not physical load-cell torque',
            'torque_phase':'pre_physics','contact_frame':'world',
            'contact_phase':'pre_physics',
            'contact_update_period_s':float(contact_sensor.cfg.update_period),
            'contact_history_length':int(contact_sensor.cfg.history_length),
            'physics_dt_s':float(bare.physics_dt),'control_dt_s':float(bare.step_dt)})
        if (len(joint_names)!=52 or len(contact_body_names)!=60 or
                abs(float(bare.physics_dt)-0.005)>1e-9 or
                abs(float(bare.step_dt)-0.05)>1e-9 or
                not callable(getattr(contact_sensor.contact_physx_view,'get_net_contact_forces',None))):
            raise ValueError('teacher live tensor/contact API contract')
        reward_capture=RewardTermCapture.from_manager(bare.reward_manager,float(bare.step_dt))
        atomic(args.output/'teacher-reward-manifest.json',reward_capture.manifest())
        state=fixture[args.anchor_id]['prephysics_state']
        if state['joint_names']!=list(robot.data.joint_names):
            raise ValueError('joint order changed')
        ids=torch.arange(4,device=bare.device)
        root=robot.data.root_state_w[ids].clone()
        radius_match=re.search(r'-r([0-9]+\.[0-9]+)-',anchor['pose_id'])
        if radius_match is None:
            raise ValueError('frozen source pose radius missing')
        radius=float(radius_match.group(1))
        requested=anchor['requested_xy_yaw']; source_pose=state['root_state_w']
        local_xy=[radius*math.cos(requested[2]),radius*math.sin(requested[2])]
        for i in range(4):
            root[i,0]=float(origins[i,0])+local_xy[0]
            root[i,1]=float(origins[i,1])+local_xy[1]
            root[i,2]=float(source_pose[2])
            root[i,3:7]=torch.as_tensor(source_pose[3:7],device=bare.device,dtype=root.dtype)
            root[i,7:13]=torch.as_tensor(source_pose[7:13],device=bare.device,dtype=root.dtype)
        robot.write_root_state_to_sim(root,env_ids=ids)
        joints=torch.as_tensor([state['joint_position']]*4,device=bare.device,dtype=robot.data.joint_pos.dtype)
        speeds=torch.as_tensor([state['joint_velocity']]*4,device=bare.device,dtype=robot.data.joint_vel.dtype)
        robot.write_joint_state_to_sim(joints,speeds,env_ids=ids)
        bare._onix_flipper_fold_mode[ids]=float(state['flipper_fold_mode_latch'])
        term=bare.action_manager._terms['flipper_pos']
        desired=joints[:,term._joint_ids]
        term._previous_targets.copy_(desired)
        term._pre_clip_processed_actions.copy_(desired)
        bare.sim.forward()
        zero=torch.zeros((4,6),device=bare.device)
        override=CommandOverride(bare.command_manager,4)
        step_index=0
        def provider():
            current=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            current[:4]=torch.as_tensor([fixture[args.anchor_id]['settling_commands'][step_index]]*4,
                                         device=bare.device,dtype=current.dtype)
            return current
        override.provider=provider
        override.set(provider())
        dem_history=[[] for _ in range(4)]
        for step_index in range(20):
            obs,_,done,_=env.step(zero)
            if bool(torch.any(done)):
                raise ValueError('termination during candidate-free settling')
            if step_index in (4,9,14,19):
                for i in range(4):dem_history[i].append(dem_frame(bare,i,eat_cfg,step_index))
            atomic(args.output/'heartbeat.json',{'completed_control_steps':step_index+1,
                                                  'elapsed_s':time.monotonic()-started,
                                                  'Path_activations':0})
        clone=clone_internal_branch_state(bare,0,range(1,4))
        # The helper lists every unreplicated field; do not silently call it equal.
        atomic(args.output/'clone-report.json',clone)
        records=[snapshot(bare,i,origins[i],fingerprints[i]['canonical_mesh_sha256'],
                          collision['sha256'],material) for i in range(4)]
        atomic(args.output/'initial-state.json',records)
        physical_gate=require_prepath_gate(records,config_json['initialization_tolerances'])
        atomic(args.output/'prepath-physical-state-gate.json',physical_gate)
        # Native histories are measured, branch-specific raycaster evidence, captured before clone.
        persist_dem_evidence(args.output,dem_history)
        native_frames=[]
        for other in range(1,4):
            for frame_index in range(4):
                metrics,_=compare_direct_frames(dem_history[0][frame_index],
                                                 dem_history[other][frame_index])
                native_frames.append({'reference_env':0,'other_env':other,
                                      'frame_index':frame_index,**metrics})
        native_pass=all(row['height_rms_m']<=1e-5 and
                        row['height_max_abs_m']<=1e-4 and row['mask_iou']>=1.0
                        for row in native_frames)
        native_diagnostic={'meaning':'measured_native_sensor_history_equality',
                           'pass':native_pass,'frames':native_frames,
                           'pairs':'env_0_vs_env_1_2_3',
                           'not_used_as_shared_input':True}
        shared=build_shared_history(dem_history)
        shared_persistence=persist_shared_history(args.output,shared,
                                                  native_diagnostic=native_diagnostic)
        shared_gate=validate_shared_history(shared,direct_dem_gate)
        _durable_json(args.output/'prepath-shared-dem-gate.json',shared_gate)
        shared_ledger=json.loads((args.output/'prepath-shared-dem-sha-ledger.json').read_text())
        experimental_inputs=[{'role':'shared_experimental_history','source_env':0,
            'frames':[{'array_file':f'prepath-shared-dem-env-{env}-frame-{frame}.npy',
                       'array_sha256':shared_ledger[f'prepath-shared-dem-env-{env}-frame-{frame}.npy'],
                       'source_frame_sha256':shared['branches'][env][frame]['source_frame_sha256']}
                      for frame in range(4)]} for env in range(4)]
        # This version intentionally evaluates the unchanged numeric gate on shared input.
        # Native DEM inequality remains visible and is never called a sensor pass.
        atomic(args.output/'prepath-gate.json',{
            'pass':physical_gate['pass'] and shared_gate['pass'],
            'physical_state':physical_gate,'shared_experimental_input':shared_gate,
            'native_sensor_history':{'pass':native_pass,'diagnostic_only':True},
            'Path_activations_before_gate':0})
        if tuple(obs['policy'].shape)!=(4,22) or tuple(obs['critic'].shape)!=(4,267):
            raise ValueError('Actor/Critic observation shape drift')
        runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
        checkpoint=Path(config_json['rl_checkpoint'])
        if sha(checkpoint)!=config_json['rl_checkpoint_sha256']:
            raise ValueError('RL checkpoint SHA drift')
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
        runner.alg.policy.load_state_dict(weights)
        policy=runner.alg.policy
        pairer=PairedPolicyNoise(bare.observation_manager,args.anchor_id,batch_seed)
        capture=ObservationCapture(bare.observation_manager)
        pairer.set_step(0)
        capture.begin_step()
        raw=bare.observation_manager.compute()
        latency_prev=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
        latency_reset=bare._latency_policy_obs_reset_mask.detach().clone()
        obs=bare._apply_policy_observation_latency(raw)
        observed=capture.finish_step(obs)
        noise_current=pairer.verify_step()
        contact_data=contact_sensor.data
        contact_now=contact_data.net_forces_w.detach().cpu().numpy().copy()
        contact_history=contact_data.net_forces_w_history.detach().cpu().numpy().copy()
        input_arrays={
            'actor_pre_noise':observed['policy_pre_noise'].detach().cpu().numpy().copy(),
            'actor_post_noise_pre_delay':observed['policy_post_noise_pre_delay'].detach().cpu().numpy().copy(),
            'actor_actual_post_delay':obs['policy'].detach().cpu().numpy().copy(),
            'critic_observation':observed['critic_observation'].detach().cpu().numpy().copy(),
            'contact_current':contact_now,'contact_history':contact_history}
        _durable_npz(args.output/'prepath-branch-inputs.npz',input_arrays)
        branch_diagnostics=diagnose_branch_inputs(**input_arrays)
        _durable_json(args.output/'prepath-branch-input-diagnostics.json',branch_diagnostics)
        actor_diff=float((obs['policy']-obs['policy'][0:1]).abs().max())
        if actor_diff>1e-4:raise ValueError('direct pre-Path Actor input gate')
        atomic(args.output/'direct-actor-gate.json',{'max_abs':actor_diff,'limit':1e-4,
            'noise_keys':list(noise_current)})
        if tuple(obs['policy'].shape)!=(4,22) or tuple(obs['critic'].shape)!=(4,267):
            raise ValueError('post-settling observation shape drift')
        baseline_errors={}
        b_slot=0  # Representative slot for the unchanged Actor noise path.
        offset=0
        expected_post=observed['policy_post_noise_pre_delay'][b_slot:b_slot+1].clone()
        noise_names=[name for name,cfg in zip(
            bare.observation_manager._group_obs_term_names['policy'],
            bare.observation_manager._group_obs_term_cfgs['policy'])
            if callable(getattr(getattr(cfg,'noise',None),'func',None))]
        original_noise=dict(zip(noise_names,(func for _,func in pairer.originals)))
        for name,cfg,dim in zip(bare.observation_manager._group_obs_term_names['policy'],
                                bare.observation_manager._group_obs_term_cfgs['policy'],
                                bare.observation_manager._group_obs_term_dim['policy']):
            width=math.prod(dim)
            if name in original_noise:
                raw_term=capture.pre_terms[name][b_slot:b_slot+1]
                devices=[torch.cuda.current_device()] if raw_term.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(key_seed(args.anchor_id,batch_seed,0,name))
                    expected=original_noise[name](raw_term,cfg.noise)
                if cfg.clip:expected=expected.clamp(min=cfg.clip[0],max=cfg.clip[1])
                if cfg.scale is not None:expected=expected*cfg.scale
                actual=observed['policy_post_noise_pre_delay'][b_slot:b_slot+1,offset:offset+width]
                baseline_errors[name]=float((expected.reshape_as(actual)-actual).abs().max())
                expected_post[:,offset:offset+width]=expected.reshape_as(actual)
            offset+=width
        if latency_prev is None or bool(latency_reset[b_slot]):
            expected_actor=expected_post
        else:
            alpha=(bare._onix_latency_seconds[b_slot]/float(bare.step_dt)).clamp(0.0,1.0)
            expected_actor=expected_post+alpha*(latency_prev[b_slot:b_slot+1]-expected_post)
        baseline_errors['post_delay_actor_input']=float((expected_actor-obs['policy'][b_slot:b_slot+1]).abs().max())
        with torch.inference_mode():
            expected_action=policy.act_inference({'policy':expected_actor,'critic':obs['critic'][b_slot:b_slot+1]})
            observed_action=policy.act_inference({'policy':obs['policy'][b_slot:b_slot+1],
                                                   'critic':obs['critic'][b_slot:b_slot+1]})
        baseline_errors['Actor_action']=float((expected_action-observed_action).abs().max())
        if set(baseline_errors)!={'base_ang_vel','joint_pos','joint_vel','projected_gravity',
                                  'post_delay_actor_input','Actor_action'} or max(baseline_errors.values())>2e-6:
            raise ValueError('instrumented B baseline noise equivalence failed: '+repr(baseline_errors))
        atomic(args.output/'gate1-noise-baseline.json',{'B_slot':b_slot,'per_term_max_abs':baseline_errors,
            'noise_terms':list(noise_current),'pre_delay_shape':list(observed['policy_post_noise_pre_delay'].shape),
            'actor_input_shape':list(observed['policy_actor_input'].shape)})
        if args.gate1_probe:
            atomic(args.output/'gate1-probe-complete.json',{'stage':'pre_Path_instrumentation_probe',
                'Path_activations':0,'baseline_errors':baseline_errors,'noise_terms':list(noise_current),
                'protocol_version':'eatnav_p1_longhorizon_g3_shared_dem_v2',
                'shared_dem_ledger_sha256':shared_persistence['ledger_sha256'],
                'shared_dem_gate_sha256':sha(args.output/'prepath-shared-dem-gate.json'),
                'native_dem_diagnostic_sha256':sha(args.output/'prepath-native-dem-diagnostic.json'),
                'branch_input_diagnostics_sha256':sha(args.output/'prepath-branch-input-diagnostics.json'),
                'branch_inputs_sha256':sha(args.output/'prepath-branch-inputs.npz')})
            return
        if not hasattr(robot.data,'applied_torque'):
            raise ValueError('pre-Path applied torque unavailable')
        with torch.inference_mode():
            pre_actions,_,pre_values=capture_actor_critic(
                policy,obs['policy'],obs['critic'],latent_layer=6)
        pre_commands=bare.command_manager._terms['base_velocity'].vel_command_b
        pre_snapshot={'schema':ROSFREE_SCHEMA+'_prepath_v1',
            'state_group':args.anchor_id,'seed':batch_seed,
            'follower_pre_activation':{'instance_count':0,'path_activations':0},
            'terrain_mesh_sha256':fingerprints[0]['canonical_mesh_sha256'],
            'terrain_material_sha256':material,
            'robot_material_sha256':live_materials['robot_effective_physx']['canonical_sha256'],
            'envs':[{'env_id':i,
                'root_state_w':values(robot.data.root_state_w[i]),
                'joint_position':values(robot.data.joint_pos[i]),
                'joint_velocity':values(robot.data.joint_vel[i]),
                'applied_torque':values(robot.data.applied_torque[i]),
                'actor_pre_noise':values(observed['policy_pre_noise'][i]),
                'actor_post_noise_pre_delay':values(observed['policy_post_noise_pre_delay'][i]),
                'actor_input':values(obs['policy'][i]),
                'actor_action':values(pre_actions[i]),
                'noise':{name:values(sample[0]) for name,sample in noise_current.items()},
                'command':values(pre_commands[i]),
                'shared_dem_frame_sha256':shared_persistence['source_frame_sha256'],
                'contact_current':values(contact_data.net_forces_w[i]),
                'contact_history':values(contact_data.net_forces_w_history[i]),
                'critic_observation':values(obs['critic'][i]),
                'critic_value':float(pre_values[i].reshape(-1)[0]),
                'episode_length':int(bare.episode_length_buf[i]),'reset_count':0}
                for i in range(4)]}
        _durable_json(args.output/'prepath-crossover-snapshot.json',pre_snapshot)
        _durable_json(args.output/'prepath-ready.json',{
            'snapshot_sha256':sha(args.output/'prepath-crossover-snapshot.json'),
            'live_material_sha256':sha(args.output/'live-material.json'),
            'Path_activations':0,'physics_steps_after_clone':0})
        await_host_prepath_approval(args.output,
            sha(args.output/'prepath-crossover-snapshot.json'),started)
        path_lookup={p['path_sha256']:p for p in contract['paths']}
        def poses():
            roots=bare.scene['robot'].data.root_state_w.detach().cpu().tolist()
            return [local_pose_from_root(roots[i],origins[i,:2]) for i in range(4)]
        initial_poses=poses()
        paths=paths_for_slots(active_assigned,path_lookup,initial_poses)
        common_goals=[path_in_map([contract['global_goal_base_xy_m']],pose)[0]
                      for pose in initial_poses]
        logical_t0=int(config_json['logical_t0_ns'])+args.batch_index*10_000_000_000
        pose_capture_steady_ns=time.monotonic_ns()
        library=compile_core(args.output/'core-build')
        control=RosFreeG3Control(paths,library=library)
        core_rows=control.issue(logical_t0,initial_poses,frame='tile_local')
        core_call_steady_ns=time.monotonic_ns()
        commands=[row.command for row in core_rows]
        diagnostics=[row.diagnostic for row in core_rows]
        if any(row['linear_recovery'] or row['angular_recovery'] for row in diagnostics):
            raise ValueError('unexpected recovery at Path activation')
        atomic(args.output/'path-activation.json',{'count':1,'stamp_ns':logical_t0,
               'trial_paths':[{'trial_id':row['trial_id'],'path_sha256':row['path_sha256'],
                               'path_role':row['path_role'],
                               'map_xy':paths[row['slot']],
                               'common_goal_map_xy':common_goals[row['slot']],
                               'experimental_dem_input':experimental_inputs[row['slot']]}
                              for row in active_assigned],
               'control_backend':'ros_free_cpp_core','core_source_sha256':g3_contract['cpp_core_source_sha256'],
               'core_library_sha256':sha(library),'core_env_ids':[row.env_id for row in core_rows],
               'pose_frame':'tile_local','core_call_count_per_env':[1]*4,
               'dem_input_role':'shared_experimental_history',
               'dem_source_env':0,
               'shared_dem_ledger_sha256':shared_persistence['ledger_sha256'],
               'shared_dem_source_frame_sha256':shared_persistence['source_frame_sha256']})
        atomic(args.output/'status.json',{'stage':'Path_activated','Path_activations':1})
        override.provider=None
        def inject_rows(rows):
            command_tensor=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            command_tensor[:4]=torch.as_tensor(rows,device=bare.device,dtype=command_tensor.dtype)
            override.set(command_tensor)
            return values(bare.command_manager._terms['base_velocity'].vel_command_b[:4])
        applied=control.inject(core_rows,inject_rows,stamp=logical_t0)
        injection_steady_ns=time.monotonic_ns()
        obs,observed,command_obs=reconcile_command_observation(
            obs,observed,bare,applied,latency_prev)
        terminal_capture={}
        original_reset=bare._reset_idx
        def capture_reset(env_ids):
            terms=bare.termination_manager
            for index in env_ids.detach().cpu().tolist():
                if 0<=index<4:
                    terminal_capture[index]={'root_state':values(robot.data.root_state_w[index]),
                        'termination_reason':[name for name in terms.active_terms
                              if bool(terms.get_term(name)[index])]}
            return original_reset(env_ids)
        bare._reset_idx=capture_reset
        traces=[[] for _ in range(4)]
        def abort_run(reason,control_step):
            for index,trace in enumerate(traces):
                with (args.output/f'abort-trace-slot-{index}.jsonl').open('x') as stream:
                    for record in trace:
                        stream.write(json.dumps(record,sort_keys=True,allow_nan=True)+'\n')
            atomic(args.output/'abort.json',{'reason':reason,'control_step':control_step,
                'Path_activations':1,'trace_lengths':[len(trace) for trace in traces]})
            raise ValueError('paired diagnostic abort: '+reason)
        ended=[False]*4
        exposures=[G3Exposure() for _ in range(4)]
        steps=g3_contract['max_control_steps']
        initial_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
        episode_gate=RosFreeSameEpisodeGate(initial_episode,int(bare.max_episode_length))
        trace_streams=[DurableTrace(args.output/f'trace-slot-{i}.jsonl') for i in range(4)]
        for stream in trace_streams:stream.__enter__()
        reward_terms=len(reward_capture.names)
        fields={
            'env_id':{'shape':[],'dtype':'int32'},'control_step':{'shape':[],'dtype':'int32'},
            'logical_stamp_ns':{'shape':[],'dtype':'int64'},
            'root_pre':{'shape':[13],'dtype':'float32'},
            'root_post':{'shape':[13],'dtype':'float32'},
            'follower_command':{'shape':[3],'dtype':'float32'},
            'injected_command':{'shape':[3],'dtype':'float32'},
            'actor_input':{'shape':[int(obs['policy'].shape[1])],'dtype':'float32'},
            'critic_input':{'shape':[int(obs['critic'].shape[1])],'dtype':'float32'},
            'actor_action':{'shape':[6],'dtype':'float32'},
            'critic_value':{'shape':[],'dtype':'float32'},
            'joint_pos_pre':{'shape':[52],'dtype':'float32'},
            'joint_vel_pre':{'shape':[52],'dtype':'float32'},
            'applied_torque_pre':{'shape':[52],'dtype':'float32'},
            'applied_torque_post':{'shape':[52],'dtype':'float32'},
            'joint_pos_target':{'shape':[52],'dtype':'float32'},
            'joint_vel_target':{'shape':[52],'dtype':'float32'},
            'joint_effort_target':{'shape':[52],'dtype':'float32'},
            'target_available':{'shape':[3],'dtype':'uint8'},
            'body_force_pre_world':{'shape':[60,3],'dtype':'float32'},
            'body_force_post_world':{'shape':[60,3],'dtype':'float32'},
            'substep_peak_force_world':{'shape':[60,3],'dtype':'float32'},
            'substep_peak_index':{'shape':[60],'dtype':'int16'},
            'substep_force_integral_world':{'shape':[60,3],'dtype':'float32'},
            'substep_contact_count':{'shape':[60],'dtype':'uint8'},
            'reward_total':{'shape':[],'dtype':'float32'},
            'reward_term_weighted':{'shape':[reward_terms],'dtype':'float32'},
            'reward_term_raw':{'shape':[reward_terms],'dtype':'float32'},
            'reward_term_raw_available':{'shape':[reward_terms],'dtype':'uint8'},
            'goal_reached_pre':{'shape':[],'dtype':'uint8'},
        }
        teacher_writer=ChunkWriter(args.output/'teacher-chunks',
            schema='eatnav_p1_teacher_chunks_v1',fixed_shapes=fields,chunk_rows=128,
            metadata={'capture_schema':'eatnav_p1_teacher_capture_v1',
                      'reward_manifest_sha256':sha(args.output/'teacher-reward-manifest.json'),
                      'index_mapping_sha256':sha(args.output/'teacher-index-mapping.json'),
                      'source_sha256':sha(__file__),
                      'shared_dem_source_sha256':shared_persistence['source_frame_sha256'],
                      'path_assignment_sha256':g3_contract['assignment_sha256'],
                      'checkpoint_sha256':config_json['rl_checkpoint_sha256'],
                      'control_dt_s':0.05,'physics_dt_s':0.005,
                      'load_phase':'pre_and_post_physics_as_named',
                      'contact_frame':'world'})
        reset_calls=[]
        instrumented_reset=bare._reset_idx
        def detect_reset(env_ids):
            reset_calls.extend(int(i) for i in env_ids.detach().cpu().tolist())
            return instrumented_reset(env_ids)
        bare._reset_idx=detect_reset
        try:
            for step in range(steps):
                stamp=logical_t0+step*STEP_NS
                active=[exposures[i].observe(step,bool(diagnostics[i]['goal_reached'])) for i in range(4)]
                if not any(active):
                    break
                before_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
                actor_begin_steady_ns=time.monotonic_ns()
                with torch.inference_mode():
                    actions,latents,critic_values=control.mark_actor(stamp,lambda: capture_actor_critic(
                        policy,obs['policy'],obs['critic'],latent_layer=6))
                if tuple(actions.shape)!=(4,6) or not bool(torch.isfinite(actions).all()):
                    raise ValueError('Actor action shape/nonfinite')
                before=robot.data.root_state_w.detach().cpu().tolist()
                records=[]
                for i in range(4):
                    records.append({'control_step':step,'logical_sim_time_ns':stamp,
                        'follower_command':commands[i], 'applied_command':applied[i],
                        'condition':conditions[i], 'candidate_slot':assigned[i]['candidate_slot'],
                        'label_exposure':active[i],
                        'follower_diagnostic':diagnostics[i],
                        'core_evidence':{'env_id':core_rows[i].env_id,
                            'pose_frame':core_rows[i].frame,'pose':core_rows[i].pose,
                            'stamp_ns':core_rows[i].stamp_ns,
                            'suppressed_after_exposure':not active[i],
                            'actor_command_slice':command_obs['actor_slice'],
                            'critic_command_slice':command_obs['critic_slice'],
                            'command_observation':command_obs['actor_seen'][i],
                            'command_observation_previous_raw':command_obs['previous_raw'][i],
                            'command_observation_latency_alpha':command_obs['latency_alpha'][i]},
                        'actor_injected_noise':{name:values(sample[0]) for name,sample in noise_current.items()},
                        'joint_position':values(robot.data.joint_pos[i]),
                        'joint_velocity':values(robot.data.joint_vel[i]),
                        'applied_torque':values(robot.data.applied_torque[i]) if hasattr(robot.data,'applied_torque') else None,
                        'contact_net_forces_w':values(contact_sensor.data.net_forces_w[i]),
                        'contact_net_forces_w_history':values(contact_sensor.data.net_forces_w_history[i]),
                        'policy_latency_seconds':float(bare._onix_latency_seconds[i]),
                        'actor_input':values(obs['policy'][i]),
                        'critic_observation':values(obs['critic'][i]),
                        'actor_pre_noise':values(observed['policy_pre_noise'][i]),
                        'actor_post_noise_pre_delay':values(observed['policy_post_noise_pre_delay'][i]),
                        'actor_action':values(actions[i]),
                        'critic_value':float(critic_values[i].reshape(-1)[0]),
                        'root_state':before[i], 'terminal':None,
                        'policy_action_mode':'act_inference',
                        'dem_input_role':'shared_experimental_history',
                        'dem_source_env':0,
                        'experimental_dem_input':experimental_inputs[i],
                        'shared_dem_source_frame_sha256':shared_persistence['source_frame_sha256']})
                action_batch=actions.clone()
                for i,done in enumerate(ended):
                    if done or not active[i]: action_batch[i]=0
                latency_prev=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
                pairer.set_step(step+1)
                capture.begin_step()
                physics_begin_steady_ns=time.monotonic_ns()
                with PhysicsSubstepTap(bare.sim,contact_sensor.contact_physx_view,
                        num_envs=4,body_count=60,physics_dt=float(bare.physics_dt),
                        substeps=10,stamp_ns=stamp) as substep_tap:
                    with torch.inference_mode():
                        next_obs,reward,done,_=control.mark_physics(stamp,lambda: env.step(action_batch))
                    substep_rows=substep_tap.finish()
                reward_row=reward_capture.capture_step(bare.reward_manager,reward,step,stamp)
                def target_row(name,index):
                    tensor=getattr(robot.data,name,None)
                    if tensor is None:
                        return [0.0]*52,0
                    if tensor.shape[1]!=52 or not bool(torch.isfinite(tensor[index]).all()):
                        raise ValueError('actuator target shape/nonfinite')
                    return values(tensor[index]),1
                post_poses=poses()
                next_pose_capture_steady_ns=time.monotonic_ns()
                after_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
                observed=capture.finish_step(next_obs)
                noise_current=pairer.verify_step()
                for i in range(4):
                    if ended[i]:continue
                    record=records[i]
                    record['reward']=float(reward[i])
                    target_pos,has_pos=target_row('joint_pos_target',i)
                    target_vel,has_vel=target_row('joint_vel_target',i)
                    target_effort,has_effort=target_row('joint_effort_target',i)
                    teacher_step={'step':step,'logical_stamp_ns':stamp,
                                  'reward':{'step':reward_row['step'],
                                            'logical_stamp_ns':reward_row['logical_stamp_ns'],
                                            'reward_compute_phase':reward_row['reward_compute_phase']},
                                  'substep':substep_rows[i],'control_dt_s':float(bare.step_dt)}
                    validate_teacher_step(teacher_step)
                    if bool(done[i]):
                        if i not in terminal_capture:
                            raise ValueError('reset before terminal-state capture')
                        record['post_root_state']=terminal_capture[i]['root_state']
                        record['terminal']={'termination_reason':terminal_capture[i]['termination_reason'],
                                            'terminated':bool(bare.reset_terminated[i]),
                                            'truncated':bool(bare.reset_time_outs[i])}
                        ended[i]=True
                    else:
                        record['post_root_state']=values(robot.data.root_state_w[i])
                    record['timing_protocol']={
                        'step':step,'stamp_ns':stamp,
                        'pose_stamp_ns':core_rows[i].stamp_ns,
                        'core_call_stamp_ns':core_rows[i].stamp_ns,
                        'command_injection_stamp_ns':stamp,
                        'actor_input_stamp_ns':stamp,
                        'physics_input_ns':stamp,
                        'next_pose_stamp_ns':stamp+STEP_NS,
                        'next_pose':post_poses[i],
                        'events':['pose','core','command_injection','actor','physics','next_pose'],
                        'pose_capture_steady_ns':pose_capture_steady_ns,
                        'core_call_steady_ns':core_call_steady_ns,
                        'command_injection_steady_ns':injection_steady_ns,
                        'actor_begin_steady_ns':actor_begin_steady_ns,
                        'physics_begin_steady_ns':physics_begin_steady_ns,
                        'next_pose_capture_steady_ns':next_pose_capture_steady_ns,
                        'episode_length_before':before_episode[i],
                        'episode_length_after':after_episode[i],
                        'terminated':bool(bare.reset_terminated[i]),
                        'truncated':bool(bare.reset_time_outs[i]),
                        'reset_called':i in reset_calls,
                    }
                    traces[i].append(record)
                    trace_streams[i].append(record)
                    raw_terms=reward_row['raw_value'][i]
                    teacher_writer.append({
                        'env_id':i,'control_step':step,'logical_stamp_ns':stamp,
                        'root_pre':record['root_state'],'root_post':record['post_root_state'],
                        'follower_command':record['follower_command'],
                        'injected_command':record['applied_command'],
                        'actor_input':record['actor_input'],
                        'critic_input':record['critic_observation'],
                        'actor_action':record['actor_action'],
                        'critic_value':record['critic_value'],
                        'joint_pos_pre':record['joint_position'],
                        'joint_vel_pre':record['joint_velocity'],
                        'applied_torque_pre':record['applied_torque'],
                        'applied_torque_post':values(robot.data.applied_torque[i]),
                        'joint_pos_target':target_pos,'joint_vel_target':target_vel,
                        'joint_effort_target':target_effort,
                        'target_available':[has_pos,has_vel,has_effort],
                        'body_force_pre_world':record['contact_net_forces_w'],
                        'body_force_post_world':values(contact_sensor.data.net_forces_w[i]),
                        'substep_peak_force_world':substep_rows[i]['peak_force_world_N'],
                        'substep_peak_index':substep_rows[i]['peak_substep'],
                        'substep_force_integral_world':substep_rows[i]['integral_force_Ns'],
                        'substep_contact_count':substep_rows[i]['contact_substep_count'],
                        'reward_total':reward_row['total_reward'][i],
                        'reward_term_weighted':reward_row['weighted_contribution'][i],
                        'reward_term_raw':[x if x is not None else 0.0 for x in raw_terms],
                        'reward_term_raw_available':[x is not None for x in raw_terms],
                        'goal_reached_pre':int(record['follower_diagnostic']['goal_reached']),
                    },ragged_contacts=[])
                    if not (pose_capture_steady_ns<=core_call_steady_ns<=injection_steady_ns<=
                            actor_begin_steady_ns<=physics_begin_steady_ns<=next_pose_capture_steady_ns):
                        raise ValueError('ROS-free causal order violated')
                try:
                    episode_gate.accept(step,stamp,core_rows,before_episode,after_episode,
                                        [bool(x) for x in done[:4]],reset_calls)
                except ValueError as exc:
                    abort_run(str(exc),step)
                if reset_calls or any(ended):
                    abort_run('same_episode_termination_or_reset',step)
                obs=next_obs
                if not bool(torch.isfinite(obs['policy']).all()):
                    abort_run('nonfinite_Actor_observation',step)
                for i in range(4):
                    if traces[i] and traces[i][-1]['control_step']==step:
                        reasons=assess_step(traces[i][-1],
                            roll_pitch_cap_rad=config_json['abort_roll_pitch_rad'],
                            speed_cap_mps=config_json['abort_root_speed_mps'],
                            angular_cap_radps=config_json['angular_command_cap_radps'])
                        if reasons:abort_run(','.join(reasons),step)
                atomic(args.output/'heartbeat.json',{'completed_control_steps':step+1,
                       'elapsed_s':time.monotonic()-started,'Path_activations':1})
                if step+1<steps and not all(x.finished_after(step) for x in exposures):
                    next_stamp=logical_t0+(step+1)*50_000_000
                    pose_capture_steady_ns=next_pose_capture_steady_ns
                    core_rows=control.issue(next_stamp,post_poses,frame='tile_local')
                    core_call_steady_ns=time.monotonic_ns()
                    commands=[row.command for row in core_rows]
                    diagnostics=[row.diagnostic for row in core_rows]
                    suppressed=[i for i,exposure in enumerate(exposures) if exposure.finished_after(step)]
                    applied=control.inject(core_rows,inject_rows,stamp=next_stamp,
                                           suppressed_envs=suppressed)
                    injection_steady_ns=time.monotonic_ns()
                    obs,observed,command_obs=reconcile_command_observation(
                        next_obs,observed,bare,applied,latency_prev)
        finally:
            bare._reset_idx=original_reset
            for stream in trace_streams:stream.__exit__(None,None,None)
        if not all(exposure.stop_exclusive is not None and
                   exposure.finished_after(len(traces[0])-1) for exposure in exposures):
            raise ValueError('G3 incomplete arrival hold or censoring')
        if episode_gate.next_step != len(traces[0]):
            raise ValueError('G3 noncontiguous timing trace')
        teacher_writer.seal()
        teacher_verify=verify_chunks(args.output/'teacher-chunks',require_complete=True)
        if teacher_verify['rows']!=4*len(traces[0]):
            raise ValueError('teacher chunk row count')
        atomic(args.output/'teacher-log-complete.json',{
            'schema':'eatnav_p1_teacher_capture_v1','rows':teacher_verify['rows'],
            'chunk_ledger_sha256':sha(args.output/'teacher-chunks/chunk-ledger.json'),
            'chunk_complete_sha256':sha(args.output/'teacher-chunks/complete.json'),
            'reward_manifest_sha256':sha(args.output/'teacher-reward-manifest.json'),
            'index_mapping_sha256':sha(args.output/'teacher-index-mapping.json')})
        atomic(args.output/'pilot-summary.json',{'anchor_id':args.anchor_id,
               'batch_index':args.batch_index,'execution_index':args.execution_index,'control_steps':len(traces[0]),
               'trial_ids':[row['trial_id'] for row in active_assigned],
               'recorded_steps':[len(trace) for trace in traces],
               'arrivals':[{'arrival_step':x.arrival_step,'stop_exclusive':x.stop_exclusive,
                            'right_censored':x.censored} for x in exposures],
               'termination':[trace[-1]['terminal'] if trace else None for trace in traces],
               'core_final':{'backend':'ros_free_cpp_core','instance_count':len(control.followers),
                             'last_stamp_ns':control.last_stamp_ns,'phase':control.phase,
                             'control_step_count':episode_gate.next_step},
               'dem_input_role':'shared_experimental_history',
               'shared_dem_ledger_sha256':shared_persistence['ledger_sha256']})
        atomic(args.output/'complete.json',{'stage':'pilot_complete',
               'Path_activations':1,'settling_control_steps':20,
               'control_steps':len(traces[0]),'anchor_id':args.anchor_id,'batch_index':args.batch_index,'execution_index':args.execution_index,
               'initial_state_sha256':sha(args.output/'initial-state.json'),
               'terrain_sha256':sha(args.output/'terrain.json'),
               'process_provenance_sha256':sha(args.output/'process-provenance.json'),
               'prepath_gate_sha256':sha(args.output/'prepath-gate.json'),
               'protocol_version':ROSFREE_SCHEMA,
               'live_material_sha256':sha(args.output/'live-material.json'),
               'prepath_crossover_snapshot_sha256':sha(args.output/'prepath-crossover-snapshot.json'),
               'prepath_approval_sha256':sha(args.output/'prepath-approval.json'),
               'prepath_physical_gate_sha256':sha(args.output/'prepath-physical-state-gate.json'),
               'prepath_shared_gate_sha256':sha(args.output/'prepath-shared-dem-gate.json'),
               'prepath_shared_ledger_sha256':sha(args.output/'prepath-shared-dem-sha-ledger.json'),
               'prepath_native_diagnostic_sha256':sha(args.output/'prepath-native-dem-diagnostic.json'),
               'prepath_branch_input_diagnostics_sha256':sha(args.output/'prepath-branch-input-diagnostics.json'),
               'prepath_branch_inputs_sha256':sha(args.output/'prepath-branch-inputs.npz'),
               'prepath_dem_ledger_sha256':sha(args.output/'prepath-dem-sha-ledger.json'),
               'direct_actor_gate_sha256':sha(args.output/'direct-actor-gate.json'),
               'path_activation_sha256':sha(args.output/'path-activation.json'),
               'path_contract_sha256':sha(args.output/'path-contract.json'),
               'pilot_summary_sha256':sha(args.output/'pilot-summary.json'),
               'contact_body_names_sha256':sha(args.output/'contact-body-names.json'),
               'teacher_log_complete_sha256':sha(args.output/'teacher-log-complete.json'),
               'trace_sha256':[sha(args.output/f'trace-slot-{i}.jsonl') for i in range(4)],
               'noise_contract':'keyed_original_additive_uniform_p1_v1'})
    finally:
        if capture is not None:capture.close()
        if pairer is not None:pairer.close()
        if override is not None: override.close()
        if env is not None: env.close()


try:
    main()
except BaseException as exc:
    args.output.mkdir(parents=True,exist_ok=True)
    atomic(args.output/'diagnostic-error.json',{'type':type(exc).__name__,
           'message':str(exc),'traceback':traceback.format_exc(),
           'Path_activations':int((args.output/'path-activation.json').exists())})
    raise
finally:
    simulation_app.close()
