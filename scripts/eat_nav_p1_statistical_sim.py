"""Candidate-free, seven-slot P1 statistical initialization validation.

The Path-enabled mode is deliberately absent until this initialization
measurement passes. This file never publishes a Path or computes Outcomes.
"""

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
parser.add_argument('--batch-index', type=int, choices=(0, 1), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaacsim.core.simulation_manager import SimulationManager
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, clone_internal_branch_state
from visualnav_transformer.eat_nav.phase_c_sim import capture_actor_critic
from visualnav_transformer.eat_nav.p1_statistical_runtime import CachedTerrainGeometry, require_prepath_gate, jsonable_values, validate_candidate_free_trace
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from rsl_rl.runners import OnPolicyRunner
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg


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
if sha(__file__) != freeze['simulator_source_sha256']:
    raise ValueError('simulator source SHA drift')
anchor=next(a for a in contract['anchors'] if a['anchor_id']==args.anchor_id)
batch_seed=manifest['batch_seeds'][args.batch_index]
assigned=[row for row in manifest['trials'] if row['anchor_id']==args.anchor_id and row['batch_index']==args.batch_index]
if len(assigned)!=7 or {row['slot'] for row in assigned}!=set(range(7)):
    raise ValueError('assignment does not cover seven slots')
if config_json['policy_action_mode']!='act_inference' or config_json['paired_policy_noise']:
    raise ValueError('inference contract drift')
if config_json['primary_control_steps']!=40 or config_json['diagnostic_control_steps']!=120:
    raise ValueError('P1 primary/diagnostic window contract')


class SevenSameTerrain(EATTerrainGenerator):
    last_fingerprints=None

    def _generate_random_terrains(self):
        if (self.cfg.num_rows,self.cfg.num_cols)!=(1,7):
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
            for col in range(7):
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
    if tensor.shape[0]==7:
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


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    pid=os.getpid()
    tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    atomic(args.output/'owned-process.json',{'pid':pid,'start_tick':tick,'Path_publications':0})
    atomic(args.output/'status.json',{'stage':'startup','Path_publications':0})
    eat_cfg=load_eat_config(args.input/'eat_nav.yaml')
    saved=yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    toggles={name:saved[name].lower()=='true' for name in
             ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,7)
    gen=cfg.scene.terrain.terrain_generator
    gen.class_type=SevenSameTerrain
    gen.num_rows,gen.num_cols=1,7
    gen.curriculum=False
    gen.difficulty_range=(.5,.5)
    gen.use_cache=False
    gen.seed=batch_seed
    gen.sub_terrains={anchor['terrain']:gen.sub_terrains[anchor['terrain']]}
    cfg.scene.terrain.max_init_terrain_level=0
    env=None;override=None
    try:
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        terrain=bare.scene.terrain
        fingerprints=SevenSameTerrain.last_fingerprints
        if not fingerprints or len(fingerprints)!=7 or len({r['canonical_mesh_sha256'] for r in fingerprints})!=1:
            raise ValueError('terrain geometry not reused')
        collision=composed_collision_fingerprint()
        material=hashlib.sha256(str(cfg.scene.terrain.physics_material).encode()).hexdigest()
        origins=bare.scene.env_origins.detach().cpu().numpy()
        atomic(args.output/'terrain.json',{'fingerprints':fingerprints,
                'world_tile_origins':origins.tolist(),'collision':collision,
                'material_sha256':material,'scene_num_envs':7,
                'sim_dt':float(bare.physics_dt),'control_dt':float(bare.step_dt)})
        env.reset()
        robot=bare.scene['robot']
        state=fixture[args.anchor_id]['prephysics_state']
        if state['joint_names']!=list(robot.data.joint_names):
            raise ValueError('joint order changed')
        ids=torch.arange(7,device=bare.device)
        root=robot.data.root_state_w[ids].clone()
        radius_match=re.search(r'-r([0-9]+\.[0-9]+)-',anchor['pose_id'])
        if radius_match is None:
            raise ValueError('frozen source pose radius missing')
        radius=float(radius_match.group(1))
        requested=anchor['requested_xy_yaw']; source_pose=state['root_state_w']
        local_xy=[radius*math.cos(requested[2]),radius*math.sin(requested[2])]
        for i in range(7):
            root[i,0]=float(origins[i,0])+local_xy[0]
            root[i,1]=float(origins[i,1])+local_xy[1]
            root[i,2]=float(source_pose[2])
            root[i,3:7]=torch.as_tensor(source_pose[3:7],device=bare.device,dtype=root.dtype)
            root[i,7:13]=torch.as_tensor(source_pose[7:13],device=bare.device,dtype=root.dtype)
        robot.write_root_state_to_sim(root,env_ids=ids)
        joints=torch.as_tensor([state['joint_position']]*7,device=bare.device,dtype=robot.data.joint_pos.dtype)
        speeds=torch.as_tensor([state['joint_velocity']]*7,device=bare.device,dtype=robot.data.joint_vel.dtype)
        robot.write_joint_state_to_sim(joints,speeds,env_ids=ids)
        bare._onix_flipper_fold_mode[ids]=float(state['flipper_fold_mode_latch'])
        term=bare.action_manager._terms['flipper_pos']
        desired=joints[:,term._joint_ids]
        term._previous_targets.copy_(desired)
        term._pre_clip_processed_actions.copy_(desired)
        bare.sim.forward()
        zero=torch.zeros((7,6),device=bare.device)
        override=CommandOverride(bare.command_manager,7)
        step_index=0
        def provider():
            current=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            current[:7]=torch.as_tensor([fixture[args.anchor_id]['settling_commands'][step_index]]*7,
                                         device=bare.device,dtype=current.dtype)
            return current
        override.provider=provider
        override.set(provider())
        for step_index in range(20):
            obs,_,done,_=env.step(zero)
            if bool(torch.any(done)):
                raise ValueError('termination during candidate-free settling')
            atomic(args.output/'heartbeat.json',{'completed_control_steps':step_index+1,
                                                  'elapsed_s':time.monotonic()-started,
                                                  'Path_publications':0})
        clone=clone_internal_branch_state(bare,0,range(1,7))
        # The helper lists every unreplicated field; do not silently call it equal.
        atomic(args.output/'clone-report.json',clone)
        records=[snapshot(bare,i,origins[i],fingerprints[i]['canonical_mesh_sha256'],
                          collision['sha256'],material) for i in range(7)]
        atomic(args.output/'initial-state.json',records)
        gate=require_prepath_gate(records,config_json['initialization_tolerances'])
        atomic(args.output/'prepath-gate.json',gate)
        if tuple(obs['policy'].shape)!=(7,22) or tuple(obs['critic'].shape)!=(7,267):
            raise ValueError('Actor/Critic observation shape drift')
        checkpoint=Path(config_json['rl_checkpoint'])
        if sha(checkpoint)!=config_json['rl_checkpoint_sha256']:
            raise ValueError('RL checkpoint SHA drift')
        runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
        runner.alg.policy.load_state_dict(weights)
        policy=runner.alg.policy
        terminal_capture={}
        original_reset=bare._reset_idx
        def capture_reset(env_ids):
            terms=bare.termination_manager
            for index in env_ids.detach().cpu().tolist():
                if 0<=index<7:
                    terminal_capture[index]={'root_state':values(robot.data.root_state_w[index]),
                         'termination_reason':[name for name in terms.active_terms
                             if bool(terms.get_term(name)[index])]}
            return original_reset(env_ids)
        bare._reset_idx=capture_reset
        traces=[[] for _ in range(7)]
        try:
            for control_step in range(120):
                with torch.inference_mode():
                    actions,_,_=capture_actor_critic(policy,obs['policy'],obs['critic'],latent_layer=6)
                if tuple(actions.shape)!=(7,6) or not bool(torch.isfinite(actions).all()):
                    raise ValueError('candidate-free Actor action shape/nonfinite')
                roots=robot.data.root_state_w.detach().cpu().tolist()
                commands=bare.command_manager._terms['base_velocity'].vel_command_b.detach().cpu().tolist()
                actors=obs['policy'].detach().cpu().tolist()
                critics=obs['critic'].detach().cpu().tolist()
                capture_actions=actions.detach().cpu().tolist()
                with torch.inference_mode():
                    obs,reward,done,_=env.step(actions)
                for index in range(7):
                    terminal=None
                    if bool(done[index]):
                        if index not in terminal_capture:
                            raise ValueError('candidate-free terminal state missing')
                        post=terminal_capture[index]['root_state']
                        terminal={'termination_reason':terminal_capture[index]['termination_reason']}
                    else:
                        post=values(robot.data.root_state_w[index])
                    traces[index].append({'control_step':control_step,
                         'actor_input':actors[index],'critic_observation':critics[index],
                         'actor_action':capture_actions[index],'command':commands[index],
                         'root_state':roots[index],'post_root_state':post,
                         'reward':float(reward[index]),'terminal':terminal})
                terminal_capture.clear()
                atomic(args.output/'heartbeat.json',{'completed_control_steps':control_step+1,
                     'elapsed_s':time.monotonic()-started,'Path_publications':0})
        finally:
            bare._reset_idx=original_reset
        for index,rows in enumerate(traces):
            validate_candidate_free_trace(rows)
            with (args.output/f'trace-slot-{index}.jsonl').open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        atomic(args.output/'complete.json',{'stage':'candidate_free_complete',
               'Path_publications':0,'settling_control_steps':20,'control_steps':120,
               'anchor_id':args.anchor_id,'batch_index':args.batch_index,
               'initial_state_sha256':sha(args.output/'initial-state.json'),
               'terrain_sha256':sha(args.output/'terrain.json'),
               'prepath_gate_sha256':sha(args.output/'prepath-gate.json'),
               'trace_sha256':[sha(args.output/f'trace-slot-{index}.jsonl') for index in range(7)]})
    finally:
        if override is not None: override.close()
        if env is not None: env.close()


try:
    main()
except BaseException as exc:
    args.output.mkdir(parents=True,exist_ok=True)
    atomic(args.output/'diagnostic-error.json',{'type':type(exc).__name__,
           'message':str(exc),'traceback':traceback.format_exc(),
           'Path_publications':0})
    raise
finally:
    simulation_app.close()
