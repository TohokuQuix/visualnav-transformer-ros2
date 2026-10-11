"""Standalone G2 600-step timing diagnostic; never use for Path quality labels."""

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
from isaacsim.core.simulation_manager import SimulationManager
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, clone_internal_branch_state
from visualnav_transformer.eat_nav.phase_c_sim import ObservationCapture, capture_actor_critic
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from visualnav_transformer.eat_nav.p1_statistical_pilot import (
    local_pose_from_root, paths_for_slots, validate_bridge_response,
)
from rsl_rl.runners import OnPolicyRunner
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.p1_statistical_runtime import CachedTerrainGeometry, require_prepath_gate, jsonable_values
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from visualnav_transformer.eat_nav.p1_paired_command import intervene
from visualnav_transformer.eat_nav.p1_paired_noise import PairedPolicyNoise, key_seed
from visualnav_transformer.eat_nav.p1_paired_integrity import assess_step
from visualnav_transformer.eat_nav.p1_longhorizon_protocol import DurableTrace, SameEpisodeGate, STEP_NS


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
g2_contract=json.loads((args.input/'g2-contract.json').read_text())
if sha(__file__) != g2_contract['g2_simulator_source_sha256']:
    raise ValueError('G2 diagnostic source SHA drift')
if g2_contract['control_steps'] != 600 or g2_contract['episode_length_s'] != 40.0:
    raise ValueError('G2 diagnostic contract drift')
anchor=next(a for a in contract['anchors'] if a['anchor_id']==args.anchor_id)
batch_seed=manifest['batch_seeds'][args.anchor_id][args.batch_index]
assigned=[row for row in manifest['trials'] if row['anchor_id']==args.anchor_id and row['batch_index']==args.batch_index]
conditions=[row['condition'] for row in sorted(assigned,key=lambda row:row['slot'])]
if len(assigned)!=4 or {row['slot'] for row in assigned}!=set(range(4)):
    raise ValueError('assignment does not cover seven slots')
if config_json['policy_action_mode']!='act_inference' or not config_json['paired_policy_noise']:
    raise ValueError('inference contract drift')
if config_json['primary_control_steps']!=40 or config_json['diagnostic_control_steps']!=120:
    raise ValueError('P1 primary/diagnostic window contract')
direct_path=next(p for p in contract['paths'] if p['bearing_deg']==0 and p['endpoint']==[0.75,0.0])
assigned=[{**row,'condition':'A','path_sha256':direct_path['path_sha256']} for row in assigned]
conditions=['A']*4


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


def accepted_events(response,stamp_ns):
    events=[event for event in response['events'] if event['accepted'] and event['stamp_ns']==stamp_ns]
    if len(events)!=5 or {event['branch'] for event in events}!=set(range(5)):
        raise ValueError('accepted exact-stamp bridge event count')
    if any(event['tf_stamp_ns']!=stamp_ns for event in events):
        raise ValueError('accepted command TF stamp mismatch')
    return {event['branch']:event for event in events}


def main():
    persistent_parent=Path('/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1')
    if not args.output.is_relative_to(persistent_parent):
        raise ValueError('G2 output must be under the host-backed persistent root')
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
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,4)
    # Diagnostic-only timeout. Physics, physical terminations, policy and follower are unchanged.
    cfg.episode_length_s=g2_contract['episode_length_s']
    gen=cfg.scene.terrain.terrain_generator
    gen.class_type=SevenSameTerrain
    gen.num_rows,gen.num_cols=1,4
    gen.curriculum=False
    gen.difficulty_range=(.5,.5)
    gen.use_cache=False
    gen.seed=batch_seed
    gen.sub_terrains={anchor['terrain']:gen.sub_terrains[anchor['terrain']]}
    cfg.scene.terrain.max_init_terrain_level=0
    env=None;override=None;capture=None;pairer=None;bridge=None
    try:
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        if abs(float(bare.step_dt)-0.05)>1e-9 or int(bare.max_episode_length)!=800:
            raise ValueError('G2 50ms/40s environment contract')
        terrain=bare.scene.terrain
        fingerprints=SevenSameTerrain.last_fingerprints
        if not fingerprints or len(fingerprints)!=4 or len({r['canonical_mesh_sha256'] for r in fingerprints})!=1:
            raise ValueError('terrain geometry not reused')
        collision=composed_collision_fingerprint()
        material=hashlib.sha256(str(cfg.scene.terrain.physics_material).encode()).hexdigest()
        origins=bare.scene.env_origins.detach().cpu().numpy()
        atomic(args.output/'terrain.json',{'fingerprints':fingerprints,
                'world_tile_origins':origins.tolist(),'collision':collision,
                'material_sha256':material,'scene_num_envs':4,
                'sim_dt':float(bare.physics_dt),'control_dt':float(bare.step_dt)})
        env.reset()
        robot=bare.scene['robot']
        contact_sensor=bare.scene['contact_forces']
        contact_body_names=list(contact_sensor.body_names)
        if len(contact_body_names)!=contact_sensor.data.net_forces_w.shape[1]:
            raise ValueError('contact body identity/count mismatch')
        atomic(args.output/'contact-body-names.json',{'body_names':contact_body_names,
            'count':len(contact_body_names),'force_frame':'world'})
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
        for step_index in range(20):
            obs,_,done,_=env.step(zero)
            if bool(torch.any(done)):
                raise ValueError('termination during candidate-free settling')
            atomic(args.output/'heartbeat.json',{'completed_control_steps':step_index+1,
                                                  'elapsed_s':time.monotonic()-started,
                                                  'Path_publications':0})
        clone=clone_internal_branch_state(bare,0,range(1,4))
        # The helper lists every unreplicated field; do not silently call it equal.
        atomic(args.output/'clone-report.json',clone)
        records=[snapshot(bare,i,origins[i],fingerprints[i]['canonical_mesh_sha256'],
                          collision['sha256'],material) for i in range(4)]
        atomic(args.output/'initial-state.json',records)
        gate=require_prepath_gate(records,config_json['initialization_tolerances'])
        atomic(args.output/'prepath-gate.json',gate)
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
                'Path_publications':0,'baseline_errors':baseline_errors,'noise_terms':list(noise_current)})
            return
        path_lookup={p['path_sha256']:p for p in contract['paths']}
        def poses():
            roots=bare.scene['robot'].data.root_state_w.detach().cpu().tolist()
            return [local_pose_from_root(roots[i],origins[i,:2]) for i in range(4)]
        initial_poses=poses()
        paths=paths_for_slots(assigned,path_lookup,initial_poses)
        logical_t0=int(config_json['logical_t0_ns'])+args.batch_index*10_000_000_000
        bridge=FollowerBridgeClient(timeout=45.)
        startup=bridge.request('reset',stamp_ns=logical_t0,
                             poses=initial_poses+[initial_poses[0]],paths=paths)
        bridge_response=startup
        commands=validate_bridge_response(startup,logical_t0,branches=5)
        if max(abs(a-b) for a,b in zip(commands[0],commands[4]))>1e-6:
            raise ValueError('production follower monitor command mismatch')
        diagnostics=startup['follower_diagnostics']
        if len(diagnostics)!=5:raise ValueError('follower diagnostic count')
        bridge_events=accepted_events(startup,logical_t0)
        if any(row['linear_recovery'] or row['angular_recovery'] for row in diagnostics[:4]):
            raise ValueError('unexpected recovery at Path activation')
        interventions=[intervene(commands[i],diagnostics[i],conditions[i],stamp_ns=logical_t0) for i in range(4)]
        applied=[row[0] for row in interventions]
        atomic(args.output/'path-publication.json',{'count':1,'stamp_ns':logical_t0,
               'trial_paths':[{'trial_id':row['trial_id'],'path_sha256':row['path_sha256'],
                               'map_xy':paths[row['slot']]} for row in assigned],
               'monitor_path_equal_slot_0':paths[4]==paths[0],
               'readiness':startup.get('readiness')})
        atomic(args.output/'status.json',{'stage':'Path_published','Path_publications':1})
        override.provider=None
        command_tensor=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
        command_tensor[:4]=torch.as_tensor(applied,device=bare.device,dtype=command_tensor.dtype)
        override.set(command_tensor)
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
                'Path_publications':1,'trace_lengths':[len(trace) for trace in traces]})
            raise ValueError('paired diagnostic abort: '+reason)
        ended=[False]*4
        steps=g2_contract['control_steps']
        initial_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
        gates=[SameEpisodeGate(steps,initial_episode_length=x,
                               episode_cap_steps=int(bare.max_episode_length)) for x in initial_episode]
        trace_streams=[DurableTrace(args.output/f'trace-slot-{i}.jsonl') for i in range(4)]
        for stream in trace_streams:stream.__enter__()
        reset_calls=[]
        instrumented_reset=bare._reset_idx
        def detect_reset(env_ids):
            reset_calls.extend(int(i) for i in env_ids.detach().cpu().tolist())
            return instrumented_reset(env_ids)
        bare._reset_idx=detect_reset
        try:
            for step in range(steps):
                stamp=logical_t0+step*STEP_NS
                before_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
                with torch.inference_mode():
                    actions,latents,critic_values=capture_actor_critic(
                        policy,obs['policy'],obs['critic'],latent_layer=6)
                if tuple(actions.shape)!=(4,6) or not bool(torch.isfinite(actions).all()):
                    raise ValueError('Actor action shape/nonfinite')
                before=robot.data.root_state_w.detach().cpu().tolist()
                records=[]
                for i in range(4):
                    records.append({'control_step':step,'logical_sim_time_ns':stamp,
                        'follower_command':commands[i], 'applied_command':applied[i],
                        'condition':conditions[i],
                        'follower_diagnostic':diagnostics[i],
                        'bridge_acceptance':bridge_events[i],
                        'intervention_reason':interventions[i][1],
                        'actor_injected_noise':{name:values(sample[0]) for name,sample in noise_current.items()},
                        'joint_position':values(robot.data.joint_pos[i]),
                        'joint_velocity':values(robot.data.joint_vel[i]),
                        'applied_torque':values(robot.data.applied_torque[i]) if hasattr(robot.data,'applied_torque') else None,
                        'contact_net_forces_w':values(bare.scene['contact_forces'].data.net_forces_w[i]) if 'contact_forces' in bare.scene.sensors else None,
                        'policy_latency_seconds':float(bare._onix_latency_seconds[i]),
                        'actor_input':values(obs['policy'][i]),
                        'actor_pre_noise':values(observed['policy_pre_noise'][i]),
                        'actor_post_noise_pre_delay':values(observed['policy_post_noise_pre_delay'][i]),
                        'actor_action':values(actions[i]),
                        'critic_value':float(critic_values[i].reshape(-1)[0]),
                        'root_state':before[i], 'terminal':None,
                        'policy_action_mode':'act_inference'})
                action_batch=actions.clone()
                for i,done in enumerate(ended):
                    if done: action_batch[i]=0
                pairer.set_step(step+1)
                capture.begin_step()
                physics_begin_steady_ns=time.monotonic_ns()
                with torch.inference_mode():
                    next_obs,reward,done,_=env.step(action_batch)
                after_episode=[int(x) for x in bare.episode_length_buf[:4].detach().cpu().tolist()]
                observed=capture.finish_step(next_obs)
                noise_current=pairer.verify_step()
                for i in range(4):
                    if ended[i]:continue
                    record=records[i]
                    record['reward']=float(reward[i])
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
                        'clock_ns':bridge_response['clock_now_ns'],
                        'tf_ns':bridge_response['tf_stamp_ns'],
                        'command_ns':bridge_events[i]['stamp_ns'],
                        'bridge_accept_ns':bridge_events[i]['stamp_ns'],
                        'physics_input_ns':stamp,
                        'events':['clock','tf','command','bridge_acceptance','physics'],
                        'command_accepted_steady_ns':bridge_events[i]['steady_ns'],
                        'physics_begin_steady_ns':physics_begin_steady_ns,
                        'episode_length_before':before_episode[i],
                        'episode_length_after':after_episode[i],
                        'terminated':bool(bare.reset_terminated[i]),
                        'truncated':bool(bare.reset_time_outs[i]),
                        'reset_called':i in reset_calls,
                    }
                    traces[i].append(record)
                    trace_streams[i].append(record)
                    if bridge_events[i]['steady_ns']>physics_begin_steady_ns:
                        raise ValueError('command accepted after physics began')
                    gates[i].accept(record['timing_protocol'])
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
                       'elapsed_s':time.monotonic()-started,'Path_publications':1})
                if step+1<steps:
                    next_stamp=logical_t0+(step+1)*50_000_000
                    next_poses=poses()
                    response=bridge.request('step',stamp_ns=next_stamp,
                        poses=next_poses+[next_poses[0]],timeout_sec=3.)
                    bridge_response=response
                    commands=validate_bridge_response(response,next_stamp,branches=5)
                    if max(abs(a-b) for a,b in zip(commands[0],commands[4]))>1e-6:
                        raise ValueError('production follower monitor command mismatch')
                    diagnostics=response['follower_diagnostics']
                    if len(diagnostics)!=5:raise ValueError('follower diagnostic count')
                    bridge_events=accepted_events(response,next_stamp)
                    # G2 observes recovery timing; recovery is not a timing-protocol failure.
                    interventions=[intervene(commands[i],diagnostics[i],conditions[i],stamp_ns=next_stamp) for i in range(4)]
                    applied=[row[0] for row in interventions]
                    command_tensor=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
                    command_tensor[:4]=torch.as_tensor(applied,device=bare.device,dtype=command_tensor.dtype)
                    override.set(command_tensor)
        finally:
            bare._reset_idx=original_reset
            for stream in trace_streams:stream.__exit__(None,None,None)
        bridge_final=bridge.request('finish')
        if not all(gate.complete for gate in gates):
            raise ValueError('G2 incomplete contiguous trace')
        atomic(args.output/'pilot-summary.json',{'anchor_id':args.anchor_id,
               'batch_index':args.batch_index,'control_steps':steps,
               'trial_ids':[row['trial_id'] for row in assigned],
               'recorded_steps':[len(trace) for trace in traces],
               'termination':[trace[-1]['terminal'] if trace else None for trace in traces],
               'bridge_final':bridge_final})
        atomic(args.output/'complete.json',{'stage':'pilot_complete',
               'Path_publications':1,'settling_control_steps':20,
               'control_steps':steps,'anchor_id':args.anchor_id,'batch_index':args.batch_index,
               'initial_state_sha256':sha(args.output/'initial-state.json'),
               'terrain_sha256':sha(args.output/'terrain.json'),
               'prepath_gate_sha256':sha(args.output/'prepath-gate.json'),
               'path_publication_sha256':sha(args.output/'path-publication.json'),
               'pilot_summary_sha256':sha(args.output/'pilot-summary.json'),
               'contact_body_names_sha256':sha(args.output/'contact-body-names.json'),
               'trace_sha256':[sha(args.output/f'trace-slot-{i}.jsonl') for i in range(4)],
               'noise_contract':'keyed_original_additive_uniform_p1_v1'})
    finally:
        if capture is not None:capture.close()
        if pairer is not None:pairer.close()
        if bridge is not None:bridge.close()
        if override is not None: override.close()
        if env is not None: env.close()


try:
    main()
except BaseException as exc:
    args.output.mkdir(parents=True,exist_ok=True)
    atomic(args.output/'diagnostic-error.json',{'type':type(exc).__name__,
           'message':str(exc),'traceback':traceback.format_exc(),
           'Path_publications':int((args.output/'path-publication.json').exists())})
    raise
finally:
    simulation_app.close()
