"""One-shot candidate-free radial terrain inventory; never imports ROS or Path clients."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--freeze', type=Path, required=True)
parser.add_argument('--batch-seed', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--pose-manifest', type=Path, required=True)
parser.add_argument('--mode', choices=('diagnostic','full'), required=True)
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
from pxr import UsdGeom, UsdPhysics
from visualnav_transformer.eat_nav.radial_contact_audit import decode_contact_buffer, orientation_metrics
from visualnav_transformer.eat_nav.contact_preflight_protocol import normalize_view_filter_paths, atomic_json, run_calibration_with_failure_marker_before_close
from visualnav_transformer.eat_nav.radial_five_pose_contract import terrain_mesh_path
from visualnav_transformer.eat_nav.contact_raw_capture import diagnostic_should_stop, validate_view_capacity, verify_frozen_view_contract
from visualnav_transformer.eat_nav.radial_lean_log import LeanContactRecorder, append_step, save_dem_frame, save_actor, write_complete_before_close
from visualnav_transformer.eat_nav.radial_shard_inventory import sensor_paths_for_env
from visualnav_transformer.eat_nav.r3_anchor_contract import select_r3_shard_rows, requested_world
from visualnav_transformer.eat_nav.radial_termination_capture import build_reset_events
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import (
    EATTerrainGenerator, make_collection_cfg,
)
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, terrain_seed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


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


def extract_dem(env, i, config):
    robot, scanner = env.scene['robot'], env.scene['height_scanner_eat']
    pos = robot.data.root_pos_w[i].detach().cpu().numpy()
    _, _, yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
    yaw = float(yaw_tensor.item())
    hits = scanner.data.ray_hits_w[i, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hits)
    start = scanner._ray_starts_w[i, 0, :2].detach().cpu().numpy()
    source = SourceGrid(height=np.where(valid, hits, 0).astype(np.float32), valid=valid,
                        origin_xy=(float(start[0]), float(start[1])), resolution_m=config.dem_resolution_m,
                        start_index_xy=(0, 0), frame_id=config.reference_frame, stamp_ns=0)
    transformed = np.asarray(robot_dem(source, (float(pos[0]), float(pos[1]), yaw), config), dtype=np.float32)
    if transformed.shape != (2, 20, 20):
        raise ValueError('transformed DEM contract changed')
    if not np.isfinite(transformed).all():
        raise ValueError('DEM nonfinite after transform')
    return {'source_height': source.height.tolist(), 'source_mask': valid.astype(int).tolist(),
            'source_origin_xy': list(source.origin_xy), 'transformed': transformed.tolist(),
            'transformed_sha256': hashlib.sha256(transformed.tobytes()).hexdigest(),
            'valid_fraction': float(np.mean(transformed[1] > .5))}




def vector(value):
    return value.detach().cpu().tolist()


def finite(value):
    return bool(np.isfinite(np.asarray(value, dtype=float)).all())


def state_snapshot(bare, index, requested):
    robot = bare.scene['robot']
    root = vector(robot.data.root_state_w[index])
    joint_pos = vector(robot.data.joint_pos[index])
    joint_vel = vector(robot.data.joint_vel[index])
    gravity = vector(robot.data.projected_gravity_b[index])
    mode = getattr(bare, '_onix_flipper_fold_mode', None)
    mode_value = None if mode is None else float(mode[index].item())
    orient = orientation_metrics(root[3:7], requested['world_yaw'])
    result = {'root_state_w': root, 'root_quaternion_order': 'wxyz',
              'requested_pose': requested,
              'xy_drift_m': float(np.linalg.norm(np.asarray(root[:2]) -
                            [requested['world_x'], requested['world_y']])),
              'joint_position': joint_pos, 'joint_velocity': joint_vel,
              'joint_names': list(robot.data.joint_names),
              'flipper_fold_mode_latch': mode_value,
              'projected_gravity_b': gravity,
              'linear_speed_mps': float(np.linalg.norm(root[7:10])),
              'angular_speed_radps': float(np.linalg.norm(root[10:13])),
              **orient}
    if not all(finite(result[k]) for k in ('root_state_w','joint_position','joint_velocity','projected_gravity_b')):
        raise ValueError('nonfinite root/joint/gravity state')
    if mode_value is None or not math.isfinite(mode_value):
        raise ValueError('flipper fold-mode latch missing')
    return result


def main():
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'owned-process.json', {
        'pid': os.getpid(),
        'proc_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19]),
        'candidate_free': True})
    atomic_json(args.output/'in-process-started.json', {'wall_ns':time.time_ns(),
        'monotonic_ns':time.monotonic_ns(), 'physics_steps':0})
    freeze = json.loads(args.freeze.read_text())
    if args.mode != freeze['run_kind']:
        raise ValueError('run mode differs from freeze')
    view_capacity = validate_view_capacity(args.mode, freeze['contact_contract']['view_capacity'],
        freeze['contact_contract'].get('observed_required_entries'))
    manifest = json.loads(args.pose_manifest.read_text())
    primary = json.loads(Path(freeze['primary_manifest_path']).read_text())['poses']
    if digest(__file__) != freeze['sim_source_sha256']:
        raise ValueError('simulator SHA drift')
    if digest(args.pose_manifest) != freeze['pose_manifest_sha256']:
        raise ValueError('pose manifest SHA drift')
    if digest(freeze['primary_manifest_path']) != freeze['primary_manifest_sha256']:
        raise ValueError('primary manifest SHA drift')
    if args.batch_seed != 20261007 or args.batch_seed not in freeze['batch_seeds']:
        raise ValueError('batch seed drift')
    for label,path in freeze['source_paths'].items():
        if digest(path) != freeze['source_sha256'][label]:
            raise ValueError('input SHA drift: '+label)
    selected = select_r3_shard_rows(manifest['poses'], primary)
    config = load_eat_config(Path(freeze['source_paths']['eat_config']))
    saved = yaml.load(Path(freeze['source_paths']['checkpoint_env']).read_text(), Loader=yaml.BaseLoader)
    toggles = {name:saved[name].lower()=='true' for name in
        ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), config, args.batch_seed,112)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type=RadialTerrainGenerator
    gen.num_rows,gen.num_cols=16,7
    gen.curriculum=False
    gen.difficulty_range=(.5,.5)
    gen.use_cache=False
    gen.seed=args.batch_seed
    gen.sub_terrains={name:gen.sub_terrains[name] for name in TERRAIN_ORDER}
    cfg.scene.terrain.max_init_terrain_level=15
    env=None
    try:
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare=env.unwrapped
        metadata=bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata
        origins=bare.scene.env_origins.detach().cpu().numpy()
        generated=bare.scene.terrain.terrain_origins.detach().cpu().numpy()
        for row in selected:
            i=row['env_index']; m=metadata[i]
            if any(m[k]!=row[v] for k,v in [('row','row'),('column','column'),('seed','terrain_seed'),('terrain_type','terrain')]):
                raise ValueError('terrain metadata mismatch '+row['pose_id'])
            if float(m['difficulty'])!=.5 or not np.allclose(origins[i],generated[row['row'],row['column']],atol=1e-5):
                raise ValueError('terrain identity/origin mismatch '+row['pose_id'])
        obs,_=env.reset()
        robot=bare.scene['robot']
        terrain_path=terrain_mesh_path('/World/ground/terrain')
        terrain_prim=omni.usd.get_context().get_stage().GetPrimAtPath(terrain_path)
        if not terrain_prim.IsValid() or not terrain_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('terrain collider missing or not CollisionAPI')
        terrain_mesh=UsdGeom.Mesh(terrain_prim)
        points=np.asarray(terrain_mesh.GetPointsAttr().Get(),dtype=np.float64)
        faces=np.asarray(terrain_mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64)
        actual_mesh_sha=hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()
        if actual_mesh_sha != freeze['terrain_mesh_sha256']:
            raise ValueError('terrain mesh SHA drift before physics')
        atomic_json(args.output/'terrain-mesh-identity.json',{
            'sha256':actual_mesh_sha,'vertex_count':len(points),'face_index_count':len(faces)})
        sim_view=SimulationManager.get_physics_sim_view()
        views={}
        for row in selected:
            i=row['env_index']
            view=sim_view.create_rigid_contact_view(
                f'/World/envs/env_{i}/onix_chassis/*',
                filter_patterns=[terrain_path], max_contact_data_count=view_capacity)
            sensors=list(view.sensor_paths)
            filters=normalize_view_filter_paths(list(view.filter_paths),[terrain_path],len(sensors))
            if not sensors or filters!=[terrain_path] or any(f'/env_{i}/onix_chassis/' not in s for s in sensors):
                raise ValueError('ambiguous Onix contact sensor paths: '+repr(sensors))
            if args.mode=='full':
                expected_sensors=freeze['contact_contract'].get('sensor_paths_by_pose',{}).get(row['pose_id'])
                if expected_sensors is None:
                    expected_sensors=sensor_paths_for_env(freeze['contact_contract']['sensor_path_template'],i)
                verify_frozen_view_contract(sensors, expected_sensors,
                    filters, freeze['contact_contract']['filter_paths'],
                    freeze['contact_contract']['expected_array_shapes'], freeze['contact_contract']['expected_array_shapes'])
            views[row['pose_id']]=view
        reset_state={r['pose_id']:state_snapshot(bare,r['env_index'],requested_world(r,origins[r['env_index']])) for r in selected}
        root=robot.data.root_state_w.clone()
        ids=torch.as_tensor([r['env_index'] for r in selected],device=bare.device)
        positions=root[ids,:3].clone(); quats=root[ids,3:7].clone()
        requested={}
        for j,row in enumerate(selected):
            i=row['env_index']; p=requested_world(row,origins[i])
            requested[row['pose_id']]=p
            expected_z=float(origins[i,2]+robot.data.default_root_state[i,2])
            if abs(float(root[i,2])-expected_z)>1e-4 or abs(float(robot.data.default_root_state[i,2])-.35)>1e-4:
                raise ValueError('RL reset z contract mismatch: '+row['pose_id'])
            if not torch.all(root[i,7:13] == 0):
                raise ValueError('RL reset root velocity must be zero: '+row['pose_id'])
            positions[j,0]=p['world_x']; positions[j,1]=p['world_y']
            theta=torch.as_tensor(p['world_yaw'],device=bare.device,dtype=root.dtype)
            quats[j]=math_utils.quat_from_euler_xyz(theta*0,theta*0,theta)
        robot.write_root_pose_to_sim(torch.cat([positions,quats],dim=1),env_ids=ids)
        bare.sim.forward()
        pre={r['pose_id']:state_snapshot(bare,r['env_index'],requested[r['pose_id']]) for r in selected}
        for r in selected:
            name=r['pose_id']; a=reset_state[name]; b=pre[name]
            if a['root_state_w'][2]!=b['root_state_w'][2] or a['joint_position']!=b['joint_position'] or a['joint_velocity']!=b['joint_velocity'] or a['root_state_w'][7:13]!=b['root_state_w'][7:13]:
                raise ValueError('radial placement altered non-XY/yaw state: '+name)
        atomic_json(args.output/'reset-pre-physics.json', {'reset':reset_state,'radial':pre,
            'requested':requested, 'terrain_metadata':{r['pose_id']:metadata[r['env_index']] for r in selected},
            'physics_steps':0})
        # Compute diagnostics without advancing the simulation. History update disabled.
        preobs=bare.observation_manager.compute(update_history=False)['policy']
        atomic_json(args.output/'prephysics-observation.json',
            {r['pose_id']:{'shape':list(preobs[r['env_index']].shape),'values':vector(preobs[r['env_index']]),
                           'finite':finite(vector(preobs[r['env_index']]))} for r in selected})
        term_names=bare.observation_manager._group_obs_term_names['policy']
        term_dims=bare.observation_manager._group_obs_term_dim['policy']
        actor_order=[f'{name}[{k}]' for name,dim in zip(term_names,term_dims)
                     for k in range(int(np.prod(dim)) if isinstance(dim,(tuple,list)) else int(dim))]
        if len(actor_order)!=22:
            raise ValueError('Actor field order is not 22 components')
        for r in selected:
            save_actor(args.output,r['pose_id'],'prephysics',vector(preobs[r['env_index']]),
                       actor_order,freeze['source_sha256']['checkpoint_env'])
        atomic_json(args.output/'contact-view.json', {r['pose_id']:{
            'sensor_paths':list(views[r['pose_id']].sensor_paths),
            'filter_paths':[terrain_path], 'capacity':views[r['pose_id']].max_contact_data_count}
            for r in selected})
        paths={r['pose_id']:(list(views[r['pose_id']].sensor_paths),[terrain_path]) for r in selected}
        recorder=LeanContactRecorder(args.output,paths,
            expected_shapes=freeze['contact_contract']['expected_array_shapes'])
        first_contacts={r['pose_id']:None for r in selected}
        selected_by_env={r['env_index']:r['pose_id'] for r in selected}
        active=set(selected_by_env.values())
        terminal_events={}
        reset_counts={i:0 for i in selected_by_env}
        last_contacts={}
        trace_steps={i:0 for i in selected_by_env}
        dem_frame_ids={i:[] for i in selected_by_env}
        original_update=bare.scene.update
        physics_count=0
        def measured_update(dt):
            nonlocal physics_count
            original_update(dt)
            physics_count+=1
            snapshots={r['pose_id']:state_snapshot(bare,r['env_index'],requested[r['pose_id']])
                       for r in selected if r['pose_id'] in active}
            atomic_json(args.output/'heartbeat.json',{'physics_step':physics_count,
                'monotonic_ns':time.monotonic_ns(),'elapsed_sec':time.monotonic()-started})
            for r in selected:
                name=r['pose_id']
                if name not in active:
                    continue
                arrays=[a.detach().cpu().numpy() for a in views[name].get_contact_data(dt=0.005)]
                contact=recorder.record(name,physics_count,arrays,views[name].max_contact_data_count)
                last_contacts[r['env_index']]=contact
                trace_steps[r['env_index']]=physics_count
                if contact['contact_count']>0 and first_contacts[name] is None:
                    first_contacts[name]=physics_count
                sample={'physics_step':physics_count,'logical_time_s':physics_count*.005,
                        'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
                        'phase':'FIRST_CONTACT_TRANSIENT' if physics_count==1 else
                            'SETTLED_ANCHOR' if physics_count>160 else 'SETTLING',
                        'state':snapshots[name], 'episode_id':reset_counts[r['env_index']],
                        'reset_counter':reset_counts[r['env_index']],
                        'post_reset_state':False, 'termination_phase':'pre_evaluation',
                        'contact':contact}
                append_step(args.output/'trace'/f'{name}.jsonl',sample)
        bare.scene.update=measured_update
        original_reset_idx=bare._reset_idx
        def capture_before_reset(env_ids):
            done_ids=[int(i) for i in env_ids.detach().cpu().tolist()]
            manager=bare.termination_manager
            terminated_ids=bare.reset_terminated.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            truncated_ids=bare.reset_time_outs.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            term_flags={name:manager.get_term(name).nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                        for name in manager.active_terms}
            timeout_terms={name for name,cfg in zip(manager.active_terms,manager._term_cfgs) if cfg.time_out}
            selected_done=[i for i in done_ids if i in selected_by_env and selected_by_env[i] in active]
            pre_states={i:state_snapshot(bare,i,requested[selected_by_env[i]]) for i in selected_done}
            pre_actor=bare.observation_manager.compute(update_history=False)['policy'] if selected_done else None
            actor_values={i:vector(pre_actor[i]) for i in selected_done}
            active_by_env={i:name for i,name in selected_by_env.items() if name in active}
            events=build_reset_events(selected=active_by_env,physics_step=physics_count,
                done_ids=done_ids,terminated_ids=terminated_ids,truncated_ids=truncated_ids,
                term_flags=term_flags,timeout_terms=timeout_terms,pre_states=pre_states,
                actor_values=actor_values,
                episode_lengths={i:int(bare.episode_length_buf[i].item()) for i in selected_done},
                reset_counts=reset_counts,trace_steps=trace_steps,dem_frame_ids=dem_frame_ids,
                contacts=last_contacts)
            for event in events:
                name=event['pose_id']; i=event['env_id']
                event['captured_wall_ns']=time.time_ns()
                event['trace_sha256']=digest(args.output/'trace'/f'{name}.jsonl')
                atomic_json(args.output/'termination'/f'{name}.json',event)
                terminal_events[name]=event
                active.remove(name)
                reset_counts[i]+=1
            atomic_json(args.output/'control-reset'/f'step-{physics_count:04d}.json',{
                'physics_step':physics_count,'logical_timestamp_ns':physics_count*5_000_000,
                'all_done_env_ids':done_ids,'selected_termination_pose_ids':[e['pose_id'] for e in events],
                'terminated_env_ids':terminated_ids,'truncated_env_ids':truncated_ids,
                'term_flags':term_flags,'post_reset_state':False})
            return original_reset_idx(env_ids)
        bare._reset_idx=capture_before_reset
        zero=torch.zeros((112,6),device=bare.device)
        frame_steps=(4,9,14,19)
        control_count=6 if args.mode=='diagnostic' else 20
        for control_step in range(control_count):
            obs,reward,done,extra=env.step(zero)
            if physics_count!=(control_step+1)*10:
                raise RuntimeError('physics/control decimation mismatch')
            done_ids=done.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            if done_ids and not (args.output/'control-reset'/f'step-{physics_count:04d}.json').exists():
                raise RuntimeError('reset occurred without pre-reset capture')
            selected_control={}
            for i,name in selected_by_env.items():
                terminal=terminal_events.get(name)
                just_terminated=terminal is not None and terminal['termination_step']==physics_count
                if terminal is not None and not just_terminated:
                    continue
                selected_control[name]={
                    'env_id':i,'episode_id':0,'reset_counter':0,
                    'terminated':bool(just_terminated),'truncated':False,'done':bool(just_terminated),
                    'term_flags':terminal['term_flags'] if just_terminated else
                        {term:False for term in bare.termination_manager.active_terms},
                    'termination_reason':terminal['termination_reason'] if just_terminated else [],
                    'state':terminal['pre_reset_state'] if just_terminated else state_snapshot(bare,i,requested[name]),
                    'actor_observation':terminal['pre_reset_actor'] if just_terminated else vector(obs['policy'][i]),
                    'contact':last_contacts[i],
                    'dem_frame_ids':list(dem_frame_ids[i]),
                    'post_reset_state':False}
            append_step(args.output/'control-events.jsonl',{
                'control_step':control_step+1,'physics_step':physics_count,
                'logical_timestamp_ns':physics_count*5_000_000,
                'all_done_env_ids':done_ids,
                'selected_active_pose_ids':sorted(active),
                'selected_terminal_pose_ids':sorted(terminal_events),
                'selected':selected_control,
                'post_reset_observation_excluded_for_terminated':True})
            if control_step in frame_steps:
                for r in selected:
                    if r['pose_id'] not in active:
                        continue
                    frame={'control_step':control_step,
                        'physics_step':physics_count,'logical_stamp_ns':physics_count*5_000_000,
                        'frame_id':config.reference_frame,'resolution_m':config.dem_resolution_m,
                        'axis':'robot-relative-grid-xy','dem':extract_dem(bare,r['env_index'],config)}
                    payload={**frame['dem'],**{k:v for k,v in frame.items() if k!='dem'}}
                    save_dem_frame(args.output,r['pose_id'],frame_steps.index(control_step),payload)
                    dem_frame_ids[r['env_index']].append(f'frame-{frame_steps.index(control_step):02d}')
        bare.scene.update=original_update
        bare._reset_idx=original_reset_idx
        expected_steps=60 if args.mode=='diagnostic' else 200
        if physics_count!=expected_steps:
            raise RuntimeError('physics step count mismatch')
        if args.mode=='diagnostic':
            if not diagnostic_should_stop(physics_count,invalid=False):
                raise RuntimeError('diagnostic budget not reached')
            atomic_json(args.output/'diagnostic-finished.json',{'physics_steps':physics_count,
                'first_contact_step':min((v for v in first_contacts.values() if v is not None),default=None),
                'sim_source_sha256':digest(__file__)})
            return
        actor=obs['policy']
        for r in selected:
            name=r['pose_id']; i=r['env_index']
            if name in terminal_events:
                event=terminal_events[name]
                atomic_json(args.output/'poses'/f'{name}.json',{
                    'pose_id':name,'env_index':i,'classification':'anchor_ineligible_physical_termination',
                    'termination_path':f'termination/{name}.json',
                    'trace_path':f'trace/{name}.jsonl',
                    'dem_paths':[f'dem/{name}/{f}.json' for f in event['dem_frame_ids']],
                    'physics_step_count':event['termination_step'],'candidate_free':True})
                continue
            save_actor(args.output,name,'settled_step_200',vector(actor[i]),
                       actor_order,freeze['source_sha256']['checkpoint_env'])
            atomic_json(args.output/'poses'/f'{name}.json', {
                'pose_id':name,'env_index':i,'reset_state':reset_state[name],
                'prephysics_state':pre[name],'trace_path':f'trace/{name}.jsonl',
                'dem_paths':[f'dem/{name}/frame-{j:02d}.json' for j in range(4)],
                'first_contact_step':first_contacts[name],
                'settled_actor_path':f'actor/{name}/settled_step_200.json',
                'action':vector(zero[i]),'control_step_count':20,'physics_step_count':200,
                'candidate_free':True})
        write_complete_before_close(args.output,{'pose_count':len(selected),'physics_steps':physics_count,
            'physical_termination_pose_ids':sorted(terminal_events),
            'elapsed_sec':time.monotonic()-started,'sim_source_sha256':digest(__file__)},env.close)
        env=None
    finally:
        if 'recorder' in locals() and physics_count<200:
            recorder.save_last_failure()
        if env is not None:
            env.close()


run_calibration_with_failure_marker_before_close(simulation_app, main, args.output)
