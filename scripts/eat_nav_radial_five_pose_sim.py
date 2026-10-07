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
from pxr import UsdPhysics
from visualnav_transformer.eat_nav.radial_contact_audit import decode_contact_buffer, orientation_metrics
from visualnav_transformer.eat_nav.contact_preflight_protocol import normalize_view_filter_paths, atomic_json, run_calibration_with_failure_marker_before_close
from visualnav_transformer.eat_nav.radial_five_pose_contract import terrain_mesh_path
from visualnav_transformer.eat_nav.contact_raw_capture import diagnostic_should_stop, validate_view_capacity, verify_frozen_view_contract
from visualnav_transformer.eat_nav.radial_lean_log import LeanContactRecorder, append_step, save_dem_frame, save_actor, write_complete_before_close
from visualnav_transformer.eat_nav.radial_shard_inventory import select_shard_rows, sensor_paths_for_env
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import (
    EATTerrainGenerator, make_collection_cfg,
)
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.radial_inventory import (
    TERRAIN_ORDER, build_pose_schedule, pose_world, terrain_seed, validate_record,
)


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
    if digest(__file__) != freeze['sim_source_sha256']:
        raise ValueError('simulator SHA drift')
    if digest(args.pose_manifest) != freeze['pose_manifest_sha256']:
        raise ValueError('pose manifest SHA drift')
    if args.batch_seed != 20261007 or args.batch_seed not in freeze['batch_seeds']:
        raise ValueError('batch seed drift')
    for label,path in freeze['source_paths'].items():
        if digest(path) != freeze['source_sha256'][label]:
            raise ValueError('input SHA drift: '+label)
    all_rows = [r for r in build_pose_schedule([args.batch_seed])]
    selected = select_shard_rows(manifest['poses'], all_rows)
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
        reset_state={r['pose_id']:state_snapshot(bare,r['env_index'],pose_world(origins[r['env_index']],r['angle_index'])) for r in selected}
        root=robot.data.root_state_w.clone()
        ids=torch.as_tensor([r['env_index'] for r in selected],device=bare.device)
        positions=root[ids,:3].clone(); quats=root[ids,3:7].clone()
        requested={}
        for j,row in enumerate(selected):
            i=row['env_index']; p=pose_world(origins[i],row['angle_index'])
            requested[row['pose_id']]=p
            expected_z=float(origins[i,2]+robot.data.default_root_state[i,2])
            if abs(float(root[i,2])-expected_z)>1e-4 or abs(float(robot.data.default_root_state[i,2])-.35)>1e-4:
                raise ValueError('RL reset z contract mismatch: '+row['pose_id'])
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
        original_update=bare.scene.update
        physics_count=0
        def measured_update(dt):
            nonlocal physics_count
            original_update(dt)
            physics_count+=1
            snapshots={r['pose_id']:state_snapshot(bare,r['env_index'],requested[r['pose_id']])
                       for r in selected}
            atomic_json(args.output/'heartbeat.json',{'physics_step':physics_count,
                'monotonic_ns':time.monotonic_ns(),'elapsed_sec':time.monotonic()-started})
            for r in selected:
                name=r['pose_id']
                arrays=[a.detach().cpu().numpy() for a in views[name].get_contact_data(dt=0.005)]
                contact=recorder.record(name,physics_count,arrays,views[name].max_contact_data_count)
                if contact['contact_count']>0 and first_contacts[name] is None:
                    first_contacts[name]=physics_count
                sample={'physics_step':physics_count,'logical_time_s':physics_count*.005,
                        'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
                        'phase':'FIRST_CONTACT_TRANSIENT' if physics_count==1 else
                            'SETTLED_ANCHOR' if physics_count>160 else 'SETTLING',
                        'state':snapshots[name],
                        'contact':contact}
                append_step(args.output/'trace'/f'{name}.jsonl',sample)
        bare.scene.update=measured_update
        zero=torch.zeros((112,6),device=bare.device)
        frame_steps=(4,9,14,19)
        control_count=6 if args.mode=='diagnostic' else 20
        for control_step in range(control_count):
            obs,reward,done,extra=env.step(zero)
            if bool(torch.any(done)):
                raise RuntimeError('unexpected episode reset during settling')
            if physics_count!=(control_step+1)*10:
                raise RuntimeError('physics/control decimation mismatch')
            if control_step in frame_steps:
                for r in selected:
                    frame={'control_step':control_step,
                        'physics_step':physics_count,'logical_stamp_ns':physics_count*5_000_000,
                        'frame_id':config.reference_frame,'resolution_m':config.dem_resolution_m,
                        'axis':'robot-relative-grid-xy','dem':extract_dem(bare,r['env_index'],config)}
                    payload={**frame['dem'],**{k:v for k,v in frame.items() if k!='dem'}}
                    save_dem_frame(args.output,r['pose_id'],frame_steps.index(control_step),payload)
        bare.scene.update=original_update
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
            'elapsed_sec':time.monotonic()-started,'sim_source_sha256':digest(__file__)},env.close)
        env=None
    finally:
        if 'recorder' in locals() and physics_count<200:
            recorder.save_last_failure()
        if env is not None:
            env.close()


run_calibration_with_failure_marker_before_close(simulation_app, main, args.output)
