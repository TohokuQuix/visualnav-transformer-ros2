"""Three-env, candidate-free P1 stepwise source/clone state diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils import math as math_utils
from isaacsim.core.simulation_manager import SimulationManager
from p1_contact_summary import summarize_contact_arrays
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride
from isaaclab.utils.math import euler_xyz_from_quat


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def vector(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def actor_values(x):
    raw = np.asarray(vector(x), dtype=np.float64)
    if raw.shape != (22,):
        raise ValueError(f'Actor shape mismatch: {raw.shape}')
    finite = np.isfinite(raw)
    return {'values': [float(v) if ok else None for v, ok in zip(raw, finite)],
            'finite': bool(finite.all()), 'nonfinite_indices': np.flatnonzero(~finite).tolist()}


manifest = json.loads(args.manifest.read_text())
for key, path_key in [('source_sha256', 'source_path'), ('fixture_sha256', 'fixture_path'),
                      ('eat_config_sha256', 'eat_config_path'), ('checkpoint_env_sha256', 'checkpoint_env_path')]:
    if sha(manifest[path_key]) != manifest[key]:
        raise ValueError(f'freeze SHA mismatch: {key}')
if sha(__file__) != manifest['source_sha256']:
    raise ValueError('running source SHA mismatch')
if sha(Path(__file__).with_name('p1_contact_summary.py')) != manifest['contact_helper_sha256']:
    raise ValueError('contact helper SHA mismatch')
if manifest['mode'] != 'R4_global_shift' or manifest['num_envs'] != 3 or manifest['physics_steps'] != 200:
    raise ValueError('diagnostic contract drift')
fixture = json.loads(Path(manifest['fixture_path']).read_text())[manifest['anchor_id']]


class ThreeSameTerrain(EATTerrainGenerator):
    """Three exact deterministic tiles with one frozen k_rail_step_field seed."""

    last_canonical = None

    def __init__(self, cfg, device):
        self.canonical = []
        super().__init__(cfg, device)
        shift_xy = np.asarray(manifest['global_shift_xy'], dtype=np.float64)
        if shift_xy.shape != (2,) or not np.isfinite(shift_xy).all():
            raise ValueError('invalid frozen global shift')
        transform = np.eye(4)
        transform[:2, 3] = shift_xy
        self.terrain_mesh.apply_transform(transform)
        self.terrain_origins[:, :, :2] += shift_xy
        for patches in self.flat_patches.values():
            patches[:, :, :, :2] += torch.as_tensor(shift_xy, device=patches.device, dtype=patches.dtype)
        ThreeSameTerrain.last_canonical = list(self.canonical)

    def _generate_random_terrains(self):
        if (self.cfg.num_rows, self.cfg.num_cols) != (1, 3):
            raise ValueError('minimal tile layout drift')
        cfg = self.cfg.sub_terrains['k_rail_step_field']
        for col in range(3):
            mesh, origin = self._get_terrain_mesh(.5, cfg)
            self._add_sub_terrain(mesh, origin, 0, col, cfg)

    def _get_terrain_mesh(self, difficulty, cfg):
        index = self._tile_index
        old_seed, old_np = self.cfg.seed, np.random.get_state()
        try:
            self.cfg.seed = int(manifest['terrain_seed'])
            np.random.seed(int(manifest['terrain_seed']))
            mesh, origin = TerrainGenerator._get_terrain_mesh(self, difficulty, cfg)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            self.canonical.append({'tile_index': index, 'seed': int(manifest['terrain_seed']),
                                   'difficulty': float(difficulty), 'origin_local': np.asarray(origin).tolist(),
                                   'vertex_count': len(vertices), 'triangle_count': len(faces),
                                   'tile_local_mesh_sha256': hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()})
            self.tile_metadata.append({'tile_index': index, 'row': 0, 'column': index,
                                       'seed': int(manifest['terrain_seed']),
                                       'terrain_type': 'k_rail_step_field', 'difficulty': float(difficulty)})
            self._tile_index += 1
            return mesh, origin
        finally:
            self.cfg.seed = old_seed
            np.random.set_state(old_np)


def item(tensor, index):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        if tensor.ndim == 0:
            return tensor.item()
        if tensor.shape[0] == 3:
            return vector(tensor[index])
    return None


def snapshot(bare, index, step, boundary):
    robot = bare.scene['robot']
    action = bare.action_manager
    terms = {}
    for name, term in action._terms.items():
        terms[name] = {field: item(getattr(term, field, None), index) for field in
                       ('_raw_actions', '_processed_actions', '_previous_targets',
                        '_pre_clip_processed_actions', '_travel_direction', '_drives_initialized')}
    buffers = {}
    for name, buffer in bare._action_latency_buffers.items():
        circ = buffer._circular_buffer
        data = circ._buffer
        buffers[name] = {'lag': item(buffer._time_lags, index),
                         'num_pushes': item(circ._num_pushes, index),
                         'pointer': circ._pointer,
                         'stored': vector(data[:, index]) if data is not None else None}
    sensor = bare.scene['contact_forces'] if bare.scene.cfg.contact_forces is not None else None
    contact = item(getattr(getattr(sensor, 'data', None), 'net_forces_w', None), index) if sensor is not None else None
    command = bare.command_manager._terms['base_velocity'].vel_command_b
    return {'physics_step': step, 'boundary': boundary,
            'root_state_w': vector(robot.data.root_state_w[index]),
            'joint_names': list(robot.data.joint_names),
            'joint_position': vector(robot.data.joint_pos[index]),
            'joint_velocity': vector(robot.data.joint_vel[index]),
            'joint_position_target': item(getattr(robot.data, 'joint_pos_target', None), index),
            'joint_velocity_target': item(getattr(robot.data, 'joint_vel_target', None), index),
            'applied_torque': item(getattr(robot.data, 'applied_torque', None), index),
            'flipper_latch': item(getattr(bare, '_onix_flipper_fold_mode', None), index),
            'action': item(action.action, index), 'previous_action': item(action.prev_action, index),
            'action_terms': terms, 'latency_buffers': buffers,
            'latency_substeps': item(getattr(bare, '_latency_substeps', None), index),
            'latency_policy_obs_prev': item(getattr(bare, '_latency_policy_obs_prev', None), index),
            'contact_force': contact, 'command': item(command, index)}


def dem(bare, index, config):
    robot, scanner = bare.scene['robot'], bare.scene['height_scanner_eat']
    pos = robot.data.root_pos_w[index].detach().cpu().numpy()
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w[index:index+1])
    hits = scanner.data.ray_hits_w[index, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hits)
    start = scanner._ray_starts_w[index, 0, :2].detach().cpu().numpy()
    source = SourceGrid(height=np.where(valid, hits, 0).astype(np.float32), valid=valid,
                        origin_xy=(float(start[0]), float(start[1])),
                        resolution_m=config.dem_resolution_m, start_index_xy=(0, 0),
                        frame_id=config.reference_frame, stamp_ns=0)
    transformed = np.asarray(robot_dem(source, (float(pos[0]), float(pos[1]), float(yaw_t.item())), config), dtype=np.float32)
    return {'transformed': transformed.tolist(), 'finite': bool(np.isfinite(transformed).all()),
            'source_mask': valid.astype(int).tolist(), 'source_origin_xy': list(source.origin_xy)}


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pid = os.getpid()
    tick = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    atomic(args.output/'owned-process.json', {'pid': pid, 'start_tick': tick,
                                             'source_sha256': sha(__file__), 'Path_publications': 0})
    atomic(args.output/'status.json', {'stage': 'startup', 'Path_publications': 0})
    config = load_eat_config(Path(manifest['eat_config_path']))
    saved = yaml.load(Path(manifest['checkpoint_env_path']).read_text(), Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), config, manifest['batch_seed'], 3)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type = ThreeSameTerrain
    gen.num_rows, gen.num_cols = 1, 3
    gen.curriculum = False
    gen.difficulty_range = (.5, .5)
    gen.use_cache = False
    gen.seed = manifest['batch_seed']
    gen.sub_terrains = {name: gen.sub_terrains[name] for name in ('k_rail_step_field',)}
    cfg.scene.terrain.max_init_terrain_level = 0
    env = None
    override = None
    try:
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
        bare = env.unwrapped
        canonical = ThreeSameTerrain.last_canonical
        if canonical is None:
            raise ValueError('canonical tile mesh unavailable from live generator')
        if len({x['tile_local_mesh_sha256'] for x in canonical}) != 1:
            raise ValueError('deterministic terrain tile geometry mismatch')
        origins = bare.scene.env_origins.detach().cpu().numpy()
        atomic(args.output/'terrain.json', {'canonical_tiles': canonical, 'world_tile_origins': origins.tolist(),
                                            'physics_material_cfg': str(cfg.scene.terrain.physics_material),
                                            'collision_group': str(cfg.scene.terrain.collision_group),
                                            'scene_num_envs': 3, 'tile_layout': [1,3],
                                            'sim_dt': float(bare.physics_dt), 'control_dt': float(bare.step_dt)})
        capacity = int(manifest['contact_capacity'])
        if capacity != 4096:
            raise ValueError('contact view capacity drift')
        sim_view = SimulationManager.get_physics_sim_view()
        terrain_path = '/World/ground/terrain/mesh'
        views = {}
        view_contract = []
        for i in range(3):
            view = sim_view.create_rigid_contact_view(
                f'/World/envs/env_{i}/onix_chassis/*',
                filter_patterns=[terrain_path], max_contact_data_count=capacity)
            sensors = list(view.sensor_paths)
            filters = list(view.filter_paths)
            if not sensors or any(f'/env_{i}/onix_chassis/' not in sensor for sensor in sensors):
                raise ValueError(f'contact sensor paths invalid for env {i}')
            views[i] = view
            view_contract.append({'env': i, 'sensor_paths': sensors,
                                  'filter_paths': filters, 'capacity': view.max_contact_data_count})
        atomic(args.output/'contact-view-contract.json', view_contract)
        env.reset()
        atomic(args.output/'after-reset.json', [snapshot(bare, i, 0, 'after_reset') for i in range(3)])
        robot = bare.scene['robot']
        ids = torch.arange(3, device=bare.device)
        source = fixture['prephysics_state']
        if source['joint_names'] != list(robot.data.joint_names):
            raise ValueError('joint name order mismatch')
        root = robot.data.root_state_w.clone()
        positions, quats = root[ids, :3].clone(), root[ids, 3:7].clone()
        for i in range(3):
            positions[i, 0] = float(origins[i, 0]) + manifest['source_anchor_local_xy'][0]
            positions[i, 1] = float(origins[i, 1]) + manifest['source_anchor_local_xy'][1]
            yaw = torch.as_tensor(manifest['source_anchor_yaw'], device=bare.device, dtype=root.dtype)
            quats[i] = math_utils.quat_from_euler_xyz(yaw*0, yaw*0, yaw)
            if manifest['mode'] == 'R1':
                positions[i, 2] = float(source['root_state_w'][2])
        robot.write_root_pose_to_sim(torch.cat([positions, quats], dim=1), env_ids=ids)
        if manifest['mode'] == 'R1':
            velocity = torch.as_tensor([source['root_state_w'][7:13]]*3, device=bare.device,
                                       dtype=root.dtype)
            robot.write_root_velocity_to_sim(velocity, env_ids=ids)
        joints = torch.as_tensor([source['joint_position']]*3, device=bare.device,
                                 dtype=robot.data.joint_pos.dtype)
        speeds = torch.as_tensor([source['joint_velocity']]*3, device=bare.device,
                                 dtype=robot.data.joint_vel.dtype)
        robot.write_joint_state_to_sim(joints, speeds, env_ids=ids)
        bare._onix_flipper_fold_mode[ids] = float(source['flipper_fold_mode_latch'])
        bare.sim.forward()
        if manifest['mode'] == 'R4_global_shift':
            term = bare.action_manager._terms['flipper_pos']
            source_targets = joints[:, term._joint_ids]
            term._previous_targets.copy_(source_targets)
            term._pre_clip_processed_actions.copy_(source_targets)
        atomic(args.output/'after-placement.json', [snapshot(bare, i, 0, 'after_placement') for i in range(3)])
        current_command = bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
        def provider():
            nonlocal current_command
            current_command = current_command.clone()
            for i in range(3):
                current_command[i] = torch.as_tensor(fixture['settling_commands'][control_step],
                                                      device=bare.device, dtype=current_command.dtype)
            return current_command
        override = CommandOverride(bare.command_manager, 3)
        control_step = 0
        override.provider = provider
        override.set(provider())
        atomic(args.output/'before-first-physics.json', [snapshot(bare, i, 0, 'before_first_physics') for i in range(3)])
        rows = []
        original_update = bare.scene.update
        physics_step = 0
        def measured_update(dt):
            nonlocal physics_step
            original_update(dt)
            physics_step += 1
            states = [snapshot(bare, i, physics_step, 'after_physics') for i in range(3)]
            for i, state in enumerate(states):
                arrays = [x.detach().cpu().numpy() for x in views[i].get_contact_data(dt=float(bare.physics_dt))]
                state['physx_contact'] = summarize_contact_arrays(arrays, capacity)
            rows.append({'physics_step': physics_step, 'states': states})
        bare.scene.update = measured_update
        frames, actors = [], []
        zero = torch.zeros((3, 6), device=bare.device)
        for control_step in range(20):
            observation, _, done, _ = env.step(zero)
            if bool(torch.any(done)):
                raise ValueError(f'early termination at control step {control_step+1}: {vector(done)}')
            actors.append({'control_step': control_step+1, 'physics_step': physics_step,
                           'post_delay_actor_input': [actor_values(observation['policy'][i]) for i in range(3)]})
            if control_step in (4,9,14,19):
                frames.append({'control_step': control_step+1, 'physics_step': physics_step,
                               'dem': [dem(bare, i, config) for i in range(3)]})
            atomic(args.output/'heartbeat.json', {'completed_control_steps': control_step+1,
                                                  'physics_step': physics_step,
                                                  'elapsed_s': time.monotonic()-started,
                                                  'Path_publications': 0})
        if physics_step != 200:
            raise ValueError(f'expected 200 physics steps, got {physics_step}')
        atomic(args.output/'physics-trace.json', rows)
        atomic(args.output/'actor-observations.json', actors)
        atomic(args.output/'dem-frames.json', frames)
        atomic(args.output/'complete.json', {'stage': 'complete', 'mode': manifest['mode'],
                                             'physics_steps': physics_step, 'Path_publications': 0,
                                             'trace_sha256': sha(args.output/'physics-trace.json'),
                                             'actors_sha256': sha(args.output/'actor-observations.json'),
                                             'dem_sha256': sha(args.output/'dem-frames.json')})
    finally:
        if override is not None:
            override.close()
        if env is not None:
            env.close()


try:
    main()
except BaseException as exc:
    args.output.mkdir(parents=True, exist_ok=True)
    atomic(args.output/'diagnostic-error.json', {'type': type(exc).__name__, 'message': str(exc),
                                                 'traceback': traceback.format_exc(),
                                                 'Path_publications': 0})
    raise
finally:
    simulation_app.close()
