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


def main():
    atomic(args.output / 'owned-process.json', {'pid': os.getpid(),
        'proc_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]),
        'candidate_free': True})
    atomic(args.output / 'heartbeat.json', {'batch_seed': args.batch_seed,
        'stage': 'pre_scene', 'monotonic_ns': time.monotonic_ns(), 'elapsed_sec': 0.0})
    freeze = json.loads(args.freeze.read_text())
    if args.batch_seed not in freeze['batch_seeds']:
        raise ValueError('batch seed outside freeze')
    for label, path in freeze['source_paths'].items():
        if digest(path) != freeze['source_sha256'][label]:
            raise ValueError('source SHA drift: ' + label)
    if digest(__file__) != freeze['sim_source_sha256']:
        raise ValueError('sim source SHA drift')
    rows = [r for r in build_pose_schedule(freeze['batch_seeds']) if r['batch_seed'] == args.batch_seed]
    if len(rows) != 112 or [r['env_index'] for r in rows] != list(range(112)):
        raise ValueError('schedule/env mapping drift')
    config = load_eat_config(Path(freeze['source_paths']['eat_config']))
    saved = yaml.load(Path(freeze['source_paths']['checkpoint_env']).read_text(), Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), config, args.batch_seed, 112)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type = RadialTerrainGenerator
    gen.num_rows, gen.num_cols = 16, 7
    gen.curriculum = False
    gen.difficulty_range = (.5, .5)
    gen.use_cache = False
    gen.seed = args.batch_seed
    gen.sub_terrains = {name: gen.sub_terrains[name] for name in TERRAIN_ORDER}
    cfg.scene.terrain.max_init_terrain_level = 15
    env = None
    started = time.monotonic()
    try:
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
        bare = env.unwrapped
        actual_metadata = bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata
        if len(actual_metadata) != 112:
            raise ValueError('terrain tile metadata count')
        for row, meta in zip(rows, actual_metadata):
            if any(meta[k] != row[v] for k, v in [('row','row'), ('column','column'),
                                                  ('seed','terrain_seed'), ('terrain_type','terrain')]):
                raise ValueError('runtime terrain identity mismatch ' + row['pose_id'])
            if meta['difficulty'] != .5:
                raise ValueError('difficulty mismatch')
        origins = bare.scene.env_origins.detach().cpu().numpy()
        if origins.shape != (112, 3) or not np.isfinite(origins).all():
            raise ValueError('runtime env origins invalid')
        # EATTerrainImporter maps env i to row-major tile i. Check XYZ against generated origins.
        generated = np.asarray(bare.scene.terrain.terrain_origins.detach().cpu().numpy())
        for row in rows:
            if not np.allclose(origins[row['env_index']], generated[row['row'], row['column']], atol=1e-5):
                raise ValueError('env-origin row/column mismatch')
        atomic(args.output / 'env-origins.json', {'batch_seed': args.batch_seed, 'origins': origins.tolist(),
                                                   'terrain_metadata': actual_metadata})
        obs, _ = env.reset()
        robot = bare.scene['robot']
        root = robot.data.default_root_state.clone()
        positions = root[:, :3] + bare.scene.env_origins
        for row in rows:
            idx = row['env_index']
            p = pose_world(origins[idx], row['angle_index'])
            positions[idx, 0], positions[idx, 1] = p['world_x'], p['world_y']
            positions[idx, 2] = p['world_z_reference'] + .55
        yaw = torch.as_tensor([r['theta_rad'] for r in rows], device=bare.device, dtype=root.dtype)
        zeros = torch.zeros_like(yaw)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        ids = torch.arange(112, device=bare.device)
        robot.write_root_pose_to_sim(torch.cat([positions, quats], dim=1), env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros((112, 6), device=bare.device, dtype=root.dtype), env_ids=ids)
        bare.sim.forward()
        frame_steps = (4, 9, 14, 19)
        histories = [[] for _ in rows]
        requested = [pose_world(origins[r['env_index']], r['angle_index']) for r in rows]
        for step in range(20):
            obs, _reward, done, _extra = env.step(torch.zeros((112, 6), device=bare.device))
            if bool(torch.any(done)):
                raise RuntimeError(f'unexpected reset/termination at settling step {step}')
            if step in frame_steps:
                for i in range(112):
                    histories[i].append({'step': step, 'logical_stamp_ns': (step + 1) * 50_000_000,
                                         'dem': extract_dem(bare, i, config)})
            atomic(args.output / 'heartbeat.json', {'batch_seed': args.batch_seed,
                  'settling_step': step, 'monotonic_ns': time.monotonic_ns(),
                  'elapsed_sec': time.monotonic() - started})
        actor = obs['policy'].detach().cpu().numpy()
        state = robot.data.root_state_w.detach().cpu().numpy()
        gravity = robot.data.projected_gravity_b.detach().cpu().numpy()
        contacts = bare.scene['contact_forces'].data.net_forces_w.detach().cpu().numpy()
        for row in rows:
            i = row['env_index']
            root_state = state[i]
            _, _, settled_yaw_t = euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
            settled_yaw = float(settled_yaw_t.item())
            err_yaw = math.atan2(math.sin(settled_yaw - requested[i]['world_yaw']),
                                 math.cos(settled_yaw - requested[i]['world_yaw']))
            hist = histories[i]
            arr = np.asarray([h['dem']['transformed'] for h in hist], dtype=np.float32)
            record = {**{k: row[k] for k in ('pose_id','batch_seed','terrain','row','column','source_family')},
                'terrain_seed': row['terrain_seed'], 'terrain_class': row['terrain_class'],
                'env_index': i, 'env_origin': origins[i].tolist(), 'requested_pose': requested[i],
                'requested_root_z_m': float(positions[i, 2]),
                'settled_root_state': root_state.tolist(),
                'settled_error_xy_m': float(np.linalg.norm(root_state[:2] - [requested[i]['world_x'], requested[i]['world_y']])),
                'settled_yaw_error_rad': abs(err_yaw),
                'settled_speed_mps': float(np.linalg.norm(root_state[7:10])),
                'settled_angular_speed_radps': float(np.linalg.norm(root_state[10:13])),
                'projected_gravity': gravity[i].tolist(), 'contact_force': contacts[i].tolist(),
                'joint_position': robot.data.joint_pos[i].detach().cpu().numpy().tolist(),
                'joint_velocity': robot.data.joint_vel[i].detach().cpu().numpy().tolist(),
                'actor_observation': actor[i].tolist(), 'actor_finite': bool(np.isfinite(actor[i]).all()),
                'dem_history': hist, 'dem_history_shape': list(arr.shape),
                'dem_valid_fraction': float(np.mean(arr[:, 1] > .5)),
                'dem_finite': bool(np.isfinite(arr).all()),
                'timestamps_monotone': [h['logical_stamp_ns'] for h in hist] == sorted({h['logical_stamp_ns'] for h in hist}),
                'unexpected_reset': False, 'termination': False,
                'penetration': None, 'penetration_diagnostic': 'not directly observed by net contact force',
                'cleanup_ok': False, 'settling_steps': 20, 'dem_frame_id': config.reference_frame,
                'dem_resolution_m': config.dem_resolution_m,
                'generated_identity_sha256': hashlib.sha256(json.dumps({
                    'batch_seed': args.batch_seed, 'terrain_seed': row['terrain_seed'],
                    'terrain': row['terrain'], 'difficulty': .5,
                    'terrain_cfg_sha256': freeze['source_sha256']['rough_cfg']}, sort_keys=True).encode()).hexdigest(),
            }
            atomic(args.output / 'poses' / f"{row['pose_id']}.json", record)
        atomic(args.output / 'sim-finished.json', {'batch_seed': args.batch_seed, 'pose_count': 112,
            'elapsed_sec': time.monotonic() - started, 'sim_source_sha256': digest(__file__)})
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


try:
    main()
except BaseException as exc:
    atomic(args.output / 'failure.json', {'batch_seed': args.batch_seed, 'error': repr(exc),
                                            'wall_ns': time.time_ns()})
    raise
