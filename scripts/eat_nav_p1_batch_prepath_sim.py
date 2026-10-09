"""One 112-env candidate-free P1 batch anchor gate. No Path publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
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
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, terrain_seed
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


plan = json.loads((args.input / 'plan.json').read_text())
freeze = json.loads((args.input / 'freeze.json').read_text())
if plan['num_envs'] != 112 or len(plan['targets']) != 28:
    raise ValueError('frozen scene shape changed')
if digest(args.input / 'plan.json') != freeze['plan_sha256']:
    raise ValueError('plan SHA drift')
fixtures = json.loads((args.input / 'anchor-fixtures.json').read_text())
if digest(args.input / 'anchor-fixtures.json') != freeze['anchor_fixtures_sha256']:
    raise ValueError('anchor fixture SHA drift')


class MixedP1TerrainGenerator(EATTerrainGenerator):
    """Generate each target's source terrain seed on a distinct tile."""

    def _generate_random_terrains(self):
        targets = {item['target_env_index']: item for item in plan['targets']}
        if (self.cfg.num_rows, self.cfg.num_cols) != (16, 7):
            raise ValueError('frozen 16x7 layout changed')
        for row in range(16):
            for col in range(7):
                index = row * 7 + col
                name = targets[index]['terrain_type'] if index in targets else TERRAIN_ORDER[col]
                cfg = self.cfg.sub_terrains[name]
                mesh, origin = self._get_terrain_mesh(.5, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)

    def _get_terrain_mesh(self, difficulty, cfg):
        index = self._tile_index
        target = next((item for item in plan['targets'] if item['target_env_index'] == index), None)
        if target:
            seed, name = target['terrain_seed'], target['terrain_type']
        else:
            col = index % 7
            seed, name = terrain_seed(20261007, col), TERRAIN_ORDER[col]
        old_seed, old_np = self.cfg.seed, np.random.get_state()
        try:
            self.cfg.seed = int(seed)
            np.random.seed(int(seed))
            mesh, origin = TerrainGenerator._get_terrain_mesh(self, difficulty, cfg)
            self.tile_metadata.append({'tile_index': index, 'row': index // 7,
                                       'column': index % 7, 'seed': int(seed),
                                       'terrain_type': name, 'difficulty': float(difficulty)})
            self._tile_index += 1
            return mesh, origin
        finally:
            self.cfg.seed = old_seed
            np.random.set_state(old_np)


def dem(env, index, config):
    robot, scanner = env.scene['robot'], env.scene['height_scanner_eat']
    pos = robot.data.root_pos_w[index].detach().cpu().numpy()
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w[index:index + 1])
    hits = scanner.data.ray_hits_w[index, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hits)
    start = scanner._ray_starts_w[index, 0, :2].detach().cpu().numpy()
    source = SourceGrid(height=np.where(valid, hits, 0).astype(np.float32), valid=valid,
                        origin_xy=(float(start[0]), float(start[1])),
                        resolution_m=config.dem_resolution_m,
                        start_index_xy=(0, 0), frame_id=config.reference_frame, stamp_ns=0)
    transformed = np.asarray(robot_dem(source, (float(pos[0]), float(pos[1]), float(yaw_t.item())), config), dtype=np.float32)
    return {'source_height': source.height.tolist(), 'source_mask': valid.astype(int).tolist(),
            'source_origin_xy': list(source.origin_xy), 'transformed': transformed.tolist(),
            'shape': list(transformed.shape), 'finite': bool(np.isfinite(transformed).all())}


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'owned-process.json', {
        'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'source_sha256': digest(__file__), 'Path_publications': 0})
    atomic_json(args.output / 'status.json', {'stage': 'startup', 'Path_publications': 0})
    config = load_eat_config(Path(freeze['eat_config']))
    saved = yaml.load(Path(freeze['checkpoint_env']).read_text(), Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), config, 20261007, 112)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type = MixedP1TerrainGenerator
    gen.num_rows, gen.num_cols = 16, 7
    gen.curriculum = False
    gen.difficulty_range = (.5, .5)
    gen.use_cache = False
    gen.seed = 20261007
    gen.sub_terrains = {name: gen.sub_terrains[name] for name in TERRAIN_ORDER}
    cfg.scene.terrain.max_init_terrain_level = 15
    env = None
    override = None
    try:
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
        bare = env.unwrapped
        metadata = bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata
        origins = bare.scene.env_origins.detach().cpu().numpy()
        for item in plan['targets']:
            index = item['target_env_index']
            row = metadata[index]
            if row['seed'] != item['terrain_seed'] or row['terrain_type'] != item['terrain_type']:
                raise ValueError(f'target terrain mismatch: {index}')
        env.reset()
        robot = bare.scene['robot']
        root = robot.data.root_state_w.clone()
        ids = torch.arange(28, device=bare.device)
        positions = root[ids, :3].clone()
        quats = root[ids, 3:7].clone()
        for item in plan['targets']:
            i = item['target_env_index']
            positions[i, 0] = float(origins[i, 0]) + item['source_anchor_local_xy'][0]
            positions[i, 1] = float(origins[i, 1]) + item['source_anchor_local_xy'][1]
            yaw = torch.as_tensor(item['source_anchor_yaw'], device=bare.device, dtype=root.dtype)
            quats[i] = math_utils.quat_from_euler_xyz(yaw*0, yaw*0, yaw)
        robot.write_root_pose_to_sim(torch.cat([positions, quats], dim=1), env_ids=ids)
        joint_pos = []
        joint_vel = []
        fold_mode = bare._onix_flipper_fold_mode
        for item in plan['targets']:
            fixture = fixtures[item['anchor_id']]
            state = fixture['prephysics_state']
            if state['joint_names'] != list(robot.data.joint_names):
                raise ValueError('frozen joint name order mismatch')
            joint_pos.append(state['joint_position'])
            joint_vel.append(state['joint_velocity'])
            fold_mode[item['target_env_index']] = float(state['flipper_fold_mode_latch'])
        robot.write_joint_state_to_sim(
            torch.as_tensor(joint_pos, device=bare.device, dtype=robot.data.joint_pos.dtype),
            torch.as_tensor(joint_vel, device=bare.device, dtype=robot.data.joint_vel.dtype),
            env_ids=ids)
        bare.sim.forward()
        atomic_json(args.output / 'status.json', {'stage': 'radial_placed', 'Path_publications': 0})
        zero = torch.zeros((112, 6), device=bare.device)
        frames = {}
        terminated = set()
        command_index = 0
        override = CommandOverride(bare.command_manager, 112)
        def command_provider():
            nonlocal command_index
            if command_index >= 20:
                raise ValueError('settling command replay overflow')
            current = bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            for item in plan['targets']:
                current[item['target_env_index']] = torch.as_tensor(
                    fixtures[item['anchor_id']]['settling_commands'][command_index],
                    device=bare.device, dtype=current.dtype)
            command_index += 1
            return current
        override.provider = command_provider
        override.set(command_provider())
        command_index = 0
        for step in range(20):
            obs, _, done, _ = env.step(zero)
            for i in range(28):
                if bool(done[i]):
                    terminated.add(i)
            if step in (4, 9, 14, 19):
                frame_index = (step + 1) // 5 - 1
                for i in range(28):
                    frames[(i, frame_index)] = dem(bare, i, config)
            atomic_json(args.output / 'heartbeat.json', {'completed_control_steps': step + 1,
                                                          'Path_publications': 0})
        if command_index != 20:
            raise ValueError(f'settling command replay count {command_index}, expected 20')
        records = []
        for item in plan['targets']:
            i = item['target_env_index']
            records.append({'physical_run_id': item['physical_run_id'],
                            'target_env_index': i, 'terrain_metadata': metadata[i],
                            'source_tile_origin_xy': item['source_tile_origin_xy'],
                            'target_tile_origin_xy': origins[i, :2].tolist(),
                            'root_state_w': robot.data.root_state_w[i].detach().cpu().tolist(),
                            'joint_position': robot.data.joint_pos[i].detach().cpu().tolist(),
                            'joint_velocity': robot.data.joint_vel[i].detach().cpu().tolist(),
                            'actor_input': obs['policy'][i].detach().cpu().tolist(),
                            'dem_frames': [frames[(i, j)] for j in range(4)],
                            'terminated_during_settling': i in terminated})
        atomic_json(args.output / 'prepath-records.json', records)
        atomic_json(args.output / 'sim-finished.json', {'native_completion': True,
                                                        'target_env_count': 28,
                                                        'settling_control_steps': 20,
                                                        'Path_publications': 0})
    finally:
        if override is not None:
            override.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == '__main__':
    main()
