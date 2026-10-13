"""ROS-free B-MI paired motion collection, four envs, 40 steps.

Each Arc is run in a fresh process with the same StateGroup seed. A reference
pre-Path record is compared before creating or publishing the Path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rl-checkpoint', type=Path, required=True)
parser.add_argument('--eat-config', type=Path, required=True)
parser.add_argument('--arc-angle', type=int, required=True)
parser.add_argument('--direction', type=int, choices=(-1, 0, 1), required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--terrain-type', required=True)
parser.add_argument('--terrain-level', type=int, required=True)
parser.add_argument('--reference-prepath', type=Path)
parser.add_argument('--condition', choices=('rough', 'mirror'), required=True)
parser.add_argument('--repeat', type=int, choices=(0, 1), required=True)
parser.add_argument('--protocol-sha256', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.bidirectional_bank import candidate, audit_candidate, SCHEMA as BANK_SCHEMA
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg
from visualnav_transformer.eat_nav.demo_terrain import configure_selected_terrain
from visualnav_transformer.eat_nav.demo_reset_policy import configure_demo_reset_policy
from visualnav_transformer.eat_nav.motion_collector import body_path_to_tile_local, compare_prepath
from visualnav_transformer.eat_nav.bidirectional_labels import motion_label_with_yaw
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core, SOURCE as CORE_SOURCE
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture
from visualnav_transformer.eat_nav.motion_identifiability import SCHEMA, mirror_audit


STEP_NS = 50_000_000
NUM_ENVS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x') as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def write_npz(path, **arrays):
    with Path(path).open('xb') as out:
        np.savez_compressed(out, **arrays)
        out.flush()
        os.fsync(out.fileno())


def pose_world(robot, index):
    xyz = robot.data.root_pos_w[index].detach().cpu().tolist()
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[index:index + 1])
    return float(xyz[0]), float(xyz[1]), float(yaw.item())


def capture_dem(bare, eat_cfg, index, logical_stamp_ns):
    scanner = bare.scene['height_scanner_eat']
    hit = scanner.data.ray_hits_w[index, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hit)
    starts = scanner._ray_starts_w[index].detach().cpu().numpy()
    source = SourceGrid(np.where(valid, hit, 0).astype(np.float32), valid,
                        (float(starts[0, 0]), float(starts[0, 1])),
                        eat_cfg.dem_resolution_m, (0, 0), eat_cfg.reference_frame,
                        logical_stamp_ns)
    transformed = robot_dem(source, pose_world(bare.scene['robot'], index), eat_cfg)
    if transformed.shape != (2, 20, 20) or not np.isfinite(transformed).all():
        raise ValueError('nonfinite DEM')
    return transformed.astype(np.float32)


def prepath_state(bare, observed, history, seed, terrain_metadata, origins):
    robot = bare.scene['robot']
    material = robot.root_physx_view.get_material_properties().detach().cpu().numpy()
    if material.ndim < 2 or not np.isfinite(material).all():
        raise ValueError('effective PhysX material missing/nonfinite')
    terrain_rows = []
    generator = bare.scene.terrain.cfg.terrain_generator
    for index in range(NUM_ENVS):
        row = int(bare.scene.terrain.terrain_levels[index])
        col = int(bare.scene.terrain.terrain_types[index])
        tile = terrain_metadata[col * generator.num_rows + row]
        terrain_rows.append(tile)
    terrain_sha = hashlib.sha256(json.dumps(terrain_rows, sort_keys=True).encode()).hexdigest()
    material_sha = hashlib.sha256(np.ascontiguousarray(material).tobytes()).hexdigest()
    identity = {'seed': seed, 'terrain_sha256': terrain_sha,
                'material_sha256': material_sha, 'terrain_tiles': terrain_rows,
                'condition': args.condition, 'repeat': args.repeat,
                'protocol_sha256': args.protocol_sha256,
                'joint_names': list(robot.data.joint_names),
                'dem_frame': 'robot_base', 'dem_resolution_m': .1,
                'dem_stamps_ns': [250_000_000, 500_000_000, 750_000_000, 1_000_000_000],
                'episode_length_s': float(bare.cfg.episode_length_s)}
    arrays = {'root': robot.data.root_state_w.detach().cpu().numpy().copy(),
              'joint': np.concatenate((robot.data.joint_pos.detach().cpu().numpy(),
                                       robot.data.joint_vel.detach().cpu().numpy()), axis=1),
              'actor_pre_noise': observed['policy_pre_noise'].detach().cpu().numpy().copy(),
              'actor_post_noise': observed['policy_post_noise_pre_delay'].detach().cpu().numpy().copy(),
              'actor_input': observed['policy_actor_input'].detach().cpu().numpy().copy(),
              'critic': observed['critic_observation'].detach().cpu().numpy().copy(),
              'dem': np.asarray(history, dtype=np.float32).transpose(1, 0, 2, 3, 4),
              'material': material}
    for name in ('_latency_substeps', '_onix_latency_seconds',
                 '_latency_policy_obs_prev', '_latency_policy_obs_reset_mask'):
        value = getattr(bare, name, None)
        if isinstance(value, torch.Tensor):
            arrays['latency'+name] = value.detach().cpu().numpy().copy()
    arrays['command'] = bare.command_manager._terms['base_velocity'].vel_command_b.detach().cpu().numpy().copy()
    local_root = arrays['root'].copy()
    local_root[:, :3] -= origins
    arrays['root_tile_local'] = local_root
    if arrays['dem'].shape != (NUM_ENVS, 4, 2, 20, 20) or \
            arrays['actor_pre_noise'].shape != (NUM_ENVS, 22):
        raise ValueError('pre-Path input shape')
    return identity, arrays


def main():
    if args.output.exists():
        raise FileExistsError('fresh Arc collection output required')
    args.output.mkdir(parents=True)
    if args.direction not in (-1, 1) or args.arc_angle not in (-30, 0, 30):
        raise ValueError('MI-A pilot accepts only forward/reverse Direct and ±30')
    path_candidate = candidate(args.direction if args.direction else 1, args.arc_angle)
    path_body = path_candidate.waypoints
    geometry = audit_candidate(path_candidate)
    if not geometry['geometry_valid']:
        raise ValueError('invalid frozen Arc geometry')
    write_json(args.output/'input-manifest.json', {
        'schema': SCHEMA, 'bank_schema': BANK_SCHEMA, 'arc_angle_deg': args.arc_angle,
        'direction': args.direction, 'arc_id': path_candidate.arc_id if args.direction else 'zero_control',
        'arc_geometry': geometry, 'path_body_xy_m': path_body.tolist(),
        'path_body_sha256': hashlib.sha256(path_body.astype('<f8').tobytes()).hexdigest(),
        'source_sha256': sha(__file__), 'core_source_sha256': sha(CORE_SOURCE),
        'checkpoint_sha256': sha(args.rl_checkpoint), 'eat_config_sha256': sha(args.eat_config),
        'seed': args.seed, 'condition': args.condition, 'repeat': args.repeat,
        'protocol_sha256': args.protocol_sha256, 'terrain_type': args.terrain_type,
        'terrain_level': args.terrain_level, 'num_envs': NUM_ENVS,
        'reference_prepath_sha256': sha(args.reference_prepath/'prepath.npz')
            if args.reference_prepath else None})
    env = override = capture = None
    followers = []
    streams = []
    try:
        eat_cfg = load_eat_config(args.eat_config)
        if eat_cfg.dem_frame_count != 4 or eat_cfg.dem_resolution_m != .1:
            raise ValueError('DEM input config mismatch')
        saved = yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),
                          Loader=yaml.BaseLoader)
        toggles = {name: saved[name].lower() == 'true' for name in
                   ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
        cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), eat_cfg, args.seed, NUM_ENVS)
        cfg.episode_length_s = 40.0
        configure_demo_reset_policy(cfg)
        if args.condition == 'mirror':
            pose = cfg.events.reset_base.params['pose_range']
            for name in ('x', 'y', 'yaw'):
                pose[name] = (0.0, 0.0)
        cfg.events.ignore_sprocket_self_collision = None
        generator = cfg.scene.terrain.terrain_generator
        generator.seed = args.seed
        configure_selected_terrain(generator, cfg.scene.terrain,
                                   args.terrain_type, args.terrain_level, NUM_ENVS)
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
        bare = env.unwrapped
        if abs(float(bare.step_dt) - .05) > 1e-9 or int(bare.max_episode_length) != 800:
            raise ValueError('20 Hz / 40 s environment contract')
        env.reset()
        runner = OnPolicyRunner(env, OnixRoughPPORunnerCfg().to_dict(), log_dir=None, device=bare.device)
        weights = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        runner.alg.policy.load_state_dict(weights)
        policy = runner.alg.policy
        robot = bare.scene['robot']
        override = CommandOverride(bare.command_manager, NUM_ENVS)
        capture = ObservationCapture(bare.observation_manager)
        zeros = torch.zeros((NUM_ENVS, 3), device=bare.device)
        override.set(zeros)
        latency_prev = (bare._latency_policy_obs_prev.detach().clone()
                        if bare._latency_policy_obs_prev is not None else None)
        capture.begin_step()
        raw = bare.observation_manager.compute()
        obs = bare._apply_policy_observation_latency(raw)
        observed = capture.finish_step(obs)
        history = []
        for settling_step in range(20):
            override.set(zeros)
            obs, observed, _ = reconcile_command_observation(
                obs, observed, bare, zeros, latency_prev)
            with torch.inference_mode():
                action = policy.act_inference(obs)
            latency_prev = (bare._latency_policy_obs_prev.detach().clone()
                            if bare._latency_policy_obs_prev is not None else None)
            capture.begin_step()
            with torch.inference_mode():
                obs, reward, done, _ = env.step(action)
            observed = capture.finish_step(obs)
            if bool(done.any()):
                raise ValueError('reset/termination during settling')
            if settling_step in (4, 9, 14, 19):
                history.append(np.stack([capture_dem(bare, eat_cfg, index,
                    (settling_step + 1) * STEP_NS) for index in range(NUM_ENVS)]))
        origins = bare.scene.env_origins[:, :3].detach().cpu().numpy()
        identity, arrays = prepath_state(bare, observed, history, args.seed,
                                        bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata,
                                        origins)
        write_json(args.output/'prepath-identity.json', identity)
        write_npz(args.output/'prepath.npz', **arrays)
        if args.condition == 'mirror':
            mirror = mirror_audit(arrays['root_tile_local'], arrays['dem'])
            write_json(args.output/'mirror-audit.json', mirror)
            if not mirror['pass']:
                raise ValueError('pre-Path mirror audit failed')
        if args.reference_prepath:
            reference_identity = json.loads((args.reference_prepath/'prepath-identity.json').read_text())
            with np.load(args.reference_prepath/'prepath.npz', allow_pickle=False) as reference:
                result = compare_prepath({**reference_identity, **{k: reference[k] for k in
                    ('root', 'joint', 'actor_pre_noise', 'dem')}},
                    {**identity, **arrays})
                extra = {}
                for key in ('actor_post_noise', 'actor_input', 'critic', 'command',
                            '_latency_substeps', '_onix_latency_seconds',
                            '_latency_policy_obs_prev', '_latency_policy_obs_reset_mask'):
                    field = 'latency'+key if key.startswith('_') else key
                    if field not in reference.files or field not in arrays:
                        extra[field] = {'pass': False, 'reason': 'missing'}
                        continue
                    a, b = reference[field], arrays[field]
                    valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                    maximum = float(np.max(np.abs(a-b))) if valid and a.size else None
                    extra[field] = {'pass': bool(valid and maximum is not None and maximum <= 1e-6),
                                    'max_abs': maximum}
                result['extra_fields'] = extra
                result['pass'] = result['pass'] and all(x['pass'] for x in extra.values())
            write_json(args.output/'prepath-pair-gate.json', result)
            if not result['pass']:
                raise ValueError('cross-candidate pre-Path gate failed')
        else:
            write_json(args.output/'prepath-pair-gate.json', {'pass': True,
                'reference': 'self_first_candidate', 'tolerance_abs': 1e-6})
        library = compile_core(args.output/'core-build')
        paths_tile = []
        origins = bare.scene.env_origins[:, :2].detach().cpu().numpy()
        for index in range(NUM_ENVS):
            path_tile = body_path_to_tile_local(path_body, pose_world(robot, index), origins[index])
            paths_tile.append(path_tile.tolist())
            if args.direction:
                followers.append(PureFollower(path_tile.tolist(), library=library))
        write_json(args.output/'path-activation.json', {'arc_angle_deg': args.arc_angle,
                   'path_body_sha256': json.loads((args.output/'input-manifest.json').read_text())['path_body_sha256'],
                   'paths_tile_xy_m': paths_tile, 'activation_stamp_ns': 1_000_000_000,
                   'direction': args.direction,
                   'count_per_env': [1 if args.direction else 0] * NUM_ENVS})
        for index in range(NUM_ENVS):
            streams.append((args.output/f'trace-env-{index}.jsonl').open('x'))
        traces = [[] for _ in range(NUM_ENVS)]
        for step in range(40):
            stamp = 1_000_000_000 + step * STEP_NS
            root_pre = robot.data.root_state_w.detach().cpu().numpy().copy()
            commands = []
            diagnostics = []
            for index in range(NUM_ENVS):
                if args.direction:
                    follower = followers[index]
                    world = pose_world(robot, index)
                    tile = (world[0] - origins[index, 0], world[1] - origins[index, 1], world[2])
                    command, diagnostic = follower.step_diagnostic(*tile, stamp)
                    if step == 0 and args.arc_angle == 0 and (
                            int(diagnostic['travel_direction']) != args.direction or
                            command[0] * args.direction <= 0 or abs(command[2]) > 1e-8):
                        raise ValueError('initial follower direction/command mismatch')
                else:
                    command, diagnostic = (0., 0., 0.), {'zero_command_control': True}
                commands.append(command)
                diagnostics.append(diagnostic)
            injected = torch.tensor(commands, dtype=zeros.dtype, device=bare.device)
            override.set(injected)
            obs, observed, command_detail = reconcile_command_observation(
                obs, observed, bare, injected, latency_prev)
            if not bool(torch.isfinite(obs['policy']).all()):
                raise ValueError('nonfinite Actor observation')
            with torch.inference_mode():
                action = policy.act_inference(obs)
            latency_prev = (bare._latency_policy_obs_prev.detach().clone()
                            if bare._latency_policy_obs_prev is not None else None)
            capture.begin_step()
            with torch.inference_mode():
                obs, reward, done, _ = env.step(action)
            observed = capture.finish_step(obs)
            root_post = robot.data.root_state_w.detach().cpu().numpy().copy()
            contact = bare.scene['contact_forces'].data.net_forces_w.detach().cpu().numpy()
            torque = robot.data.applied_torque.detach().cpu().numpy()
            joint_velocity = robot.data.joint_vel.detach().cpu().numpy()
            for index, stream in enumerate(streams):
                row = {'control_step': step, 'logical_sim_time_ns': stamp,
                       'root_state': root_pre[index].tolist(),
                       'post_root_state': root_post[index].tolist(),
                       'follower_command': list(commands[index]),
                       'follower_diagnostic': diagnostics[index],
                       'path_tracking_error_m': diagnostics[index].get('cross_track_error', None),
                       'injected_command': command_detail['injected'][index],
                       'command_previous_raw': command_detail['previous_raw'][index],
                       'command_latency_alpha': command_detail['latency_alpha'][index],
                       'actor_command': command_detail['actor_seen'][index],
                       'actor_observation': obs['policy'][index].detach().cpu().tolist(),
                       'actor_action': action[index].detach().cpu().tolist(),
                       'reward': float(reward[index]), 'contact_net_forces_w': contact[index].tolist(),
                       'applied_torque': torque[index].tolist(),
                       'joint_velocity': joint_velocity[index].tolist(),
                       'body_velocity_xy_mps': root_pre[index, 7:9].tolist(),
                       'terminal': {'termination_reason': 'reset_or_termination'} if bool(done[index]) else None}
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                traces[index].append(row)
            if bool(done.any()):
                raise ValueError('reset/termination before 40-step label completion')
        labels = [motion_label_with_yaw(trace) for trace in traces]
        if not all(label['available'] for label in labels):
            raise ValueError('unavailable 40-step motion label')
        write_json(args.output/'motion-labels.json', {'schema': SCHEMA, 'labels': labels})
        write_json(args.output/'complete.json', {'schema': SCHEMA, 'status': 'complete',
                  'control_steps': 40, 'env_count': NUM_ENVS,
                  'prepath_sha256': sha(args.output/'prepath.npz'),
                  'trace_sha256': [sha(args.output/f'trace-env-{i}.jsonl') for i in range(NUM_ENVS)]})
    except Exception as exc:
        write_json(args.output/'failure.json', {'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc(),
                   'path_published': (args.output/'path-activation.json').exists()})
        raise
    finally:
        for stream in streams:
            stream.close()
        for follower in followers:
            follower.close()
        if capture is not None:
            capture.close()
        if override is not None:
            override.close()
        if env is not None:
            env.close()
        files = {str(path.relative_to(args.output)): sha(path) for path in args.output.rglob('*')
                 if path.is_file() and path.name not in ('sha-ledger.json', 'host-seal.json')}
        write_json(args.output/'sha-ledger.json', files)
        write_json(args.output/'host-seal.json', {'schema': SCHEMA,
                   'ledger_sha256': sha(args.output/'sha-ledger.json'),
                   'file_count': len(files)})


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
