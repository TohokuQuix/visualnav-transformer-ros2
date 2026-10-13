"""Isaac-owned EAT demo: ROS goals over IPC, frozen evaluator, C++ follower, RL Actor.

This is a demonstration series. It does not reuse sealed G3 comparisons or make
claims about evaluator generalization. ROS transports goal/state only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import select
import socket
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--rl-checkpoint', type=Path, required=True)
parser.add_argument('--bank', type=Path, required=True)
parser.add_argument('--eat-config', type=Path, required=True)
parser.add_argument('--port', type=int, default=29553)
parser.add_argument('--seed', type=int, default=2026101101)
parser.add_argument('--max-steps', type=int, default=3000)
parser.add_argument('--replan-steps', type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
import onix_isaac_lab.tasks  # noqa: F401
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg
from visualnav_transformer.eat_nav.demo_contract import (
    CHECKPOINT_SHA256, DATASET_SHA256, RL_CHECKPOINT_SHA256, candidates_for_goal, load_bank,
    select_best, sensor_stamp_valid, sha256, validate_evaluator_sample,
)
from visualnav_transformer.eat_nav.evaluator.inference import load_checkpoint
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation


STEP_NS = 50_000_000
PROTOCOL = 'eatnav_ros2_isaac_gui_demo_v1'


def durable_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def pose_map(robot):
    position = robot.data.root_pos_w[0].detach().cpu().tolist()
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
    return (float(position[0]), float(position[1]), float(yaw.item()))


def dem_from_sensor(bare, eat_cfg, stamp_ns):
    robot, scanner = bare.scene['robot'], bare.scene['height_scanner_eat']
    hits = scanner.data.ray_hits_w[0, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hits)
    starts = scanner._ray_starts_w[0].detach().cpu().numpy()
    pose = pose_map(robot)
    source = SourceGrid(np.where(valid, hits, 0).astype(np.float32), valid,
                        (float(starts[0, 0]), float(starts[0, 1])),
                        eat_cfg.dem_resolution_m, (0, 0), eat_cfg.reference_frame,
                        int(stamp_ns))
    result = robot_dem(source, pose, eat_cfg)
    actual_update_ns = round(float(scanner._timestamp_last_update[0]) * 1e9)
    # The raycaster updates every 250 ms, independently of the 50 ms control
    # boundary. Its latest sample can be one sensor period old at that boundary.
    # Preserve both clocks and reject a missing update beyond period + one step.
    age_ns = stamp_ns - actual_update_ns
    if result.shape != (2, 20, 20) or not np.isfinite(result).all() or \
            not sensor_stamp_valid(stamp_ns, actual_update_ns):
        raise ValueError(f'DEM shape/finiteness/sensor timestamp mismatch: age_ns={age_ns}')
    return result, actual_update_ns


class GoalWire:
    def __init__(self, port):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(1)
        self.listener.settimeout(120)
        self.conn, _ = self.listener.accept()
        self.conn.setblocking(False)
        self.pending = b''
        self.last_receive_wall = time.monotonic()

    def poll(self):
        messages = []
        while select.select([self.conn], [], [], 0)[0]:
            chunk = self.conn.recv(1 << 16)
            if not chunk:
                raise ConnectionError('ROS frontend disconnected')
            self.last_receive_wall = time.monotonic()
            self.pending += chunk
            while b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                messages.append(json.loads(line))
        return messages

    def send(self, value):
        self.conn.sendall((json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode())

    def close(self):
        self.conn.close()
        self.listener.close()


def main():
    if args.output.exists():
        raise FileExistsError('fresh persistent output root required')
    args.output.mkdir(parents=True)
    ledger = {'protocol': PROTOCOL, 'source_sha256': sha256(__file__),
              'checkpoint_sha256': sha256(args.checkpoint),
              'rl_checkpoint_sha256': sha256(args.rl_checkpoint),
              'bank_sha256': sha256(args.bank),
              'eat_config_sha256': sha256(args.eat_config), 'seed': args.seed,
              'max_steps': args.max_steps, 'replan_steps': args.replan_steps}
    durable_json(args.output/'input-ledger.json', ledger)
    if ledger['checkpoint_sha256'] != CHECKPOINT_SHA256 or \
            ledger['rl_checkpoint_sha256'] != RL_CHECKPOINT_SHA256:
        raise ValueError('Evaluator/RL checkpoint SHA mismatch')
    if args.replan_steps < 5 or args.replan_steps % 5 or args.max_steps <= 20:
        raise ValueError('replan interval must align to 250 ms DEM history')
    bank = load_bank(args.bank)
    eat_cfg = load_eat_config(args.eat_config)
    # PyTorch 2.7 defaults to weights_only=True. The SHA-pinned pilot carries
    # TorchVersion metadata; allow exactly that class without changing the
    # evaluator loader or enabling arbitrary pickle objects.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        evaluator, payload = load_checkpoint(args.checkpoint, device='cpu',
                                             expected_label_type='common_goal_progress',
                                             expected_dataset_sha256=DATASET_SHA256)
    if eat_cfg.dem_frame_count != 4 or eat_cfg.dem_resolution_m != .1:
        raise ValueError('DEM history config mismatch')
    saved = yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),
                      Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), eat_cfg, args.seed, 4)
    cfg.episode_length_s = 40.0
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows, generator.num_cols = 2, 2
    generator.curriculum = False
    generator.difficulty_range = (.5, .5)
    generator.use_cache = False
    generator.seed = args.seed
    generator.sub_terrains = {'boxes': generator.sub_terrains['boxes']}
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.viewer.origin_type = 'asset_root'
    cfg.viewer.asset_name = 'robot'
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (4.5, -3.8, 2.4)
    cfg.viewer.lookat = (0., 0., .9)
    env = None
    override = None
    capture = None
    wire = None
    follower = None
    trace = None
    try:
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
        bare = env.unwrapped
        if tuple((float(bare.step_dt), int(bare.max_episode_length))) != (.05, 800):
            raise ValueError('RL control/episode contract mismatch')
        env.reset()
        runner = OnPolicyRunner(env, OnixRoughPPORunnerCfg().to_dict(), log_dir=None, device=bare.device)
        weights = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        runner.alg.policy.load_state_dict(weights)
        policy = runner.alg.policy
        robot = bare.scene['robot']
        override = CommandOverride(bare.command_manager, 4)
        capture = ObservationCapture(bare.observation_manager)
        library = compile_core(args.output/'core-build')
        zeros = torch.zeros((4, 3), device=bare.device)
        override.set(zeros)
        latency_prev_for_reconcile = (bare._latency_policy_obs_prev.detach().clone()
                                      if bare._latency_policy_obs_prev is not None else None)
        capture.begin_step()
        raw = bare.observation_manager.compute()
        obs = bare._apply_policy_observation_latency(raw)
        observed = capture.finish_step(obs)
        if tuple(obs['policy'].shape) != (4, 22) or tuple(obs['critic'].shape) != (4, 267):
            raise ValueError('Actor observation contract mismatch')
        durable_json(args.output/'startup-ready.json', {'protocol': PROTOCOL,
            'actor_shape': list(obs['policy'].shape), 'critic_shape': list(obs['critic'].shape),
            'control_dt_s': float(bare.step_dt), 'checkpoint_sha256': ledger['checkpoint_sha256'],
            'rl_checkpoint_sha256': ledger['rl_checkpoint_sha256']})
        wire = GoalWire(args.port)
        trace = (args.output/'trace.jsonl').open('x')
        history = collections.deque(maxlen=4)
        goal_map = None
        arrived = False
        selected = None
        paths_map = {}
        scores = {}
        last_replan_step = -args.replan_steps
        run_start = time.monotonic()
        next_wall = run_start
        for step in range(args.max_steps):
            stamp_ns = step * STEP_NS
            if not simulation_app.is_running():
                break
            for message in wire.poll():
                if message.get('type') == 'cancel':
                    goal_map = None
                    arrived = False
                    selected = None
                    paths_map = {}
                    scores = {}
                    if follower is not None:
                        follower.close()
                        follower = None
                elif message.get('type') == 'goal':
                    age = stamp_ns - int(message.get('stamp_ns', -1))
                    if message.get('frame') != 'map' or not 0 <= age <= 500_000_000:
                        continue
                    value = np.asarray(message.get('xy'), dtype=np.float64)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        continue
                    goal_map = value
                    arrived = False
                    last_replan_step = step - args.replan_steps
                else:
                    raise ValueError('unknown ROS transport message')
            pose = pose_map(robot)
            if step % 5 == 0:
                frame, sensor_stamp = dem_from_sensor(bare, eat_cfg, stamp_ns)
                history.append((stamp_ns, frame, sensor_stamp))
            if goal_map is not None and not arrived and len(history) == 4 and step % args.replan_steps == 0 and \
                    step - last_replan_step >= args.replan_steps:
                body, paths_map, candidate_info = candidates_for_goal(bank, pose, goal_map)
                candidate_paths = np.stack([body[name] for name in ('direct', 'left_20', 'right_20')])
                dem = np.stack([row[1] for row in history])
                actor_state = observed['policy_pre_noise'][0].detach().cpu().numpy()
                validate_evaluator_sample(dem, actor_state, candidate_paths,
                                          dem_stamps_ns=[row[0] for row in history],
                                          state_stamp_ns=stamp_ns, now_ns=stamp_ns)
                begin = time.monotonic_ns()
                result = evaluator.predict(dem, actor_state, candidate_paths,
                                           np.ones((1, 3), dtype=bool))[0]
                elapsed_ms = (time.monotonic_ns() - begin) / 1e6
                selected = select_best(result)
                scores = dict(zip(('direct', 'left_20', 'right_20'), map(float, result)))
                tile_origin = bare.scene.env_origins[0, :2].detach().cpu().numpy()
                selected_tile = paths_map[selected] - tile_origin
                if follower is not None:
                    follower.close()
                follower = PureFollower(selected_tile.tolist(), library=library)
                last_replan_step = step
                trace.write(json.dumps({'event': 'selection', 'step': step, 'stamp_ns': stamp_ns,
                                        'selected': selected, 'scores_m': scores,
                                        'goal_map': goal_map.tolist(), 'candidate_info': candidate_info,
                                        'inference_latency_ms': elapsed_ms}, allow_nan=False) + '\n')
            command = (0., 0., 0.)
            diagnostic = {}
            if follower is not None:
                tile_origin = bare.scene.env_origins[0, :2].detach().cpu().numpy()
                local = (pose[0] - tile_origin[0], pose[1] - tile_origin[1], pose[2])
                command, diagnostic = follower.step_diagnostic(*local, stamp_ns)
                if diagnostic['goal_reached'] and command != (0., 0., 0.):
                    raise ValueError('follower Goal nonzero command')
                if diagnostic['goal_reached'] and goal_map is not None and \
                        np.linalg.norm(np.asarray(pose[:2]) - goal_map) <= .15:
                    arrived = True
            commands = torch.zeros((4, 3), device=bare.device)
            commands[0] = torch.tensor(command, dtype=commands.dtype, device=commands.device)
            override.set(commands)
            obs, observed, command_detail = reconcile_command_observation(
                obs, observed, bare, commands, latency_prev_for_reconcile)
            if not bool(torch.isfinite(obs['policy']).all()):
                raise ValueError('Actor input nonfinite')
            with torch.inference_mode():
                action = policy.act_inference(obs)
            if tuple(action.shape) != (4, 6) or not bool(torch.isfinite(action).all()):
                raise ValueError('Actor action shape/nonfinite')
            latency_prev_for_reconcile = (bare._latency_policy_obs_prev.detach().clone()
                                          if bare._latency_policy_obs_prev is not None else None)
            capture.begin_step()
            with torch.inference_mode():
                next_obs, reward, done, _ = env.step(action)
            observed = capture.finish_step(next_obs)
            if bool(torch.any(done)):
                goal_map = None
                arrived = False
                selected = None
                paths_map = {}
                scores = {}
                if follower is not None:
                    follower.close()
                    follower = None
                history.clear()
            obs = next_obs
            post_pose = pose_map(robot)
            status = 'arrived' if arrived else ('active' if follower is not None else 'idle')
            state = {'type': 'state', 'stamp_ns': stamp_ns + STEP_NS,
                     'pose_map': post_pose,
                     'pose_z_m': float(robot.data.root_pos_w[0, 2]),
                     'velocity_body': bare.scene['robot'].data.root_lin_vel_b[0].detach().cpu().tolist()[:2] +
                                      [float(bare.scene['robot'].data.root_ang_vel_b[0, 2])],
                     'actor_pre_noise': observed['policy_pre_noise'][0].detach().cpu().tolist(),
                     'dem': history[-1][1].tolist() if history else None,
                     'dem_stamp_ns': history[-1][0] if history else None,
                     'dem_sensor_update_ns': history[-1][2] if history else None,
                     'selected': selected, 'candidate_paths_map': {k: v.tolist() for k, v in paths_map.items()},
                     'scores_m': scores, 'goal_map': goal_map.tolist() if goal_map is not None else None,
                     'command': command, 'status': status}
            wire.send(state)
            trace.write(json.dumps({'event': 'step', 'step': step, 'stamp_ns': stamp_ns,
                                    'pose_pre': pose, 'pose_post': post_pose, 'command': command,
                                    'actor_action': action[0].detach().cpu().tolist(),
                                    'actor_input_command': command_detail['actor_seen'][0],
                                    'follower': diagnostic, 'reward': float(reward[0]),
                                    'done': bool(done[0]), 'selected': selected,
                                    'dem_sensor_update_ns': history[-1][2] if history else None,
                                    'dem_sensor_age_ns': stamp_ns - history[-1][2] if history else None},
                                   allow_nan=False) + '\n')
            trace.flush()
            os.fsync(trace.fileno())
            next_wall += .05
            time.sleep(max(0., next_wall - time.monotonic()))
        durable_json(args.output/'complete.json', {'status': 'completed', 'steps': step + 1})
    except Exception as exc:
        durable_json(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                                  'traceback': traceback.format_exc()})
        raise
    finally:
        if trace is not None:
            trace.close()
        if follower is not None:
            follower.close()
        if wire is not None:
            wire.close()
        if capture is not None:
            capture.close()
        if override is not None:
            override.close()
        if env is not None:
            env.close()
        files = {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*')
                 if p.is_file() and p.name not in ('sha-ledger.json', 'host-seal.json')}
        durable_json(args.output/'sha-ledger.json', files)
        durable_json(args.output/'host-seal.json', {'ledger_sha256': sha256(args.output/'sha-ledger.json'),
                                                   'file_count': len(files)})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            durable_json(args.output/'failure.json', {'type': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc(),
                'stage': 'preflight_or_startup'})
        raise
    finally:
        simulation_app.close()
