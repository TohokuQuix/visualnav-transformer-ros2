"""Isaac GUI source and actuator for the separated ROS Motion-14 nodes.

This is a demonstration series. It does not reuse sealed G3 comparisons or make
claims about predictor generalization. ROS owns inference and Path selection.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
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
parser.add_argument('--eat-config', type=Path, required=True)
parser.add_argument('--rl-env-config', type=Path, required=True)
parser.add_argument('--port', type=int, default=29553)
parser.add_argument('--seed', type=int, default=2026101101)
parser.add_argument('--max-steps', type=int, default=3000)
parser.add_argument('--replan-steps', type=int, default=10)
parser.add_argument('--num-envs', type=int, choices=(1, 4), default=1)
parser.add_argument('--terrain-type', default='boxes')
parser.add_argument('--terrain-level', type=int, choices=range(5), default=0)
parser.add_argument('--residual-diagnostics', action='store_true')
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
from visualnav_transformer.eat_nav.demo_terrain import configure_selected_terrain
from visualnav_transformer.eat_nav.demo_reset_policy import (
    DemoSensorClock, assert_demo_flipper_pose, assert_finite_sim_state, configure_demo_reset_policy,
)
from visualnav_transformer.eat_nav.demo_turn_escape import TurnStuckEscape
from visualnav_transformer.eat_nav.demo_residual import capture_physical_state
from visualnav_transformer.eat_nav.demo_contract import (
    RL_CHECKPOINT_SHA256, sensor_stamp_valid, sha256,
)
from visualnav_transformer.eat_nav.motion_14_predictor import CANDIDATE_IDS
from visualnav_transformer.eat_nav.demo_motion_14 import (
    DEMO_CHECKPOINT_SHA256, candidates_from_frozen_bank,
)
from visualnav_transformer.eat_nav.ros2_phase3_gui_contract import (
    validate_selection, validate_gated_command, validate_source_frame,
)
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation


STEP_NS = 50_000_000
PROTOCOL = 'eatnav_ros2_phase3_gui_adopted14_v1'
RESIDUAL_PROTOCOL = 'eatnav_ros2_phase3_gui_adopted14_residual_v1'


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


def dem_from_sensor(bare, eat_cfg, stamp_ns, sensor_clock):
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
    sensor_now_s = float(scanner._timestamp[0])
    sensor_update_s = float(scanner._timestamp_last_update[0])
    # Both values use Isaac's same accumulated float32 sensor clock. Their
    # difference is stable after the absolute clock drifts beyond 1 ms.
    actual_update_ns = sensor_clock.sensor_update_on_logical_clock(
        stamp_ns, sensor_now_s, sensor_update_s)
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
    protocol = RESIDUAL_PROTOCOL if args.residual_diagnostics else PROTOCOL
    ledger = {'protocol': protocol, 'source_sha256': sha256(__file__),
              'checkpoint_sha256': sha256(args.checkpoint),
              'rl_checkpoint_sha256': sha256(args.rl_checkpoint),
              'candidate_bank_source_sha256': sha256(Path(__file__).parent/'src/visualnav_transformer/eat_nav/bidirectional_bank.py'),
              'motion_model_source_sha256': sha256(Path(__file__).parent/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
              'motion_adapter_source_sha256': sha256(Path(__file__).parent/'src/visualnav_transformer/eat_nav/demo_motion_14.py'),
              'candidate_order': list(CANDIDATE_IDS),
              'eat_config_sha256': sha256(args.eat_config), 'seed': args.seed,
              'rl_env_config_sha256': sha256(args.rl_env_config),
              'max_steps': args.max_steps, 'replan_steps': args.replan_steps,
              'num_envs': args.num_envs, 'terrain_type': args.terrain_type,
              'terrain_level': args.terrain_level,
              'residual_diagnostics': args.residual_diagnostics,
              'reset_policy': 'tilt_over_only', 'flipper_fold_mode': 0.0,
              'turn_escape_source_sha256': sha256(Path(__file__).parent/'src/visualnav_transformer/eat_nav/demo_turn_escape.py')}
    durable_json(args.output/'input-ledger.json', ledger)
    if ledger['checkpoint_sha256'] != DEMO_CHECKPOINT_SHA256 or \
            ledger['rl_checkpoint_sha256'] != RL_CHECKPOINT_SHA256:
        raise ValueError('Motion-14 predictor/RL checkpoint SHA mismatch')
    if args.replan_steps < 5 or args.replan_steps % 5 or args.max_steps <= 20:
        raise ValueError('replan interval must align to 250 ms DEM history')
    eat_cfg = load_eat_config(args.eat_config)
    # The model is loaded only by the separate ROS Model Worker. The Isaac
    # process verifies its bytes for provenance but never runs inference.
    if eat_cfg.dem_frame_count != 4 or eat_cfg.dem_resolution_m != .1:
        raise ValueError('DEM history config mismatch')
    saved = yaml.load(args.rl_env_config.read_text(),
                      Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), eat_cfg, args.seed, args.num_envs)
    cfg.episode_length_s = 40.0
    configure_demo_reset_policy(cfg)
    # The stock startup USD FilteredPairs edit runs only in a single-env
    # scene and invalidates the active PhysX articulation tensor view. It is
    # already skipped by the stock event for the existing four-env demo.
    # Disable this edit here so one-env uses the same effective collision
    # setup as the verified four-env path; reset events and dynamics remain.
    cfg.events.ignore_sprocket_self_collision = None
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = args.seed
    terrain_selection = configure_selected_terrain(
        generator, cfg.scene.terrain, args.terrain_type, args.terrain_level, args.num_envs)
    # This demo fixes the selected row over episode resets; the original
    # terrain generator still determines the difficulty within that row.
    terrain_config = {
        **terrain_selection, 'seed': args.seed,
        'single_env_startup_stage_edit': 'disabled_to_match_multi_env_effective_collisions',
        'generator_config': generator.to_dict(),
        'selected_subterrain_config': generator.sub_terrains[args.terrain_type].to_dict(),
    }
    durable_json(args.output/'terrain-config.json', json.loads(json.dumps(terrain_config, default=str)))
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
    turn_escape = TurnStuckEscape()
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
        assert_finite_sim_state(robot)
        initial_flipper_pose = assert_demo_flipper_pose(robot, cfg, list(range(args.num_envs)))
        if not bool(torch.all(bare._onix_flipper_fold_mode == 0)):
            raise ValueError('demo flipper fold mode must be zero')
        override = CommandOverride(bare.command_manager, args.num_envs)
        capture = ObservationCapture(bare.observation_manager)
        library = compile_core(args.output/'core-build')
        zeros = torch.zeros((args.num_envs, 3), device=bare.device)
        override.set(zeros)
        latency_prev_for_reconcile = (bare._latency_policy_obs_prev.detach().clone()
                                      if bare._latency_policy_obs_prev is not None else None)
        capture.begin_step()
        raw = bare.observation_manager.compute()
        obs = bare._apply_policy_observation_latency(raw)
        observed = capture.finish_step(obs)
        if tuple(obs['policy'].shape) != (args.num_envs, 22) or tuple(obs['critic'].shape) != (args.num_envs, 267):
            raise ValueError('Actor observation contract mismatch')
        terrain_importer = bare.scene.terrain
        tile_metadata = terrain_importer.cfg.terrain_generator.eat_nav_tile_metadata
        selected_tiles = []
        for env_index in range(args.num_envs):
            row = int(terrain_importer.terrain_levels[env_index])
            col = int(terrain_importer.terrain_types[env_index])
            if row != args.terrain_level or col != env_index:
                raise ValueError('terrain tile selection mismatch')
            tile = tile_metadata[col * generator.num_rows + row]
            if tile['terrain_type'] != args.terrain_type:
                raise ValueError('generated terrain type mismatch')
            selected_tiles.append({'env_id': env_index, 'row': row, 'col': col,
                                   'tile': tile,
                                   'terrain_origin': bare.scene.env_origins[env_index].detach().cpu().tolist(),
                                   'robot_spawn_pose': robot.data.root_pos_w[env_index].detach().cpu().tolist()})
        durable_json(args.output/'terrain-provenance.json', {
            **terrain_selection, 'selected_tiles': selected_tiles,
            'terrain_config_sha256': sha256(args.output/'terrain-config.json'),
            'simulator_source_sha256': sha256(__file__)})
        durable_json(args.output/'startup-ready.json', {'protocol': protocol,
            'actor_shape': list(obs['policy'].shape), 'critic_shape': list(obs['critic'].shape),
            'control_dt_s': float(bare.step_dt), 'checkpoint_sha256': ledger['checkpoint_sha256'],
            'rl_checkpoint_sha256': ledger['rl_checkpoint_sha256'],
            'num_envs': args.num_envs, 'terrain_type': args.terrain_type,
            'terrain_level': args.terrain_level, 'reset_policy': 'tilt_over_only',
            'flipper_fold_modes': bare._onix_flipper_fold_mode.detach().cpu().tolist(),
            'flipper_joint_pose_order': ['rear_right', 'rear_left', 'front_right', 'front_left'],
            'flipper_joint_pose_rad': initial_flipper_pose,
            'turn_escape': 'policy_preprocessing_bridge_launch_defaults'})
        wire = GoalWire(args.port)
        trace = (args.output/'trace.jsonl').open('x')
        history = collections.deque(maxlen=4)
        sensor_clock = DemoSensorClock(STEP_NS)
        goal_map = None
        arrived = False
        selected = None
        selected_generation = 0
        paths_map = {}
        scores = {}
        episode_id = 0
        last_replan_step = -args.replan_steps
        run_start = time.monotonic()
        next_wall = run_start

        def process_ipc(message, stamp_ns, step):
            nonlocal goal_map, arrived, selected, selected_generation
            nonlocal paths_map, scores, follower, last_replan_step
            if message.get('type') in ('cancel', 'invalidate'):
                goal_map = None
                arrived = False
                selected = None
                selected_generation = 0
                paths_map = {}
                scores = {}
                history.clear()
                if follower is not None:
                    follower.close()
                    follower = None
                turn_escape.reset()
                trace.write(json.dumps({'event': message['type'], 'step': step,
                                        'stamp_ns': stamp_ns})+'\n')
            elif message.get('type') == 'goal':
                age = stamp_ns - int(message.get('stamp_ns', -1))
                if message.get('frame') != 'map' or not 0 <= age <= 500_000_000:
                    return
                value = np.asarray(message.get('xy'), dtype=np.float64)
                if value.shape != (2,) or not np.isfinite(value).all():
                    return
                goal_map = value
                arrived = False
                selected = None
                selected_generation = 0
                paths_map = {}
                scores = {}
                if follower is not None:
                    follower.close()
                    follower = None
                history.clear()
                turn_escape.reset()
                last_replan_step = step - args.replan_steps
                trace.write(json.dumps({'event': 'goal', 'step': step,
                                        'stamp_ns': stamp_ns,
                                        'goal_map': value.tolist()})+'\n')
            elif message.get('type') == 'selection':
                index = validate_selection(message, stamp_ns)
                if goal_map is None or message['generation'] < selected_generation:
                    raise ValueError('selection without live goal/generation')
                pose_now = pose_map(robot)
                body, mapped, candidate_info = candidates_from_frozen_bank(pose_now)
                role = CANDIDATE_IDS[index]
                origin = bare.scene.env_origins[0, :2].detach().cpu().numpy()
                tile_path = mapped[role] - origin
                if follower is not None:
                    follower.close()
                follower = PureFollower(tile_path.tolist(), library=library)
                selected = role
                selected_generation = int(message['generation'])
                paths_map = mapped
                scores = {}
                last_replan_step = step
                trace.write(json.dumps({'event': 'ros_selection', 'step': step,
                                        'stamp_ns': stamp_ns, 'role': role,
                                        'generation': selected_generation,
                                        'wire_path_sha256': message['wire_path_sha256'],
                                        'candidate_info': candidate_info})+'\n')
            else:
                raise ValueError('unexpected ROS IPC message')

        def await_ipc(expected, stamp_ns, step, timeout_s=3.0):
            deadline = time.monotonic()+timeout_s
            while time.monotonic() < deadline:
                for incoming in wire.poll():
                    if incoming.get('type') == expected:
                        return incoming
                    process_ipc(incoming, stamp_ns, step)
                    if goal_map is None:
                        raise RuntimeError('goal invalidated during ROS wait')
                time.sleep(.002)
            raise TimeoutError(f'ROS {expected} missing at logical stamp {stamp_ns}')

        for step in range(args.max_steps):
            stamp_ns = step * STEP_NS
            if not simulation_app.is_running():
                break
            for message in wire.poll():
                process_ipc(message, stamp_ns, step)
            pose = pose_map(robot)
            assert_finite_sim_state(robot)
            if args.residual_diagnostics:
                try:
                    contact_sensor = bare.scene['contact_forces']
                except KeyError:
                    contact_sensor = None
                physical_pre = capture_physical_state(robot, contact_sensor)
            if step % 5 == 0:
                frame, sensor_stamp = dem_from_sensor(bare, eat_cfg, stamp_ns, sensor_clock)
                history.append((stamp_ns, frame, sensor_stamp))
            source = {'type': 'source', 'stamp_ns': stamp_ns,
                      'episode_id': episode_id, 'pose_map': pose,
                      'pose_z_m': float(robot.data.root_pos_w[0, 2]),
                      'actor_pre_noise': observed['policy_pre_noise'][0].detach().cpu().tolist(),
                      'dem': history[-1][1].tolist() if step % 5 == 0 else None,
                      'dem_stamp_ns': stamp_ns if step % 5 == 0 else None,
                      'dem_sensor_update_ns': sensor_stamp if step % 5 == 0 else None}
            validate_source_frame(source)
            wire.send(source)
            if goal_map is not None and not arrived and len(history) == 4 and step % args.replan_steps == 0 and \
                    step - last_replan_step >= args.replan_steps:
                incoming = await_ipc('selection', stamp_ns, step)
                process_ipc(incoming, stamp_ns, step)
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
            follower_command = command
            measured_yaw_rate = float(robot.data.root_ang_vel_b[0, 2])
            command, turn_escape_diagnostic = turn_escape.apply(
                follower_command, measured_yaw_rate, stamp_ns)
            if follower is not None:
                wire.send({'type': 'command_request', 'stamp_ns': stamp_ns,
                           'generation': selected_generation,
                           'command': [float(command[0]), float(command[2])]})
                gated = await_ipc('gated_command', stamp_ns, step)
                vx, wz = validate_gated_command(gated, stamp_ns, selected_generation)
                if abs(vx-command[0]) > 1e-6 or abs(wz-command[2]) > 1e-6:
                    raise ValueError('ROS command gate altered active follower command')
                command = (vx, 0., wz)
            commands = torch.zeros((args.num_envs, 3), device=bare.device)
            commands[0] = torch.tensor(command, dtype=commands.dtype, device=commands.device)
            override.set(commands)
            obs, observed, command_detail = reconcile_command_observation(
                obs, observed, bare, commands, latency_prev_for_reconcile)
            diagnostic_extra_pre = {}
            if args.residual_diagnostics:
                actor_slice = slice(*command_detail['actor_slice'])
                diagnostic_extra_pre = {
                    'physical_pre': physical_pre,
                    'command_injected': command_detail['injected'][0],
                    'actor_pre_noise_command': observed['policy_pre_noise'][0, actor_slice].detach().cpu().tolist(),
                    'actor_post_noise_pre_delay_command': observed['policy_post_noise_pre_delay'][0, actor_slice].detach().cpu().tolist(),
                    'actor_post_delay_command': command_detail['actor_seen'][0],
                    'latency_alpha': command_detail['latency_alpha'][0],
                    'observation_sample_count_added': command_detail['observation_sample_count_added'],
                }
            if not bool(torch.isfinite(obs['policy']).all()):
                raise ValueError('Actor input nonfinite')
            with torch.inference_mode():
                action = policy.act_inference(obs)
            if tuple(action.shape) != (args.num_envs, 6) or not bool(torch.isfinite(action).all()):
                raise ValueError('Actor action shape/nonfinite')
            latency_prev_for_reconcile = (bare._latency_policy_obs_prev.detach().clone()
                                          if bare._latency_policy_obs_prev is not None else None)
            capture.begin_step()
            with torch.inference_mode():
                next_obs, reward, done, _ = env.step(action)
            observed = capture.finish_step(next_obs)
            assert_finite_sim_state(robot)
            reset_env_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
            if reset_env_ids:
                reset_flipper_pose = assert_demo_flipper_pose(robot, cfg, reset_env_ids)
                if bool(torch.any(bare.reset_time_outs)) or not bool(torch.all(
                        bare._onix_flipper_fold_mode == 0)):
                    raise ValueError('unexpected demo reset reason or flipper mode')
                if 0 in reset_env_ids:
                    episode_id += 1
                    wire.send({'type': 'reset', 'stamp_ns': stamp_ns + STEP_NS})
                    sensor_clock.after_fall(step)
                    latency_prev_for_reconcile = (bare._latency_policy_obs_prev.detach().clone()
                                                  if bare._latency_policy_obs_prev is not None else None)
                    goal_map = None
                    arrived = False
                    selected = None
                    paths_map = {}
                    scores = {}
                    if follower is not None:
                        follower.close()
                        follower = None
                    turn_escape.reset()
                    history.clear()
                trace.write(json.dumps({'event': 'fall_reset', 'step': step,
                                        'stamp_ns': stamp_ns + STEP_NS,
                                        'env_ids': reset_env_ids,
                                        'sensor_epoch_origin_ns': sensor_clock.episode_origin_ns,
                                        'flipper_joint_pose_rad': reset_flipper_pose,
                                        'flipper_fold_mode': 0.0}) + '\n')
            obs = next_obs
            post_pose = pose_map(robot)
            diagnostic_extra = {}
            if args.residual_diagnostics:
                diagnostic_extra = {
                    **diagnostic_extra_pre,
                    'physical_post': capture_physical_state(robot, contact_sensor),
                }
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
                     'scores_m': scores,
                     'score_definition': 'predicted_2s_global_goal_distance_reduction_m',
                     'goal_map': goal_map.tolist() if goal_map is not None else None,
                     'command': command, 'follower_command': follower_command,
                     'turn_escape': turn_escape_diagnostic, 'status': status}
            trace.write(json.dumps({'event': 'step', 'step': step, 'stamp_ns': stamp_ns,
                                    'pose_pre': pose, 'pose_post': post_pose, 'command': command,
                                    'follower_command': follower_command,
                                    'turn_escape': turn_escape_diagnostic,
                                    'actor_action': action[0].detach().cpu().tolist(),
                                    'actor_input_command': command_detail['actor_seen'][0],
                                    'follower': diagnostic, 'reward': float(reward[0]),
                                    'done': bool(done[0]), 'selected': selected,
                                    'episode_length_post': int(bare.episode_length_buf[0]),
                                    'flipper_fold_mode': float(bare._onix_flipper_fold_mode[0]),
                                    'sensor_epoch_origin_ns': sensor_clock.episode_origin_ns,
                                    'dem_sensor_update_ns': history[-1][2] if history else None,
                                    'dem_sensor_age_ns': stamp_ns - history[-1][2] if history else None,
                                    **diagnostic_extra},
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
