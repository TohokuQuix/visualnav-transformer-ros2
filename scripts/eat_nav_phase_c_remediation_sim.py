"""Few-state Phase C branch smoke with real ROS follower and frozen RL Actor."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import resource
import subprocess

started_total = time.monotonic()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/sim'))
parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'))
parser.add_argument('--reference', type=Path, default=Path('/tmp/eatnav-phase-c/reference/best.pt'))
parser.add_argument('--reference-config', type=Path, default=Path('/tmp/eatnav-phase-c/reference-eat.yaml'))
parser.add_argument('--resume', action='store_true')
parser.add_argument('--case', choices=['duplicate','recorded_command','recorded_observation','recorded_action','fixed_branch_repeat','serial_duplicate','serial_recorded_command','serial_candidate','candidate_smoke'], required=True)
parser.add_argument('--recorded', type=Path, default=None)
parser.add_argument('--repeat-id', type=int, default=0)
parser.add_argument('--fixed-path-json', type=Path, default=None)
parser.add_argument('--fixed-candidate-json', type=Path, default=None)
parser.add_argument('--candidate-index', type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


args.output.mkdir(parents=True, exist_ok=True)
source = {'rl_checkpoint_sha256': sha256_file(args.checkpoint),
          'reference_sha256': sha256_file(args.reference),
          'reference_config_sha256': sha256_file(args.reference_config),
          'mode': 'remediation_diagnostic', 'case': args.case, 'repeat_id': args.repeat_id, 'seed': 20260929, 'branches': 8,
          'steps': 40, 'terrain_type': 'random_rough',
          'recorded_trace_sha256': sha256_file(args.recorded) if args.recorded else None,
          'fixed_path_sha256': sha256_file(args.fixed_path_json) if args.fixed_path_json else None,
          'fixed_candidate_sha256': sha256_file(args.fixed_candidate_json) if args.fixed_candidate_json else None,
          'candidate_index': args.candidate_index}
case_name = args.case + (f'-{args.repeat_id}' if args.case in ('fixed_branch_repeat','serial_duplicate','serial_recorded_command','serial_candidate') else '')
case_marker = args.output/f'{case_name}.marker.json'
if args.resume and case_marker.exists():
    marker = json.loads(case_marker.read_text())
    if (marker['source'] != source or
        marker['summary_sha256'] != sha256_file(args.output/f'{case_name}.json') or
        marker['trace_sha256'] != sha256_file(args.output/f'{case_name}.npz')):
        raise ValueError('resume marker/source mismatch')
    print('PHASE_C_SMOKE_RESUME_SKIP', json.dumps(marker), flush=True)
    raise SystemExit(0)
atomic_json(args.output/'status.json', {'stage': 'initializing', 'source': source})
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks  # noqa
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from visualnav_transformer.eat_nav.phase_c_clone_cfg import make_phase_c_clone_cfg
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture, capture_actor_critic, clone_internal_branch_state


def local_poses(env):
    robot = env.scene['robot']
    pos = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    return torch.cat((pos, yaw[:, None]), dim=1).detach().cpu().numpy().tolist()


def map_path(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    points = [[0.0, 0.0]] + np.asarray(points, dtype=float).tolist()
    return [[float(x+c*px-s*py), float(y+s*px+c*py)] for px, py in points]


def dem_at_source(env, config):
    robot = env.scene['robot']
    scanner = env.scene['height_scanner_eat']
    hits = scanner.data.ray_hits_w[0, :, 2].detach().cpu().numpy().reshape(config.dem_height_cells, config.dem_width_cells)
    valid = np.isfinite(hits)
    start = scanner._ray_starts_w[0, 0, :2].detach().cpu().numpy()
    pose = local_poses(env)[0]
    origin = env.scene.env_origins[0, :2].detach().cpu().numpy()
    source = SourceGrid(height=np.where(valid, hits, 0).astype(np.float32), valid=valid,
                        origin_xy=(float(start[0]), float(start[1])), resolution_m=config.dem_resolution_m,
                        start_index_xy=(0,0), frame_id=config.reference_frame, stamp_ns=0)
    world_pose = (pose[0]+float(origin[0]), pose[1]+float(origin[1]), pose[2])
    return robot_dem(source, world_pose, config), float(valid.mean())


def rms_to_first(array):
    array = np.asarray(array, dtype=float)
    return float(np.sqrt(np.mean((array-array[:, :1])**2)))


def run_case(case, duplicate, reference, config):
    recorded = np.load(args.recorded) if args.recorded else None
    if (case.startswith('recorded_') or case in ('fixed_branch_repeat','serial_duplicate','serial_recorded_command','serial_candidate')) and recorded is None:
        raise ValueError('recorded control needs baseline trace')
    saved = yaml.load((args.checkpoint.parent/'params'/'env.yaml').read_text(), Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    cfg = make_phase_c_clone_cfg(OnixRoughEnvCfg(**toggles), config, seed=20260929,
                                 branches=8, terrain_type='random_rough', difficulty=0.5)
    env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0', cfg=cfg))
    bare = env.unwrapped
    runner = OnPolicyRunner(env, OnixRoughPPORunnerCfg().to_dict(), log_dir=None, device=bare.device)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    runner.alg.policy.load_state_dict(weights)
    policy = runner.alg.policy
    env.reset()
    clone_report = clone_internal_branch_state(bare, 0, list(range(1,8)))
    serial_override = serial_capture = serial_obs = serial_captured = serial_anchor = None
    if case in ('serial_duplicate','serial_recorded_command','serial_candidate'):
        prefix_command_step = 0
        def prefix_command():
            nonlocal prefix_command_step
            index = min(prefix_command_step, 14)
            prefix_command_step += 1
            return np.repeat(recorded['command'][index, :1], 8, axis=0)
        serial_override = CommandOverride(bare.command_manager, 8, provider=prefix_command)
        serial_capture = ObservationCapture(bare.observation_manager)
        serial_override.set(prefix_command())
        serial_capture.begin_step()
        initial_raw = bare.observation_manager.compute()
        serial_obs = bare._apply_policy_observation_latency(initial_raw)
        serial_captured = serial_capture.finish_step(serial_obs)
        history = [dem_at_source(bare, config)[0]]
        for prefix_step in range(15):
            prefix_action = torch.as_tensor(np.repeat(recorded['actor_action'][prefix_step, :1], 8, axis=0), device=bare.device)
            serial_capture.begin_step()
            with torch.inference_mode():
                serial_obs, _, done, _ = env.step(prefix_action)
            serial_captured = serial_capture.finish_step(serial_obs)
            if bool(done.any()): raise ValueError('reset during serial prefix')
            if prefix_step in (4,9,14): history.append(dem_at_source(bare, config)[0])
        with torch.inference_mode():
            _,anchor_latent,_ = capture_actor_critic(policy,serial_obs['policy'],serial_obs['critic'],latent_layer=6)
        robot = bare.scene['robot']
        local_root = robot.data.root_state_w[0].detach().clone()
        local_root[:3] -= bare.scene.env_origins[0]
        serial_anchor = {'root_state':local_root.cpu().numpy().tolist(),
                         'joint_position':robot.data.joint_pos[0].detach().cpu().numpy().tolist(),
                         'joint_velocity':robot.data.joint_vel[0].detach().cpu().numpy().tolist(),
                         'actor_input':serial_obs['policy'][0].detach().cpu().numpy().tolist(),
                         'actor_latent':anchor_latent[0].detach().cpu().numpy().tolist(),
                         'critic_observation':serial_obs['critic'][0].detach().cpu().numpy().tolist(),
                         'history_frame_count':len(history),'prefix_steps':15}
    poses = local_poses(bare)
    if case not in ('serial_duplicate','serial_recorded_command','serial_candidate') and float(np.max(np.abs(np.asarray(poses)-np.asarray(poses)[0]))) > 1e-4:
        raise ValueError('initial local root poses differ after cloning')
    dem, valid_fraction = dem_at_source(bare, config)
    if valid_fraction < config.dem_min_valid_fraction:
        raise ValueError('initial DEM is too sparse for candidate smoke')
    frame_stack = np.stack(history) if case in ('serial_duplicate','serial_recorded_command','serial_candidate') else np.repeat(dem[None], config.dem_frame_count, axis=0)
    if serial_anchor is not None:
        serial_anchor['dem_history_sha256'] = hashlib.sha256(frame_stack.tobytes()).hexdigest()
        serial_anchor['dem_history_frame_delta_rms'] = float(np.sqrt(np.mean(np.diff(frame_stack.astype(float), axis=0)**2)))
    input_dem = torch.from_numpy(frame_stack[None]).to(bare.device)
    goal = torch.tensor([[0.8,0.0]], device=bare.device)
    with torch.inference_mode():
        candidates = [reference.sample(input_dem, goal, None, seed=1001+i)[0].detach().cpu().numpy() for i in range(8)]
    paths = [map_path(candidates[0] if duplicate else candidates[i], poses[i]) for i in range(8)]
    generated_candidate_paths = [map_path(candidates[i], poses[0]) for i in range(8)]
    if case == 'serial_candidate':
        if not 0 <= args.candidate_index < 8: raise ValueError('candidate index outside K=8')
        if args.fixed_candidate_json is not None:
            generated_candidate_paths = json.loads(args.fixed_candidate_json.read_text())['generated_candidate_paths']
        paths = [generated_candidate_paths[args.candidate_index]] * 8
    if args.fixed_path_json is not None:
        fixed = json.loads(args.fixed_path_json.read_text())['candidate_paths']
        if len(fixed) != 8 or any(len(path) != config.len_traj_pred+1 for path in fixed):
            raise ValueError('fixed Path shape mismatch')
        paths = fixed
    bridge = None if case in ('recorded_command','recorded_observation','recorded_action','fixed_branch_repeat','serial_recorded_command') else FollowerBridgeClient(timeout=45.0)
    if bridge is not None:
        bridge.request('reset', poses=poses, paths=paths)
    command_step = 15 if case == 'serial_recorded_command' else 0
    def command_provider():
        nonlocal command_step
        if bridge is None:
            index = min(command_step, len(recorded['command']) - 1)
            command_step += 1
            return np.repeat(recorded['command'][index, :1], 8, axis=0)
        response = bridge.request('step', poses=local_poses(bare), timeout_sec=2.0)
        if not response['fresh']:
            raise RuntimeError('stale follower command: ' + response.get('error',''))
        return response['commands']
    override = serial_override or CommandOverride(bare.command_manager, 8, provider=command_provider)
    capture = serial_capture or ObservationCapture(bare.observation_manager)
    override.provider = command_provider
    override.set(command_provider())
    if case in ('serial_duplicate','serial_recorded_command','serial_candidate'):
        obs = serial_obs
        current_capture = serial_captured
        serial_anchor['follower_command_at_anchor'] = bare.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().tolist()
    else:
        capture.begin_step()
        initial_raw = bare.observation_manager.compute()
        obs = bare._apply_policy_observation_latency(initial_raw)
        current_capture = capture.finish_step(obs)
    start_xy = np.asarray(local_poses(bare))[:, :2]
    trace = {key: [] for key in ('pre_noise','post_noise','actor_input','critic_obs','actor_latent',
                                  'actor_action','critic_value','command','xy','reward','contact_max','dem_height',
                                  'root_state','joint_position','joint_velocity','contact_force','raycast',
                                  'processed_action','joint_target')}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(40):
        actor_input = obs['policy']
        if case == 'recorded_observation':
            actor_input = torch.as_tensor(np.repeat(recorded['actor_input'][step, :1], 8, axis=0), device=bare.device)
        with torch.inference_mode():
            action, latent, value = capture_actor_critic(policy, actor_input, obs['critic'], latent_layer=6)
        if case == 'recorded_observation' and float((action-action[:1]).abs().max()) > 1e-6:
            raise ValueError('identical recorded Actor input produced different actions')
        if case in ('recorded_action','fixed_branch_repeat'):
            action = torch.as_tensor(np.repeat(recorded['actor_action'][step, :1], 8, axis=0), device=bare.device)
        if action.shape != (8,6) or latent.shape != (8,256) or obs['policy'].shape != (8,22) or obs['critic'].shape != (8,267):
            raise ValueError('Actor/Critic/action shape mismatch')
        trace['pre_noise'].append(current_capture['policy_pre_noise'].cpu().numpy())
        trace['post_noise'].append(current_capture['policy_post_noise_pre_delay'].cpu().numpy())
        trace['actor_input'].append(actor_input.cpu().numpy())
        trace['critic_obs'].append(current_capture['critic_observation'].cpu().numpy())
        trace['actor_latent'].append(latent.cpu().numpy())
        trace['actor_action'].append(action.cpu().numpy())
        trace['critic_value'].append(value.cpu().numpy())
        trace['command'].append(bare.command_manager.get_command('base_velocity').detach().cpu().numpy())
        root_pre = bare.scene['robot'].data.root_state_w.detach().clone()
        root_pre[:, :3] -= bare.scene.env_origins
        trace['root_state'].append(root_pre.cpu().numpy())
        trace['joint_position'].append(bare.scene['robot'].data.joint_pos.detach().cpu().numpy())
        trace['joint_velocity'].append(bare.scene['robot'].data.joint_vel.detach().cpu().numpy())
        capture.begin_step()
        with torch.inference_mode():
            next_obs, reward, done, _ = env.step(action)
        current_capture = capture.finish_step(next_obs)
        trace['xy'].append(np.asarray(local_poses(bare))[:, :2]-start_xy)
        trace['reward'].append(reward.detach().cpu().numpy())
        force = torch.linalg.vector_norm(bare.scene['contact_forces'].data.net_forces_w,dim=-1).max(dim=-1).values
        trace['contact_max'].append(force.detach().cpu().numpy())
        trace['contact_force'].append(bare.scene['contact_forces'].data.net_forces_w.detach().cpu().numpy())
        rays = bare.scene['height_scanner_eat'].data.ray_hits_w.detach().clone()
        rays[:, :, :3] -= bare.scene.env_origins[:, None, :3]
        trace['raycast'].append(torch.nan_to_num(rays).cpu().numpy())
        processed_terms = []
        for term_name in ('flipper_pos', 'right_track_vel', 'left_track_vel'):
            term = bare.action_manager._terms[term_name]
            processed = getattr(term, '_processed_actions', None)
            if processed is None: processed = getattr(term, '_processed', None)
            if processed is None: raise ValueError(f'processed action unavailable: {term_name}')
            processed_terms.append(processed.reshape(8, -1))
        trace['processed_action'].append(torch.cat(processed_terms, dim=1).detach().cpu().numpy())
        target = getattr(bare.scene['robot'].data, 'joint_pos_target', None)
        if target is None: raise ValueError('robot joint target unavailable')
        trace['joint_target'].append(target.detach().cpu().numpy())
        heights = bare.scene['height_scanner_eat'].data.ray_hits_w[:, :, 2]
        trace['dem_height'].append(torch.nan_to_num(heights).detach().cpu().numpy())
        if bool(done.any()):
            raise ValueError('episode reset during 40-step few-state smoke')
        obs = next_obs
        if step in (0,9,19,29,39):
            atomic_json(args.output/'status.json', {'stage': case, 'step': step+1, 'source': source})
            print('PHASE_C_SMOKE',case,step+1,'elapsed',round(time.monotonic()-started,2),flush=True)
    arrays = {key:np.asarray(values) for key,values in trace.items()}
    destination = args.output/f'{case_name}.npz'
    temp = args.output/f'{case_name}.tmp.npz'
    np.savez_compressed(temp, **arrays)
    temp.replace(destination)
    summary = {'case': case, 'duplicate': duplicate, 'runtime_sec': time.monotonic()-started,
               'clone_report': clone_report, 'initial_local_pose': poses, 'serial_anchor': serial_anchor,
               'generated_candidate_paths': generated_candidate_paths,
               'dem_valid_fraction': valid_fraction,
               'initial_pre_noise_rms': float(np.sqrt(np.mean((arrays['pre_noise'][0]-arrays['pre_noise'][0,:1])**2))),
               'initial_actor_input_rms': float(np.sqrt(np.mean((arrays['actor_input'][0]-arrays['actor_input'][0,:1])**2))),
               'branch_rms': {key:rms_to_first(arrays[key]) for key in ('pre_noise','post_noise','actor_input',
                       'actor_latent','actor_action','command','xy','critic_value','reward','contact_max','dem_height')},
               'contact_force_max': float(arrays['contact_max'].max()),
               'reward_range': [float(arrays['reward'].min()),float(arrays['reward'].max())],
               'critic_range': [float(arrays['critic_value'].min()),float(arrays['critic_value'].max())],
               'trace_sha256': sha256_file(destination),
               'candidate_paths': paths,
               'policy_term_names': list(current_capture['policy_term_names']),
               'policy_term_dims': [list(v) for v in current_capture['policy_term_dims']],
               'gpu_peak_torch_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
               'cpu_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'gpu_total_used_mib_end': int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip().split()[0])}
    capture.close(); override.close();
    if bridge is not None: bridge.close()
    env.close()
    return summary


def main():
    config = load_eat_config(args.reference_config)
    reference, _, _ = load_checkpoint(args.reference, config)
    reference = reference.to('cuda' if torch.cuda.is_available() else 'cpu')
    cases = {}
    started = time.monotonic()
    try:
        for case, duplicate in [(args.case, args.case != 'candidate_smoke')]:
            atomic_json(args.output/'status.json', {'stage': case+'_starting', 'source': source})
            cases[case] = run_case(case, duplicate, reference, config)
            atomic_json(args.output/f'{case_name}.json', cases[case])
        result = {'source': source, 'cases': cases, 'runtime_sec': time.monotonic()-started,
                  'formal_rollout_performed': False, 'total_process_seconds': time.monotonic()-started_total}
        atomic_json(args.output/f'{case_name}-result.json', result)
        atomic_json(case_marker, {'source': source,
                    'summary_sha256': sha256_file(args.output/f'{case_name}.json'),
                    'trace_sha256': sha256_file(args.output/f'{case_name}.npz')})
        atomic_json(args.output/f'{case_name}-complete.marker.json', {'source': source,
                    'result_sha256': sha256_file(args.output/f'{case_name}-result.json')})
        atomic_json(args.output/'status.json', {'stage': 'complete', 'source': source})
        print('PHASE_C_SMOKE_COMPLETE',json.dumps({'runtime_sec':result['runtime_sec']}),flush=True)
    except Exception as exc:
        import traceback
        atomic_json(args.output/'status.json', {'stage': 'failed', 'error':repr(exc), 'traceback':traceback.format_exc(), 'source':source})
        raise
    finally:
        simulation_app.close()


if __name__ == '__main__':
    main()
