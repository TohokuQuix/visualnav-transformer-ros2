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
parser.add_argument('--output', type=Path, default=Path('/tmp/eatnav-phase-c/smoke'))
parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'))
parser.add_argument('--reference', type=Path, default=Path('/tmp/eatnav-phase-c/reference/best.pt'))
parser.add_argument('--reference-config', type=Path, default=Path('/tmp/eatnav-phase-c/reference-eat.yaml'))
parser.add_argument('--resume', action='store_true')
parser.add_argument('--case', choices=['duplicate','candidate_smoke'], required=True)
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
          'mode': 'few_state_smoke', 'case': args.case, 'seed': 20260929, 'branches': 8,
          'steps': 40, 'terrain_type': 'random_rough'}
case_marker = args.output/f'{args.case}.marker.json'
if args.resume and case_marker.exists():
    marker = json.loads(case_marker.read_text())
    if (marker['source'] != source or
        marker['summary_sha256'] != sha256_file(args.output/f'{args.case}.json') or
        marker['trace_sha256'] != sha256_file(args.output/f'{args.case}.npz')):
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
    poses = local_poses(bare)
    if float(np.max(np.abs(np.asarray(poses)-np.asarray(poses)[0]))) > 1e-4:
        raise ValueError('initial local root poses differ after cloning')
    dem, valid_fraction = dem_at_source(bare, config)
    if valid_fraction < config.dem_min_valid_fraction:
        raise ValueError('initial DEM is too sparse for candidate smoke')
    input_dem = torch.from_numpy(np.repeat(dem[None], config.dem_frame_count, axis=0)[None]).to(bare.device)
    goal = torch.tensor([[0.8,0.0]], device=bare.device)
    with torch.inference_mode():
        candidates = [reference.sample(input_dem, goal, None, seed=1001+i)[0].detach().cpu().numpy() for i in range(8)]
    paths = [map_path(candidates[0] if duplicate else candidates[i], poses[i]) for i in range(8)]
    bridge = FollowerBridgeClient(timeout=3.0)
    bridge.request('reset', poses=poses, paths=paths)
    def command_provider():
        response = bridge.request('step', poses=local_poses(bare), timeout_sec=2.0)
        if not response['fresh']:
            raise RuntimeError('stale follower command: ' + response.get('error',''))
        return response['commands']
    override = CommandOverride(bare.command_manager, 8, provider=command_provider)
    capture = ObservationCapture(bare.observation_manager)
    override.set(command_provider())
    capture.begin_step()
    initial_raw = bare.observation_manager.compute()
    initial_obs = bare._apply_policy_observation_latency(initial_raw)
    current_capture = capture.finish_step(initial_obs)
    obs = initial_obs
    start_xy = np.asarray(local_poses(bare))[:, :2]
    trace = {key: [] for key in ('pre_noise','post_noise','actor_input','critic_obs','actor_latent',
                                  'actor_action','critic_value','command','xy','reward','contact_max','dem_height')}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(40):
        with torch.inference_mode():
            action, latent, value = capture_actor_critic(policy, obs['policy'], obs['critic'], latent_layer=6)
        if action.shape != (8,6) or latent.shape != (8,256) or obs['policy'].shape != (8,22) or obs['critic'].shape != (8,267):
            raise ValueError('Actor/Critic/action shape mismatch')
        trace['pre_noise'].append(current_capture['policy_pre_noise'].cpu().numpy())
        trace['post_noise'].append(current_capture['policy_post_noise_pre_delay'].cpu().numpy())
        trace['actor_input'].append(current_capture['policy_actor_input'].cpu().numpy())
        trace['critic_obs'].append(current_capture['critic_observation'].cpu().numpy())
        trace['actor_latent'].append(latent.cpu().numpy())
        trace['actor_action'].append(action.cpu().numpy())
        trace['critic_value'].append(value.cpu().numpy())
        trace['command'].append(bare.command_manager.get_command('base_velocity').detach().cpu().numpy())
        capture.begin_step()
        with torch.inference_mode():
            next_obs, reward, done, _ = env.step(action)
        current_capture = capture.finish_step(next_obs)
        trace['xy'].append(np.asarray(local_poses(bare))[:, :2]-start_xy)
        trace['reward'].append(reward.detach().cpu().numpy())
        force = torch.linalg.vector_norm(bare.scene['contact_forces'].data.net_forces_w,dim=-1).max(dim=-1).values
        trace['contact_max'].append(force.detach().cpu().numpy())
        heights = bare.scene['height_scanner_eat'].data.ray_hits_w[:, :, 2]
        trace['dem_height'].append(torch.nan_to_num(heights).detach().cpu().numpy())
        if bool(done.any()):
            raise ValueError('episode reset during 40-step few-state smoke')
        obs = next_obs
        if step in (0,9,19,29,39):
            atomic_json(args.output/'status.json', {'stage': case, 'step': step+1, 'source': source})
            print('PHASE_C_SMOKE',case,step+1,'elapsed',round(time.monotonic()-started,2),flush=True)
    arrays = {key:np.asarray(values) for key,values in trace.items()}
    destination = args.output/f'{case}.npz'
    temp = args.output/f'{case}.tmp.npz'
    np.savez_compressed(temp, **arrays)
    temp.replace(destination)
    summary = {'case': case, 'duplicate': duplicate, 'runtime_sec': time.monotonic()-started,
               'clone_report': clone_report, 'initial_local_pose': poses,
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
    capture.close(); override.close(); bridge.close(); env.close()
    return summary


def main():
    config = load_eat_config(args.reference_config)
    reference, _, _ = load_checkpoint(args.reference, config)
    reference = reference.to('cuda' if torch.cuda.is_available() else 'cpu')
    cases = {}
    started = time.monotonic()
    try:
        for case, duplicate in [(args.case, args.case == 'duplicate')]:
            atomic_json(args.output/'status.json', {'stage': case+'_starting', 'source': source})
            cases[case] = run_case(case, duplicate, reference, config)
            atomic_json(args.output/f'{case}.json', cases[case])
        result = {'source': source, 'cases': cases, 'runtime_sec': time.monotonic()-started,
                  'formal_rollout_performed': False, 'total_process_seconds': time.monotonic()-started_total}
        atomic_json(args.output/'result.json', result)
        atomic_json(case_marker, {'source': source,
                    'summary_sha256': sha256_file(args.output/f'{args.case}.json'),
                    'trace_sha256': sha256_file(args.output/f'{args.case}.npz')})
        atomic_json(args.output/'complete.marker.json', {'source': source,
                    'result_sha256': sha256_file(args.output/'result.json')})
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
