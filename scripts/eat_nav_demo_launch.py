#!/usr/bin/env python3
"""Start one persistent GUI demo attempt; never manage the external router."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ONIX = Path('/home/kojima/lib/onix_isaac_lab')
CHECKPOINT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/'
                  'motion-14-training-20261014-04/training/full-seed123/best.pt')
TRAINING_ROOT = CHECKPOINT.parents[2]
DEMO_CANDIDATE_ORDER = tuple(json.loads((REPO/'docs/eat-nav/motion-predictor-vnext/'
                                         'motion_14_training_contract.json').read_text())['candidate_order'])
DEMO_CANDIDATE_COUNT = len(DEMO_CANDIDATE_ORDER)
RL = ONIX/'logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'


def run(*cmd):
    subprocess.run(cmd, check=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def resource_sample(root, sim, frontend):
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)
    stat = os.statvfs(root)
    return {'wall_ns': time.time_ns(), 'sim_docker_client_pid': sim.pid,
            'frontend_pid': frontend.pid if frontend else None,
            'gpu': gpu.stdout.strip() if gpu.returncode == 0 else None,
            'gpu_query_error': gpu.stderr.strip() if gpu.returncode else None,
            'host_mem_available_kb': next((int(line.split()[1]) for line in
                 Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')), None),
            'filesystem_used_bytes': (stat.f_blocks - stat.f_bfree) * stat.f_frsize}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=2026101101)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--replan-steps', type=int, default=10)
    parser.add_argument('--num-envs', type=int, choices=(1, 4), default=1)
    parser.add_argument('--terrain-type', choices=(
        'k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv',
        'random_rough', 'boxes', 'hf_pyramid_slope', 'k_rail_field'), default='boxes')
    parser.add_argument('--terrain-level', type=int, choices=range(5), default=0)
    parser.add_argument('--residual-diagnostics', action='store_true')
    parser.add_argument('--port', type=int, default=29553)
    parser.add_argument('--output-root', type=Path, default=ONIX)
    args = parser.parse_args()
    results = json.loads((TRAINING_ROOT/'results.json').read_text())
    if DEMO_CANDIDATE_COUNT != 14 or results['demo_adoption']['decision'] != 'pass' or \
            results['selected_full_seed'] != 123 or \
            results['variants']['full']['123']['checkpoint_sha256'] != sha(CHECKPOINT):
        raise ValueError('adopted fourteen-Path checkpoint/result mismatch')
    series = 'cancel-residual-v11-motion14' if args.residual_diagnostics else 'terrain-v10-motion14'
    schema = ('eatnav_ros2_isaac_gui_demo_cancel_residual_v11_motion14'
              if args.residual_diagnostics else 'eatnav_ros2_isaac_gui_demo_terrain_v10_motion14')
    attempt = f'.eatnav-ros2-gui-demo-{series}-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    root = args.output_root/attempt
    root.mkdir(mode=0o755)
    staging = f'/tmp/eatnav-demo-{attempt.removeprefix(".eatnav-")}'
    run('docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', staging+'/src', staging+'/config')
    run('docker', 'cp', str(REPO/'src/visualnav_transformer'), 'isaac-lab-onix:'+staging+'/src/')
    for source, target in (
        (REPO/'scripts/eat_nav_demo_isaac.py', staging+'/sim.py'),
        (REPO/'config/eat_nav.yaml', staging+'/config/eat_nav.yaml'),
        (CHECKPOINT, staging+'/config/checkpoint.pt'),
    ):
        run('docker', 'cp', str(source), 'isaac-lab-onix:'+target)
    manifest = {'schema': schema, 'attempt': attempt,
                'source_git_head': git_head(),
                'container_staging': staging, 'checkpoint_sha256': sha(CHECKPOINT),
                'predictor': 'adopted Motion-14 Full seed 123; lowest aggregate validation loss',
                'selection_metric': 'predicted_2s_global_goal_distance_reduction_m',
                'candidate_geometry_matches_training': True,
                'candidate_bank': 'eatnav_forward_reverse_path_bank_v1',
                'candidate_count': DEMO_CANDIDATE_COUNT,
                'candidate_order': list(DEMO_CANDIDATE_ORDER),
                'training_results_sha256': sha(TRAINING_ROOT/'results.json'),
                'rl_checkpoint_sha256': sha(RL),
                'source_sha256': sha(REPO/'scripts/eat_nav_demo_isaac.py'),
                'frontend_sha256': sha(REPO/'scripts/eat_nav_demo_ros_frontend.py'),
                'launcher_sha256': sha(__file__),
                'rviz_config_sha256': sha(REPO/'config/eat_nav_demo.rviz'),
                'contract_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_contract.py'),
                'motion_adapter_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_motion_14.py'),
                'motion_model_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
                'env_cfg_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_isaac_env_cfg.py'),
                'reset_policy_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_reset_policy.py'),
                'turn_escape_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_turn_escape.py'),
                'follower_core_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/p1_follower_core.py'),
                'observation_adapter_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/p1_g3_rosfree_control.py'),
                'candidate_bank_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py'),
                'eat_config_sha256': sha(REPO/'config/eat_nav.yaml'),
                'headless': args.headless, 'seed': args.seed, 'port': args.port,
                'max_steps': args.max_steps, 'replan_steps': args.replan_steps,
                'num_envs': args.num_envs, 'terrain_type': args.terrain_type,
                'terrain_level': args.terrain_level,
                'reset_policy': 'tilt_over_only', 'flipper_fold_mode': 0.0,
                'reset_flipper_pose': 'front_folded_rear_extended',
                'turn_escape': 'policy_preprocessing_bridge_launch_defaults',
                'residual_diagnostics': args.residual_diagnostics,
                'terrain_selection_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/demo_terrain.py')}
    (root/'launch-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    # The simulator's sealed output is separate from launcher logs and manifest.
    output = root/'sim-output'
    sim_cmd = ['docker', 'exec', '-e', f'PYTHONPATH={staging}/src:/workspace/isaaclab_onix/source/onix_isaac_lab:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks',
               'isaac-lab-onix', '/isaac-sim/python.sh', staging+'/sim.py',
               '--output', '/workspace/isaaclab_onix/'+attempt+'/sim-output',
               '--checkpoint', staging+'/config/checkpoint.pt',
               '--rl-checkpoint', '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
               '--eat-config', staging+'/config/eat_nav.yaml',
               '--seed', str(args.seed), '--max-steps', str(args.max_steps),
               '--replan-steps', str(args.replan_steps), '--port', str(args.port),
               '--num-envs', str(args.num_envs), '--terrain-type', args.terrain_type,
               '--terrain-level', str(args.terrain_level)]
    if args.headless:
        sim_cmd.append('--headless')
    if args.residual_diagnostics:
        sim_cmd.append('--residual-diagnostics')
    frontend_cmd = ['/usr/bin/python3', str(REPO/'scripts/eat_nav_demo_ros_frontend.py'),
                    '--port', str(args.port)]
    sim = frontend = None
    try:
        with (root/'simulator.log').open('x') as sim_log, (root/'ros-frontend.log').open('x') as ros_log:
            sim = subprocess.Popen(sim_cmd, stdout=sim_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while not (output/'startup-ready.json').exists():
                if (output/'failure.json').exists():
                    raise RuntimeError('Isaac startup diagnostic failed; inspect sim-output/failure.json')
                if sim.poll() is not None:
                    raise RuntimeError(f'Isaac startup failed, exit={sim.returncode}; inspect simulator.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Isaac startup timeout')
                time.sleep(.5)
            frontend = subprocess.Popen(frontend_cmd, stdout=ros_log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp'})
            with (root/'resource-series.jsonl').open('x') as resource_log:
                while sim.poll() is None and frontend.poll() is None:
                    resource_log.write(json.dumps(resource_sample(root, sim, frontend)) + '\n')
                    resource_log.flush()
                    os.fsync(resource_log.fileno())
                    time.sleep(1.)
            # On normal bounded completion the simulator closes IPC first;
            # the frontend then exits before the Isaac process finishes its
            # own Kit shutdown. A completed sealed run is the authority.
            if frontend.poll() is not None and sim.poll() is None:
                if not (output/'complete.json').exists():
                    if (output/'failure.json').exists():
                        sim.wait(timeout=60)
                        raise RuntimeError('Isaac diagnostic failed; inspect sim-output/failure.json')
                    raise RuntimeError(f'ROS frontend exited before sim completion, exit={frontend.returncode}')
                sim.wait(timeout=60)
            if sim.returncode != 0:
                raise RuntimeError(f'Isaac demo exited, exit={sim.returncode}')
            if not (output/'complete.json').exists() or not (output/'host-seal.json').exists():
                raise RuntimeError('Isaac completion/seal evidence missing')
    finally:
        for process in (frontend, sim):
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
    print(root)


if __name__ == '__main__':
    main()
