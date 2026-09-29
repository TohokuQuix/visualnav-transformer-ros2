"""Freeze the approved coverage-003 nominal clock diagnostic inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


OLD = Path('/tmp/eatnav-candidate-coverage/command-preview-long-horizon')
OUT = Path('/tmp/eatnav-candidate-coverage/clock-domain-diagnostic')
FOLLOWER = Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')
REPO = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen output differs: {path}')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def approved_runs(identities: dict) -> list[dict]:
    runs = identities['runs'][5:8]
    assert len(runs) == 3
    assert [(r['order'], r['repeat'], r['state_id'], r['candidate_id']) for r in runs] == [
        (5, 0, 'coverage-003', 'bank-nominal'),
        (6, 1, 'coverage-003', 'bank-nominal'),
        (7, 2, 'coverage-003', 'bank-nominal'),
    ]
    assert len({r['map_path_sha256'] for r in runs}) == 1
    return runs


def main() -> None:
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True)
    if status != '?? scripts/eat_nav_clock_diagnostic_stage0.py\n':
        raise RuntimeError('unexpected worktree content before input freeze: ' + repr(status))
    ids = json.loads((OLD / 'long-horizon-run-identities.json').read_text())
    runs = approved_runs(ids)
    state = json.loads(Path('/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json').read_text())['train'][3]
    assert state['state_id'] == 'coverage-003'
    source = {}
    for name, path in {
        'follower_source': FOLLOWER / 'src/simple_trajectory_follower.cpp',
        'follower_header': FOLLOWER / 'include/simple_trajectory_follower/simple_trajectory_follower.hpp',
        'follower_binary': Path('/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node'),
        'follower_parameters': FOLLOWER / 'config/onix.yaml',
        'bridge_source': REPO / 'scripts/eat_nav_phase_c_ros_bridge.py',
        'simulator_runner_source': REPO / 'scripts/eat_nav_command_preview_long_horizon_sim.py',
        'host_runner_source': REPO / 'scripts/eat_nav_command_preview_long_horizon_run.py',
        'run_identities': OLD / 'long-horizon-run-identities.json',
        'nominal_map_path': Path('/tmp/eatnav-candidate-coverage/pilot-inputs/state-003/bank-nominal-map-path.json'),
    }.items():
        source[name] = {'path': str(path), 'sha256': sha(path)}
    assert source['nominal_map_path']['sha256'] == runs[0]['map_path_sha256']
    originals = []
    for r in runs:
        folder = OLD / 'long-horizon/runs' / f"long-{r['order']:03d}"
        originals.append({
            'run_id': f"long-{r['order']:03d}",
            'summary': {'path': str(folder / 'live-0.json'), 'sha256': sha(folder / 'live-0.json')},
            'trace': {'path': str(folder / 'live-0.trace.json.gz'), 'sha256': sha(folder / 'live-0.trace.json.gz')},
        })
    config = {
        'state_id': 'coverage-003', 'episode_key': state['episode_key'],
        'anchor_step': state['step'], 'candidate_id': 'bank-nominal',
        'run_order': [0, 1, 2], 'repeat_count': 3,
        'control_period_sec': 0.05, 'simulator_step_sec': 0.05,
        'control_steps': 120, 'prefix_step_cap': 40,
        'motion_window_sec': 20.0, 'stuck_ratio_threshold': 0.25,
        'duplicate_xy_rms_limit_m': 0.0105,
        'nominal_path_sha256': runs[0]['map_path_sha256'],
        'follower_parameters_sha256': source['follower_parameters']['sha256'],
        'production_follower_source_sha256': source['follower_source']['sha256'],
        'production_follower_binary_sha256': source['follower_binary']['sha256'],
        'forbidden': ['candidate_comparison', 'unfinished_stage_b', 'model_retraining',
                      'follower_control_change', 'timer_change', 'use_sim_time', 'clock_publisher'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'clock-diagnostic-config.yaml'
    rendered = yaml.safe_dump(config, sort_keys=True)
    if config_path.exists() and config_path.read_text() != rendered:
        raise ValueError('frozen config differs')
    if not config_path.exists():
        config_path.write_text(rendered)
    atomic_json(OUT / 'clock-diagnostic-run-identities.json', {
        'runs': [{**r, 'diagnostic_order': i} for i, r in enumerate(runs)],
        'originals': originals,
        'state': {'state_id': state['state_id'], 'episode_key': state['episode_key'],
                  'step': state['step']},
    })
    atomic_json(OUT / 'clock-source-audit.json', {
        'git': {'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip(),
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'status_porcelain_at_freeze': status,
                'status_porcelain_before_implementation': ''},
        'source_sha256': source,
        'original_trace_sha256': originals,
        'clock_and_state_locations': {
            'wall_control_timer': 'simple_trajectory_follower.cpp:154-155 create_wall_timer(1/control_rate_hz)',
            'node_clock_sample': 'simple_trajectory_follower.cpp:373 get_clock()->now()',
            'motion_dt_and_distance': 'simple_trajectory_follower.cpp:716-748 dt=(now-last_motion_sample_time).seconds; abs(vx)*dt',
            'motion_prune': 'simple_trajectory_follower.cpp:757-771 by node-clock elapsed > stuck_window_sec',
            'window_ready': 'simple_trajectory_follower.cpp:803-810 back.stamp-front.stamp >= stuck_window_sec',
            'linear_stuck_ratio': 'simple_trajectory_follower.cpp:773-785 actual/commanded <= threshold',
            'progress_stall': 'simple_trajectory_follower.cpp:830-842 node-clock elapsed',
            'tf_lookup': 'simple_trajectory_follower.cpp:672-683 latest transform TimePointZero',
            'path_activation_reset': 'simple_trajectory_follower.cpp:244-266 resetMotionTracking and resetProgressTracking',
            'reverse_transition': 'simple_trajectory_follower.cpp:1067-1092 beginLinearStuckRecovery; reverse active travel direction',
            'bridge_node_clock': 'eat_nav_phase_c_ros_bridge.py:88,104,114 publish TF and Path with node clock',
            'bridge_command_stamp': 'eat_nav_phase_c_ros_bridge.py:54-69 TwistStamped stamp freshness',
            'bridge_step_wait': 'eat_nav_phase_c_ros_bridge.py:269-303 monotonic deadline and 20ms polling',
            'system_steady_clock_use': 'follower control does not directly use system or steady clock; wall timer uses steady scheduling',
        },
        'follower_timer_type': 'rclcpp create_wall_timer',
        'follower_node_clock_type': 'RCL_ROS_TIME with use_sim_time=false (system-time fallback)',
        'simulator_nominal_time_per_control_step_sec': 0.05,
        'original_duplicate_xy_rms_m': 0.035260755456820767,
    })
    atomic_json(OUT / 'stage-d0-complete.json', {
        'status': 'complete', 'config_sha256': sha(config_path),
        'identities_sha256': sha(OUT / 'clock-diagnostic-run-identities.json'),
        'audit_sha256': sha(OUT / 'clock-source-audit.json'),
    })
    print('D0_FROZEN', OUT)


if __name__ == '__main__':
    main()
