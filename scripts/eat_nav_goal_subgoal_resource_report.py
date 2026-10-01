"""Summarize measured full-prefix timing and forecast the frozen 48 runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np

PREFIX = {'train-003': 1621, 'train-004': 421, 'train-019': 436, 'train-021': 3241}
RESETS = {'train-003': 4, 'train-004': 1, 'train-019': 1, 'train-021': 8}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def describe(seconds):
    values = np.asarray(seconds, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite step times')
    return {'count': int(values.size), 'mean_sec': float(values.mean()),
            'median_sec': float(np.median(values)),
            'p90_sec': float(np.percentile(values, 90)),
            'p95_sec': float(np.percentile(values, 95)),
            'p99_sec': float(np.percentile(values, 99)),
            'max_sec': float(values.max())}


def event_ns(events, name):
    matches = [row['monotonic_ns'] for row in events if row['event'] == name]
    if len(matches) != 1:
        raise ValueError(f'expected one timing event {name}, got {len(matches)}')
    return matches[0]


def suffix_timing(sim_log: str, host_elapsed_sec: float, sim_elapsed_sec: float):
    def one(pattern):
        values = re.findall(pattern, sim_log, re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f'suffix boundary missing or duplicated: {pattern}, count={len(values)}')
        return int(values[0])
    reset_start = one(r'^DIAG_BRIDGE_REQUEST_ENTRY reset \d+ (\d+)$')
    reset_end = one(r'^DIAG_BRIDGE_REQUEST_RETURN reset \d+ (\d+)$')
    first_step = one(r'^DIAG_ENV_STEP_ENTRY 0 (\d+)$')
    final_step = one(r'^DIAG_ENV_STEP_RETURN 119 (\d+)$')
    finish = one(r'^DIAG_BRIDGE_REQUEST_RETURN finish (?:None|\d+) (\d+)$')
    app_close = one(r'^DIAG_APP_CLOSE_ENTRY (\d+)$')
    return {'follower_bridge_setup_sec': (reset_end - reset_start) / 1e9,
            'suffix_step_and_command_sec': (final_step - first_step) / 1e9,
            'post_suffix_trace_save_sec': (app_close - finish) / 1e9,
            'suffix_total_sec': (app_close - reset_start) / 1e9,
            'host_copy_cleanup_sec': max(0., host_elapsed_sec - sim_elapsed_sec),
            'source': 'latest-source readiness-002 120-step exact-stamp parity'}


def summarize_probe(events, durations_ns, host_start_ns, host_return_ns, run_elapsed_sec,
                    reset_neighborhood=2):
    if len(durations_ns) != PREFIX['train-003']:
        raise ValueError('train-003 must have exactly 1621 step durations')
    reset_steps = [row['global_step'] for row in events if row['event'] == 'episode_reset']
    if len(reset_steps) != RESETS['train-003']:
        raise ValueError('train-003 reset count differs from frozen episode ID')
    duration_sec = [value / 1e9 for value in durations_ns]
    near = {step for center in reset_steps for step in range(max(0, center - reset_neighborhood),
                                                             min(len(duration_sec), center + reset_neighborhood + 1))}
    normal = [value for step, value in enumerate(duration_sec) if step not in near]
    by_episode = {}
    episode = 0
    for step, value in enumerate(duration_sec):
        by_episode.setdefault(str(episode), []).append(value)
        if step in reset_steps:
            episode += 1
    reset_values = [duration_sec[step] for step in reset_steps]
    prefix_start = event_ns(events, 'prefix_replay_start')
    anchor_reached = event_ns(events, 'target_anchor_reached')
    anchor_checked = event_ns(events, 'anchor_verification_complete')
    cleanup_start = event_ns(events, 'cleanup_start')
    prefix_wall_sec = (anchor_reached - prefix_start) / 1e9
    step_sum_sec = sum(duration_sec)
    if prefix_wall_sec + 1e-6 < step_sum_sec:
        raise ValueError('step durations exceed measured prefix wall time')
    return {
        'all_step': describe(duration_sec), 'normal_step_excluding_reset_neighborhood': describe(normal),
        'reset_step': describe(reset_values), 'by_episode': {key: describe(value) for key, value in by_episode.items()},
        'reset_indices': reset_steps, 'reset_neighborhood_steps': reset_neighborhood,
        'reset_overhead_mean_sec': max(0., statistics.mean(reset_values) - statistics.mean(normal)),
        'reset_overhead_max_sec': max(0., max(reset_values) - statistics.mean(normal)),
        'docker_exec_to_isaac_sec': (event_ns(events, 'isaac_start') - host_start_ns) / 1e9,
        'scene_construction_sec': (event_ns(events, 'scene_construction_complete') - event_ns(events, 'isaac_start')) / 1e9,
        'initial_reset_and_policy_setup_sec': (event_ns(events, 'initial_reset_complete') - event_ns(events, 'scene_construction_complete')) / 1e9,
        'prefix_wall_sec': prefix_wall_sec, 'prefix_env_step_sum_sec': step_sum_sec,
        'prefix_non_step_overhead_sec': prefix_wall_sec - step_sum_sec,
        'prefix_non_step_overhead_per_step_sec': (prefix_wall_sec - step_sum_sec) / len(duration_sec),
        'anchor_verification_sec': (anchor_checked - anchor_reached) / 1e9,
        'cleanup_and_container_return_sec': (host_return_ns - cleanup_start) / 1e9,
        'anchor_host_total_sec': run_elapsed_sec,
        'fixed_anchor_overhead_sec': run_elapsed_sec - prefix_wall_sec,
    }


def forecast(probe: dict, suffix: dict, *, storage_bytes_per_run: int, observed_gpu_peak_mib: int,
             observed_anon_peak_bytes: int):
    normal = probe['normal_step_excluding_reset_neighborhood']
    overhead = probe['prefix_non_step_overhead_per_step_sec']
    fixed = probe['fixed_anchor_overhead_sec']
    suffix_extra = suffix['suffix_total_sec'] + suffix['host_copy_cleanup_sec']

    def scenario(step_sec, reset_extra, fixed_extra, suffix_sec):
        states = {}
        total = 0.
        for name, count in PREFIX.items():
            prefix = count * (step_sec + overhead) + RESETS[name] * reset_extra
            anchor = fixed_extra + prefix
            live = anchor + suffix_sec
            states[name] = {'prefix_steps': count, 'expected_resets': RESETS[name],
                            'prefix_sec': prefix, 'anchor_sec': anchor, 'fresh_live_sec': live,
                            'eleven_live_sec': 11 * live, 'state_total_sec': anchor + 11 * live}
            total += anchor + 11 * live
        return {'states': states, 'total_sec': total, 'total_hours': total / 3600}

    point = scenario(normal['mean_sec'], probe['reset_overhead_mean_sec'], fixed, suffix_extra)
    conservative = scenario(normal['p95_sec'], probe['reset_overhead_max_sec'], fixed, suffix_extra)
    worst = scenario(normal['max_sec'], probe['reset_overhead_max_sec'],
                     max(fixed, probe['anchor_host_total_sec'] - probe['prefix_env_step_sum_sec']), suffix_extra)
    return {'status': 'measured_forecast', 'formula': 'sum over four states of anchor_s + 11*(launch_s + full_prefix_s + 120_step_suffix_s + cleanup_s)',
            'source_state': 'train-003', 'anchor_runs': 4, 'live_runs': 44,
            'nominal_duplicates': 12, 'candidate_runs': 32,
            'point': point, 'conservative_p95_step': conservative,
            'conservative_plus_20_percent_sec': conservative['total_sec'] * 1.2,
            'conservative_plus_20_percent_hours': conservative['total_hours'] * 1.2,
            'worst_observed': worst,
            'storage_prediction_bytes': storage_bytes_per_run * 48,
            'observed_gpu_peak_mib': observed_gpu_peak_mib,
            'observed_anon_peak_bytes': observed_anon_peak_bytes,
            'assumptions': ['fresh Docker/Isaac process for every anchor and live run',
                            'each live run replays the complete global prefix',
                            'train-003 measured normal step and reset overhead extrapolated to other frozen states',
                            '120-step suffix and follower setup from prior exact-stamp readiness-002',
                            'no previous failed/stalled probe time added to normal step distribution']}


def write_report(folder: Path, probe: dict, forecast_result: dict):
    target = folder / 'runtime-forecast.json'
    if target.exists():
        raise FileExistsError(target)
    atomic(target, {'probe': probe, 'forecast': forecast_result,
                    'generated_unix': time.time()})
    return {'path': str(target), 'sha256': sha(target)}
