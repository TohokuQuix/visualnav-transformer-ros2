"""Forecast the frozen 48 fresh-process runs from the measured resource probe."""
from __future__ import annotations

import json
from pathlib import Path

from eat_nav_goal_subgoal_v2_select import OUT, CONFIG, sha, atomic

TIMING = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    schedule = read(OUT / 'run-schedule-v2.json')
    frozen = read(OUT / 'v4.done.json')
    measured = read(TIMING / 'runtime-forecast.json')
    resource = read(TIMING / 'resource-anchor-result.json')
    thresholds = read(TIMING / 'resource-thresholds.json')
    if sha(OUT / 'run-schedule-v2.json') != frozen['schedule_sha256']:
        raise ValueError('v2 schedule SHA drift')
    if sha(TIMING / 'runtime-forecast.json') != '7159bbf7d8769ae6625f82e808ed0d78a6087dede265ee5d36b453f23fdcf3e3':
        raise ValueError('resource timing forecast SHA drift')
    if resource['status'] != 'pass' or thresholds['anon_shmem_cap_bytes'] != 6 * 2**30:
        raise ValueError('resource probe or threshold drift')
    runs = schedule['runs']
    if len(runs) != 48 or [r['order'] for r in runs] != list(range(48)):
        raise ValueError('48-run order drift')
    by_state = []
    for index in range(4):
        group = runs[index * 12:(index + 1) * 12]
        if len(group) != 12 or [r['kind'] for r in group] != ['anchor'] + ['duplicate'] * 3 + ['candidate'] * 8:
            raise ValueError('run group drift')
        if any(r['episode_key'][2] != 0 or r['prefix_steps'] > 80 for r in group):
            raise ValueError('episode-zero short-prefix condition failed')
        by_state.append({'state_id': group[0]['state_id'], 'terrain': group[0]['terrain'],
                         'prefix_steps': group[0]['prefix_steps'], 'expected_past_episode_resets': 0,
                         'fresh_processes': 12, 'live_processes': 11})
    probe = measured['probe']
    suffix = measured['forecast']['suffix_timing']
    suffix_sec = suffix['suffix_total_sec'] + suffix['host_copy_cleanup_sec']
    overhead = probe['prefix_non_step_overhead_per_step_sec']
    fixed = probe['fixed_anchor_overhead_sec']

    def scenario(step_sec, fixed_sec):
        states = []
        total = 0.
        for state in by_state:
            prefix = state['prefix_steps'] * (step_sec + overhead)
            anchor = fixed_sec + prefix
            live = anchor + suffix_sec
            whole = anchor + 11 * live
            states.append({**state, 'prefix_sec': prefix, 'launch_scene_reset_anchor_cleanup_sec': fixed_sec,
                           'anchor_sec': anchor, 'live_sec': live, 'state_total_sec': whole})
            total += whole
        return {'states': states, 'total_sec': total, 'total_hours': total / 3600}

    normal = probe['normal_step_excluding_reset_neighborhood']
    point = scenario(normal['mean_sec'], fixed)
    conservative = scenario(normal['p95_sec'], fixed)
    worst = scenario(normal['max_sec'], max(fixed, probe['anchor_host_total_sec'] - probe['prefix_env_step_sum_sec']))
    margin = conservative['total_sec'] * 1.2
    forecast = {'source_probe_sha256': sha(TIMING / 'runtime-forecast.json'),
                'source_anchor_sha256': sha(TIMING / 'resource-anchor-result.json'),
                'schedule_sha256': sha(OUT / 'run-schedule-v2.json'), 'source_state': 'train-003',
                'formula': 'sum over 4 states of anchor_s + 11*(launch_s + full_prefix_s + 120-step suffix_s + cleanup_s)',
                'anchor_processes': 4, 'live_processes': 44, 'full_prefix_step_total': 12 * sum(s['prefix_steps'] for s in by_state),
                'suffix_step_total': 44 * 120, 'state_prefix_steps': by_state,
                'measured_step_distribution_sec': normal,
                'measured_launch_scene_initial_reset_sec': {
                    'docker_exec_to_isaac': probe['docker_exec_to_isaac_sec'],
                    'scene_construction': probe['scene_construction_sec'],
                    'initial_reset_and_policy_setup': probe['initial_reset_and_policy_setup_sec'],
                    'anchor_verification': probe['anchor_verification_sec'],
                    'cleanup_and_container_return': probe['cleanup_and_container_return_sec']},
                'measured_suffix_sec': suffix, 'point': point,
                'conservative_p95_step': conservative,
                'conservative_plus_20_percent_sec': margin,
                'conservative_plus_20_percent_hours': margin / 3600,
                'worst_observed': worst,
                'storage_estimate_bytes': measured['forecast']['storage_prediction_bytes'],
                'gpu_estimate_mib': resource['gpu_peak_mib'],
                'anon_shmem_estimate_bytes': resource['anon_shmem_peak_bytes'],
                'host_mem_available_min_bytes': resource['host_mem_available_min_bytes'],
                'thresholds_sha256': sha(TIMING / 'resource-thresholds.json'),
                'assumptions': ['same fresh Docker/Isaac launch for every process',
                                'each episode-zero state uses recorded 100-env actions from initial reset',
                                'no past episode reset in a prefix of at most 80 steps',
                                'train-003 normal step distribution extrapolated to four frozen terrains',
                                '120-step suffix from current-source exact-stamp parity']}
    if margin > 14400 or worst['total_sec'] > 14400:
        decision = 'v2_runtime_gate_fail'
    elif (forecast['storage_estimate_bytes'] > 800_000_000 or
          forecast['gpu_estimate_mib'] > thresholds['gpu_cap_mib'] or
          forecast['anon_shmem_estimate_bytes'] > thresholds['anon_shmem_cap_bytes'] or
          forecast['host_mem_available_min_bytes'] <= thresholds['host_mem_available_min_bytes']):
        decision = 'v2_runtime_gate_fail'
    else:
        decision = 'pass'
    forecast['decision'] = decision
    atomic(OUT / 'runtime-forecast-v2.json', forecast)
    atomic(OUT / 'v5.done.json', {'decision': decision,
                                  'forecast_sha256': sha(OUT / 'runtime-forecast-v2.json')})
    print('V5_RUNTIME', decision, 'point_h', round(point['total_hours'], 3),
          'margin_h', round(margin / 3600, 3), 'worst_h', round(worst['total_hours'], 3))


if __name__ == '__main__':
    main()
