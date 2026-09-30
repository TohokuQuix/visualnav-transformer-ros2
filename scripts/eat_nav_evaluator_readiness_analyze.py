"""Analyze completed train-only counterfactual rollouts; no candidate selection."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import os
import numpy as np
from scipy.stats import spearmanr

from visualnav_transformer.eat_nav.command_preview_long_horizon import post_root
from visualnav_transformer.eat_nav.feasible_transfer import cluster_labels, command_cluster_labels
from visualnav_transformer.eat_nav.evaluator_readiness_analysis import oracle_curve, readiness
from eat_nav_exact_stamp_analyze import metrics_at

REPO = Path('/tmp/eatnav-vnt-candidate-coverage')
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
ROOT = Path(os.environ.get('EAT_READINESS_ROOT', '/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot'))
INPUT = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
OUT = ROOT / 'analysis'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError('analysis artifact changed: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)


def padded(array, count=120):
    a = np.asarray(array, dtype=float)
    if len(a) == 0:
        raise ValueError('empty trajectory')
    return np.concatenate((a, np.repeat(a[-1:], count-len(a), axis=0)), axis=0)


def extra_metrics(trace, metrics):
    rewards = np.asarray([x['reward'] for x in trace], dtype=float)
    forces = np.asarray([x['contact_force'] for x in trace], dtype=float)
    peak = np.linalg.norm(forces, axis=-1).max(axis=-1)
    positions = np.asarray([post_root(row)[:2] for row in trace], dtype=float)
    initial = np.asarray(trace[0]['root_state'][:2], dtype=float)
    realized = np.diff(np.vstack((initial, positions)), axis=0)
    command = np.asarray([row['command'] for row in trace], dtype=float)
    commanded = float(.05 * np.linalg.norm(command[:, :2], axis=1).sum())
    realized_distance = float(np.linalg.norm(realized, axis=1).sum())
    return {'discounted_return': float(np.dot(.99**np.arange(len(rewards)), rewards)),
            'contact_integral_n_sec': float(.05 * peak.sum()),
            'contact_peak_n': float(peak.max()),
            'commanded_motion_m': commanded,
            'realized_path_length_m': realized_distance,
            'realized_over_commanded_motion_ratio': realized_distance / commanded if commanded > 1e-9 else None,
            'observed_steps': len(trace),
            'termination_reason': trace[-1].get('terminal'),
            'six_second_observed': metrics['status'] == 'observed'}


def input_audit(summary, state):
    anchor = summary['post_follower_anchor']
    dem = np.asarray(anchor['dem_history'], dtype=float)
    goal = np.asarray(state['goal'], dtype=float)
    pre = np.asarray(anchor['policy_pre_noise'], dtype=float)
    post = np.asarray(anchor['policy_post_noise_pre_delay'], dtype=float)
    actor = np.asarray(anchor['policy_actor_input'], dtype=float)
    latent = np.asarray(anchor['actor_latent'], dtype=float)
    return {'finite_dem_goal_actor_latent': bool(np.isfinite(dem).all() and np.isfinite(goal).all()
             and np.isfinite(pre).all() and np.isfinite(post).all() and np.isfinite(actor).all()
             and np.isfinite(latent).all()),
            'dem_shape': list(dem.shape), 'actor_shape': list(actor.shape),
            'latent_shape': list(latent.shape),
            'same_step_capture_available': bool(pre.shape == post.shape == actor.shape == (22,) and latent.shape == (256,)),
            'joint_position_available': anchor.get('joint_position') is not None,
            'joint_effort_available': anchor.get('joint_effort') is not None,
            'previous_action_available': anchor.get('action_manager_previous_action') is not None,
            'critic_input_excluded': True, 'future_input_excluded': True}


def state_results(state, entries, thresholds):
    index = int(state['state_id'][-3:])
    table = {r['candidate_id']: r for r in read(INPUT/f'state-{index:03d}'/'candidate-table.json')}
    nominal = [r for r in entries if r['kind'] == 'nominal']
    if len(nominal) != 3 or not read(ROOT/'duplicate-gates'/f"{state['state_id']}.json")['pass']:
        raise ValueError('nominal duplicate gate missing')
    analysis = []
    plans, commands, realized = [], [], []
    for entry in entries:
        if entry['kind'] == 'anchor' or (entry['kind'] == 'nominal' and entry['repeat'] != 0):
            continue
        folder = ROOT/'runs'/entry['run_id']
        marker = read(folder/'run-complete.json')
        if marker['identity'] != entry:
            raise ValueError('candidate marker identity mismatch')
        for name, digest in marker['output_sha256'].items():
            if sha(folder/name) != digest:
                raise ValueError('candidate output SHA mismatch')
        summary = read(folder/'live-0.json')
        with gzip.open(folder/'live-0.trace.json.gz', 'rt') as stream:
            trace = json.load(stream)
        if not trace or len(trace) > 120:
            raise ValueError('trace length invalid')
        two = metrics_at(summary, trace, 40)
        six = metrics_at(summary, trace, 120)
        candidate = table[entry['candidate_id']]
        base_path = np.asarray(read(INPUT/f'state-{index:03d}'/f"{entry['candidate_id']}-base-path.json"), dtype=float)
        if base_path.shape != (9, 2) or not np.isfinite(base_path).all():
            raise ValueError('planned Path shape/nonfinite')
        planned = base_path[1:]
        command = padded([row['command'] for row in trace])
        roots = padded([post_root(row)[:2] for row in trace])
        roots -= np.asarray(trace[0]['root_state'][:2], dtype=float)
        audit = input_audit(summary, state)
        if not audit['finite_dem_goal_actor_latent'] or not audit['same_step_capture_available']:
            raise ValueError('online input nonfinite or Actor/latent step mismatch')
        record = {'run_id': entry['run_id'], 'state_id': state['state_id'],
                  'terrain_type': state['terrain_type'], 'candidate_id': entry['candidate_id'],
                  'kind': entry['kind'], 'seed': entry['seed'],
                  'base_path_sha256': candidate['base_path_sha256'],
                  'constraint_valid': candidate['constraints']['valid'],
                  'constraint_reasons': candidate['constraints']['reasons'],
                  'planned_metrics': candidate['planned_metrics'],
                  'two_second': two, 'six_second': six,
                  'extra': extra_metrics(trace, six), 'input_audit': audit,
                  'planned_waypoints_base': planned.tolist(),
                  'planned_delta_base': np.diff(base_path, axis=0).tolist(),
                  'trace_sha256': sha(folder/'live-0.trace.json.gz')}
        analysis.append(record)
        plans.append(planned)
        commands.append(command)
        realized.append(roots)
    if len(analysis) != 14:
        raise ValueError('nominal + 13 candidate outcomes required')
    path_modes = cluster_labels(np.asarray(plans), threshold=thresholds['planned_path_cluster_rms_m'])
    command_modes = command_cluster_labels(np.asarray(commands),
        linear_threshold=thresholds['command_linear_rms_mps'],
        angular_threshold=thresholds['command_angular_rms_radps'])
    realized_modes = cluster_labels(np.asarray(realized), threshold=thresholds['realized_trajectory_cluster_rms_m'])
    for row, p, c, r in zip(analysis, path_modes, command_modes, realized_modes):
        row['mode_id'] = {'planned': p, 'command': c, 'realized': r}
    observed = [r for r in analysis if r['six_second']['status'] == 'observed']
    valid = [r for r in observed if r['constraint_valid']]
    def spread(key):
        values = [r['six_second'][key] for r in valid]
        return float(max(values)-min(values)) if len(values) >= 2 else None
    ranges = {'state_id': state['state_id'], 'terrain_type': state['terrain_type'],
              'valid_observed_count': len(valid), 'delta_v_range': spread('delta_v'),
              'advantage_range': spread('advantage_h'), 'drawdown_range': spread('drawdown'),
              'recovery_range': spread('recovery'),
              'progress_range': spread('forward_goal_progress_m'),
              'nomad_distinct_ranks': len({r['six_second']['delta_v'] for r in valid if r['kind'] == 'nomad'})}
    metrics = {}
    for key in ('delta_v', 'advantage_h', 'drawdown', 'recovery', 'forward_goal_progress_m'):
        rows = [{'candidate_id': r['candidate_id'], 'kind': r['kind'],
                 'constraint_valid': r['constraint_valid'],
                 key: r['six_second'].get(key) if r['six_second']['status'] == 'observed' else None}
                for r in analysis]
        metrics[key] = oracle_curve(rows, epsilon=thresholds['oracle_delta_v_epsilon'], metric=key)
    correlations = {}
    labels = ('delta_v', 'advantage_h', 'drawdown', 'recovery', 'forward_goal_progress_m')
    for i, first in enumerate(labels):
        for second in labels[i+1:]:
            x = np.asarray([r['six_second'][first] for r in valid], dtype=float)
            y = np.asarray([r['six_second'][second] for r in valid], dtype=float)
            value = spearmanr(x, y).statistic if len(x) >= 3 and np.ptp(x) and np.ptp(y) else None
            correlations[f'{first}__{second}'] = float(value) if value is not None and np.isfinite(value) else None
    diversity = {'state_id': state['state_id'], 'terrain_type': state['terrain_type'],
                 'nomad_valid_precision': sum(r['constraint_valid'] for r in analysis if r['kind']=='nomad')/8,
                 'bank_valid_precision': sum(r['constraint_valid'] for r in analysis if r['kind']=='bank')/5,
                 'nomad_mode_count': {space:len({r['mode_id'][space] for r in analysis if r['kind']=='nomad'})
                                       for space in ('planned','command','realized')},
                 'bank_only_valid_modes': {space: sorted({r['mode_id'][space] for r in analysis if r['kind']=='bank' and r['constraint_valid']}
                                  - {r['mode_id'][space] for r in analysis if r['kind']=='nomad' and r['constraint_valid']})
                                  for space in ('planned','command','realized')},
                 'nomad_only_valid_modes': {space: sorted({r['mode_id'][space] for r in analysis if r['kind']=='nomad' and r['constraint_valid']}
                                  - {r['mode_id'][space] for r in analysis if r['kind']=='bank' and r['constraint_valid']})
                                  for space in ('planned','command','realized')}}
    return analysis, ranges, metrics, correlations, diversity


def main():
    if read(ROOT/'status.json')['stage'] != 'RAW_COMPLETE':
        raise RuntimeError('all 112 outcomes must be complete before analysis')
    frozen = read(DOC/'schedule.json')['runs']
    selection = read(DOC/'selection.json')['selected']
    thresholds = read(DOC/'analysis-thresholds.json')
    outcomes, ranges, oracle, correlation, diversity = [], [], [], [], []
    for index, state in enumerate(selection):
        group = frozen[17*index:17*(index+1)]
        rows, spread, metric, corr, modes = state_results(state, group, thresholds)
        outcomes.extend(rows); ranges.append(spread)
        oracle.append({'state_id':state['state_id'], 'terrain_type':state['terrain_type'], 'metrics':metric})
        correlation.append({'state_id':state['state_id'], 'spearman':corr})
        diversity.append(modes)
    if len(outcomes) != 98:
        raise ValueError('98 unique nominal/candidate outcomes missing')
    relative = readiness(ranges, min_states=thresholds['relative_support_min_states'],
                         delta_v_min=thresholds['relative_delta_v_range_min'],
                         secondary_min=thresholds['relative_advantage_or_drawdown_range_min'])
    safety = defaultdict(lambda:{'positive_states':set(), 'negative_states':set()})
    for row in outcomes:
        sid=row['state_id'];m=row['six_second'];extra=row['extra']
        flags={'stack':bool(m.get('stack')) if m['status']=='observed' else False,
               'excessive_tilt':bool(m.get('maximum_tilt_rad',0)>thresholds['excessive_tilt_rad']),
               'termination':bool(extra['termination_reason']),
               'recovery':bool(m.get('recovery',0)>thresholds['relative_advantage_or_drawdown_range_min']),
               'large_contact':bool(extra['contact_peak_n']>thresholds['large_contact_n']),
               'tracking_failure':bool(m.get('tracking_error_rms_m',0)>thresholds['tracking_failure_rms_m'])}
        for name,flag in flags.items():
            safety[name]['positive_states' if flag else 'negative_states'].add(sid)
    safety_result={name:{key:sorted(values) for key,values in groups.items()} for name,groups in safety.items()}
    safety_ready=all(len(v.get('positive_states',[]))>=thresholds['safety_positive_negative_min_distinct_states']
                     and len(v.get('negative_states',[]))>=thresholds['safety_positive_negative_min_distinct_states']
                     for v in safety_result.values())
    input_rows=[r['input_audit'] for r in outcomes]
    input_result={'finite_and_same_step':all(r['finite_dem_goal_actor_latent'] and r['same_step_capture_available'] for r in input_rows),
                  'joint_effort_captured':all(r['joint_effort_available'] for r in input_rows),
                  'previous_action_captured':all(r['previous_action_available'] for r in input_rows),
                  'online_future_or_critic_leakage':False,
                  'deployable':all(r['joint_effort_available'] for r in input_rows)}
    report={'state_count':7,'unique_nominal_and_candidate_outcomes':98,
            'nominal_duplicate_runs_excluded_from_oracle':14,
            'relative_outcome':relative,'safety_head_ready':safety_ready,
            'safety_support':safety_result,'input_readiness':input_result,
            'statistical_unit':'state','validation_used':False,'heldout_used':False,
            'retraining':False,'single_composite_score_created':False,
            'caveat':'Six-second metrics are censored for candidates terminating before 120 steps; termination remains a safety label.'}
    for name, data in [('per-candidate-outcomes.json',outcomes),('label-variance.json',ranges),
                       ('oracle-coverage.json',oracle),('label-correlations.json',correlation),
                       ('three-space-diversity.json',diversity),('readiness.json',report)]:
        atomic_once(OUT/name,data)
    atomic_once(OUT/'analysis.done.json',{'stage':'coverage_and_readiness_complete',
                'source_status_sha256':sha(ROOT/'status.json'),
                'output_sha256':{name:sha(OUT/name) for name in
                  ('per-candidate-outcomes.json','label-variance.json','oracle-coverage.json',
                   'label-correlations.json','three-space-diversity.json','readiness.json')}})
    print('READINESS_ANALYSIS_COMPLETE',relative,'safety',safety_ready,flush=True)


if __name__=='__main__':
    main()
