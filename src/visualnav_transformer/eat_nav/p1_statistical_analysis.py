"""Outcome extraction and descriptive batch-level P1 pilot comparisons."""

from __future__ import annotations

import math
import random
import statistics

from scipy.stats import t as student_t

from .p1_statistical_protocol import validate_assignment


METRICS = ('discounted_reward_2s', 'global_goal_progress_2s_m')


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def extract_diagnostic(trace: list[dict], goal: tuple[float, float]) -> dict:
    """Existing six-second global-goal progress/termination auxiliary readout."""
    if not trace or len(trace)>120 or [row['control_step'] for row in trace]!=list(range(len(trace))):
        raise ValueError('diagnostic trace step continuity')
    terminal=next((row for row in trace if row.get('terminal')),None)
    if terminal is not None:
        return {'valid_to_6s':False,'global_goal_progress_6s_m':None,
                'first_terminal_step':terminal['control_step'],
                'termination_reason':terminal['terminal'].get('termination_reason')}
    if len(trace)!=120:
        raise ValueError('unterminated diagnostic trace must reach step 119')
    first=trace[0]['root_state'];final=trace[119]['post_root_state']
    if len(first)!=13 or len(final)!=13 or not all(_finite(v) for v in first+final):
        raise ValueError('diagnostic root state shape/nonfinite')
    w,x,y,z=first[3:7]
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    target_x=first[0]+goal[0]*math.cos(yaw)-goal[1]*math.sin(yaw)
    target_y=first[1]+goal[0]*math.sin(yaw)+goal[1]*math.cos(yaw)
    return {'valid_to_6s':True,
            'global_goal_progress_6s_m':math.hypot(*goal)-math.hypot(
                target_x-final[0],target_y-final[1]),
            'first_terminal_step':None,'termination_reason':None}


def extract_outcome(trace: list[dict], goal: tuple[float, float], gamma: float = .99) -> dict:
    primary=trace[:40]
    if not primary or [row['control_step'] for row in primary] != list(range(len(primary))):
        raise ValueError('primary trace must start at step 0 without gaps')
    if not 0 < gamma <= 1 or len(goal) != 2:
        raise ValueError('invalid Outcome contract')
    terminal = next((row for row in primary if row.get('terminal')), None)
    if len(primary)<40 and (terminal is None or terminal is not primary[-1]):
        raise ValueError('short primary trace without final physical terminal')
    if terminal is not None:
        return {'valid_primary': False, 'discounted_reward_2s': None,
                'global_goal_progress_2s_m': None,
                'goal_related_reward_2s': None,
                'first_terminal_step': terminal['control_step'],
                'termination_reason': terminal['terminal'].get('termination_reason')}
    rewards = [row.get('reward') for row in trace[:40]]
    if not all(_finite(x) for x in rewards):
        raise ValueError('nonfinite or missing primary reward')
    first = trace[0]['root_state']
    final = trace[39]['post_root_state']
    if len(first) != 13 or len(final) != 13 or not all(_finite(x) for x in first+final):
        raise ValueError('root state invalid')
    w, x, y, z = first[3:7]
    yaw = math.atan2(2*(w*z + x*y), 1-2*(y*y+z*z))
    target_x = first[0] + goal[0]*math.cos(yaw) - goal[1]*math.sin(yaw)
    target_y = first[1] + goal[0]*math.sin(yaw) + goal[1]*math.cos(yaw)
    goal_related = [row.get('goal_related_reward') for row in trace[:40]]
    return {'valid_primary': True,
            'discounted_reward_2s': sum(gamma**step * value for step, value in enumerate(rewards)),
            'global_goal_progress_2s_m': math.hypot(*goal) -
            math.hypot(target_x-final[0], target_y-final[1]),
            'goal_related_reward_2s': sum(gamma**step * value for step, value in enumerate(goal_related))
            if all(_finite(x) for x in goal_related) else None,
            'first_terminal_step': None, 'termination_reason': None}


def _percentile(values, probability):
    values = sorted(values)
    index = (len(values)-1)*probability
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def _metric_summary(contrasts, candidate_values, direct_values, margin, seed):
    n = len(contrasts)
    if n == 0:
        return {'independent_contrasts': [], 'mean_difference': None,
                'confidence_interval_95': None, 'recommended_total_batches': 'unestimable'}
    mean = statistics.mean(contrasts)
    if n == 1:
        return {'independent_contrasts': contrasts, 'mean_difference': mean,
                'confidence_interval_95': None, 'recommended_total_batches': 'unestimable',
                'candidate_sd': None, 'direct_sd': statistics.stdev(direct_values) if len(direct_values)>1 else None}
    rng = random.Random(seed)
    resampled = [statistics.mean(rng.choices(contrasts, k=n)) for _ in range(2000)]
    sd = statistics.stdev(contrasts)
    half_width = margin/2
    tcrit = float(student_t.ppf(.975, n-1))
    count = max(n, math.ceil((tcrit*sd/half_width)**2)) if sd > 0 else n
    return {'independent_contrasts': contrasts, 'mean_difference': mean,
            'confidence_interval_95': [_percentile(resampled,.025), _percentile(resampled,.975)],
            'batch_contrast_sd': sd,
            'candidate_sd': statistics.stdev(candidate_values) if len(candidate_values)>1 else None,
            'direct_sd': statistics.stdev(direct_values) if len(direct_values)>1 else None,
            'recommended_total_batches': count,
            'practical_margin': margin,
            'descriptive_effect_category': ('at_least_practical_margin' if abs(mean)>=margin
                                            else 'below_practical_margin_point_estimate_only')}


def analyze(rollouts: list[dict], manifest: dict, config: dict) -> dict:
    if manifest.get('paired_policy_noise') or config.get('paired_policy_noise'):
        raise ValueError('pairing requires independently verified aligned streams')
    validate_assignment(manifest)
    expected = {t['trial_id'] for t in manifest['trials']}
    if len(rollouts) != len(expected) or {r.get('trial_id') for r in rollouts} != expected:
        raise ValueError('trial coverage mismatch')
    by_id = {r['trial_id']: r for r in rollouts}
    results = []
    direct_diagnostics = []
    for group in sorted({t['anchor_id'] for t in manifest['trials']}):
        rows = [t for t in manifest['trials'] if t['anchor_id'] == group]
        for batch in (0, 1):
            direct = [(r,by_id[r['trial_id']]) for r in rows if r['batch_index']==batch and
                      r['path_identity']=='direct_reference']
            direct_diagnostics.append({'anchor_id': group, 'batch_index': batch,
                                       'slots': [r['slot'] for r,_ in direct],
                                       'values': {metric:[o[metric] for _,o in direct] for metric in METRICS}})
        alternatives = sorted({r['path_sha256'] for r in rows if r['path_identity']!='direct_reference'})
        for path_sha in alternatives:
            item = {'anchor_id': group, 'path_sha256': path_sha,
                    'independent_batch_count': 2,
                    'confidence_interval_kind': 'pilot_descriptive_underpowered_cluster_bootstrap',
                    'paired_policy_noise': False}
            for metric in METRICS:
                contrasts, candidates, references = [], [], []
                for batch in (0, 1):
                    block = [r for r in rows if r['batch_index']==batch]
                    candidate = next(by_id[r['trial_id']][metric] for r in block if r['path_sha256']==path_sha)
                    direct = [by_id[r['trial_id']][metric] for r in block if r['path_identity']=='direct_reference']
                    if _finite(candidate) and all(_finite(v) for v in direct):
                        contrasts.append(candidate-statistics.mean(direct))
                        candidates.append(candidate)
                        references.extend(direct)
                margin = config['minimum_meaningful_discounted_reward'] if metric=='discounted_reward_2s' else config['minimum_meaningful_progress_m']
                item[metric] = _metric_summary(contrasts,candidates,references,margin,
                                               int(path_sha[:8],16))
                item[metric]['complete_batch_count'] = len(contrasts)
            results.append(item)
    missing_primary = [r['trial_id'] for r in rollouts if not r['valid_primary']]
    return {'schema': 'p1_statistical_pilot_analysis_v1',
            'analysis_unit': 'StateGroup x independent batch',
            'policy_action_mode': 'act_inference', 'paired_policy_noise': False,
            'confirmatory_significance_claim': False,
            'multiplicity_for_later_confirmation': 'Holm within 4 alternatives x 2 primary metrics per StateGroup',
            'trial_count': len(rollouts), 'incomplete_primary_trial_ids': missing_primary,
            'direct_reference_diagnostics': direct_diagnostics,
            'comparisons': results}
