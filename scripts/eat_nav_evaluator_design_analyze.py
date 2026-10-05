"""Descriptive, StateGroup-level label analysis of sealed E1 outcomes."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import kendalltau, spearmanr

from eat_nav_e1_analyze import load_trace
from eat_nav_evaluator_design_audit import NEW, OLD, OLD_CORRECTED, main as provenance_audit, read, sha

OUTPUT = Path(__file__).resolve().parents[1] / 'docs/quality-identifiability-gate/evaluator-design-review'
CANDIDATES = ('direct_reference', 'left_1', 'left_2', 'left_3', 'right_1', 'right_2', 'right_3')
METRICS = {
    'delta_v': 1, 'H_step_advantage': 1, 'cumulative_reward': 1,
    'progress_m': 1, 'drawdown': 1, 'cross_track_0_2s_p95_m': -1,
    'endpoint_error_at_2s_m': -1, 'maximum_raw_joint_effort': -1,
    'maximum_roll_pitch_rad': -1, 'maximum_contact_n': -1,
}


def write(name: str, value):
    path = OUTPUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def input_records():
    old = read(OLD_CORRECTED / 'corrected-outcome-results.json')['per_run']['runs']
    old_geometry = read(OLD_CORRECTED / 'corrected-geometry-results.json')['runs']
    new = read(NEW / 'analysis/corrected-outcome-records.json')['records']
    records = []
    for pid, saved in old.items():
        trace = load_trace(OLD / 'runs' / pid / 'live-0.trace.json.gz')
        records.append({'process_id': pid, 'state_group_id': saved['state_group_id'],
                        'terrain': 'pyramid_stairs_inv' if saved['state_group_id'].endswith('000') else 'random_rough',
                        'candidate_id': saved['candidate_id'], 'repeat_id': saved['repeat_id'],
                        'source': 'reused_corrected_E1', 'outcome': saved['corrected'],
                        'geometry': old_geometry[pid]['corrected'],
                        'cumulative_reward': float(sum(r['reward'] for r in trace)),
                        'reward_sequence_sha256': saved['reward_sequence_sha256'],
                        'critic_sequence_sha256': saved['critic_sequence_sha256']})
    for saved in new:
        pid = saved['process_id']
        trace = load_trace(NEW / 'runs' / pid / 'live-0.trace.json.gz')
        records.append({'process_id': pid, 'state_group_id': saved['state_group_id'],
                        'terrain': saved['terrain'], 'candidate_id': saved['candidate_id'],
                        'repeat_id': int(pid.rsplit('-', 1)[-1]) if saved['role'] == 'nominal_duplicate' else 0,
                        'source': 'expanded_E1', 'outcome': saved['outcome'],
                        'geometry': saved['geometry'],
                        'cumulative_reward': float(sum(r['reward'] for r in trace)),
                        'reward_sequence_sha256': hashlib.sha256(json.dumps([r['reward'] for r in trace], separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                        'critic_sequence_sha256': hashlib.sha256(json.dumps([r['critic_value'] for r in trace], separators=(',', ':'), allow_nan=False).encode()).hexdigest()})
    return sorted(records, key=lambda r: (r['state_group_id'], CANDIDATES.index(r['candidate_id']), r['repeat_id']))


def metric_value(record, metric):
    if metric.startswith('outcome.'):
        return record['outcome'].get(metric.split('.', 1)[1])
    if metric.startswith('geometry.'):
        return record['geometry'].get(metric.split('.', 1)[1])
    if metric == 'cumulative_reward':
        return record['cumulative_reward']
    if metric in record['geometry']:
        return record['geometry'][metric]
    return record['outcome'].get(metric)


def pairwise_agreement(a, b, floor_a=0, floor_b=0):
    comparable = agree = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        da, db = a[i]-a[j], b[i]-b[j]
        if abs(da) <= floor_a or abs(db) <= floor_b:
            continue
        comparable += 1
        agree += int(da*db > 0)
    return {'agreement': agree/comparable if comparable else None,
            'comparable_pairs': comparable, 'descriptive_pairs': len(a)*(len(a)-1)//2}


def analyze_group(sid, rows):
    by_id = defaultdict(list)
    for r in rows:
        by_id[r['candidate_id']].append(r)
    if list(by_id) != list(CANDIDATES) or len(by_id['direct_reference']) != 3:
        raise ValueError(f'{sid}: candidate composition mismatch')
    directs = by_id['direct_reference']
    noise = {}
    candidates = {}
    for metric, direction in METRICS.items():
        rep = [metric_value(r, metric) for r in directs]
        if any(v is None or not math.isfinite(v) for v in rep):
            raise ValueError(f'{sid}.{metric}: direct value missing/nonfinite')
        pairs = [a-b for a, b in itertools.combinations(rep, 2)]
        noise[metric] = {'range': max(rep)-min(rep), 'std_sample': statistics.stdev(rep),
                         'pairwise_rms': math.sqrt(sum(x*x for x in pairs)/len(pairs)),
                         'max_abs_difference': max(abs(x) for x in pairs),
                         'duplicate_variation_floor': max(abs(x) for x in pairs),
                         'direct_values': rep, 'direction': 'higher' if direction > 0 else 'lower'}
        candidates[metric] = {cid: float(statistics.median(metric_value(r, metric) for r in by_id[cid]))
                              for cid in CANDIDATES}
    relative = {}
    for metric, vals in candidates.items():
        baseline = vals['direct_reference']
        floor = noise[metric]['duplicate_variation_floor']
        relative[metric] = {cid: {'signed_improvement': METRICS[metric]*(v-baseline),
                                  'absolute_difference': abs(v-baseline),
                                  'effect_to_duplicate_ratio': abs(v-baseline)/floor if floor else None,
                                  'ratio_status': 'defined' if floor else 'denominator_zero'}
                            for cid, v in vals.items() if cid != 'direct_reference'}
    rankings = {}
    for metric, vals in candidates.items():
        rankings[metric] = (sorted(CANDIDATES, key=lambda cid: (-METRICS[metric]*vals[cid], cid))
                            if len(set(vals.values())) > 1 else None)
    concordance = {}
    primary = 'delta_v'
    pvals = [METRICS[primary]*candidates[primary][cid] for cid in CANDIDATES]
    for metric in METRICS:
        if metric == primary:
            continue
        vals = [METRICS[metric]*candidates[metric][cid] for cid in CANDIDATES]
        sr = spearmanr(pvals, vals).statistic
        kt = kendalltau(pvals, vals).statistic
        noise_aware = pairwise_agreement(pvals, vals,
            noise[primary]['duplicate_variation_floor'], noise[metric]['duplicate_variation_floor'])
        raw = pairwise_agreement(pvals, vals)
        supported = rankings[metric] is not None
        concordance[metric] = {'spearman': float(sr) if np.isfinite(sr) else None,
                               'kendall': float(kt) if np.isfinite(kt) else None,
                               'pairwise_raw': raw, 'pairwise_duplicate_tie_aware': noise_aware,
                               'rank_supported':supported,
                               'top1_exact_match': rankings[primary][0] == rankings[metric][0] if supported else None,
                               'top3_overlap_count': len(set(rankings[primary][:3]) & set(rankings[metric][:3])) if supported else None,
                               'rank_inversion_candidates': [cid for cid in CANDIDATES if
                                   (rankings[primary].index(cid) <= 2) != (rankings[metric].index(cid) <= 2)] if supported else None}
    all_pairs = {}
    for a, b in itertools.combinations(METRICS, 2):
        key = a+'__'+b
        if rankings[a] is None or rankings[b] is None:
            all_pairs[key] = {'spearman':None,'kendall':None,'pairwise_raw':None,
                              'pairwise_duplicate_tie_aware':None,'top1_match':None,
                              'top3_overlap_count':None,'status':'NA_constant_label'}
            continue
        aa = [METRICS[a]*candidates[a][cid] for cid in CANDIDATES]
        bb = [METRICS[b]*candidates[b][cid] for cid in CANDIDATES]
        all_pairs[key] = {'spearman':float(spearmanr(aa,bb).statistic),
                          'kendall':float(kendalltau(aa,bb).statistic),
                          'pairwise_raw':pairwise_agreement(aa,bb),
                          'pairwise_duplicate_tie_aware':pairwise_agreement(aa,bb,
                              noise[a]['duplicate_variation_floor'],noise[b]['duplicate_variation_floor']),
                          'top1_match':rankings[a][0]==rankings[b][0],
                          'top3_overlap_count':len(set(rankings[a][:3]) & set(rankings[b][:3])),
                          'status':'descriptive'}
    delta_floor = noise['delta_v']['duplicate_variation_floor']
    delta_vals = candidates['delta_v']
    distinguishable = [cid for cid in CANDIDATES if cid != 'direct_reference' and
                       abs(delta_vals[cid]-delta_vals['direct_reference']) > delta_floor]
    return {'terrain': rows[0]['terrain'], 'state_group_id': sid, 'noise': noise,
            'candidate_values': candidates, 'candidate_vs_direct': relative,
            'rankings': rankings, 'concordance_vs_delta_v': concordance,
            'all_label_pairs':all_pairs,
            'delta_v_candidate_exceeds_duplicate_count': len(distinguishable),
            'delta_v_distinguishable_candidates': distinguishable,
            'metric_ranges': {m: max(vals.values())-min(vals.values()) for m, vals in candidates.items()}}


def all_direct_noise(rows):
    direct = [r for r in rows if r['candidate_id'] == 'direct_reference']
    keys = ({'outcome.'+key for key in direct[0]['outcome']} |
            {'geometry.'+key for key in direct[0]['geometry']} |
            {'cumulative_reward'} | set(METRICS))
    result = {}
    for key in sorted(keys):
        values = [metric_value(r, key) for r in direct]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values):
            differences = [a-b for a,b in itertools.combinations(values, 2)]
            result[key] = {'range': max(values)-min(values),
                           'std_sample': statistics.stdev(values),
                           'pairwise_rms': math.sqrt(sum(x*x for x in differences)/3),
                           'max_abs_difference': max(abs(x) for x in differences),
                           'duplicate_variation_floor': max(abs(x) for x in differences),
                           'direct_values': values}
        else:
            result[key] = {'direct_values': values, 'numeric_noise': None,
                           'reason': 'non-numeric or missing outcome'}
    return result


def horizon_diagnostic(groups):
    result = {}
    for sid, rows in groups.items():
        by_id = defaultdict(list)
        for r in rows:
            root = OLD if sid.startswith('g0e-') else NEW
            trace = load_trace(root/'runs'/r['process_id']/'live-0.trace.json.gz')
            critic = np.asarray([x['critic_value'] for x in trace], dtype=float)
            reward = np.asarray([x['reward'] for x in trace], dtype=float)
            # 40 rewards over [0,2 s), bootstrap at the pre-step V at t=2 s.
            by_id[r['candidate_id']].append({'delta_v_0_2s':float(critic[40]-critic[0]),
                'advantage_0_2s':float(np.dot(.99**np.arange(40),reward[:40])+.99**40*critic[40]-critic[0]),
                'reward_0_2s':float(reward[:40].sum())})
        values = {cid:{key:float(statistics.median(v[key] for v in by_id[cid]))
                       for key in ('delta_v_0_2s','advantage_0_2s','reward_0_2s')}
                  for cid in CANDIDATES}
        rank = lambda key: sorted(CANDIDATES,key=lambda cid:(-values[cid][key],cid))
        rho = spearmanr([values[c]['delta_v_0_2s'] for c in CANDIDATES],
                        [values[c]['advantage_0_2s'] for c in CANDIDATES]).statistic
        result[sid] = {'candidate_values':values, 'delta_advantage_spearman':float(rho),
                       'delta_top1_equals_advantage':rank('delta_v_0_2s')[0]==rank('advantage_0_2s')[0]}
    return {'status':'exploratory_only_not_teacher_selection','window_sec':2,
            'bootstrap':'trace[40] pre-step Critic at t=2 s; rewards trace[0:40]',
            'gamma':.99,'groups':result,
            'state_group_macro_delta_advantage_spearman':statistics.mean(v['delta_advantage_spearman'] for v in result.values()),
            'top1_match_count':sum(v['delta_top1_equals_advantage'] for v in result.values())}


def main():
    provenance = provenance_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = input_records()
    groups = defaultdict(list)
    for r in records:
        groups[r['state_group_id']].append(r)
    results = {sid: analyze_group(sid, rows) for sid, rows in groups.items()}
    slim = {key: value for key, value in provenance.items() if key != 'groups'}
    slim['groups'] = {sid: {'terrain': results[sid]['terrain'], 'runs': rows} for sid, rows in provenance['groups'].items()}
    slim['source_commit'] = 'e6c0bff11c017644e2bee5e375c024ded5a234bb'
    slim['coordinate_correction_fields'] = ['cross_track_p95_m', 'endpoint_error_m', 'progress_m']
    slim['old_first_direct_marker_exception'] = 'transferred artifact: local stage marker absent; run-complete, protocol-valid, and output SHA verified'
    write('source-provenance.json', slim)
    with (OUTPUT / 'stategroup-candidate-table.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as f:
            for r in records:
                f.write((json.dumps(r, sort_keys=True, allow_nan=False) + '\n').encode())
    write('direct-repeat-noise.json', {sid: all_direct_noise(rows) for sid, rows in groups.items()})
    write('label-ranking-concordance.json', {sid: {'terrain': g['terrain'],
        'rankings': g['rankings'], 'concordance_vs_delta_v': g['concordance_vs_delta_v'],
        'all_label_pairs':g['all_label_pairs'],
        'delta_v_distinguishable_candidates': g['delta_v_distinguishable_candidates']}
        for sid, g in results.items()})
    summary = {}
    for metric in METRICS:
        vals = [g['concordance_vs_delta_v'][metric] for g in results.values()] if metric != 'delta_v' else []
        summary[metric] = {'state_group_macro_spearman': statistics.mean(v['spearman'] for v in vals if v['spearman'] is not None) if vals and any(v['spearman'] is not None for v in vals) else None,
                           'state_group_macro_kendall': statistics.mean(v['kendall'] for v in vals if v['kendall'] is not None) if vals and any(v['kendall'] is not None for v in vals) else None,
                           'state_group_macro_raw_pairwise_agreement': statistics.mean(v['pairwise_raw']['agreement'] for v in vals if v['pairwise_raw']['agreement'] is not None) if vals and any(v['pairwise_raw']['agreement'] is not None for v in vals) else None,
                           'state_group_top1_match_count': sum(v['top1_exact_match'] for v in vals if v['top1_exact_match'] is not None) if vals and any(v['top1_exact_match'] is not None for v in vals) else None}
    all_scalar = {}
    all_relative = {}
    for sid, rows in groups.items():
        by_id = defaultdict(list)
        for row in rows:
            by_id[row['candidate_id']].append(row)
        keys = ({'outcome.'+key for key in rows[0]['outcome']} |
                {'geometry.'+key for key in rows[0]['geometry']} |
                {'cumulative_reward'} | set(METRICS))
        all_scalar[sid] = {}
        all_relative[sid] = {}
        floor_by_key = all_direct_noise(rows)
        for key in sorted(keys):
            values = {cid: [metric_value(row, key) for row in by_id[cid]] for cid in CANDIDATES}
            if not all(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in xs) for xs in values.values()):
                continue
            medians = {cid: float(statistics.median(xs)) for cid,xs in values.items()}
            floor = floor_by_key[key]['duplicate_variation_floor']
            all_scalar[sid][key] = medians
            all_relative[sid][key] = {cid: {'absolute_difference':abs(medians[cid]-medians['direct_reference']),
                'effect_to_duplicate_ratio':abs(medians[cid]-medians['direct_reference'])/floor if floor else None,
                'ratio_status':'defined' if floor else 'denominator_zero'} for cid in CANDIDATES if cid!='direct_reference'}
    directions = {m: 'higher_is_better' if d > 0 else 'lower_is_better' for m, d in METRICS.items()}
    directions['maximum_contact_n'] = 'lower_peak_contact_is_a_diagnostic_proxy_not_universal_quality'
    directions['maximum_raw_joint_effort'] = 'lower_peak_effort_is_a_diagnostic_proxy_but_all_values_saturated'
    write('evaluator-label-vector.json', {'directions': directions,
        'candidate_values_by_group': {sid: g['candidate_values'] for sid, g in results.items()},
        'candidate_vs_direct_by_group': {sid: g['candidate_vs_direct'] for sid, g in results.items()},
        'all_scalar_candidate_values_by_group':all_scalar,
        'all_scalar_candidate_vs_direct_by_group':all_relative,
        'additional_labels': {'initial_critic': 'calibration covariate, no monotonic quality direction',
                              'final_critic': 'higher is better only within fixed checkpoint, contextualized by initial value',
                              'recovery': 'diagnostic; larger recovery can follow worse drawdown',
                              'along_path_progress_0_2s_m': 'higher is better, primary Path horizon',
                              'reach_within_0_2s_0_2m': 'encounter event, true preferred over false',
                              'stack_proxy': 'false preferred; no positive support',
                              'verified_stack': 'false preferred; no positive support',
                              'termination': 'cause-specific; never infer failure from timeout',
                              'censored': 'mask only; never convert to negative',
                              'contact': 'maximum_contact_n is diagnostic, no universal monotonic quality direction',
                              'raw_joint_effort': 'maximum_raw_joint_effort is diagnostic and saturated at 3000 in these runs'},
        'missing_policy': 'NA, no zero imputation; censored and infrastructure-invalid excluded from teacher',
        'endpoint_encounter': 'reach_within_0_2s_0_2m; descriptive only',
        'stack_and_termination': 'all live records negative, insufficient positive support'})
    write('critic-label-audit.json', {'state_group_macro_concordance': summary,
        'groups_delta_v_exceeding_direct_repeat_noise': sum(g['delta_v_candidate_exceeds_duplicate_count'] > 0 for g in results.values()),
        'groups': {sid: {'terrain': g['terrain'], 'delta_v_candidate_exceeds_duplicate_count': g['delta_v_candidate_exceeds_duplicate_count'],
                         'metric_ranges': g['metric_ranges'], 'concordance_vs_delta_v': g['concordance_vs_delta_v']}
                   for sid, g in results.items()},
        'statistical_unit': 'StateGroup', 'candidate_pairs_independent': False})
    write('horizon-diagnostic.json', horizon_diagnostic(groups))
    return results, summary


if __name__ == '__main__':
    results, summary = main()
    print(json.dumps({'groups': len(results), 'delta_v_beyond_noise': sum(g['delta_v_candidate_exceeds_duplicate_count'] > 0 for g in results.values()), 'macro': summary}, sort_keys=True))
