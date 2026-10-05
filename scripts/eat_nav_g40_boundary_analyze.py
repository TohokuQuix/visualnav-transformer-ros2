"""Group-level bootstrap sensitivity from offline Critic queries."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau,spearmanr
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/quality-identifiability-gate/g40-boundary-freeze'
PRIOR=ROOT/'docs/quality-identifiability-gate/g40-teacher-review'
RAW=Path('/tmp/eatnav-g40-boundary-terminal-raw.json')
CANDIDATES=('direct_reference','left_1','left_2','left_3','right_1','right_2','right_3')
CONDITIONS=('zero_recorded','hold_candidate','canonical','direct_reference','reward_only')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def compare_rankings(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=(7,) or b.shape!=(7,):raise ValueError('seven candidate scores required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite candidate score')
    comparable=[(i,j) for i,j in itertools.combinations(range(7),2) if a[i]!=a[j] and b[i]!=b[j]]
    agree=sum(bool((a[i]-a[j])*(b[i]-b[j])>0) for i,j in comparable)
    direct=[i for i in range(1,7) if a[i]!=a[0] and b[i]!=b[0]]
    direct_agree=sum(bool((a[i]-a[0])*(b[i]-b[0])>0) for i in direct)
    top_a=sorted(range(7),key=lambda i:(-a[i],CANDIDATES[i]))
    top_b=sorted(range(7),key=lambda i:(-b[i],CANDIDATES[i]))
    rho=float(spearmanr(a,b).statistic) if len(set(a))>1 and len(set(b))>1 else None
    tau=float(kendalltau(a,b).statistic) if rho is not None else None
    return {'spearman':rho,'kendall':tau,
            'pairwise_sign_agreement':agree/len(comparable) if comparable else None,
            'comparable_pairs':len(comparable),'descriptive_pairs':21,
            'top1_agreement':top_a[0]==top_b[0],
            'top1_a':CANDIDATES[top_a[0]],'top1_b':CANDIDATES[top_b[0]],
            'top3_overlap':len(set(top_a[:3])&set(top_b[:3])),
            'direct_sign_agreement':direct_agree/len(direct) if direct else None,
            'direct_comparable':len(direct),
            'direct_sign_reversal_candidates':[
                CANDIDATES[i] for i in direct if (a[i]-a[0])*(b[i]-b[0])<0],
            'wholesale_reversal':len(comparable)==21 and agree==0,
            'pairwise_rank_reversal_count':len(comparable)-agree}

def stability_gate(group_results, config=None):
    if config is None:config={'hold_vs_canonical_macro_kendall_min':.8,
                              'hold_vs_canonical_top1_agreement_min_groups':6,
                              'hold_vs_canonical_mean_top3_overlap_min':2.,
                              'hold_vs_canonical_direct_sign_agreement_min':.9,
                              'no_single_terrain_wholesale_rank_reversal':True}
    if len(group_results)!=7:raise ValueError('seven independent StateGroups required')
    required=('kendall','direct_sign_agreement')
    missing=any(g[k] is None for g in group_results for k in required)
    macro_kendall=float(np.mean([g['kendall'] for g in group_results if g['kendall'] is not None])) if not missing else None
    top1=sum(g['top1_agreement'] for g in group_results)
    top3=float(np.mean([g['top3_overlap'] for g in group_results]))
    direct=float(np.mean([g['direct_sign_agreement'] for g in group_results if g['direct_sign_agreement'] is not None])) if not missing else None
    wholesale=any(g['wholesale_reversal'] for g in group_results)
    gates={
        'macro_kendall':not missing and macro_kendall>=config['hold_vs_canonical_macro_kendall_min'],
        'top1':top1>=config['hold_vs_canonical_top1_agreement_min_groups'],
        'top3':top3>=config['hold_vs_canonical_mean_top3_overlap_min'],
        'direct_sign':not missing and direct>=config['hold_vs_canonical_direct_sign_agreement_min'],
        'no_wholesale_reversal':not wholesale}
    return {'status':'g40_bootstrap_stable' if all(gates.values()) else 'g40_bootstrap_sensitive',
            'gate_checks':gates,'macro_kendall':macro_kendall,'top1_agreement_groups':top1,
            'mean_top3_overlap':top3,'macro_direct_sign_agreement':direct,
            'wholesale_reversal_terrain_count':sum(g['wholesale_reversal'] for g in group_results)}

def main():
    config=yaml.safe_load((OUT/'boundary-analysis-config.yaml').read_text())
    b0=json.loads((OUT/'source-provenance.json').read_text())
    key=str((OUT/'boundary-analysis-config.yaml').relative_to(ROOT))
    if sha(OUT/'boundary-analysis-config.yaml')!=b0['source_files_sha256'][key]:
        raise ValueError('analysis config changed after B0 freeze')
    equivalence=json.loads((OUT/'critic-offline-replay-equivalence.json').read_text())
    if equivalence['status']!='pass' or equivalence['maximum_absolute_difference']>config['critic_replay']['absolute_tolerance']:
        raise ValueError('Critic replay gate not passed')
    raw=json.loads(RAW.read_text())
    if raw['equivalence']['status']!='pass' or len(raw['terminal_values'])!=252:
        raise ValueError('terminal query input invalid')
    prior={r['process_id']:r for r in (json.loads(x) for x in gzip.open(PRIOR/'g40-per-candidate.jsonl.gz','rt'))}
    grouped=defaultdict(dict)
    for record in raw['terminal_values']:
        pid=record['process_id'];name=record['condition']
        if pid not in prior or name in grouped[pid]:raise ValueError('run identity/condition mismatch')
        grouped[pid][name]=record
    if len(grouped)!=63 or any(set(v)!=set(CONDITIONS)-{'reward_only'} for v in grouped.values()):
        raise ValueError('terminal query coverage mismatch')
    records=[]
    discount=config['gamma']**config['path_transition_count']
    for pid,values in grouped.items():
        previous=prior[pid]
        zero=values['zero_recorded']
        if abs(zero['terminal_value']-previous['v40'])>config['critic_replay']['absolute_tolerance']:
            raise ValueError(f'{pid}: recorded terminal Critic mismatch')
        terminal={k:values[k]['terminal_value'] for k in values}
        terminal['zero_recorded']=previous['v40']
        g={k:previous['R_gamma_40']+discount*v for k,v in terminal.items()}
        g['reward_only']=previous['R_gamma_40']
        if abs(g['zero_recorded']-previous['G40'])>1e-9:
            raise ValueError(f'{pid}: prior G40 mismatch')
        records.append({
            'process_id':pid,'state_group_id':previous['state_group_id'],
            'terrain':previous['terrain'],'candidate_id':previous['candidate_id'],
            'repeat_id':previous['repeat_id'],'source_trace_sha256':previous['trace_sha256'],
            'reward_discounted_2s':previous['R_gamma_40'],'terminal_value':terminal,
            'terminal_command':{k:values[k]['command'] for k in values},
            'G40':g,'bootstrap_contribution':{k:discount*v for k,v in terminal.items()},
            'V0':previous['v0'],
            'A40':{k:value-previous['v0'] for k,value in g.items()},
            'deltaV40':{k:value-previous['v0'] for k,value in terminal.items()},
            'map_metrics':previous['map_metrics'],'critic_drawdown':previous['critic_drawdown'],
            'body_metrics':previous['body_metrics'],
            'termination_mask':not previous['body_metrics']['termination'],
            'recorded_V40_replay_absolute_difference':abs(zero['terminal_value']-previous['v40'])})
    records.sort(key=lambda r:(r['state_group_id'],CANDIDATES.index(r['candidate_id']),r['repeat_id']))
    with gzip.open(OUT/'terminal-value-per-candidate.jsonl.gz','wt') as f:
        for row in records:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    groups=defaultdict(list)
    for row in records:groups[row['state_group_id']].append(row)
    comparisons={};group_values={}
    combos=list(itertools.combinations(CONDITIONS,2))
    for sid,rows in sorted(groups.items()):
        by=defaultdict(list)
        for row in rows:by[row['candidate_id']].append(row)
        if set(by)!=set(CANDIDATES) or len(by['direct_reference'])!=3:
            raise ValueError(f'{sid}: group composition mismatch')
        scores={condition:{cid:float(statistics.median(r['G40'][condition] for r in by[cid]))
                           for cid in CANDIDATES} for condition in CONDITIONS}
        group_values[sid]={'terrain':rows[0]['terrain'],'scores':scores,
                           'recorded_zero_run_count':sum(max(abs(v) for v in r['terminal_command']['zero_recorded'])<1e-12 for r in rows)}
        comparisons[sid]={f'{a}_vs_{b}':compare_rankings([scores[a][c] for c in CANDIDATES],
                                                        [scores[b][c] for c in CANDIDATES])
                          for a,b in combos}
    macro={}
    for a,b in combos:
        key=f'{a}_vs_{b}';vals=[comparisons[sid][key] for sid in sorted(groups)]
        def mean(name):
            v=[x[name] for x in vals if x[name] is not None]
            return float(np.mean(v)) if v else None
        macro[key]={'spearman':mean('spearman'),'kendall':mean('kendall'),
                    'pairwise_sign_agreement':mean('pairwise_sign_agreement'),
                    'top1_agreement_groups':sum(x['top1_agreement'] for x in vals),
                    'mean_top3_overlap':mean('top3_overlap'),
                    'direct_sign_agreement':mean('direct_sign_agreement'),
                    'wholesale_reversal_groups':sum(x['wholesale_reversal'] for x in vals)}
    gate=stability_gate([comparisons[sid]['hold_candidate_vs_canonical'] for sid in sorted(groups)],config['stability_gate'])
    save('bootstrap-teacher-comparison.json',{'statistical_unit':'StateGroup','candidate_pairs':'descriptive_not_independent',
        'group_values':group_values,'group_comparisons':comparisons,'macro':macro,
        'terminal_query_raw_sha256':sha(RAW),'gamma_power_40':discount,
        'recorded_condition_caveat':'zero_recorded is literal zero for 57/63 runs, not 63/63'})
    save('bootstrap-rank-stability.json',{'gate':gate,'source_config_sha256':sha(OUT/'boundary-analysis-config.yaml'),
        'group_hold_vs_canonical':{sid:comparisons[sid]['hold_candidate_vs_canonical'] for sid in sorted(groups)},
        'thresholds':config['stability_gate'],'statistical_unit':'StateGroup'})
    return gate

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
