"""Secondary state-level strata for frozen GT(P/C/R) rankings."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest

ROOT=Path('/tmp/eatnav-candidate-coverage/feasible-transfer')

def read(name):return json.loads((ROOT/name).read_text())

def metrics(rows,kind,*,epsilon=.01):
    if not rows:return {'candidate_count':0,'pairwise_support':0,'pairwise_accuracy':None,'top1_recall':None,'top3_recall':None,'spearman':None}
    p=np.asarray([np.mean(r[kind],axis=0)[0] for r in rows],dtype=float)
    y=np.asarray([r['delta_v'] for r in rows],dtype=float)
    order=np.argsort(-p,kind='stable');best=np.max(y)
    pairs=[(i,j) for i,j in combinations(range(len(y)),2) if abs(y[i]-y[j])>epsilon]
    correct=sum((p[i]-p[j])*(y[i]-y[j])>0 for i,j in pairs)
    rho=spearmanr(y,p).statistic if len(y)>2 and np.ptp(y)>1e-8 and np.ptp(p)>1e-8 else None
    return {'candidate_count':len(rows),'candidate_ids':[r['candidate_id'] for r in rows],
        'generators':[r['generator'] for r in rows],'seeds':[r['seed'] for r in rows],
        'pairwise_support':len(pairs),'pairwise_accuracy':correct/len(pairs) if pairs else None,
        'top1_candidate_id':rows[int(order[0])]['candidate_id'],
        'top1_recall':bool(y[order[0]]>=best-epsilon),
        'top3_recall':bool(any(y[i]>=best-epsilon for i in order[:3])),
        'spearman':float(rho) if rho is not None and np.isfinite(rho) else None,
        'predicted_score_spread':float(np.ptp(p)),'true_delta_v_spread':float(np.ptp(y)),
        'calibration_bias':float(np.mean(p-y)),'calibration_rmse':float(np.sqrt(np.mean((p-y)**2))),
        'ensemble_disagreement':float(np.mean([np.std(np.asarray(r[kind])[:,0]) for r in rows]))}

def run():
    scoring=read('planned-command-realized-comparison.json')['candidate_rows']
    feasibility=read('feasibility-by-candidate.json')['rows']
    coverage=read('nomad-feasible-coverage.json')['states']
    ood=read('trajectory-ood-comparison.json')['rows']
    if len(scoring)!=91 or len(feasibility)!=91:raise ValueError('91 candidate rows required')
    detail={(r['state_id'],r['candidate_id']):r for r in feasibility}
    ood_map={(r['state_id'],r['candidate_id'],r['variant']):r for r in ood}
    cluster={r['state_id']:r['cluster_count']['command'] for r in coverage}
    by_state=defaultdict(list)
    for row in scoring:by_state[row['state_id']].append(row)
    states=[]
    for sid,rows in by_state.items():
        terr=rows[0]['terrain']
        groups={
            'all':rows,
            'nomad':[r for r in rows if r['generator']=='nomad'],
            'fixed_bank':[r for r in rows if r['generator']=='bank'],
            'constraint_valid':[r for r in rows if detail[sid,r['candidate_id']]['condition']['constraint_valid']],
            'constraint_invalid':[r for r in rows if not detail[sid,r['candidate_id']]['condition']['constraint_valid']],
            'planned_in_distribution':[r for r in rows if not ood_map[sid,r['candidate_id'],'P']['over_frozen_train_p95']],
            'planned_ood':[r for r in rows if ood_map[sid,r['candidate_id'],'P']['over_frozen_train_p95']],
            'nominal':[r for r in rows if r['candidate_id']=='bank-nominal'],
        }
        row={'state_id':sid,'terrain':terr,'command_cluster_count':cluster[sid],
            'command_cluster_stratum':'single' if cluster[sid]==1 else 'multi',
            'candidate_ids':[r['candidate_id'] for r in rows],
            'groups':{g:{kind:metrics(subset,kind) for kind in ('P','C','R')} for g,subset in groups.items()}}
        states.append(row)
    aggregate={}
    for group in states[0]['groups']:
        aggregate[group]={}
        for kind in ('P','C','R'):
            entries=[s['groups'][group][kind] for s in states if s['groups'][group][kind]['pairwise_accuracy'] is not None]
            aggregate[group][kind]={'state_support_count':len(entries),
                'state_macro_pairwise_accuracy':float(np.mean([e['pairwise_accuracy'] for e in entries])) if entries else None,
                'pairwise_support_sum_descriptive_only':sum(e['pairwise_support'] for e in entries),
                'top1_recall_count_within_group':sum(e['top1_recall'] for e in entries)}
    cluster_strata={}
    for label in ('single','multi'):
        subset=[s for s in states if s['command_cluster_stratum']==label]
        cluster_strata[label]={'state_ids':[s['state_id'] for s in subset],
            'count':len(subset),'GT_P_state_macro_pairwise_accuracy':float(np.mean([s['groups']['all']['P']['pairwise_accuracy'] for s in subset])) if subset else None}
    atomic_once(ROOT/'evaluator-strata-detail.json',{'states':states,'aggregate':aggregate,
        'command_cluster_state_strata':cluster_strata,'statistical_unit':'state',
        'state_count':7,'candidate_count':91,'tie_epsilon':.01,
        'source_sha256':{'scoring':digest(ROOT/'planned-command-realized-comparison.json'),
                         'feasibility':digest(ROOT/'feasibility-by-candidate.json'),
                         'ood':digest(ROOT/'trajectory-ood-comparison.json')}})
    print('FEASIBLE_TRANSFER_STRATA_COMPLETE',len(states),flush=True)

if __name__=='__main__':run()
