"""Analyze completed train pilot oracle outcomes without selecting candidates."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics
import yaml
from visualnav_transformer.eat_nav.candidate_coverage_oracle import oracle_best,recall_curve,outcome_spread
from scripts.eat_nav_candidate_coverage_pilot import verified_case,ROOT
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest
from scripts.eat_nav_duplicate_recovery_smoke_analyze import metrics,candidate_effect,command_divergence

METRICS=('delta_v','advantage_h','drawdown','recovery','progress_m')


def load_run(run):
    if not verified_case(run):return None
    folder=ROOT/'pilot/runs'/run['run_id']
    stage='anchor' if run['kind']=='anchor' else 'live'
    summary=folder/f'{stage}-0.json';trace=folder/f'{stage}-0.trace.json.gz'
    s=json.loads(summary.read_text())
    with gzip.open(trace,'rt') as stream:t=json.load(stream)
    return s,t,{'summary_sha256':digest(summary),'trace_sha256':digest(trace),
                'marker_sha256':digest(folder/f'{stage}-0.marker.json')}


def command_clusters(rows,linear=.03,angular=.05):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            a,b=rows[i]['command_series'],rows[j]['command_series']
            n=min(len(a),len(b))
            if not n:continue
            lv=math.sqrt(sum((a[k][0]-b[k][0])**2 for k in range(n))/n)
            av=math.sqrt(sum((a[k][2]-b[k][2])**2 for k in range(n))/n)
            if lv<linear and av<angular:parent[find(j)]=find(i)
    return len({find(i) for i in range(len(rows))})


def analyze(output_dir=None):
    schedule_path=ROOT/'stage0/frozen/pilot-schedule.json'
    selection_path=ROOT/'stage0/frozen/pilot-selection.json'
    config_path=Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/pilot-config.yaml')
    epsilon_path=ROOT/'pilot/oracle-epsilon.json'
    schedule=json.loads(schedule_path.read_text())['runs']
    selection=json.loads(selection_path.read_text())['selected']
    config=yaml.safe_load(config_path.read_text())
    epsilon=json.loads(epsilon_path.read_text())['epsilon']
    feasibility_path=Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/feasible-set/feasible-set-config.yaml')
    feasibility=yaml.safe_load(feasibility_path.read_text())['feasibility']
    if len(schedule)!=119 or len(selection)!=7 or digest(selection_path)!='1bb7865e81d2f0f52fff550037bb2e07b560b73f5de553b6183203bdaa969ddc':
        raise ValueError('frozen pilot input changed')
    out=Path(output_dir) if output_dir is not None else ROOT/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    outcomes=[];states=[];invalid=[]
    for state in selection:
        state_id=state['state_id']
        runs=[r for r in schedule if r['state_id']==state_id]
        gate=ROOT/'pilot/duplicate-gates'/f'{state_id}.json'
        if not gate.exists():
            states.append({'state_id':state_id,'terrain_type':state['terrain_type'],
                           'status':'measurement_invalid_or_pending','candidate_count':0})
            continue
        duplicate=json.loads(gate.read_text())
        if not duplicate['measurement_valid']:raise ValueError('invalid duplicate gate')
        baseline=load_run(next(r for r in runs if r['kind']=='duplicate'))
        if baseline is None:raise ValueError('duplicate gate without baseline trace')
        reference_trace=baseline[1]
        offline_path=ROOT/'stage1'/f'{state_id}.json'
        offline=json.loads(offline_path.read_text())
        follower_path=ROOT/'stage1'/f'{state_id}-follower.json'
        follower=json.loads(follower_path.read_text()) if follower_path.exists() else None
        input_folder=ROOT/'pilot-inputs'/f"state-{int(state_id.rsplit('-',1)[1]):03d}"
        offline_by_id={r['candidate_id']:r for r in offline['candidates']}
        follower_by_id={r['candidate_id']:r for r in follower['candidates']} if follower else {}
        cases=[]
        for run in runs:
            if run['kind'] not in ('nomad','bank'):continue
            data=load_run(run)
            if data is None:continue
            summary,trace,source=data
            cid=(f"nomad-seed-{run['seed']}" if run['kind']=='nomad' else f"bank-{run['name']}")
            proposal=offline_by_id[cid]
            map_file=input_folder/f'{cid}-map-path.json'
            if digest(map_file)!=summary['fixed_map_path_sha256']:
                raise ValueError(f'candidate fixed Path SHA mismatch: {run["run_id"]}')
            map_path=json.loads(map_file.read_text())
            ood={'ood_over_train_p95':proposal.get('ood_over_train_p95',False),
                 'constraint_valid':proposal['constraints']['valid'],
                 'constraint_reasons':proposal['constraints']['reasons'],
                 'ood_distance':proposal['ood_distance']}
            result=metrics(summary,trace,state['goal'],map_path,ood)
            effect=candidate_effect(reference_trace,trace)
            divergence=command_divergence(reference_trace,trace)
            terminal=result['terminal'] or {}
            reason=terminal.get('termination_reason',[])
            finite=all(math.isfinite(v) for root in result['root_xy'] for v in root)
            safe=bool(not terminal.get('terminated',False) and finite and
                      math.degrees(result['body_tilt_max_rad'])<=feasibility['tilt_max_deg'])
            result.update({'state_id':state_id,'terrain_type':state['terrain_type'],
                'candidate_id':cid,'run_id':run['run_id'],'seed':run['seed'],
                'kind':run['kind'],'proposal_valid':bool(proposal['constraints']['valid'] and not ood['ood_over_train_p95']),
                'safe_noncatastrophic':safe,'termination_reason':reason,
                'stack':bool(result['progress_m']<feasibility['progress_min_m'] or math.hypot(result['root_xy'][-1][0]-result['root_xy'][0][0],
                                                               result['root_xy'][-1][1]-result['root_xy'][0][1])<feasibility['stack_displacement_min_m']),
                'tilt_over_85deg':bool(math.degrees(result['body_tilt_max_rad'])>feasibility['tilt_max_deg']),
                'xy_effect_vs_duplicate_m':effect,'duplicate_xy_rms_m':duplicate['duplicate_xy_rms_m'],
                'command_divergence_vs_duplicate':divergence,
                'path_metrics':proposal['path_metrics'],'path_points':proposal['points'],
                'initial_follower_command':follower_by_id.get(cid,{}).get('initial_command'),
                'source_sha256':source,'map_path_sha256':digest(map_file)})
            outcomes.append(result);cases.append(result)
            if not safe or result['stack'] or result['tilt_over_85deg']:
                invalid.append({'state_id':state_id,'candidate_id':cid,'safe':safe,
                                'stack':result['stack'],'tilt_over_85deg':result['tilt_over_85deg'],
                                'termination_reason':reason})
        complete=len(cases)==13
        row={'state_id':state_id,'terrain_type':state['terrain_type'],'episode_key':state['episode_key'],
             'status':'complete' if complete else 'partial','candidate_count':len(cases),
             'duplicate_xy_rms_m':duplicate['duplicate_xy_rms_m'],
             'nomad_k8_path_geometry':offline['sets']['nomad_k8']['geometry'],
             'nomad_k8_initial_command_clusters':follower['sets']['nomad_k8']['command_cluster_count'] if follower else None,
             'nomad_k8_rollout_command_clusters':command_clusters([c for c in cases if c['kind']=='nomad']) if sum(c['kind']=='nomad' for c in cases)>=2 else None,
             'nomad_k8_delta_v_spread':outcome_spread([c for c in cases if c['kind']=='nomad'],'delta_v')}
        if complete:
            ordered=[f'nomad-seed-{seed}' for seed in config['nomad_seeds_pilot']]
            bank_ids=[f'bank-{name}' for name in config['bank_names']]
            row['oracle']={}
            for metric in METRICS:
                row['oracle'][metric]={
                    'nomad_recall_curve':recall_curve(cases,ordered,metric=metric,
                                                     epsilon=epsilon[metric],ks=[1,2,4,8]),
                    'nomad_k1_best':oracle_best(cases,ordered[:1],metric),
                    'nomad_k8_best':oracle_best(cases,ordered,metric),
                    'fixed_bank_best':oracle_best(cases,bank_ids,metric),
                    'hybrid_best':oracle_best(cases,ordered+bank_ids,metric),
                    'nominal':oracle_best(cases,['bank-nominal'],metric),
                    'pool_spread':outcome_spread(cases,metric)}
            row['nomad_better_than_nominal']=bool(row['oracle']['delta_v']['nomad_k8_best'] is not None and
                row['oracle']['delta_v']['nominal'] is not None and
                row['oracle']['delta_v']['nomad_k8_best']>row['oracle']['delta_v']['nominal']+epsilon['delta_v'])
            row['nomad_better_than_bank']=bool(row['oracle']['delta_v']['nomad_k8_best'] is not None and
                row['oracle']['delta_v']['fixed_bank_best'] is not None and
                row['oracle']['delta_v']['nomad_k8_best']>row['oracle']['delta_v']['fixed_bank_best']+epsilon['delta_v'])
            row['safe_candidate_count']=sum(c['safe_noncatastrophic'] for c in cases)
            row['safe_recall_at_k']={str(k):(any(c['candidate_id'] in ordered[:k] and c['safe_noncatastrophic'] for c in cases) if row['safe_candidate_count'] else None) for k in (1,2,4,8)}
            row['nomad_k8_ood_count']=offline['sets']['nomad_k8']['ood_count']
            row['nomad_k8_constraint_failure_count']=offline['sets']['nomad_k8']['constraint_failure_count']
            row['nomad_better_count']=sum(c['kind']=='nomad' and c['delta_v'] is not None and row['oracle']['delta_v']['nominal'] is not None and c['delta_v']>row['oracle']['delta_v']['nominal']+epsilon['delta_v'] for c in cases)
            row['nomad_worse_count']=sum(c['kind']=='nomad' and c['delta_v'] is not None and row['oracle']['delta_v']['nominal'] is not None and c['delta_v']<row['oracle']['delta_v']['nominal']-epsilon['delta_v'] for c in cases)
            row['nomad_safe_count']=sum(c['safe_noncatastrophic'] for c in cases if c['kind']=='nomad')
            row['best_delta_v_candidate_ids']=[c['candidate_id'] for c in cases if c['delta_v'] is not None and
                row['oracle']['delta_v']['hybrid_best'] is not None and
                c['delta_v']>=row['oracle']['delta_v']['hybrid_best']-epsilon['delta_v']]
            row['best_delta_v_proposal_valid']=all(next(c for c in cases if c['candidate_id']==cid)['proposal_valid']
                for cid in row['best_delta_v_candidate_ids']) if row['best_delta_v_candidate_ids'] else None
        states.append(row)
    complete=[r for r in states if r['status']=='complete']
    summary={'selection_sha256':digest(selection_path),'schedule_sha256':digest(schedule_path),
             'epsilon_sha256':digest(epsilon_path),'feasibility_config_sha256':digest(feasibility_path),'complete_state_count':len(complete),
             'partial_or_invalid_state_count':7-len(complete),'completed_candidate_count':len(outcomes),
             'heldout_used':False,'validation_used':False,'states':states,
             'delta_v_recall_support_count':sum(r['oracle']['delta_v']['nomad_recall_curve'][0]['pool_best'] is not None for r in complete),
             'nomad_recall_at_k':{str(k):(
                 sum(r['oracle']['delta_v']['nomad_recall_curve'][j]['recalled'] for r in complete)
                 /sum(r['oracle']['delta_v']['nomad_recall_curve'][0]['pool_best'] is not None for r in complete))
                 if any(r['oracle']['delta_v']['nomad_recall_curve'][0]['pool_best'] is not None for r in complete)
                 else None for j,k in enumerate((1,2,4,8))},
             'nomad_better_than_nominal_state_count':sum(r.get('nomad_better_than_nominal',False) for r in complete),
             'nomad_better_than_bank_state_count':sum(r.get('nomad_better_than_bank',False) for r in complete),
             'terrain_support':[r['terrain_type'] for r in complete],
             'best_seed_counts':dict(Counter(cid for r in complete for cid in r['best_delta_v_candidate_ids'] if cid.startswith('nomad-'))),
             'formal_validation_started':False,'retraining_started':False,'real_robot_started':False}
    summary['safe_candidate_recall_at_k']={str(k):(sum(r['safe_recall_at_k'][str(k)] for r in complete if r['safe_recall_at_k'][str(k)] is not None)/sum(r['safe_recall_at_k'][str(k)] is not None for r in complete) if any(r['safe_recall_at_k'][str(k)] is not None for r in complete) else None) for k in (1,2,4,8)}
    atomic_once(out/'oracle-outcomes.json',{'candidate_count':len(outcomes),'rows':outcomes})
    atomic_once(out/'oracle-summary.json',summary)
    atomic_once(out/'failure-stack-tilt.json',{'count':len(invalid),'rows':invalid})
    files=[p for p in (ROOT/'pilot').rglob('*') if p.is_file()]
    runtime={'run_count_complete':sum(verified_case(r) for r in schedule),'total_bytes':sum(p.stat().st_size for p in files),
             'log_bytes':sum(p.stat().st_size for p in files if p.suffix=='.log'),
             'trace_bytes':sum(p.stat().st_size for p in files if p.name.endswith('.trace.json.gz')),
             'budget_start':json.loads((ROOT/'pilot/budget-start.json').read_text()),
             'status':json.loads((ROOT/'pilot/status.json').read_text())}
    atomic_once(out/'runtime-storage.json',runtime)
    lines=['# Candidate coverage train pilot', '',
        f"Completed terrain states: {len(complete)}/7; completed candidate outcomes: {len(outcomes)}/91.",
        f"ΔV oracle Recall@K among complete states: {summary['nomad_recall_at_k']}.",
        f"NoMaD K8 beats nominal in {summary['nomad_better_than_nominal_state_count']}/{len(complete)} complete states; beats the fixed bank in {summary['nomad_better_than_bank_state_count']}/{len(complete)}.",
        '', 'Only complete-only train states were used. Formal validation, retraining, and real robot control were not started.',
        'The K32 set was assessed offline only; its oracle outcome is unavailable in this pilot.',
        'The reported recall applies only to the listed train terrain states and the evaluated 13-candidate pool.',
        '', '## State table', '',
        '| Terrain | Status | ΔV K1 | ΔV K8 | bank best | hybrid best | K8 command clusters | duplicate XY RMS |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in states:
        oracle=r.get('oracle',{}).get('delta_v',{})
        fmt=lambda x:'NA' if x is None else f'{x:.3f}'
        lines.append(f"| {r['terrain_type']} | {r['status']} | {fmt(oracle.get('nomad_k1_best'))} | {fmt(oracle.get('nomad_k8_best'))} | {fmt(oracle.get('fixed_bank_best'))} | {fmt(oracle.get('hybrid_best'))} | {r.get('nomad_k8_rollout_command_clusters') or 'NA'} | {fmt(r.get('duplicate_xy_rms_m'))} |")
    report=out/'pilot-report.md';content='\n'.join(lines)+'\n'
    if report.exists():
        if report.read_text()!=content:raise FileExistsError('pilot report differs on reanalysis')
    else:
        temp=report.with_name(report.name+'.tmp');temp.write_text(content);temp.replace(report)
    print('ORACLE_ANALYSIS',len(complete),'complete states',len(outcomes),'candidate outcomes',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--output',type=Path,default=None);a=p.parse_args()
    if not a.allow_partial and json.loads((ROOT/'pilot/status.json').read_text())['state']!='raw_runs_complete':
        raise ValueError('pilot still running or stopped early; use --allow-partial for completed cases')
    analyze(a.output)

if __name__=='__main__':main()
