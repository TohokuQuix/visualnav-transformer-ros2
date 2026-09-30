"""Create five explicit decisions and small reproducible pilot artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
from statistics import median

REPO=Path('/tmp/eatnav-vnt-candidate-coverage')
DOC=REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
ROOT=Path(os.environ.get('EAT_READINESS_ROOT', '/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot'))
AN=ROOT/'analysis'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def once(path,data):
    content=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise FileExistsError('final artifact differs: '+str(path))
        return
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content);temp.replace(path)

def main():
    if read(ROOT/'status.json')['stage']!='RAW_COMPLETE' or read(ROOT/'postprocess-status.json')['stage']!='COMPLETE':
        raise RuntimeError('raw and postprocessing must complete before final decision')
    selection=read(DOC/'selection.json')['selected']
    labels=read(AN/'label-variance.json')
    oracle=read(AN/'oracle-coverage.json')
    diversity=read(AN/'three-space-diversity.json')
    ready=read(AN/'readiness.json')
    ood=read(AN/'ood-results.json')
    score=read(AN/'frozen-evaluator.json')
    if not all(len(x)==7 for x in (selection,labels,oracle,diversity,score['states'])) or len(ood['rows'])!=294:
        raise ValueError('state or OOD result support mismatch')
    runs=read(DOC/'schedule.json')['runs']
    markers=[]
    durations=[]
    gpu=[]
    for entry in runs:
        path=ROOT/'runs'/entry['run_id']/'run-complete.json'
        marker=read(path)
        if marker['identity']!=entry:raise ValueError('run identity changed')
        markers.append({'run_id':entry['run_id'],'state_id':entry['state_id'],
                        'kind':entry['kind'],'candidate_id':entry['candidate_id'],
                        'marker_sha256':sha(path),
                        'trace_sha256':marker['output_sha256'].get('live-0.trace.json.gz')})
        durations.append(marker['elapsed_sec'])
        if marker['gpu_peak_mib'] is not None:gpu.append(marker['gpu_peak_mib'])
    once(DOC/'artifact-sha-ledger.json',{'raw_root':str(ROOT),
         'raw_status_sha256':sha(ROOT/'status.json'),
         'analysis_sha256':{p.name:sha(p) for p in AN.glob('*.json')},
         'runs':markers,'raw_traces_in_git':False})
    recall8=[];multimode=[];nomad_ranges=[]
    for i,state in enumerate(selection):
        curve=oracle[i]['metrics']['delta_v']
        point=curve['k'][-1] if curve['status']=='observed' else None
        recall8.append(point['recall'] if point else None)
        multimode.append(diversity[i]['nomad_mode_count']['realized']>=2)
        nomad_ranges.append(labels[i]['delta_v_range'])
    supported=sum(value is not None for value in recall8)
    if supported<7:
        candidate='inconclusive'
    elif sum(recall8)>=4 and sum(multimode)>=4:
        candidate='promising'
    else:
        candidate='insufficient'
    relative='ready_for_scaled_collection' if ready['relative_outcome']['ready'] else 'insufficient_label_variation'
    safety='ready_for_safety_head' if ready['safety_head_ready'] else 'insufficient_failure_coverage'
    inputs='deployable' if ready['input_readiness']['deployable'] else 'incomplete'
    if inputs!='deployable':next_action='measurement_remediation'
    elif relative!='ready_for_scaled_collection':next_action='harder_state/longer_horizon_sampling'
    elif candidate=='insufficient':next_action='candidate_generator_redesign'
    else:next_action='scaled_counterfactual_train_collection'
    ood_counts={variant:sum(r['over_frozen_train_p95'] for r in ood['rows'] if r['variant']==variant)
                for variant in ('P','C','R')}
    evaluator={}
    for horizon in ('two_second','six_second'):
        evaluator[horizon]={}
        for subset in ('all','nomad_only','bank_only','constraint_valid'):
            values=[state['rank_metrics'][horizon][subset] for state in score['states']]
            observed=[v for v in values if v['status']=='observed']
            evaluator[horizon][subset]={'supported_states':len(observed),
                'top1_recall_count':sum(v['top1_recall'] for v in observed),
                'top3_recall_count':sum(v['top3_recall'] for v in observed),
                'macro_pairwise_accuracy':sum(v['pairwise_accuracy'] for v in observed if v['pairwise_accuracy'] is not None)/
                    max(1,sum(v['pairwise_accuracy'] is not None for v in observed)),
                'macro_oracle_regret':sum(v['oracle_regret'] for v in observed)/len(observed) if observed else None}
    runtime={'run_count':119,'rollout_count':112,'wall_elapsed_sec':read(ROOT/'status.json')['elapsed_sec'],
             'median_process_sec':median(durations),'max_process_sec':max(durations),
             'gpu_peak_mib':max(gpu) if gpu else None,
             'raw_root_bytes':read(ROOT/'status.json')['size_bytes']}
    if os.environ.get('EAT_READINESS_ROOT'):
        runtime.update({'additional_wall_cap_sec':16200,'storage_cap_bytes':1500000000,
                        'old_elapsed_sec':read(ROOT/'status.json')['old_elapsed_sec'],
                        'fresh_elapsed_sec':read(ROOT/'status.json')['fresh_elapsed_sec']})
    else:
        runtime.update({'wall_cap_sec':25200,'storage_cap_bytes':1000000000})
    once(DOC/'runtime-storage.json',runtime)
    decision={'candidate_coverage':candidate,'relative_outcome_dataset':relative,
              'safety_dataset':safety,'evaluator_input_schema':inputs,'next_action':next_action,
              'evidence':{'k8_recall_count':sum(x is True for x in recall8),
                 'k8_recall_supported_states':supported,'nomad_multiple_realized_modes_states':sum(multimode),
                 'relative_outcome':ready['relative_outcome'],
                 'safety_support':ready['safety_support'],
                 'input_readiness':ready['input_readiness'],
                 'ood_over_train_p95_by_variant':ood_counts,
                 'frozen_evaluator':evaluator},
              'formal_validation_started':False,'retraining_started':False,
              'full_collection_started':False,'real_robot_test_started':False}
    once(DOC/'final-decision.json',decision)
    rows=[]
    for i,state in enumerate(selection):
        curve=oracle[i]['metrics']['delta_v']
        value=curve['k'][-1] if curve['status']=='observed' else None
        rows.append(f"| {state['terrain_type']} | {diversity[i]['nomad_valid_precision']:.2f} | {diversity[i]['nomad_mode_count']['realized']} | {labels[i]['delta_v_range']} | {value['recall'] if value else 'NA'} |")
    report='''# Counterfactual Evaluator readiness pilot

Train-only, seven-terrain, fresh-process, exact-stamp diagnostic. Each of 7 states has 3 nominal repeats, 8 frozen NoMaD seeds, and the unchanged 5-path bank. The 7 anchor checks are separate from 112 rollouts.

## Decisions

'''
    report+='\n'.join(f'- **{key}:** `{value}`' for key,value in decision.items() if key not in ('evidence','formal_validation_started','retraining_started','full_collection_started','real_robot_test_started'))+'\n'
    report+='\n## State-level evidence\n\n| Terrain | NoMaD valid precision | NoMaD realized modes | valid 6s ΔV range | feasible-union Recall@8 |\n|---|---:|---:|---:|---|\n'+'\n'.join(rows)+'\n'
    if os.environ.get('EAT_READINESS_ROOT'):
        report+='\n## Measurement protocol\n\nThe first 100 runs use the frozen original protocol. The remaining 19 use a fresh bridge and eight fresh followers per run after three outcome-blind equivalence replays passed. The old command-delivery failure remains mechanistically inconclusive; its observed boundary is between follower publish return and bridge adoption. Failure attempt 0, fault injection, diagnostic replay, and equivalence replays are excluded from all outcome estimates.\n'
    report+=f'''\nThe feasible union contains nominal, constraint-valid NoMaD K=8, and constraint-valid bank candidates. Recall@K compares NoMaD prefixes against this union with ε=0.01. Seven terrain states are descriptive support, not formal generalization.

P/C/R two-second OOD counts over frozen train P95: {ood_counts}. The Phase B GT(P) model is frozen; its results are secondary and never selected candidates. Its macro metrics are in `final-decision.json` and raw score artifact `{AN/'frozen-evaluator.json'}`.

Runtime: {runtime['wall_elapsed_sec']/3600:.2f} h, median {runtime['median_process_sec']:.1f} s/process, peak GPU {runtime['gpu_peak_mib']} MiB, saved {runtime['raw_root_bytes']/1e6:.1f} MB. Raw traces, callback logs, and run markers remain under `{ROOT}` with SHA ledger in `artifact-sha-ledger.json`.

### Limits and next approval

Input readiness reflects the fields actually captured at the anchor. The current trace does not include a separate joint-effort field, so `incomplete` means that deployable schema parity remains unproven. Early termination censors six-second continuous labels but remains a safety label. The seven states are all train split; no validation, held-out, retraining, production follower edit, full collection, or real-robot test was performed.

A scaled train/validation/held-out counterfactual collection needs a separate design and approval. At the measured sequential median, 70 states × 16 rollouts would require roughly {70*16*runtime['median_process_sec']/3600:.1f} h before startup/retry reserve; storage scales to roughly {70/7*runtime['raw_root_bytes']/1e9:.1f} GB. A candidate split is 50 train, 10 validation, 10 held-out states with terrain-seed and episode separation, contingent on resolving the stated gate results and validating parallel throughput.

Reproduce status: `cat {ROOT/'status.json'}`; `cat {ROOT/'postprocess-status.json'}`; `tail -n 100 {ROOT/'runner.log'}`; `tmux ls`; `tmux capture-pane -pt eatnav-readiness-full`.
'''
    target=DOC/'final-report.md'
    if target.exists() and target.read_text()!=report:raise FileExistsError('final report changed')
    if not target.exists():target.write_text(report)
    once(DOC/'finalize.done.json',{'report_sha256':sha(target),
         'decision_sha256':sha(DOC/'final-decision.json'),
         'artifact_ledger_sha256':sha(DOC/'artifact-sha-ledger.json')})
    print('READINESS_FINAL_REPORT_COMPLETE',decision['candidate_coverage'],relative,safety,inputs,flush=True)

if __name__=='__main__':main()
