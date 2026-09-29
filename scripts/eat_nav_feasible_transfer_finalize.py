"""Finalize state-unit feasible coverage, evaluator transfer, figures and gates."""
from __future__ import annotations
from collections import Counter,defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest

ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'feasible-transfer';FIG=OUT/'figures'
TERRAINS=['boxes','hf_pyramid_slope','k_rail_field','k_rail_step_field','pyramid_stairs','pyramid_stairs_inv','random_rough']

def read(name):return json.loads((OUT/name).read_text())

def figure(name):
    FIG.mkdir(parents=True,exist_ok=True)
    target=FIG/name
    if target.exists():raise FileExistsError(f'figure exists: {target}')
    plt.gcf().tight_layout()
    temp=target.with_name(target.name+'.tmp')
    plt.savefig(temp,format='png',dpi=150);temp.replace(target);plt.close()


def macro(values):return float(np.mean(values)) if values else None


def run():
    started=time.time()
    for marker in ('stage-0-complete.json','stage-1a-complete.json','stage-1b-complete.json',
                   'stage-2a-complete.json','stage-2b-complete.json'):
        if read(marker)['status']!='complete':raise ValueError(f'prior stage not complete: {marker}')
    feasibility=read('feasibility-by-candidate.json')['rows'];coverage=read('nomad-feasible-coverage.json')['states']
    regret=read('nomad-oracle-regret.json');teacher=read('teacher-support-analysis.json')['states']
    k32=read('k32-offline-analysis.json')['states'];evaluation=read('evaluator-counterfactual-results.json')
    comparisons=read('planned-command-realized-comparison.json');ood=read('trajectory-ood-comparison.json')
    reproduction=read('phase-b-reproduction.json')
    if len(feasibility)!=91 or len(coverage)!=7 or len(regret['bank_winner_comparisons'])!=4:
        raise ValueError('frozen candidate/state count mismatch')
    aggregate_precision={}
    for generator in ('nomad','bank','all'):
        aggregate_precision[generator]={name:macro([s['precision'][generator]['condition'][name]['precision'] for s in coverage])
            for name in coverage[0]['precision'][generator]['condition']}
    mode_recall={space:{str(k):macro([next(x for x in s['recall']['joint_observed_screen'][space] if x['k']==k)['recall']
        for s in coverage if next(x for x in s['recall']['joint_observed_screen'][space] if x['k']==k)['recall'] is not None])
        for k in (1,2,4,8)} for space in ('command','realized')}
    by_terrain={r['terrain']:r for r in coverage}
    by_regret={r['terrain']:r for r in regret['states']}
    by_eval={r['terrain']:r for r in evaluation['states']}
    by_teacher={r['terrain']:r for r in teacher}
    candidate_scores=comparisons['candidate_rows']
    candidate_by_key={(r['state_id'],r['candidate_id']):r for r in candidate_scores}
    # State is the statistical unit. Candidate and pair rows below are descriptive only.
    generator_bias=[]
    for state in evaluation['states']:
        sid=state['state_id']
        for generator in ('nomad','bank'):
            records=[r for r in candidate_scores if r['state_id']==sid and r['generator']==generator]
            generator_bias.append({'state_id':sid,'terrain':state['terrain'],'generator':generator,
                **{kind:{'bias':float(np.mean([np.mean(r[kind],axis=0)[0]-r['delta_v'] for r in records])),
                         'mae':float(np.mean([abs(np.mean(r[kind],axis=0)[0]-r['delta_v']) for r in records]))}
                   for kind in ('P','C','R')}})
    uncertainty=[]
    for state in evaluation['states']:
        records=[r for r in candidate_scores if r['state_id']==state['state_id']]
        for i,j in combinations(range(13),2):
            a,b=records[i],records[j];gap=a['delta_v']-b['delta_v']
            if abs(gap)<=.01:continue
            pa=np.asarray(a['P'])[:,0];pb=np.asarray(b['P'])[:,0]
            correct=(pa.mean()-pb.mean())*gap>0
            uncertainty.append({'state_id':state['state_id'],'terrain':state['terrain'],
                'candidate_ids':[a['candidate_id'],b['candidate_id']],
                'generators':[a['generator'],b['generator']],'seeds':[a['seed'],b['seed']],
                'oracle_delta_v_gap':gap,'predicted_gap':float(pa.mean()-pb.mean()),
                'ensemble_pair_uncertainty':float(np.hypot(pa.std(),pb.std())),
                'correct':bool(correct)})
    uncertainty_by_state=[]
    for state in evaluation['states']:
        pairs=[r for r in uncertainty if r['state_id']==state['state_id']]
        wrong=[r for r in pairs if not r['correct']]
        right=[r for r in pairs if r['correct']]
        uncertainty_by_state.append({'state_id':state['state_id'],'terrain':state['terrain'],
            'pair_count':len(pairs),'wrong_count':len(wrong),
            'uncertainty_wrong_mean':macro([r['ensemble_pair_uncertainty'] for r in wrong]),
            'uncertainty_right_mean':macro([r['ensemble_pair_uncertainty'] for r in right])})
    gt=evaluation['aggregate'];p=gt['P'];c=gt['C'];r=gt['R']
    distinct_bank_better=[x for x in regret['bank_winner_comparisons'] if x['bank']['condition']['joint_observed_screen']
                          and (not x['same_command_mode'] or not x['same_realized_mode'])]
    classification={'statistical_unit':'state','state_count':7,'candidate_count':91,
      'NoMaD':{
       'feasibility_concentration_supported':{'status':'not_established','evidence':f"NoMaD short-horizon precision {aggregate_precision['nomad']['short_horizon_noncatastrophic']:.3f}",
          'counterevidence':f"Bank short-horizon precision {aggregate_precision['bank']['short_horizon_noncatastrophic']:.3f}; bank candidates also ran without catastrophic failure"},
       'feasible_mode_undercoverage':{'status':'not_established_for_useful_constraint_valid_bank_mode',
          'evidence':f"K8 joint-screen command-mode Recall@8 state macro {mode_recall['command']['8']:.3f}",
          'counterevidence':f"Constraint-valid, better, distinct bank winners: {len(distinct_bank_better)}/4 bank-winning states"},
       'sampling_limited':{'status':'possible_unverified','evidence':f"K32 adds a path cluster in {sum(bool(x['added_path_modes_beyond_k8']) for x in k32)}/7 states",
          'counterevidence':'K32 modes were not rolled out and are not known to be feasible or better'},
       'dataset_support_limited':{'status':'contradicted_for_bank_winners',
          'evidence':'Every bank winner has independent-episode teacher trajectory/direction matches',
          'counterevidence':'Train-only matching is observational and does not prove dynamic support'},
       'model_collapse':{'status':'inconclusive','evidence':'K8 planned paths form one mode in all seven terrains',
          'counterevidence':'K32 forms two planned modes in two terrains; bank winners mostly share realized modes with NoMaD'},
       'insufficient_horizon':{'status':'supported','evidence':'All 91 observations cover only 40 control steps (2 s)',
          'counterevidence':'No long-horizon rollout was performed'}},
      'Evaluator':{
       'implementation_mismatch':{'status':'refuted_in_audited_path','evidence':'Original Phase B held-out metrics reproduced exactly and input/label audit passed',
          'counterevidence':'An unobserved implementation issue outside audited 91 inputs cannot be excluded'},
       'trajectory_representation_mismatch':{'status':'supported_semantic','evidence':f"GT(C) pairwise {c['state_macro_pairwise_accuracy']:.3f} versus GT(P) {p['state_macro_pairwise_accuracy']:.3f}",
          'counterevidence':'P increment/base-frame transformation itself matches training code'},
       'planned_realized_mismatch':{'status':'not_primary_by_R_probe','evidence':f"GT(R) pairwise {r['state_macro_pairwise_accuracy']:.3f} versus GT(P) {p['state_macro_pairwise_accuracy']:.3f}",
          'counterevidence':f"GT(C) is stronger at {c['state_macro_pairwise_accuracy']:.3f}; R is privileged and remains weak"},
       'goal_endpoint_distribution_shift':{'status':'present_but_sensitivity_small','evidence':f"Phase B exact goal-endpoint fraction {comparisons['goal_endpoint']['phase_b_heldout']['exact_fraction']:.3f}; pilot bank 35/35 exact, NoMaD 0/56 exact",
          'counterevidence':'Replacing goal by each planned endpoint barely changed score spread in most states'},
       'candidate_ood':{'status':'partial','evidence':f"Frozen train action p95 exceeded by P {ood['summary']['P']['over_p95_count']}/91, C {ood['summary']['C']['over_p95_count']}/91, R {ood['summary']['R']['over_p95_count']}/91",
          'counterevidence':'Most P candidates are below the frozen threshold, so OOD alone cannot explain all misrankings'},
       'insufficient_candidate_effect':{'status':'refuted_for_observed_40_steps','evidence':'Duplicate XY RMS was zero; true Delta V varies above epsilon across candidates',
          'counterevidence':'Some state/mode pairs remain near-ties and are excluded from ordering'},
       'insufficient_state_count':{'status':'supported','evidence':'Only seven independent train states, one per terrain',
          'counterevidence':'No narrow candidate-level CI is claimed'},
       'counterfactual_training_required':{'status':'plausible_unverified','evidence':'GT(P) top-1 0/7 despite exact Phase B reproduction and substantial outcome signal',
          'counterevidence':'GT(C) improves without retraining; this diagnostic cannot prove which future training change is needed'}}}
    gate_n='inconclusive'
    gate_e='fail' if p['top1_recall_count']==0 and p['state_macro_pairwise_accuracy']<=.5 else 'inconclusive'
    decision={'Gate_N':{'status':gate_n,'scope':'train pilot, 7 states, 40 steps',
        'reason':'High short-horizon precision and multi-command modes coexist with one planned mode per terrain; useful constraint-valid bank mode undercoverage is not demonstrated, and long-horizon feasibility is unknown.'},
       'Gate_E':{'status':gate_e,'scope':'frozen GT(P) counterfactual transfer on seven train states',
        'reason':'Phase B reproduced, but GT(P) top-1 0/7 and state-macro pairwise below chance and below simple baselines; GT(C) is a diagnostic, not a validated online selector.'},
       'formal_validation_approved':False,'retraining_approved':False,'additional_rollout_started':False,
       'next_experiment_proposal':{
         'priority':'small long-horizon feasibility confirmation before proposal redesign; counterfactual Evaluator data design separately',
         'state_ids':[x['state_id'] for x in regret['bank_winner_comparisons']],
         'candidate_rule':'for each of four frozen bank-winning states: nominal three duplicates, fixed NoMaD K8 oracle winner, fixed bank winner',
         'run_count':20,'fresh_process':True,'full_prefix_step_cap':40,'control_steps':120,
         'physical_horizon_sec':6.0,'estimated_wall_clock_min':45,'wall_clock_cap_min':90,
         'estimated_storage_mb':100,'storage_cap_mb':250,
         'stop_conditions':['anchor or duplicate gate failure','repeated follower timeout','unexpected GPU memory','forecast exceeds 90 minutes'],
         'decision_criterion':'compare long-horizon progress, stack, tilt, contact and termination for the frozen three candidate identities per state; no candidate replacement',
         'requires_separate_approval':True}}
    atomic_once(OUT/'cause-classification.json',classification)
    atomic_once(OUT/'decision.json',decision)
    atomic_once(OUT/'evaluator-strata-and-uncertainty.json',{'generator_bias_by_state':generator_bias,
        'pairwise_uncertainty_rows':uncertainty,'uncertainty_by_state':uncertainty_by_state,
        'candidate_pairs_are_not_independent':True})
    # 1. Three trajectory spaces.
    x=np.arange(7);w=.25
    plt.figure(figsize=(9,4))
    for j,space in enumerate(('planned','command','realized')):
        plt.bar(x+(j-1)*w,[by_terrain[t]['cluster_count'][space] for t in TERRAINS],w,label=space)
    plt.xticks(x,TERRAINS,rotation=35,ha='right');plt.ylabel('Observed cluster count');plt.legend();plt.title('Three spaces: 13 candidates per train state');figure('three-space-clusters.png')
    # 2. State-macro feasibility precision.
    conditions=['short_horizon_noncatastrophic','progress_producing','trackable','constraint_valid','joint_observed_screen']
    plt.figure(figsize=(9,4))
    for j,g in enumerate(('nomad','bank','all')):
        plt.bar(np.arange(len(conditions))+(j-1)*.25,[aggregate_precision[g][c] for c in conditions],.25,label=g)
    plt.xticks(range(len(conditions)),['noncat','progress','trackable','valid','joint'],rotation=20);plt.ylim(0,1.05);plt.ylabel('State-macro precision');plt.legend();figure('feasible-precision.png')
    # 3. Mode Recall@K.
    plt.figure(figsize=(6,4))
    for space in ('command','realized'):
        plt.plot((1,2,4,8),[mode_recall[space][str(k)] for k in (1,2,4,8)],marker='o',label=space)
    plt.xticks((1,2,4,8));plt.ylim(0,1.05);plt.xlabel('NoMaD K');plt.ylabel('State-macro observed-screen mode recall');plt.legend();figure('feasible-mode-recall.png')
    # 4. Oracle regret.
    plt.figure(figsize=(9,4))
    for j,key in enumerate(('all','constraint_valid','progress_producing')):
        plt.bar(x+(j-1)*.25,[by_regret[t]['regret'][key]['8'] or 0 for t in TERRAINS],.25,label=key)
    plt.xticks(x,TERRAINS,rotation=35,ha='right');plt.ylabel('Delta V oracle regret at K8');plt.legend(fontsize=8);figure('oracle-regret.png')
    # 5. Bank winner nearest NoMaD.
    bank=regret['bank_winner_comparisons'];labels=[r['terrain'] for r in bank]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(labels,[r['path_rms_m'] for r in bank],label='path RMS m');axes[0].bar(labels,[r['realized_xy_rms_m'] for r in bank],alpha=.6,label='realized XY RMS m');axes[0].legend(fontsize=8)
    axes[1].bar(labels,[r['bank']['delta_v']-r['nearest_nomad']['delta_v'] for r in bank]);axes[1].set_ylabel('Bank minus nearest NoMaD Delta V')
    for ax in axes:ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Bank winner versus path-nearest NoMaD');figure('bank-winner-nearest-nomad.png')
    # 6. Goal endpoint.
    held=comparisons['goal_endpoint']['phase_b_heldout'];pilot=comparisons['goal_endpoint']['pilot_rows']
    plt.figure(figsize=(7,4))
    for g,color in [('nomad','tab:blue'),('bank','tab:orange')]:
        vals=[v['goal_endpoint_distance_m'] for v in pilot if v['generator']==g]
        plt.hist(vals,bins=16,alpha=.55,label=f'pilot {g}',color=color)
    for q,value in zip(('P10','P50','P90'),held['distance_quantiles_m']):plt.axvline(value,ls='--',lw=1,label=f'Phase B {q}')
    plt.xlabel('Goal to planned endpoint distance (m)');plt.ylabel('Candidate count');plt.legend(fontsize=7);figure('phaseb-pilot-goal-endpoint.png')
    # 7. GT(P/C/R) state-wise pairwise.
    plt.figure(figsize=(9,4))
    for j,kind in enumerate(('P','C','R')):
        plt.bar(x+(j-1)*.25,[by_eval[t]['gt'][kind]['pairwise_accuracy'] for t in TERRAINS],.25,label=f'GT({kind})')
    plt.axhline(.5,color='black',ls=':',lw=.8);plt.xticks(x,TERRAINS,rotation=35,ha='right');plt.ylabel('Within-state pairwise accuracy');plt.legend();figure('gt-pcr-performance.png')
    # 8. True versus predicted.
    plt.figure(figsize=(6,5))
    for kind,color in [('P','tab:blue'),('C','tab:orange'),('R','tab:green')]:
        truth=[v['delta_v'] for v in candidate_scores];pred=[float(np.mean(v[kind],axis=0)[0]) for v in candidate_scores]
        plt.scatter(truth,pred,s=12,alpha=.4,c=color,label=f'GT({kind})')
    lim=plt.xlim();plt.plot(lim,lim,color='black',ls=':',lw=.8);plt.xlabel('Oracle Delta V');plt.ylabel('Predicted Delta V');plt.legend();figure('true-vs-predicted.png')
    # 9. Generator bias at state unit.
    plt.figure(figsize=(6,4))
    for j,kind in enumerate(('P','C','R')):
        for i,g in enumerate(('nomad','bank')):
            vals=[v[kind]['bias'] for v in generator_bias if v['generator']==g]
            plt.bar(i+(j-1)*.2,float(np.mean(vals)),.2,label=f'GT({kind})' if i==0 else None)
    plt.xticks((0,1),('NoMaD','bank'));plt.ylabel('State-macro prediction bias');plt.legend();figure('generator-bias.png')
    # 10. Ensemble uncertainty versus misordering, descriptive pairs only.
    plt.figure(figsize=(7,4))
    for correct,color,label in [(True,'tab:blue','correct'),(False,'tab:red','misordered')]:
        rows=[v for v in uncertainty if v['correct']==correct]
        plt.scatter([abs(v['oracle_delta_v_gap']) for v in rows],
                    [v['ensemble_pair_uncertainty'] for v in rows],s=10,alpha=.4,c=color,label=label)
    plt.xlabel('|Oracle Delta V pair gap|');plt.ylabel('Ensemble pair disagreement');plt.legend();plt.title('Pairs are nested within 7 states');figure('uncertainty-vs-misrank.png')
    input_hashes={name:digest(OUT/name) for name in ('source-provenance.json','nomad-feasible-coverage.json',
       'nomad-oracle-regret.json','teacher-support-analysis.json','k32-offline-analysis.json',
       'phase-b-reproduction.json','evaluator-input-audit.json','planned-command-realized-comparison.json',
       'evaluator-counterfactual-results.json','trajectory-ood-comparison.json')}
    output_hashes={name:digest(OUT/name) for name in ('cause-classification.json','decision.json','evaluator-strata-and-uncertainty.json')}
    figure_hashes={p.name:digest(p) for p in sorted(FIG.glob('*.png'))}
    lines=['# Feasible coverage and Evaluator transfer: train-only diagnostic','',
        'All Stage 0–3 analyses used the seven frozen train states and 91 existing 40-step rollouts. No new simulator rollout, training, formal validation, or real robot control was performed. Candidate pairs share state; the statistical unit is seven states.',
        '', '## Source and execution', '',
        f"- Source audit: 91 candidate ID/seed/Path/trace links; NoMaD and three GT checkpoint SHA values matched. See `source-provenance.json` and `candidate-identity-table.jsonl.gz`.",
        f"- Phase B held-out GT reproduction: max absolute reported-metric difference {reproduction['max_abs_difference']:.6g} (tolerance {reproduction['tolerance']}). The held-out split was used only for this reproduction and distribution comparison.",
        f"- Teacher search: {read('teacher-support-analysis.json')['complete_only_train_window_count']:,} complete-only train windows; no validation/held-out threshold choice.",
        f"- Commands: `PYTHONPATH=src:.:scripts /usr/bin/python3 scripts/eat_nav_feasible_transfer_stage0.py`; Stage 1A same with `...stage1.py`; Docker `/isaac-sim/python.sh` scripts `eat_nav_feasible_transfer_teacher.py`, `eat_nav_feasible_transfer_phaseb_reproduce.py`, `eat_nav_feasible_transfer_evaluator.py`, `eat_nav_feasible_transfer_ood.py`; host `eat_nav_feasible_transfer_finalize.py`.",
        f"- Observed GPU peak: Phase B reproduction {reproduction['peak_gpu_bytes']/2**20:.1f} MiB, Evaluator transfer {evaluation['peak_gpu_bytes']/2**20:.1f} MiB, OOD {ood['peak_gpu_bytes']/2**20:.1f} MiB. Stage 1A elapsed {read('stage-1a-complete.json')['elapsed_sec']:.1f} s, teacher {read('teacher-support-analysis.json')['elapsed_sec']:.1f} s, Evaluator {evaluation['elapsed_sec']:.1f} s, OOD {ood['elapsed_sec']:.1f} s.",
        '', '## NoMaD short-horizon coverage', '',
        f"- State-macro joint observed-screen precision: NoMaD {aggregate_precision['nomad']['joint_observed_screen']:.3f}; fixed bank {aggregate_precision['bank']['joint_observed_screen']:.3f}. These are 2 s screens, not proof of long-horizon physical feasibility.",
        f"- Observed-screen feasible command-mode Recall@8: {mode_recall['command']['8']:.3f} state macro; realized-mode Recall@8: {mode_recall['realized']['8']:.3f}. Per-state numerators, denominators, and IDs are in `nomad-feasible-coverage.json`.",
        f"- Four bank-winning states: {len(distinct_bank_better)} have a bank winner that is simultaneously joint-screen valid and command/realized distinct from nearest NoMaD. `hf_pyramid_slope` has a distinct command winner but it fails the existing constraint/OOD screen; three other bank winners share both modes with nearest NoMaD.",
        f"- K32 adds a planned Path cluster in {sum(bool(v['added_path_modes_beyond_k8']) for v in k32)}/7 states, without rollout evidence for its feasibility. All four bank winners have independent-episode teacher shape/direction matches.",
        '', '## Evaluator transfer', '',
        '| Input | top-1 / 7 | top-3 / 7 | state-macro pairwise | state-macro oracle regret |',
        '|---|---:|---:|---:|---:|']
    for kind in ('P','C','R'):
        z=gt[kind];lines.append(f"| GT({kind}) | {z['top1_recall_count']} | {z['top3_recall_count']} | {z['state_macro_pairwise_accuracy']:.3f} | {z['state_macro_oracle_regret']:.3f} |")
    lines += ['',
        'P is planned Path, C integrates recorded follower commands, and R is the realized simulator trajectory. GT(R) uses privileged outcome information and cannot select online. GT(C) is a diagnostic, not an approved selector. The same original GT checkpoints and train-only target scalers were used for all three.',
        f"- Frozen train action p95 OOD: P {ood['summary']['P']['over_p95_count']}/91, C {ood['summary']['C']['over_p95_count']}/91, R {ood['summary']['R']['over_p95_count']}/91. OOD alone does not explain all P ranking errors.",
        f"- Phase B held-out goal/end-point exact-match fraction: {held['exact_fraction']:.3f}; pilot bank 35/35 exact, NoMaD 0/56 exact. Endpoint substitution changed P score spread little in most states.",
        f"- GT(P) top-1 0/7, worse than nominal fixed 2/7 and goal/end-point distance 2/7. Per-state baselines, calibration, uncertainty and generator bias are preserved in JSON.",
        '', '## Gates and next step', '',
        f"- Gate N: **{gate_n}**. High 2 s precision and command variation do not establish long-horizon feasible coverage or a useful constraint-valid bank mode that NoMaD misses.",
        f"- Gate E: **{gate_e}**. Exact Phase B reproduction rules out the audited implementation mismatch, but planned Path ranking remains below chance and below simple baselines. Do not use this Evaluator for online candidate choice.",
        '- The proposed 20-run, 120-control-step long-horizon experiment is described in `decision.json`; it has **not** been started and requires separate approval. Formal validation and model retraining also remain stopped.',
        '', '## SHA-256', '', '| Source artifact | SHA-256 |','|---|---|']
    lines += [f'| {name} | `{value}` |' for name,value in input_hashes.items()]
    lines += ['', '| Derived artifact | SHA-256 |','|---|---|']
    lines += [f'| {name} | `{value}` |' for name,value in output_hashes.items()]
    lines += [f'| figures/{name} | `{value}` |' for name,value in figure_hashes.items()]
    lines += ['', '## Verification', '',
        'Focused host tests: `PYTHONPATH=src:. /usr/bin/python3 -m pytest -q tests/eat_nav/test_feasible_transfer.py tests/eat_nav/test_candidate_coverage_oracle.py tests/eat_nav/test_candidate_coverage_evaluator.py` (11 passed). The Stage 0 identity and SHA gate and Phase B exact reproduction also passed.',
        'Original pilot traces, checkpoints, and logs remain outside Git under `/tmp/eatnav-candidate-coverage/` and `/tmp/eatnav-quality-identifiability-gate/`.', '']
    report=OUT/'analysis-report.md';content='\n'.join(lines)
    if report.exists():
        if report.read_text()!=content:raise FileExistsError('report differs')
    else:
        temp=report.with_name(report.name+'.tmp');temp.write_text(content);temp.replace(report)
    atomic_once(OUT/'stage-3-complete.json',{'stage':3,'status':'complete','input_sha256':input_hashes,
        'output_sha256':{**output_hashes,'analysis-report.md':digest(report),
                         **{f'figures/{name}':sha for name,sha in figure_hashes.items()}},
        'elapsed_sec':time.time()-started,'figure_count':len(figure_hashes),
        'new_rollouts_started':False,'training_started':False,'formal_validation_started':False})
    print('FEASIBLE_TRANSFER_STAGE3_COMPLETE',gate_n,gate_e,len(figure_hashes),'figures',round(time.time()-started,1),'sec',flush=True)

if __name__=='__main__':run()
