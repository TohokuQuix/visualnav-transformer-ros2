"""Generate Phase A comparison and explicit blocked-forward-evaluator decision."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

REMOTE=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup/output')
DOCS=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/relative-quality-followup'
PLOTS=('probe-ablation','scalar-vs-multi','reward-free-vs-advantage')


def read(name):return json.loads((REMOTE/name).read_text())


def write(name,payload):
    p=REMOTE/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(t,p)


def inside():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from scipy.stats import spearmanr
    from visualnav_transformer.eat_nav.relative_quality_followup_probe import FIELDS
    from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci
    config=yaml.safe_load(Path('/tmp/eatnav-train/relative-quality-followup-config.yaml').read_text())
    results=read('probe-ablation-results.json');training=read('probe-ablation-training-summary.json')
    matches=read('probe-ablation-matched-pairs.json')['metrics']
    perms=read('probe-ablation-permutation.json')
    actor=read('actor-latent-provenance.json')
    old_root=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/output')
    previous=json.loads((old_root/'relative-quality-probe-results.json').read_text())
    old_stats=json.loads((old_root/'relative-quality-correlations.json').read_text())
    raw=np.load('/tmp/eatnav-quality-identifiability-gate/relative-quality/splits/test.npz')
    errors=np.load('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup/probe-ablation-window-errors.npz')
    multi_gain=np.mean([errors[f'probe_delta-seed{seed}_mse']-
                        errors[f'probe_multi_reward_free-seed{seed}_mse'] for seed in config['probe_seeds']],axis=0)
    multi_vs_delta_ci=episode_cluster_ci(multi_gain,raw['episode'],seed=config['bootstrap_seed'],
                                          repeats=config['bootstrap_repeats'])[0]
    return_v0=float(spearmanr(raw['return_h'],raw['v0']).statistic)
    adv_v0=float(spearmanr(raw['advantage_h'],raw['v0']).statistic)
    names=['probe_0','probe_abs','probe_delta','probe_recovery','probe_return',
           'probe_adv','probe_drawdown','probe_delta_drawdown','probe_relative_triplet',
           'probe_multi_reward_free','probe_multi_no_v0','probe_multi']
    bykind={k:[r for r in results['runs'] if r['kind']==k] for k in names}
    if any(len(v)!=3 for v in bykind.values()):raise ValueError('missing three-seed Probe result')
    mean={k:{field:float(np.mean([r[field] for r in bykind[k]])) for field in
              ('heldout_window_mse','heldout_physical_rms_m','heldout_r2')} for k in names}
    mean_rms0=mean['probe_0']['heldout_physical_rms_m']
    curves={}
    for run in results['runs']:
        history_path=(Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup/probes')/
                      f"{run['kind']}-seed{run['seed']}/history.json") if run['kind'] in config['probe_kinds'] else (
                      Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/probes')/
                      f"{run['kind']}-seed{run['seed']}/history.json")
        curves[f"{run['kind']}-seed{run['seed']}"]=json.loads(history_path.read_text())
    write('probe-ablation-curves.json',curves)
    (REMOTE/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    color=['#6b7583' if k in ('probe_0','probe_abs') else '#3c7b9a' if len(k.split('_'))<=2 else '#a66b43' for k in names]
    axes[0].bar(range(len(names)),[mean[k]['heldout_physical_rms_m'] for k in names],color=color)
    axes[0].set_xticks(range(len(names)),[k.replace('probe_','') for k in names],rotation=50,ha='right')
    axes[0].set_ylabel('held-out physical trajectory RMS (m)')
    axes[0].set_title('Validation-selected three-seed Probe ablation')
    for k in config['probe_kinds']:
        history_path=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup/probes')/f'{k}-seed11/history.json'
        history=json.loads(history_path.read_text())
        axes[1].plot([h['epoch'] for h in history],[h['validation_loss'] for h in history],label=k.replace('probe_',''))
    axes[1].set_xlabel('epoch');axes[1].set_ylabel('aggregate validation MSE')
    axes[1].set_title('Seed 11 validation curves; all train/val histories in /tmp')
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(REMOTE/'plots/probe-ablation.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for k in names:
        x=bykind[k][0]['parameter_count'];y=mean[k]['heldout_physical_rms_m']
        ax.scatter(x,y,s=80);ax.annotate(k.replace('probe_',''),(x,y),xytext=(3,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('trainable parameters');ax.set_ylabel('held-out physical RMS (m)')
    ax.set_title('Scalar versus vector Quality; modest parameter differences')
    fig.tight_layout();fig.savefig(REMOTE/'plots/scalar-vs-multi.png',dpi=150);plt.close(fig)
    selected=['probe_0','probe_abs','probe_adv','probe_delta','probe_drawdown',
              'probe_multi','probe_multi_reward_free','probe_multi_no_v0','probe_relative_triplet']
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(range(len(selected)),
        [(mean_rms0-mean[k]['heldout_physical_rms_m'])/mean_rms0*100 for k in selected],color='#4b8875')
    ax.set_xticks(range(len(selected)),[k.replace('probe_','') for k in selected],rotation=42,ha='right')
    ax.set_ylabel('physical RMS reduction from fixed-q Probe 0 (%)')
    ax.set_title('Reward-derived versus Critic-only relative Quality')
    fig.tight_layout();fig.savefig(REMOTE/'plots/reward-free-vs-advantage.png',dpi=150);plt.close(fig)
    proxy=old_stats['goal_dem_proxy_heldout'];common=results['common_support_terrains']
    decisions={}
    for kind in config['probe_kinds']:
        c=results['comparison'][kind];match=matches[kind]['cross_minus_same']['trajectory_rms_m']
        supported=[c['by_terrain'][t]['paired_mse_gain'] for t in common
                   if c['by_terrain'].get(t,{}).get('status')=='supported']
        checks={'episode_ci_positive':c['gain_ci95_episode'][0]>0,
                'rms_reduction_at_least_5pct':c['relative_physical_rms_gain']>=config['min_relative_rms_gain'],
                'delta_r2_at_least_0p02':c['delta_r2']>=config['min_delta_r2'],
                'beats_all_terrain_shuffles':perms[kind]['observed_outperforms_all_shuffles'],
                'common_support_direction':bool(supported) and all(v>0 for v in supported),
                'matched_trajectory_difference':match.get('ci_95',[0])[0]>0,
                'not_single_episode_dependent':c['minimum_leave_one_episode_out_gain']>0 and c['tilt_excluded_gain']>0,
                'better_than_absolute_q':c['gain_over_reference_ci95_episode']['probe_abs'][0]>0}
        proxy_r2={field.removeprefix('z_'):proxy[field.removeprefix('z_')]['heldout_r2']
                  for field in FIELDS[kind]}
        checks['goal_dem_proxy_not_sufficient']=(min(proxy_r2.values())<.1 and
            checks['episode_ci_positive'])
        ref0=[r for r in results['runs'] if r['kind']=='probe_0']
        baseline_underfit=any(r['validation_loss']>=training['mean_trajectory_validation_mse']*
             (1-config['min_validation_gain_over_mean']) or r['underfit_cap'] for r in ref0)
        underfit=baseline_underfit or not c['validation_better_than_mean'] or c['validation_improving_at_cap']
        decision='inconclusive' if underfit else 'pass' if all(checks.values()) else 'fail'
        decisions[kind]={'status':decision,'checks':checks,'underfit':underfit,
           'matched_rms_gap_m':match.get('observed_mean_difference'),
           'matched_rms_gap_ci95':match.get('ci_95'),
           'goal_dem_linear_proxy_r2_by_component':proxy_r2}
    def ci_better(a,b):
        return results['comparison'][a]['gain_over_reference_ci95_episode'][b][0]>0
    questions={
      'delta_beats_absolute_q':ci_better('probe_delta','probe_abs'),
      'delta_beats_advantage':ci_better('probe_delta','probe_adv'),
      'delta_beats_drawdown':ci_better('probe_delta','probe_drawdown'),
      'reward_free_multi_at_least_existing_multi':
         results['comparison']['probe_multi_reward_free']['gain_over_reference_ci95_episode']['probe_multi'][0]>=0,
      'reward_free_multi_point_rms_vs_existing_multi_pct':
         100*(mean['probe_multi']['heldout_physical_rms_m']-mean['probe_multi_reward_free']['heldout_physical_rms_m'])/
         mean['probe_multi']['heldout_physical_rms_m'],
      'v0_improves_reward_free_vs_triplet':
         results['comparison']['probe_multi_reward_free']['gain_over_reference_ci95_episode']['probe_relative_triplet'][0]>0,
      'v0_improves_adv_multi_vs_no_v0':
         results['comparison']['probe_multi_no_v0']['gain_over_reference_ci95_episode']['probe_multi'][1]<0,
      'recovery_single_pass':decisions['probe_recovery']['status']=='pass',
      'return_v0_spearman':return_v0,'advantage_v0_spearman':adv_v0,
      'return_initial_state_confounding_supported':abs(return_v0)>abs(adv_v0),
      'scalar_vs_multi_parameter_difference':bykind['probe_multi'][0]['parameter_count']-bykind['probe_adv'][0]['parameter_count'],
      'reward_free_multi_vs_delta_mse_gain_ci95':multi_vs_delta_ci,
      'reward_free_multi_beats_delta_ci':multi_vs_delta_ci[0]>0}
    forward_status='inconclusive'
    decision={'phase_a':decisions,'phase_a_questions':questions,
      'phase_b':{'status':forward_status,'training_performed':False,
                 'reason':'recorded 22-D actor observation absent; body-state input/window correspondence cannot be checked',
                 'candidate_ranking_claim':False},
      'oracle_best_of_k_evidence':'not yet established by forward evaluator; no rollout performed',
      'common_support_terrains':common,'all_seven_terrain_generalization_claim':False,
      'raw_data_only':True}
    write('relative-quality-followup-decision.json',decision)
    fcfg='''# Forward evaluator specification retained for a future observation-complete dataset
schema_version: 1
status: blocked_actor_observation_missing
required_body_state: exact_pre_action_actor_observation
actor_latent: only_if_exact_recomputation_and_action_consistency_pass
candidate_trajectory_shape: [8, 2]
variants: [E-state, E-traj, E-state-traj, E-latent-traj]
outputs: [critic_delta, advantage_h, drawdown, recovery, return_h]
training_started: false
heldout_evaluation_started: false
'''
    (REMOTE/'forward-evaluator-config.yaml').write_text(fcfg)
    (REMOTE/'forward-evaluator-results.json').write_text(json.dumps({'status':'not_run_input_gate_failed',
      'metrics':None,'actor_observation_status':actor['status']},indent=2)+'\n')
    (REMOTE/'forward-evaluator-training-summary.json').write_text(json.dumps({'status':'not_run_input_gate_failed',
      'runs':[],'checkpoints':[]},indent=2)+'\n')
    lines=['# Relative Quality Probe ablation','',
           'The 61,948 frozen aligned windows, action normalization, source labels, three seeds and earlier Probe checkpoints were verified before training. All new runs started from scratch and selected checkpoints using aggregate validation loss only. The 29 terminal-endpoint windows are excluded equally.','',
           '| Probe | params | selected epochs | held-out MSE | physical RMS m | R² | RMS gain vs fixed q | decision |',
           '|---|---:|---|---:|---:|---:|---:|---|']
    for k in names:
        c=results['comparison'].get(k,previous['comparison'].get(k,{}))
        lines.append(f"| {k} | {bykind[k][0]['parameter_count']:,} | "
          f"{','.join(str(r['best_epoch']) for r in bykind[k])} | {mean[k]['heldout_window_mse']:.4f} | "
          f"{mean[k]['heldout_physical_rms_m']:.4f} | {mean[k]['heldout_r2']:.3f} | "
          f"{100*c.get('relative_physical_rms_gain',0):.1f}% | {decisions.get(k,{}).get('status','reference')} |")
    lines+=['','## Paired comparisons and controls','']
    for k in config['probe_kinds']:
        c=results['comparison'][k];d=decisions[k];p=perms[k]
        lines.append(f"- **{k}**: fixed-q episode CI {c['gain_ci95_episode']}; absolute-q episode CI "
           f"{c['gain_over_reference_ci95_episode']['probe_abs']}; matched trajectory RMS gap "
           f"{d['matched_rms_gap_m']:.3f} m, CI {d['matched_rms_gap_ci95']}; terrain shuffles "
           f"{p['shuffle_count']} (all beaten={p['observed_outperforms_all_shuffles']}); "
           f"minimum leave-one-episode-out gain {c['minimum_leave_one_episode_out_gain']:.4f}; "
           f"decision **{d['status']}**.")
    delta_ci=results['comparison']['probe_delta']['gain_over_reference_ci95_episode']
    reward_free_ci=results['comparison']['probe_multi_reward_free']['gain_over_reference_ci95_episode']
    no_v0_ci=results['comparison']['probe_multi_no_v0']['gain_over_reference_ci95_episode']
    lines+=['',
      f"On identical held-out windows, delta beat advantage (episode CI {delta_ci['probe_adv']}) and drawdown (CI {delta_ci['probe_drawdown']}). "
      f"Reward-free multi was worse than the existing advantage-containing multi (old-minus-new MSE CI {reward_free_ci['probe_multi']}); "
      f"adding V0 to the Critic-only triplet improved prediction (CI {reward_free_ci['probe_relative_triplet']}). "
      f"The advantage-containing model with V0 also beat its no-V0 variant (with-V0-minus-no-V0 MSE CI {no_v0_ci['probe_multi']}). "
      f"Reward-free multi beat scalar delta on matched held-out windows (delta-minus-multi MSE CI {multi_vs_delta_ci})."]
    lines+=['','## Questions requested for Phase A','']
    for k,v in questions.items():lines.append(f'- {k}: {v}')
    lines+=['','## Terrain support and paired MSE gain','',
            '| Terrain | count | '+ ' | '.join(config['probe_kinds']) + ' |',
            '|---|---:|'+'---:|'*len(config['probe_kinds'])]
    for terrain in sorted(results['comparison']['probe_delta']['by_terrain']):
        count=results['comparison']['probe_delta']['by_terrain'][terrain]['count']
        cells=[]
        for kind in config['probe_kinds']:
            item=results['comparison'][kind]['by_terrain'][terrain]
            cells.append(f"{item['paired_mse_gain']:.4f}" if item['status']=='supported' else 'NA: insufficient support')
        lines.append(f"| {terrain} | {count} | "+' | '.join(cells)+' |')
    lines+=['','All 36 train/validation curves are saved in [probe-ablation-curves.json](probe-ablation-curves.json).']
    lines+=['','The scalar and vector Probes share the exact DEM encoder, Quality-token path, trajectory head and optimizer. Vector width adds 64, 128 or 192 first-layer parameters; this small difference alone does not prove that dimensionality, rather than capacity, caused the gain. A positive post-hoc Probe result does not establish online candidate ranking. Train-only thresholds were used for matched vector controls. Different metrics select different matched pairs, so pair RMS gaps are not a head-to-head ranking; the Probe comparison uses identical held-out windows. All 100-shuffle expansions followed the frozen boundary rule.','',
            f"Common train/validation/held-out terrain support: {common}. Terrain-specific NA cells retain 'insufficient support'; no all-seven-terrain claim is made.",'',
            '## Figures','','- [Probe ablation](plots/probe-ablation.png)',
            '- [Scalar versus multi](plots/scalar-vs-multi.png)',
            '- [Reward-free versus advantage](plots/reward-free-vs-advantage.png)']
    (REMOTE/'probe-ablation-report.md').write_text('\n'.join(lines)+'\n')
    forward=['# Forward candidate evaluator','',
      '**Stopped before training:** the raw record lacks the exact 22-D pre-action actor observation. '
      'The collector consumed `obs["policy"]` to compute action but serialized only action, command, pose, twist, DEM, reward and Critic. '
      'Joint positions/velocities and observation noise realizations were not saved, so neither the observation nor a latent can be reconstructed exactly. '
      'Substituting twist/command would violate the specified body-state input and would not permit an action-consistency check.','',
      f"Actor checkpoint SHA: `{actor['actor_checkpoint_sha256']}`. Raw records scanned: {actor['raw_record_count']:,}. "
      f"Observation statuses: {actor['actor_observation_status_counts']}. Empirical normalization was disabled, "
      'but observation corruption was enabled. Actor latent was not saved.','',
      'E-state, E-traj, E-state-traj and E-latent-traj were not trained or evaluated. '
      'No predictive signal, candidate ranking ability or body-state benefit can be claimed from this Phase B. '
      'A future dataset needs the exact pre-action `obs[policy]` per session/env/episode/step and action, '
      'with checkpoint provenance. This would require separate collection authorization; none was performed here.']
    (REMOTE/'forward-evaluator-report.md').write_text('\n'.join(forward)+'\n')
    summary=['# Relative Quality follow-up','',
       'Phase A completed on the frozen extended-pilot windows. See [Probe ablation](probe-ablation-report.md) '
       'and [machine-readable decision](relative-quality-followup-decision.json).','',
       'Phase B stopped at the actor-input gate. See [actor provenance](actor-latent-provenance.json) '
       'and [forward evaluator report](forward-evaluator-report.md). No simulator rollout, Diffusion retraining, '
       'full collection or reranker deployment was performed.']
    (REMOTE/'relative-quality-followup.md').write_text('\n'.join(summary)+'\n')
    print('FOLLOWUP_REPORT_COMPLETE',flush=True)


def main():
    if '--inside' in sys.argv:inside();return
    subprocess.run(['docker','cp',str(Path(__file__).resolve()),
      'isaac-lab-onix:/tmp/eatnav-train/eat_nav_relative_followup_report.py'],check=True)
    subprocess.run(['docker','exec','-e','PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src',
      'isaac-lab-onix','/isaac-sim/python.sh','/tmp/eatnav-train/eat_nav_relative_followup_report.py','--inside'],check=True)
    names=['probe-ablation-report.md','forward-evaluator-config.yaml','forward-evaluator-report.md',
           'relative-quality-followup.md','relative-quality-followup-decision.json',
           'forward-evaluator-results.json','forward-evaluator-training-summary.json',
           'probe-ablation-curves.json'] + [f'plots/{x}.png' for x in PLOTS]
    for name in names:
        target=DOCS/name;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.tmp')
        subprocess.run(['docker','cp',f'isaac-lab-onix:{REMOTE/name}',str(temp)],check=True)
        os.replace(temp,target)


if __name__=='__main__':main()
