"""Generate frozen relative-quality figures, decision, and report from diagnostics."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

REMOTE = Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/output')
LOCAL = Path(__file__).resolve().parents[1] / 'docs/quality-identifiability-gate/relative-quality'
PLOTS = ('relative-quality-distributions', 'terrain-confounding', 'goal-confounding',
         'quality-vs-trajectory-difference', 'probe-comparison',
         'advantage-drawdown-recovery')
METRICS = ('quality_abs', 'delta_v', 'advantage_h', 'drawdown', 'recovery', 'return_h')
DISPLAY = {'quality_abs':'absolute q', 'delta_v':'Critic delta', 'advantage_h':'H-step advantage',
           'drawdown':'drawdown', 'recovery':'recovery', 'return_h':'H-step return'}


def get(name):
    return json.loads((REMOTE / name).read_text())


def output(name, obj):
    path = REMOTE / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def inside():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    REMOTE.mkdir(parents=True, exist_ok=True)
    (REMOTE / 'plots').mkdir(exist_ok=True)
    cfg = yaml.safe_load(Path('/tmp/eatnav-train/relative-quality-config.yaml').read_text())
    extraction = get('relative-quality-extraction.json')
    summary = get('relative-quality-summary.json')
    correlations = get('relative-quality-correlations.json')
    pairs = get('relative-quality-matched-pairs.json')
    multi_pairs = get('relative-quality-multi-matched-pairs.json')
    pairs['metrics']['multi'] = multi_pairs['metrics']['multi']
    training = get('relative-quality-probe-training-summary.json')
    probe = get('relative-quality-probe-results.json')
    permutation = get('relative-quality-permutation.json')
    values=[]
    with gzip.open(REMOTE / 'relative-quality-values.jsonl.gz','rt') as source:
        for line in source:
            x=json.loads(line)
            if x['split']=='test' and x['primary_available']:
                values.append(x)

    def save(name):
        plt.tight_layout()
        plt.savefig(REMOTE / 'plots' / (name + '.png'), dpi=160)
        plt.close()

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,metric in zip(axes.flat,METRICS):
        x=np.array([r[metric] for r in values],float)
        ax.hist(x,bins=45,color='#3a78a2',alpha=.85)
        ax.set_title(DISPLAY[metric]);ax.set_ylabel('held-out windows')
    save(PLOTS[0])

    terrain=sorted(summary['test']['by_terrain'])
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    xs=np.arange(len(terrain));width=.13
    for j,metric in enumerate(METRICS):
        means=[summary['test']['by_terrain'][t]['metrics'][metric]['mean'] for t in terrain]
        z=((np.asarray(means)-summary['train']['by_metric'][metric]['mean']) /
           max(1e-8,summary['train_only_scalers'].get(metric,{}).get('std',1)))
        axes[0].bar(xs+(j-2.5)*width,z,width,label=DISPLAY[metric])
    axes[0].set_xticks(xs,terrain,rotation=35,ha='right');axes[0].set_ylabel('terrain mean, train-standardized')
    eta=[correlations['test'][m]['terrain_eta_squared'] for m in METRICS]
    axes[1].bar(np.arange(len(METRICS)),eta,color='#aa6d43')
    axes[1].set_xticks(np.arange(len(METRICS)),[DISPLAY[m] for m in METRICS],rotation=40,ha='right')
    axes[1].set_ylabel('held-out terrain eta squared')
    axes[0].legend(fontsize=8);save(PLOTS[1])

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    rng=np.random.default_rng(613);sample=rng.choice(len(values),size=min(2000,len(values)),replace=False)
    for metric in ('quality_abs','advantage_h','drawdown'):
        axes[0].scatter([values[i]['goal_distance_m'] for i in sample],
                        [values[i][metric] for i in sample],s=5,alpha=.18,label=DISPLAY[metric])
    axes[0].set_xlabel('relative goal distance (m)');axes[0].set_ylabel('raw metric');axes[0].legend()
    for j,m in enumerate(METRICS):
        axes[1].bar(j-.17,correlations['test'][m]['goal_distance_m'],.34,color='#447ca8')
        axes[1].bar(j+.17,correlations['test'][m]['dem_height_std_m'],.34,color='#cb8752')
    axes[1].set_xticks(range(len(METRICS)),[DISPLAY[m] for m in METRICS],rotation=38,ha='right')
    axes[1].set_ylabel('Spearman rho');axes[1].legend(['goal distance','DEM height std'])
    save(PLOTS[2])

    fig,ax=plt.subplots(figsize=(11,5))
    observed=[];lower=[];upper=[]
    for m in METRICS:
        v=pairs['metrics'][m]['cross_minus_same']['trajectory_rms_m']
        observed.append(v['observed_mean_difference']);lower.append(v['ci_95'][0]);upper.append(v['ci_95'][1])
    ax.bar(range(len(METRICS)),observed,color='#3c7799')
    ax.errorbar(range(len(METRICS)),observed,
                yerr=[np.asarray(observed)-lower,np.asarray(upper)-observed],fmt='none',c='black',capsize=4)
    ax.axhline(0,c='gray',lw=1);ax.set_xticks(range(len(METRICS)),[DISPLAY[m] for m in METRICS],rotation=30,ha='right')
    ax.set_ylabel('cross minus same trajectory RMS (m), episode bootstrap CI');save(PLOTS[3])

    fig,axes=plt.subplots(1,2,figsize=(14,5));kinds=['probe_0','probe_abs','probe_adv','probe_drawdown','probe_multi']
    for k in kinds:
        curves=[r for r in training['runs'] if r['kind']==k]
        for run in curves:
            history=json.loads((Path(run['checkpoint']).parent/'history.json').read_text())
            axes[0].plot([h['epoch'] for h in history],[h['validation_loss'] for h in history],
                         alpha=.4,label=k if run['seed']==11 else None)
    axes[0].set_xlabel('epoch');axes[0].set_ylabel('aggregate validation MSE');axes[0].legend(fontsize=8)
    mean=[np.mean([r['heldout_physical_rms_m'] for r in probe['runs'] if r['kind']==k]) for k in kinds]
    axes[1].bar(range(len(kinds)),mean,color='#4c8a75');axes[1].set_xticks(range(len(kinds)),kinds,rotation=35,ha='right')
    axes[1].set_ylabel('held-out physical trajectory RMS (m)');save(PLOTS[4])

    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,m in zip(axes,('advantage_h','drawdown','recovery')):
        ax.scatter([values[i][m] for i in sample],[values[i]['forward_progress_m'] for i in sample],s=5,alpha=.2)
        ax.set_xlabel(DISPLAY[m]);ax.set_ylabel('forward progress (m)')
        ax.set_title('Spearman %.3f'%correlations['test'][m]['forward_progress_m'])
    save(PLOTS[5])

    common=sorted(set.intersection(*(set(summary[s]['by_terrain']) for s in ('train','val','test'))))
    import torch
    from visualnav_transformer.eat_nav.relative_quality_probe import quality_tensor,tensorize,predict
    torch.set_num_threads(4)
    ckpt=torch.load('/tmp/eatnav-train/pilot-quality-best.pt',map_location='cpu',weights_only=True)
    norm=ckpt['action_statistics']
    raw=np.load('/tmp/eatnav-quality-identifiability-gate/relative-quality/splits/test.npz')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    test=tensorize(raw,np.asarray(norm['mean'],dtype=np.float32),np.asarray(norm['std'],dtype=np.float32),device)
    episode=np.asarray(raw['episode']);keys=[tuple(e) for e in episode]
    tilt_episodes={tuple(r['episode_key']) for r in values if 'tilt_over' in r['episode_terminal_reason']}
    baseline_by_seed={}
    for run in training['runs']:
        if run['kind']=='probe_0':
            q=quality_tensor(raw,'probe_0',device)
            pred=predict(run['checkpoint'],test,q,cfg['probe_batch_size'])
            baseline_by_seed[run['seed']]=(pred-test['action']).square().mean(dim=(1,2)).cpu().numpy()
    concentration={}
    for kind in ('probe_adv','probe_drawdown','probe_multi'):
        gains=[]
        for run in training['runs']:
            if run['kind']!=kind:continue
            q=quality_tensor(raw,kind,device)
            pred=predict(run['checkpoint'],test,q,cfg['probe_batch_size'])
            gains.append(baseline_by_seed[run['seed']]-(pred-test['action']).square().mean(dim=(1,2)).cpu().numpy())
        gain=np.mean(gains,axis=0)
        unique=list(dict.fromkeys(keys))
        leave=[float(gain[np.array([e!=key for e in keys])].mean()) for key in unique]
        non_tilt=np.array([key not in tilt_episodes for key in keys])
        concentration[kind]={'episode_count':len(unique),'tilt_episode_count':len(tilt_episodes),
                             'minimum_leave_one_episode_out_gain':min(leave),
                             'tilt_excluded_gain':float(gain[non_tilt].mean()) if non_tilt.any() else None,
                             'all_leave_one_out_positive':all(x>0 for x in leave)}
    decisions={}
    for kind,m in [('probe_adv','advantage_h'),('probe_drawdown','drawdown'),('probe_multi','multi')]:
        v=probe['comparison'][kind]
        paired=pairs['metrics'].get(m)
        checks={
            'positive_episode_ci':v['gain_ci95_episode'][0]>0,
            'physical_rms_gain_5pct':v['relative_physical_rms_gain']>=cfg['min_relative_rms_gain'],
            'delta_r2_0.02':v['delta_r2']>=cfg['min_delta_r2'],
            'beats_every_permutation':permutation[kind]['observed_outperforms_all_shuffles'],
            'better_than_absolute_q':v['relative_physical_rms_gain_vs_abs']>=cfg['min_relative_rms_gain_vs_abs'],
            'common_terrain_direction':all(v['by_terrain'][t]['paired_mse_gain']>0 for t in common
                                           if v['by_terrain'].get(t,{}).get('status')=='supported')
                and any(v['by_terrain'].get(t,{}).get('status')=='supported' for t in common),
            'matched_trajectory_difference':paired is not None and
                paired['cross_minus_same']['trajectory_rms_m']['ci_95'][0]>0,
            'not_single_episode_dependent':concentration[kind]['all_leave_one_out_positive'] and
                concentration[kind]['tilt_excluded_gain']>0,
            'not_goal_dem_proxy_only':m=='multi' or correlations['goal_dem_proxy_heldout'][m]['heldout_r2']<.1,
        }
        probe0_runs=[r for r in training['runs'] if r['kind']=='probe_0']
        probe0_underfit=any(r['best_validation_loss']>=training['mean_trajectory_validation_mse']*(1-cfg['min_probe_validation_gain_over_mean']) or r['validation_improving_at_cap'] for r in probe0_runs)
        underfit=(probe0_underfit or not v['validation_better_than_mean'] or v['validation_improving_at_cap'] or
                  not probe['comparison']['probe_abs']['validation_better_than_mean'] or
                  probe['comparison']['probe_abs']['validation_improving_at_cap'])
        status='inconclusive' if underfit else 'pass' if all(checks.values()) else 'fail'
        decisions[kind]={'status':status,'checks':checks,'underfit':underfit,
                         'episode_concentration':concentration[kind]}
    win=sorted(((m,pairs['metrics'][m]['cross_minus_same']['trajectory_rms_m']['observed_mean_difference'])
                for m in METRICS),key=lambda x:x[1],reverse=True)
    decision={'relative_candidates':decisions,'matching_ranking_by_rms_gap':win,
              'heldout_supported_terrains':sorted(summary['test']['by_terrain']),
              'common_support_terrains':common,'scope':'post-hoc static diagnostic only',
              'oracle_best_of_k_recommendation':'consider only after interpretation of gates; no rollout approved',
              'all_terrain_generalization_claim':False}
    output('relative-quality-decision.json',decision)

    lines=['# EAT-Nav relative trajectory quality diagnostic','',
           '## Main result','',
           f"Matched pairs rank **{DISPLAY[win[0][0]]}** first for trajectory RMS separation "
           f"({win[0][1]:.3f} m cross-minus-control); this is a diagnostic association, not a causal ranking.",'',
           '## Scope and timing','',
           f"Frozen complete windows: train {extraction['complete_counts']['train']:,}, validation "
           f"{extraction['complete_counts']['val']:,}, held-out {extraction['complete_counts']['test']:,}. "
           f"Primary aligned windows: {extraction['aligned_primary_counts']}. "
           'The 29 terminal-endpoint windows have auxiliary terminal returns but no complete bootstrap Critic and are excluded equally from all Probe comparisons.',
           'The collector records a pre-action Critic, then a post-transition reward/pose. We anchor at the recorded post-step pose and use the next record Critic. Each 2 s window spans 40 control rewards with gamma=0.99 per control step. True terminal bootstraps zero; timeout uses the PPO time-limit correction. Reward decomposition was not saved. All future-derived labels are post-hoc and unavailable to online inference.','',
           '## Held-out matched neighborhoods','',
           '| Metric | Cross | Control | RMS gap (m) | episode bootstrap 95% CI |','|---|---:|---:|---:|---|']
    for m in METRICS+('multi',):
        p=pairs['metrics'][m];a=p['cross_minus_same']['trajectory_rms_m']
        lines.append(f"| {DISPLAY.get(m,'Q-multi')} | {p['cross']['count']} | {p['same']['count']} | {a['observed_mean_difference']:.3f} | [{a['ci_95'][0]:.3f}, {a['ci_95'][1]:.3f}] |")
    lines+=['','Thresholds came only from train. Pairs share terrain and similar DEM/goal; no episode or window is reused within each comparison. Different metrics select different pairs, so gap sizes are not a direct head-to-head causal comparison.','',
            '## Confounding','',
            '| Metric | goal Spearman | DEM roughness Spearman | progress Spearman | goal+DEM held-out R² |','|---|---:|---:|---:|---:|']
    for m in METRICS:
        c=correlations['test'][m];r=correlations['goal_dem_proxy_heldout'][m]['heldout_r2']
        lines.append(f"| {DISPLAY[m]} | {c['goal_distance_m']:.3f} | {c['dem_height_std_m']:.3f} | {c['forward_progress_m']:.3f} | {r:.3f} |")
    lines+=['','## Validation-selected deterministic Probes','',
            f"Mean-trajectory validation MSE: {probe['mean_trajectory_val_mse']:.4f}. Mean-trajectory held-out MSE: {probe['mean_trajectory_test_mse']:.4f}.",'',
            '| Probe | held-out MSE | physical RMS (m) | R² | ΔR² vs fixed q | RMS gain vs fixed q | status |',
            '|---|---:|---:|---:|---:|---:|---|']
    for k in kinds:
        rows=[r for r in probe['runs'] if r['kind']==k]
        v=probe['comparison'].get(k,{})
        lines.append(f"| {k} | {np.mean([r['heldout_window_mse'] for r in rows]):.4f} | "
                     f"{np.mean([r['heldout_physical_rms_m'] for r in rows]):.3f} | "
                     f"{np.mean([r['heldout_r2'] for r in rows]):.3f} | "
                     f"{v.get('delta_r2',0):.3f} | {v.get('relative_physical_rms_gain',0)*100:.1f}% | "
                     f"{decisions.get(k,{}).get('status','reference')} |")
    lines+=['','The scalar Probe uses the same Quality MLP/token path for fixed q=0.5 and real q. Multi adds only 192 parameters. The train-only action scale and identical seeds/order/optimizer were used. Epoch selection used aggregate validation only; held-out was evaluated once.','',
            '## Gate details','']
    for k,v in decisions.items():
        p=permutation[k];c=probe['comparison'][k]
        lines.append(f"- **{k}: {v['status']}**. Episode CI for held-out MSE gain {c['gain_ci95_episode']}; "
                     f"permutations {p['shuffle_count']}, all beaten={p['observed_outperforms_all_shuffles']}; "
                     f"checks={v['checks']}; underfit={v['underfit']}.")
    lines+=['','Terrain-specific results are NA when a held-out terrain has fewer than the configured q-bin support. '
            f"The train/validation/held-out common support is {common}; held-out supports {decision['heldout_supported_terrains']}. "
            'No claim is made for all seven terrains. Timeout is not labeled as locomotion failure. '
            'A single tilt episode cannot support independent-window significance.','',
            '## Interpretation and next step','',
            'The rank and Probe gates above measure associations in the existing on-policy log. A positive relative label result would motivate an explicitly approved oracle best-of-K candidate rollout from shared simulator states. It does not authorize a new rollout, full collection, or model/loss change. If no candidate passes, the log is insufficient to conclude that a scalar relative score is useful; counterfactual candidate rollouts may be needed. Reward decomposition was not saved, so progress reward and body stability cannot be separated.','',
            '## Figures','']
    for name in PLOTS:
        lines.append(f'- [{name}](plots/{name}.png)')
    (REMOTE/'relative-quality-report.md').write_text('\n'.join(lines)+'\n')
    print('RELATIVE_REPORT_COMPLETE',flush=True)


def main():
    if '--inside' in sys.argv:
        inside();return
    subprocess.run(['docker','cp',str(Path(__file__).resolve()),
                    'isaac-lab-onix:/tmp/eatnav-train/eat_nav_relative_quality_report.py'],check=True)
    subprocess.run(['docker','exec','-e',
                    'PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src',
                    'isaac-lab-onix','/isaac-sim/python.sh',
                    '/tmp/eatnav-train/eat_nav_relative_quality_report.py','--inside'],check=True)
    names=['relative-quality-report.md','relative-quality-decision.json',
           'relative-quality-multi-match-threshold.json','relative-quality-multi-matched-pairs.json']+[f'plots/{n}.png' for n in PLOTS]
    for name in names:
        target=LOCAL/name;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.tmp')
        subprocess.run(['docker','cp',f'isaac-lab-onix:{REMOTE/name}',str(temp)],check=True)
        os.replace(temp,target)


if __name__=='__main__':
    main()
