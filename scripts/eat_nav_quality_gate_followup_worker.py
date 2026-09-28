"""Docker-side stages for the immutable Quality identifiability follow-up."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset, _nearest, _records
from visualnav_transformer.eat_nav.quality_followup import (
    atomic_json, is_complete_window, pair_id, select_ladder_pairs,
    sha256_file, validate_complete_splits,
)
from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings
from visualnav_transformer.eat_nav.quality_gate_overfit import prepare_overfit_pairs
from visualnav_transformer.eat_nav.quality_gate_probe import tensorize_windows
from visualnav_transformer.eat_nav.quality_followup_training import train_probe_resumable, train_overfit_resumable

BASE=Path('/tmp/eatnav-train')
SOURCE=BASE/'quality-gate-output'
ROOT=BASE/'followup'
STAGES=ROOT/'stages'
MANIFEST=Path('/tmp/eatnav-extended/output/manifest.json')
QUALITY=BASE/'quality.yaml'
FOLLOWUP=BASE/'followup-config.yaml'
PROBE=BASE/'probe-config.yaml'
OVERFIT=BASE/'overfit-config.yaml'


def config_and_inputs():
    spec=yaml.safe_load(FOLLOWUP.read_text())
    expected={'manifest_sha256':MANIFEST,'quality_config_sha256':QUALITY,
              'frozen_pair_sha256':SOURCE/'matched-pairs.json',
              'prior_gate_a_sha256':SOURCE/'probe-results.json',
              'prior_gate_b_sha256':SOURCE/'overfit-results.json'}
    for name,path in expected.items():
        actual=sha256_file(path)
        if actual!=spec[name]:raise ValueError(f'{name} mismatch: {actual} != {spec[name]}')
    audit=json.loads((SOURCE/'data-audit.json').read_text())
    if audit['split_hash']!=spec['split_hash']:raise ValueError('frozen split hash mismatch')
    config=load_eat_config(QUALITY)
    if config.len_traj_pred!=8:raise ValueError('follow-up pre-registration requires 8-point trajectory')
    return spec,config


def complete_rows(dataset, point_count):
    return [dict(dataset[i],window_index=i) for i in range(len(dataset))
            if is_complete_window(dataset.windows[i],point_count)]


def load_split(split,config):
    dataset=EATDataset(MANIFEST,split,config)
    return dataset,complete_rows(dataset,config.len_traj_pred)


def verify_episode_sources(rows_by_split,config):
    episodes=_records(MANIFEST)
    dt=round(config.trajectory_interval_sec*1e9)
    hist=round(config.dem_history_interval_sec*1e9)
    tol=round(min(config.trajectory_interval_sec,config.dem_history_interval_sec,.05)*.4*1e9)
    by_episode={key:([r['timestamp_ns'] for r in value],value) for key,value in episodes.items()}
    checked=0
    for split,windows in rows_by_split.items():
        for w in windows:
            key=tuple(w['episode_key']);times,source=by_episode[key]
            idx=int(w['local_step']);now=source[idx]
            if int(now['step'])!=int(w['step']) or now['terrain_seed']!=w['terrain_seed']:
                raise ValueError('complete window source identity mismatch')
            for offset in range(config.context_size,-1,-1):
                j=_nearest(times,now['timestamp_ns']-offset*hist,tol)
                if j is None or source[j]['dem'] is None:
                    raise ValueError('context episode/source mismatch')
            xy=[];c,s=math.cos(now['yaw']),math.sin(now['yaw'])
            for point in range(1,config.len_traj_pred+1):
                j=_nearest(times,now['timestamp_ns']+point*dt,tol)
                if j is None or j<=idx or source[j]['timestamp_ns']>times[-1]:
                    raise ValueError('trajectory crosses terminal/reset or lacks real point')
                dx,dy=source[j]['pose']-now['pose'];xy.append((c*dx+s*dy,-s*dx+c*dy))
            actions=np.diff(np.vstack((np.zeros((1,2)),np.asarray(xy))),axis=0)
            if not np.allclose(actions,w['action'],atol=1e-5):
                raise ValueError('target routing mismatch with source episode')
            goal=np.asarray(xy[-1],dtype=np.float32)
            radius=np.linalg.norm(goal)
            if radius>config.goal_max_radius_m:goal*=config.goal_max_radius_m/radius
            if not np.allclose(goal,w['goal'],atol=1e-5):
                raise ValueError('goal source mismatch')
            checked+=1
    return checked


def assert_finite_tree(value,label):
    if isinstance(value,float) and not np.isfinite(value):raise ValueError(f'nonfinite {label} output')
    if isinstance(value,dict):
        for item in value.values():assert_finite_tree(item,label)
    if isinstance(value,list):
        for item in value:assert_finite_tree(item,label)


def audit_stage(spec,config):
    output=STAGES/'01_audit';output.mkdir(parents=True,exist_ok=True)
    datasets={};splits={}
    for split in ('train','val','test'):
        datasets[split]=EATDataset(MANIFEST,split,config)
        splits[split]=[datasets[split][i] for i in range(len(datasets[split]))]
    maps=[datasets[s].split_map for s in ('train','val','test')]
    if maps[0]!=maps[1] or maps[1]!=maps[2]:raise ValueError('dataset split maps disagree')
    split_hash=hashlib.sha256(json.dumps(maps[0],sort_keys=True).encode()).hexdigest()
    if split_hash!=spec['split_hash']:raise ValueError('split leakage or changed split map')
    ecdf=[datasets[s].ecdf.values for s in ('train','val','test')]
    if ecdf[0]!=ecdf[1] or ecdf[1]!=ecdf[2]:raise ValueError('train ECDF differs across splits')
    counts=validate_complete_splits(splits,config.len_traj_pred,spec['expected_complete_windows']['test'])
    for split in counts:
        if counts[split]['complete']!=spec['expected_complete_windows'][split]:
            atomic_json(output/'complete-window-counts.json',{'by_split':counts,'error':'count mismatch'})
            raise ValueError(f'{split} complete window count mismatch')
    rows_by_split={s:[w for w in splits[s] if is_complete_window(w,config.len_traj_pred)] for s in splits}
    checked=verify_episode_sources(rows_by_split,config)
    identity_path=output/'complete-window-identities.jsonl.gz'
    with identity_path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            for split in ('train','val','test'):
                for i,w in enumerate(splits[split]):
                    if not is_complete_window(w,config.len_traj_pred):continue
                    row={'split':split,'window_index':i,'terrain_seed':w['terrain_seed'],
                         'episode_key':w['episode_key'],'step':w['step'],'terrain_type':w['terrain_type']}
                    zipped.write((json.dumps(row,sort_keys=True)+'\n').encode())
    counts.update({'manifest_sha256':sha256_file(MANIFEST),'quality_config_sha256':sha256_file(QUALITY),
                   'split_hash':split_hash,'episode_source_checks':checked,'cross_episode_sources':0,
                   'window_identities_sha256':sha256_file(identity_path)})
    atomic_json(output/'complete-window-counts.json',counts)
    frozen=json.loads((SOURCE/'matched-pairs.json').read_text())
    ladder=select_ladder_pairs(frozen['selected_pairs'],cap=spec['max_episode_uses'])
    report={'source_split':'train','frozen_pair_manifest_sha256':sha256_file(SOURCE/'matched-pairs.json'),
            'selection_rule':'B1 closest to 64-pair median RMS; B8 terrain diversity then RMS proximity; pair-ID tie; B64 original order',
            'subsets':{name:{'count':len(rows),'pair_ids':[pair_id(r) for r in rows],
                             'selected_pairs':rows} for name,rows in ladder.items()}}
    atomic_json(output/'overfit-ladder-pairs.json',report)
    support={'splits':{}}
    for split,windows in rows_by_split.items():
        grouped={}
        for w in windows:
            terrain=w['terrain_type'];qbin=min(int(w['quality']*4),3)
            grouped.setdefault(terrain,Counter())[qbin]+=1
        support['splits'][split]={'q_by_terrain':{terrain:{'bins':{str(i):int(c[i]) for i in range(4)}} for terrain,c in grouped.items()}}
    atomic_json(output/'complete-support.json',support)
    print('FOLLOWUP_AUDIT_DONE',json.dumps({'counts':{s:counts[s]['complete'] for s in splits},
                                            'B1':report['subsets']['B1']['pair_ids'],
                                            'B8':report['subsets']['B8']['pair_ids']}),flush=True)


def probe_schedule(settings,include_shuffles=10):
    rows=[('probe_0',s,None) for s in settings.training_seeds]
    rows += [('probe_q',s,None) for s in settings.training_seeds]
    rows += [('probe_q_shuffled',settings.training_seeds[i%len(settings.training_seeds)],i)
             for i in range(include_shuffles)]
    return rows


def probe_training(spec,config,smoke=False):
    output=ROOT/'smoke'/'probe' if smoke else STAGES/'02_gate_a_training'
    output.mkdir(parents=True,exist_ok=True)
    settings=load_gate_settings(PROBE)
    train,train_rows=load_split('train',config)
    val,val_rows=load_split('val',config)
    if (len(train_rows),len(val_rows))!=(spec['expected_complete_windows']['train'],spec['expected_complete_windows']['val']):
        raise ValueError('complete-only Probe train/validation count mismatch')
    if smoke:train_rows=train_rows[:512];val_rows=val_rows[:128]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_data=tensorize_windows(train_rows,train.stats,device)
    val_data=tensorize_windows(val_rows,train.stats,device)
    if len(train_data['q'])!=len(train_rows) or len(val_data['q'])!=len(val_rows):raise ValueError('partial window entered Probe')
    torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    start=time.monotonic();runs=[]
    schedule=probe_schedule(settings)
    if smoke:schedule=schedule[:1]
    fingerprint=sha256_file(FOLLOWUP)+':'+sha256_file(STAGES/'01_audit'/'complete-window-identities.jsonl.gz')
    for kind,seed,shuffle in schedule:
        name=f'{kind}-seed{seed}' + (f'-shuffle{shuffle}' if shuffle is not None else '')
        result=train_probe_resumable(train_data,val_data,kind=kind,seed=seed,shuffle_seed=shuffle,
                                     settings=settings,output=output/name,fingerprint=fingerprint,
                                     max_epochs=2 if smoke else None)
        runs.append(result)
        atomic_json(output/'summary.json',{'mode':'smoke' if smoke else 'train','device':device,
            'train_windows':len(train_rows),'validation_windows':len(val_rows),
            'wall_seconds':time.monotonic()-start,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'runs':runs,'heldout_used':False,'complete_only':True})
    print('FOLLOWUP_PROBE_TRAIN_DONE',len(runs),flush=True)


def probe_evaluation(spec,config):
    output=STAGES/'03_gate_a_evaluation';output.mkdir(parents=True,exist_ok=True)
    settings=load_gate_settings(PROBE)
    training=STAGES/'02_gate_a_training'/'summary.json'
    expanded=output/'expanded-summary.json'
    def evaluate():
        cmd=[sys.executable,str(BASE/'eat_nav_quality_gate_evaluate.py'),
             '--manifest',str(MANIFEST),'--quality-config',str(QUALITY),'--gate-config',str(PROBE),
             '--runs-summary',str(expanded if expanded.exists() else training),'--pair-controls',str(SOURCE/'heldout-pair-controls.json'),
             '--audit',str(STAGES/'01_audit'/'complete-support.json'),
             '--output',str(output/'gate-a-complete-results.json'),'--complete-only']
        subprocess.run(cmd,check=True)
        report=json.loads((output/'gate-a-complete-results.json').read_text())
        if report['heldout_window_count']!=6208:raise ValueError('held-out complete-only result count mismatch')
        return report
    report=evaluate()
    if report['status']=='needs_100_shuffles':
        train,train_rows=load_split('train',config);val,val_rows=load_split('val',config)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        train_data=tensorize_windows(train_rows,train.stats,device)
        val_data=tensorize_windows(val_rows,train.stats,device)
        summary=json.loads((expanded if expanded.exists() else training).read_text())
        present={r['shuffle_seed'] for r in summary['runs'] if r['kind']=='probe_q_shuffled'}
        fingerprint=sha256_file(FOLLOWUP)+':'+sha256_file(STAGES/'01_audit'/'complete-window-identities.jsonl.gz')
        for i in range(settings.initial_permutations,settings.final_permutations):
            if i in present:continue
            seed=settings.training_seeds[i%len(settings.training_seeds)]
            name=f'probe_q_shuffled-seed{seed}-shuffle{i}'
            run=train_probe_resumable(train_data,val_data,kind='probe_q_shuffled',seed=seed,shuffle_seed=i,
                                      settings=settings,output=output/'extra-shuffles'/name,fingerprint=fingerprint)
            summary['runs'].append(run)
            atomic_json(expanded,summary)
        report=evaluate()
    if report['status']=='needs_100_shuffles':raise ValueError('Gate A permutation escalation incomplete')
    assert_finite_tree(report,'Gate A')
    print('FOLLOWUP_GATE_A_EVALUATED',report['status'],report['decision_metrics']['improvement'],flush=True)


def b_stage(spec,config,label,smoke=False):
    stage={'B1':'04_gate_b1','B8':'05_gate_b8','B64':'06_gate_b64'}[label]
    output=ROOT/'smoke'/label if smoke else STAGES/stage
    output.mkdir(parents=True,exist_ok=True)
    pairs=json.loads((STAGES/'01_audit'/'overfit-ladder-pairs.json').read_text())['subsets'][label]
    train=EATDataset(MANIFEST,'train',config)
    windows=[train[i] for i in range(len(train))]
    ids=[(int(r['low_index']),int(r['high_index'])) for r in pairs['selected_pairs']]
    for row,(lo,hi) in zip(pairs['selected_pairs'],ids):
        if windows[lo]['terrain_seed']!=row['low_terrain_seed'] or windows[hi]['terrain_seed']!=row['high_terrain_seed']:
            raise ValueError('frozen B pair source identity mismatch')
        if tuple(windows[lo]['episode_key'])!=tuple(row['low_episode']) or tuple(windows[hi]['episode_key'])!=tuple(row['high_episode']):
            raise ValueError('frozen B pair episode mismatch')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    batch=prepare_overfit_pairs(windows,ids,device)
    if len(batch['q'])!=2*len(ids) or not torch.equal(batch['dem'][0::2],batch['dem'][1::2]) or not torch.equal(batch['goal'][0::2],batch['goal'][1::2]):
        raise ValueError('B canonical DEM/goal or pair count mismatch')
    if not torch.equal(batch['action'][0::2],torch.from_numpy(np.stack([windows[lo]['action'] for lo,hi in ids])).to(device)) or not torch.equal(batch['action'][1::2],torch.from_numpy(np.stack([windows[hi]['action'] for lo,hi in ids])).to(device)):
        raise ValueError('B target routing mismatch')
    if not torch.all(batch['q'][0::2]<batch['q'][1::2]):raise ValueError('B low/high q routing mismatch')
    settings=load_gate_settings(OVERFIT)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();runs=[]
    fingerprint=sha256_file(FOLLOWUP)+':'+sha256_file(STAGES/'01_audit'/'overfit-ladder-pairs.json')+':'+label
    seeds=spec['training_seeds'][:1] if smoke else spec['training_seeds']
    for seed in seeds:
        run=train_overfit_resumable(config,train.stats,batch,seed=seed,settings=settings,
                                   output=output/f'seed-{seed}',fingerprint=fingerprint,
                                   max_epochs=2 if smoke else spec['max_overfit_epochs'],
                                   checkpoint_interval=1 if smoke else spec['overfit_checkpoint_interval_epochs'])
        runs.append(run)
        atomic_json(output/'summary.json',{'mode':'smoke' if smoke else 'train','train_only':True,
            'pair_count':len(ids),'training_seeds':seeds,'scratch_start':True,
            'wall_seconds':time.monotonic()-start,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'runs':runs})
    if not smoke:
        subset={'source_split':'train','selected_count':len(ids),'selected_pairs':pairs['selected_pairs']}
        atomic_json(output/'subset-pairs.json',subset)
        cmd=[sys.executable,str(BASE/'eat_nav_quality_gate_overfit_evaluate.py'),
             '--manifest',str(MANIFEST),'--quality-config',str(QUALITY),'--overfit-config',str(OVERFIT),
             '--matched-pairs',str(output/'subset-pairs.json'),'--runs-summary',str(output/'summary.json'),
             '--output',str(output/'results.json')]
        subprocess.run(cmd,check=True)
        result=json.loads((output/'results.json').read_text())
        assert_finite_tree(result,'Gate B')
    print('FOLLOWUP_B_DONE',label,len(ids),len(runs),flush=True)


def report_stage(spec,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=STAGES/'07_report';plots=output/'plots';plots.mkdir(parents=True,exist_ok=True)
    counts=json.loads((STAGES/'01_audit'/'complete-window-counts.json').read_text())
    ladder=json.loads((STAGES/'01_audit'/'overfit-ladder-pairs.json').read_text())
    gatea=json.loads((STAGES/'03_gate_a_evaluation'/'gate-a-complete-results.json').read_text())
    previous=json.loads((SOURCE/'probe-results.json').read_text())
    b={label:json.loads((STAGES/f'{number}_gate_{label.lower()}'/'results.json').read_text())
       for label,number in (('B1','04'),('B8','05'),('B64','06'))}
    summaries={label:json.loads((STAGES/f'{number}_gate_{label.lower()}'/'summary.json').read_text())
       for label,number in (('B1','04'),('B8','05'),('B64','06'))}
    probe_summary=json.loads((STAGES/'02_gate_a_training'/'summary.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4))
    for run in probe_summary['runs']:
        if run['kind']=='probe_q_shuffled':continue
        hist=json.loads((Path(run['checkpoint']).parent/'history.json').read_text())
        ax.plot([r['epoch'] for r in hist],[r['validation_loss'] for r in hist],label=f"{run['kind']} {run['training_seed']}")
    ax.axhline(gatea['mean_trajectory_validation_mse'],color='k',linestyle='--',label='complete train mean')
    ax.set(xlabel='Epoch',ylabel='Aggregate validation MSE',title='Complete-only Probe learning curves')
    ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(plots/'complete-probe-learning-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,label in zip(axes,('B1','B8','B64')):
        for run in summaries[label]['runs']:
            hist=json.loads((Path(run['checkpoint']).parent/'history.json').read_text())
            ax.plot([r['epoch'] for r in hist],[r['fixed_bank_loss'] for r in hist],label=f"seed {run['training_seed']}")
        ax.axhline(spec['overfit_loss_ceiling'],color='red',linestyle='--')
        ax.set(title=label,xlabel='Epoch',ylabel='Fixed-bank loss');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(plots/'overfit-ladder-loss.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,label in zip(axes,('B1','B8','B64')):
        for row in b[label]['q_sweep']:
            ax.plot(row['q_values'],row['forward_progress_median_m'],marker='o',label=f"seed {row['training_seed']}")
        ax.set(title=label,xlabel='q',ylabel='Forward progress (m)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(plots/'overfit-ladder-q-sweep.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label in ('B1','B8','B64'):
        metrics=b[label]['median_metrics']
        axes[0].scatter(metrics['target_effect_m'],metrics['generated_effect_m'],label=label)
        axes[1].bar([label],[metrics['effect_sampling_ratio']])
    axes[0].plot([0,1.2],[0,.6],linestyle='--',color='orange')
    axes[0].set(xlabel='Target RMS difference (m)',ylabel='Generated q effect (m)');axes[0].legend()
    axes[1].axhline(1,color='red',linestyle='--');axes[1].set(ylabel='q effect / sampling variation')
    fig.tight_layout();fig.savefig(plots/'overfit-ladder-effects.png',dpi=160);plt.close(fig)
    gatea_summary={'primary_complete_only':{'status':gatea['status'],'heldout_windows':gatea['heldout_window_count'],
          'paired_mse_improvement':gatea['decision_metrics']['improvement'],'ci_95':gatea['bootstrap_ci_95'],
          'relative_rms_improvement':gatea['decision_metrics']['relative_rms_improvement'],
          'delta_r2':gatea['decision_metrics']['delta_r2'],'permutations':len(gatea['decision_metrics']['shuffled'])},
          'secondary_partial_inclusive':{'status':previous['status'],'heldout_windows':previous['heldout_window_count'],
          'paired_mse_improvement':previous['decision_metrics']['improvement'],'ci_95':previous['bootstrap_ci_95'],
          'relative_rms_improvement':previous['decision_metrics']['relative_rms_improvement'],
          'delta_r2':previous['decision_metrics']['delta_r2'],'permutations':len(previous['decision_metrics']['shuffled'])}}
    atomic_json(output/'gate-a-comparison.json',gatea_summary)
    lines=['# Gate A complete-only versus partial-inclusive','','| Metric | Primary: complete-only | Secondary: partial-inclusive |',
           '| --- | ---: | ---: |']
    for key in ('status','heldout_windows','paired_mse_improvement','ci_95','relative_rms_improvement','delta_r2','permutations'):
        lines.append(f"| {key} | {gatea_summary['primary_complete_only'][key]} | {gatea_summary['secondary_partial_inclusive'][key]} |")
    (output/'gate-a-comparison.md').write_text('\n'.join(lines)+'\n')
    report={'overall':'follow-up completed; scientific gates evaluated','full_collection_allowed':False,
            'gate_a':gatea_summary,'gate_b_ladder':{label:{'status':b[label]['status'],
                'decision_metrics':b[label]['decision_metrics'],'median_metrics':b[label]['median_metrics'],
                'per_seed':b[label]['per_seed'],'epochs':{str(r['training_seed']):r['epochs_ran'] for r in summaries[label]['runs']},
                'stop_reasons':{str(r['training_seed']):r['stop_reason'] for r in summaries[label]['runs']}}
                for label in ('B1','B8','B64')},
            'complete_counts':counts,'ladder_pair_ids':{label:ladder['subsets'][label]['pair_ids'] for label in ladder['subsets']},
            'manifest_sha256':sha256_file(MANIFEST),'followup_config_sha256':sha256_file(FOLLOWUP),
            'prior_gate_a_sha256':spec['prior_gate_a_sha256'],'prior_gate_b_sha256':spec['prior_gate_b_sha256'],
            'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,
                       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    atomic_json(output/'quality-identifiability-followup.json',report)
    atomic_json(output/'overfit-ladder-results.json',{label:b[label] for label in ('B1','B8','B64')})
    atomic_json(output/'overfit-ladder-training-summary.json',summaries)
    md=['# EAT-Nav Quality identifiability follow-up','',
        f"Status: complete-only Gate A **{gatea['status']}**; B1 **{b['B1']['status']}**, B8 **{b['B8']['status']}**, B64 **{b['B64']['status']}**. Full collection is not authorized by this follow-up.",'',
        'The previous Gate A/B artifacts are unchanged; this report is a separate follow-up.','',
        '## Data integrity','',
        f"Complete windows: train {counts['train']['complete']:,}, validation {counts['val']['complete']:,}, held-out {counts['test']['complete']:,}. Episode source checks: {counts['episode_source_checks']:,}; cross-episode: {counts['cross_episode_sources']}.",'',
        '## Gate A','', (output/'gate-a-comparison.md').read_text(),'']
    md += ['Model selection used aggregate validation loss. Held-out was evaluated only after selection. Terrain support and seed-level metrics are in `../03_gate_a_evaluation/gate-a-complete-results.json`.','',
           '## Gate B ladder','', '| Stage | Pairs | Gate | Median fixed loss | Generated/target | Correct low/high | Direction | q/sampling |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for label in ('B1','B8','B64'):
        m=b[label]['median_metrics'];loss=np.median([r['final_fixed_bank_loss'] for r in summaries[label]['runs']])
        md.append(f"| {label} | {b[label]['pair_count']} | {b[label]['status']} | {loss:.4f} | {m['effect_fraction']:.3f} | {m['correct_selection_fraction']:.3f} | {m['direction_fraction']:.3f} | {m['effect_sampling_ratio']:.3f} |")
    md += ['', 'Each B stage starts from scratch and uses the frozen nested pair IDs in `../01_audit/overfit-ladder-pairs.json`. Loss plateau before 0.05 yields inconclusive for expressivity. The per-seed training curves, stop reasons, q sweep and target selection are in stage JSON and plots.','',
           '## Figures','',
           '- [Complete Probe curves](plots/complete-probe-learning-curves.png)',
           '- [B1/B8/B64 loss curves](plots/overfit-ladder-loss.png)',
           '- [q sweeps](plots/overfit-ladder-q-sweep.png)',
           '- [Generated/target and sampling effects](plots/overfit-ladder-effects.png)','',
           '## Reconnect and provenance','',
           'Run: `tmux ls`; inspect: `tmux capture-pane -pt eatnav-quality-followup`; attach: `tmux attach -t eatnav-quality-followup`. Log: `/tmp/eatnav-quality-identifiability-gate/followup/runner.log`; status: `/tmp/eatnav-quality-identifiability-gate/followup/status.json`.','',
           f"Input manifest SHA-256 `{report['manifest_sha256']}`; follow-up config SHA-256 `{report['followup_config_sha256']}`; split hash `{spec['split_hash']}`. Exact model checkpoint hashes are saved in the runner stage markers.",'',
           'No full collection, model architecture change, loss change, actor latent, embodiment token, or Critic distillation was performed.']
    (output/'quality-identifiability-followup.md').write_text('\n'.join(md)+'\n')
    print('FOLLOWUP_REPORT_DONE',json.dumps({'gate_a':gatea['status'],'b':{k:v['status'] for k,v in b.items()}}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,
        choices=('audit','gate_a_training','gate_a_evaluation','gate_b1','gate_b8','gate_b64','report','smoke'))
    args=p.parse_args();torch.set_num_threads(4)
    spec,config=config_and_inputs()
    if args.stage=='audit':audit_stage(spec,config)
    elif args.stage=='gate_a_training':probe_training(spec,config)
    elif args.stage=='gate_a_evaluation':probe_evaluation(spec,config)
    elif args.stage=='gate_b1':b_stage(spec,config,'B1')
    elif args.stage=='gate_b8':b_stage(spec,config,'B8')
    elif args.stage=='gate_b64':b_stage(spec,config,'B64')
    elif args.stage=='report':report_stage(spec,config)
    elif args.stage=='smoke':
        probe_training(spec,config,True)
        for label in ('B1','B8','B64'):b_stage(spec,config,label,True)
        print('FOLLOWUP_SMOKE_DONE',flush=True)

if __name__=='__main__':main()
