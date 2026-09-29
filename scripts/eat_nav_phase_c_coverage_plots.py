"""Fixed eleven-panel figure set for Phase C coverage diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES=('teacher-mode-count','teacher-path-vs-command','ddpm-step-variance','denoising-effective-rank',
         'k8-vs-k32','sampling-coverage-ood','residual-pca-basis','raw-vs-structured',
         'path-vs-command-cluster','evaluator-score-spread','duplicate-vs-candidate-effect')

def read(root,name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else {}

def draw(root):
    data={name:read(root,name+'.json') for name in ('teacher-multimodality','denoising-variance-trace',
        'sampling-grid-results','sampling-coverage-decision','structured-proposal-results','structured-coverage-decision',
        'evaluator-score-spread','serial-smoke-results')}
    out=root/'figures';out.mkdir(exist_ok=True)
    for name in FIGURES:
        fig,ax=plt.subplots(figsize=(7,4));ax.set_title(name.replace('-',' '));done=False
        teacher=data['teacher-multimodality'];trace=data['denoising-variance-trace'];grid=data['sampling-grid-results']
        structured=data['structured-proposal-results'];score=data['evaluator-score-spread'];serial=data['serial-smoke-results']
        if name=='teacher-mode-count' and teacher.get('command_mode_count_histogram'):
            h=teacher['command_mode_count_histogram'];ax.bar(list(map(int,h)),list(h.values()));ax.set_xlabel('follower command modes');done=True
        elif name=='teacher-path-vs-command' and (root/'train-neighborhoods.jsonl.gz').exists():
            import gzip
            with gzip.open(root/'train-neighborhoods.jsonl.gz','rt') as stream:
                rows=[json.loads(x) for x in stream]
            ax.scatter([r['path_cluster_count'] for r in rows],[r['command_cluster_count'] for r in rows],alpha=.2)
            ax.set_xlabel('Path clusters');ax.set_ylabel('Command clusters');done=True
        elif name in ('ddpm-step-variance','denoising-effective-rank') and trace.get('rows'):
            rows=trace['rows'];key='candidate_variance' if name=='ddpm-step-variance' else 'effective_rank'
            y=np.asarray([[s['waypoint'][key] for s in r['trace_stats']] for r in rows])
            ax.plot(np.median(y,axis=0),marker='o');ax.fill_between(np.arange(y.shape[1]),np.quantile(y,.1,axis=0),np.quantile(y,.9,axis=0),alpha=.2)
            ax.set_xlabel('DDPM step');ax.set_ylabel(key);done=True
        elif name=='k8-vs-k32' and grid.get('rows'):
            x=[r['conditions'] for r in grid['rows']];ax.scatter([z['baseline_k8']['command_cluster_count'] for z in x],[z['baseline_k32']['command_cluster_count'] for z in x])
            ax.set_xlabel('K8 command clusters');ax.set_ylabel('K32 command clusters');done=True
        elif name=='sampling-coverage-ood' and data['sampling-coverage-decision'].get('conditions'):
            h=data['sampling-coverage-decision']['conditions'];labels=list(h)
            ax.bar(np.arange(len(h))-.2,[h[k]['command_fraction'] for k in labels],.4,label='command')
            ax.bar(np.arange(len(h))+.2,[h[k]['ood_fraction'] for k in labels],.4,label='OOD')
            ax.set_xticks(np.arange(len(h)),labels,rotation=35,ha='right');ax.legend();done=True
        elif name=='residual-pca-basis' and (root/'residual-basis.npz').exists():
            with np.load(root/'residual-basis.npz') as z:
                if 'basis' in z:
                    for i,p in enumerate(z['basis']):ax.plot(p[:,0],p[:,1],marker='o',label=f'PC{i+1}')
                    ax.legend();done=True
        elif name=='raw-vs-structured' and structured.get('rows') and grid.get('rows'):
            ax.scatter([r['conditions']['baseline_k8']['shape_spread_m'] for r in grid['rows']],
                       [r['shape_spread_m'] for r in structured['rows']])
            ax.set_xlabel('raw shape spread m');ax.set_ylabel('structured shape spread m');done=True
        elif name=='path-vs-command-cluster' and grid.get('rows'):
            rows=[r['conditions']['baseline_k32'] for r in grid['rows']]
            ax.scatter([r['path_cluster_count'] for r in rows],[r['command_cluster_count'] for r in rows])
            ax.set_xlabel('Path clusters');ax.set_ylabel('Command clusters');done=True
        elif name=='evaluator-score-spread' and score.get('rows'):
            ax.hist([r['primary_composite_spread'] for r in score['rows']],bins=15)
            ax.set_xlabel('primary composite spread');done=True
        elif name=='duplicate-vs-candidate-effect' and serial.get('rows'):
            ax.scatter([r['duplicate_xy_rms_m'] for r in serial['rows']],[r['candidate_xy_effect_rms_m'] for r in serial['rows']])
            ax.set_xlabel('duplicate RMS m');ax.set_ylabel('candidate effect RMS m');done=True
        if not done:ax.text(.5,.5,'not run / no support',ha='center',va='center',transform=ax.transAxes)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=130);plt.close(fig)
    return [str(out/(name+'.png')) for name in FIGURES]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print('\n'.join(draw(a.root)))
