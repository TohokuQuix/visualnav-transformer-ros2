"""Render descriptive figures from the completed train-only pilot analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(analysis:Path):
    summary=json.loads((analysis/'oracle-summary.json').read_text())
    outcomes=json.loads((analysis/'oracle-outcomes.json').read_text())['rows']
    states=[r for r in summary['states'] if r['status']=='complete']
    if len(states)!=7 or len(outcomes)!=91:
        raise ValueError('all seven states and 91 candidate outcomes required')
    terrain=[r['terrain_type'] for r in states]
    by_state={r['state_id']:[c for c in outcomes if c['state_id']==r['state_id']] for r in states}
    def save(name):
        fig=plt.gcf();fig.tight_layout();target=analysis/name
        if target.exists():raise FileExistsError(f'figure already exists: {target}')
        fig.savefig(target.with_name(target.name+'.tmp'),format='png',dpi=160)
        target.with_name(target.name+'.tmp').replace(target)
        plt.close(fig)
    ks=[1,2,4,8]
    fig,ax=plt.subplots(figsize=(5.7,3.6))
    ax.plot(ks,[summary['nomad_recall_at_k'][str(k)] for k in ks],marker='o',label='Delta V oracle best recall')
    ax.plot(ks,[summary['safe_candidate_recall_at_k'][str(k)] for k in ks],marker='s',label='Safe candidate recall')
    ax.set(xlabel='NoMaD K',ylabel='Fraction of supported states',ylim=(-.05,1.05),xticks=ks,title='Train pilot: Oracle Recall@K (7 states)')
    ax.grid(alpha=.25);ax.legend(fontsize=8);save('recall-at-k.png')
    fig,ax=plt.subplots(figsize=(7,4))
    for state in states:
        curve=state['oracle']['delta_v']['nomad_recall_curve']
        y=[v['best'] if v['best'] is not None else np.nan for v in curve]
        ax.plot(ks,y,marker='.',alpha=.75,label=state['terrain_type'])
    ax.set(xlabel='NoMaD K',ylabel='Best Delta V',xticks=ks,title='Oracle best outcome as K grows')
    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);save('k-vs-oracle-best.png')
    xs=[];ys=[];cs=[];effects=[]
    for state in states:
        cases=by_state[state['state_id']]
        nominal=next(c for c in cases if c['candidate_id']=='bank-nominal')
        base=np.asarray(nominal['path_points'],dtype=float)
        for c in cases:
            path=np.asarray(c['path_points'],dtype=float)
            xs.append(float(np.sqrt(np.mean(np.sum((path-base)**2,axis=1)))))
            ys.append(float(c['command_divergence_vs_duplicate']['rms_l2']))
            cs.append('tab:blue' if c['kind']=='nomad' else 'tab:orange')
            effects.append(None if c['delta_v'] is None or nominal['delta_v'] is None else c['delta_v']-nominal['delta_v'])
    fig,ax=plt.subplots(figsize=(5.7,3.7));ax.scatter(xs,ys,c=cs,alpha=.7,s=24)
    ax.set(xlabel='Path RMS from nominal (m)',ylabel='Follower command RMS from nominal',title='Path diversity vs command diversity')
    ax.grid(alpha=.25);save('path-vs-command.png')
    fig,ax=plt.subplots(figsize=(5.7,3.7))
    for label,color in [('NoMaD','tab:blue'),('Fixed bank','tab:orange')]:
        sel=[i for i,v in enumerate(cs) if v==color and effects[i] is not None]
        ax.scatter([ys[i] for i in sel],[effects[i] for i in sel],label=label,c=color,alpha=.7,s=24)
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Follower command RMS from nominal',ylabel='Delta V improvement over nominal',title='Command diversity vs oracle outcome')
    ax.grid(alpha=.25);ax.legend(fontsize=8);save('command-vs-outcome.png')
    x=np.arange(len(states));width=.23
    fig,ax=plt.subplots(figsize=(9,4))
    for j,(key,label) in enumerate([('nomad_k8_best','NoMaD K8'),('fixed_bank_best','Fixed bank'),('hybrid_best','Hybrid')]):
        vals=[r['oracle']['delta_v'][key] if r['oracle']['delta_v'][key] is not None else np.nan for r in states]
        ax.bar(x+(j-1)*width,vals,width,label=label)
    ax.set(xticks=x,xticklabels=terrain,ylabel='Best Delta V',title='Candidate set comparison (train pilot)')
    ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);save('nomad-vs-bank-vs-hybrid.png')
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].bar(x-.18,[r['nomad_k8_path_geometry']['path_cluster_count'] for r in states],.36,label='Path clusters')
    axes[0].bar(x+.18,[r['nomad_k8_rollout_command_clusters'] for r in states],.36,label='Command clusters')
    axes[0].set(ylabel='K8 cluster count',title='Terrain coverage of proposal modes');axes[0].legend(fontsize=8)
    axes[1].bar(x,[r['nomad_k8_delta_v_spread']['p90_minus_p10'] if r['nomad_k8_delta_v_spread']['p90_minus_p10'] is not None else np.nan for r in states])
    axes[1].set(ylabel='K8 Delta V P90-P10',xticks=x,xticklabels=terrain)
    axes[1].tick_params(axis='x',rotation=35);save('terrain-coverage.png')
    print('COVERAGE_FIGURES_COMPLETE',analysis,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,default=Path('/tmp/eatnav-candidate-coverage/analysis'))
    render(p.parse_args().analysis)

if __name__=='__main__':main()
