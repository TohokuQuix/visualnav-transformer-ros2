"""Render standalone figures from frozen Gate A/B/layer JSON results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    for name in ('probe-summary','overfit-summary','gate-a','gate-b','layers','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    probe=json.loads(a.probe_summary.read_text())
    overfit=json.loads(a.overfit_summary.read_text())
    gatea=json.loads(a.gate_a.read_text())
    gateb=json.loads(a.gate_b.read_text())
    layers=json.loads(a.layers.read_text())
    fig,ax=plt.subplots(figsize=(8,4))
    for run in probe['runs']:
        if run['kind']=='probe_q_shuffled':continue
        history=json.loads((Path(run['checkpoint']).parent/'history.json').read_text())
        ax.plot([x['epoch'] for x in history],[x['validation_loss'] for x in history],
                alpha=.75,label=f"{run['kind']} seed {run['training_seed']}")
    ax.axhline(gatea['mean_trajectory_validation_mse'],color='k',linestyle='--',label='mean trajectory')
    ax.set(xlabel='Epoch',ylabel='Aggregate validation normalized MSE',title='Deterministic probe learning curves')
    ax.legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(a.output/'probe-learning-curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for run in overfit['runs']:
        history=json.loads((Path(run['checkpoint']).parent/'history.json').read_text())
        ax.plot([x['epoch'] for x in history],[x['fixed_bank_loss'] for x in history],label=f"seed {run['training_seed']}")
    ax.set(xlabel='Epoch',ylabel='Fixed-bank diffusion loss',title='Scratch same-condition overfit')
    ax.legend();fig.tight_layout();fig.savefig(a.output/'overfit-learning-curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for row in gateb['q_sweep']:
        ax.plot(row['q_values'],row['forward_progress_median_m'],marker='o',label=f"seed {row['training_seed']}")
    ax.set(xlabel='q target',ylabel='Median generated forward progress (m)',title='Same DEM and goal, paired diffusion noise')
    ax.legend();fig.tight_layout();fig.savefig(a.output/'q-sweep.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5))
    for row in gateb['per_seed']:
        ax.scatter(row['target_effect_m'],row['generated_effect_m'],label=f"seed {row['training_seed']}")
    extent=max([row['target_effect_m'] for row in gateb['per_seed']]+[row['generated_effect_m'] for row in gateb['per_seed']]+[.01])
    ax.plot([0,extent],[0,extent],color='gray',linestyle='--')
    ax.plot([0,extent],[0,.5*extent],color='orange',linestyle=':')
    ax.set(xlabel='Median target pair RMS difference (m)',ylabel='Median generated q RMS difference (m)',title='Gate B target versus generated effect')
    ax.legend();fig.tight_layout();fig.savefig(a.output/'target-vs-generated-effect.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    rows=gateb['per_seed']
    ax.bar([str(x['training_seed']) for x in rows],[x['effect_sampling_ratio'] for x in rows])
    ax.axhline(1,color='red',linestyle='--')
    ax.set(xlabel='Scratch training seed',ylabel='q effect / same-q sampling variation',title='Gate B effect-to-sampling ratio')
    fig.tight_layout();fig.savefig(a.output/'effect-sampling-ratio.png',dpi=170);plt.close(fig)
    values=layers['relative_sensitivity']
    order=sorted(values,key=lambda name:(0 if name=='quality_mlp' else 1 if 'token_input' in name else
        2 if 'transformer_layer' in name else 3 if name=='final_context' else
        4 if name.startswith('film') else 5 if name.startswith('down') else
        6 if name.startswith('mid') else 7 if name.startswith('up') else 8,name))
    fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[3,1]})
    axes[0].bar(range(len(order)),[max(values[name],1e-12) for name in order])
    axes[0].set_yscale('log');axes[0].set_xticks(range(len(order)),order,rotation=75,ha='right',fontsize=7)
    axes[0].set_ylabel('Relative q activation change S_l')
    axes[1].plot(range(len(layers['epsilon_delta_by_timestep'])),layers['epsilon_delta_by_timestep'],marker='o')
    axes[1].set(xlabel='Diffusion timestep k',ylabel='Paired predicted-noise Δε')
    fig.suptitle('Quality signal through the existing EAT network')
    fig.tight_layout();fig.savefig(a.output/'layer-q-sensitivity.png',dpi=170);plt.close(fig)
    print('PLOTS_DONE',len(list(a.output.glob('*.png'))),flush=True)

if __name__=='__main__':main()
