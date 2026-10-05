"""Small offline figures from the sealed 40-step teacher audit."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'docs/quality-identifiability-gate/g40-teacher-review'
rows = [json.loads(x) for x in gzip.open(OUT/'g40-per-candidate.jsonl.gz','rt')]
groups = json.loads((OUT/'g40-stategroup-summary.json').read_text())['groups']
ranking = json.loads((OUT/'teacher-rank-concordance-2s.json').read_text())['macro']
bellman = json.loads((OUT/'bellman-consistency.json').read_text())
reward = json.loads((OUT/'reward-component-audit.json').read_text())
candidates=('direct_reference','left_1','left_2','left_3','right_1','right_2','right_3')

def save(name):
    plt.tight_layout();plt.savefig(OUT/name,format='svg');plt.close()
    path=OUT/name
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

fig,axs=plt.subplots(7,1,figsize=(9,15),sharex=True)
for ax,(sid,g) in zip(axs,groups.items()):
    ax.bar(range(7),[g['candidate_values']['G40'][c] for c in candidates]);ax.set_ylabel(g['terrain'],fontsize=8)
axs[0].set_title('G40 by StateGroup; direct is index 0')
axs[-1].set_xticks(range(7),candidates,rotation=40,ha='right');save('fig-01-g40-by-stategroup.svg')

fig,ax=plt.subplots(figsize=(9,4));ax.scatter([r['G40'] for r in rows],[r['six_sec_deltaV'] for r in rows],s=15)
ax.set(xlabel='G40',ylabel='6 s delta V',title='Run-level values; StateGroup is the statistical unit')
save('fig-02-g40-vs-six-second-deltav.svg')

names=['A40','progress','R_gamma_40','six_sec_deltaV','endpoint_error','tilt']
fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(len(names)),[ranking[n]['spearman_mean'] for n in names])
ax.axhline(0,color='black',lw=.5);ax.set(ylim=(-1,1),ylabel='7-group mean Spearman vs G40',title='StateGroup macro rank comparison')
ax.set_xticks(range(len(names)),names,rotation=25,ha='right');save('fig-03-teacher-rank-comparison.svg')

fig,ax=plt.subplots(figsize=(9,4));active=[x for x in reward['terms'] if x['weight']!=0]
ax.barh(range(len(active)),[x['weight'] for x in active]);ax.set_yticks(range(len(active)),[x['name'] for x in active],fontsize=7)
ax.set(xlabel='configured weight (not observed contribution)',title='Reward contributions: NA; only configured weights are available')
save('fig-04-reward-contribution-na.svg')

fig,ax=plt.subplots(figsize=(9,4))
for sid,g in groups.items():
    direct=next(r for r in rows if r['state_group_id']==sid and r['candidate_id']=='direct_reference')
    ax.plot(np.arange(41)*.05,direct['critic_0_40'],label=g['terrain'])
ax.set(xlabel='time (s)',ylabel='Critic V',title='Direct run Critic trajectories and drawdown')
ax.legend(fontsize=6,ncol=2);save('fig-05-critic-drawdown.svg')

fig,ax=plt.subplots(figsize=(9,4));ax.plot(np.arange(40)*.05,bellman['all']['per_step_mean'])
ax.axhline(0,color='black',lw=.5);ax.set(xlabel='time (s)',ylabel='mean TD residual',title='0-2 s per-step TD residual; 63 runs')
save('fig-06-td-residual-time.svg')

fig,ax=plt.subplots(figsize=(9,4));data=[[r['horizon_residual'][str(h)] for r in rows] for h in (10,20,40)]
ax.boxplot(data,labels=['0.5 s','1 s','2 s']);ax.set(ylabel='H-step return minus V0',title='Bellman diagnostic by horizon')
save('fig-07-horizon-bellman.svg')

fig,ax=plt.subplots(figsize=(9,4))
for sid,g in groups.items():
    group=[r for r in rows if r['state_group_id']==sid and r['candidate_id']!='direct_reference']
    ax.scatter([r['map_metrics']['endpoint_error_m'] for r in group],[r['G40'] for r in group],s=20,label=g['terrain'])
ax.set(xlabel='endpoint error at 2 s (m)',ylabel='G40',title='Task feasibility and embodiment return')
ax.legend(fontsize=6,ncol=2);save('fig-08-task-vs-embodiment.svg')

fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
for x,label in zip([.12,.37,.62,.87],['DEM + goal + Path','multi-head estimates','safety -> task -> G40','tie: direct / replan']):
    ax.text(x,.5,label,ha='center',va='center',bbox={'boxstyle':'round','fc':'#e8f0fa'},fontsize=10)
for a,b in [( .22,.27),(.47,.52),(.72,.77)]:ax.annotate('',xy=(b,.5),xytext=(a,.5),arrowprops={'arrowstyle':'->'})
ax.set_title('Proposed multi-head evaluator and lexicographic selection (prospective)')
save('fig-09-multihead-selection.svg')
