"""Review figures for the offline bootstrap and input-contract freeze."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/g40-boundary-freeze'
data=json.loads((OUT/'bootstrap-teacher-comparison.json').read_text())
rows=[json.loads(x) for x in gzip.open(OUT/'terminal-value-per-candidate.jsonl.gz','rt')]
groups=data['group_values']
candidates=('direct_reference','left_1','left_2','left_3','right_1','right_2','right_3')
conds=('zero_recorded','hold_candidate','canonical','reward_only')

def save(name):
    plt.tight_layout();path=OUT/name;plt.savefig(path,format='svg');plt.close()
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

fig,axes=plt.subplots(7,1,figsize=(10,15),sharex=True)
for ax,(sid,g) in zip(axes,groups.items()):
    subset=[r for r in rows if r['state_group_id']==sid and r['candidate_id']!='direct_reference']
    for j,name in enumerate(('zero_recorded','hold_candidate','canonical')):
        ax.scatter([j]*len(subset),[r['terminal_value'][name] for r in subset],s=12)
    ax.set_ylabel(g['terrain'],fontsize=8)
axes[0].set_title('Terminal Critic value by command condition (non-direct candidates)')
axes[-1].set_xticks(range(3),('recorded','hold','canonical'));save('fig-01-terminal-value-conditions.svg')

fig,axes=plt.subplots(7,1,figsize=(10,14),sharex=True)
for ax,(sid,g) in zip(axes,groups.items()):
    rank=[]
    for cond in conds:
        scores=g['scores'][cond]
        order=sorted(candidates,key=lambda c:(-scores[c],c))
        rank.append([order.index(c)+1 for c in candidates])
    im=ax.imshow(rank,vmin=1,vmax=7,aspect='auto',cmap='viridis_r')
    ax.set_yticks(range(4),('recorded','hold','canonical','reward'),fontsize=7)
    ax.set_ylabel(g['terrain'],fontsize=8)
axes[0].set_title('Within-StateGroup candidate ranks (1 = highest teacher)')
axes[-1].set_xticks(range(7),candidates,rotation=30,ha='right');fig.colorbar(im,ax=axes,shrink=.3)
save('fig-02-g40-ranks-by-group.svg')

keys=('zero_recorded_vs_hold_candidate','zero_recorded_vs_canonical','hold_candidate_vs_canonical',
      'canonical_vs_reward_only','hold_candidate_vs_reward_only')
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(keys));ax.bar(x-.18,[data['macro'][k]['kendall'] for k in keys],.36,label='Kendall')
ax.bar(x+.18,[data['macro'][k]['spearman'] for k in keys],.36,label='Spearman')
ax.axhline(.8,color='red',ls='--',lw=.8,label='hold/canonical gate')
ax.set_xticks(x,[k.replace('_recorded','').replace('_candidate','').replace('_only','') for k in keys],rotation=25,ha='right')
ax.set(ylabel='7-group macro',ylim=(0,1.05));ax.legend();save('fig-03-rank-correlations.svg')

fig,ax=plt.subplots(figsize=(9,4));ax.bar(x-.18,[data['macro'][k]['top1_agreement_groups'] for k in keys],.36,label='top-1 groups')
ax.bar(x+.18,[data['macro'][k]['mean_top3_overlap'] for k in keys],.36,label='mean top-3 overlap')
ax.set_xticks(x,[k.replace('_recorded','').replace('_candidate','').replace('_only','') for k in keys],rotation=25,ha='right')
ax.legend();save('fig-04-top1-top3-change.svg')

fig,ax=plt.subplots(figsize=(9,4));sids=list(groups)
reversals=[len(data['group_comparisons'][sid]['hold_candidate_vs_canonical']['direct_sign_reversal_candidates']) for sid in sids]
ax.bar(range(7),reversals);ax.set_xticks(range(7),[groups[s]['terrain'] for s in sids],rotation=25,ha='right')
ax.set(ylabel='direct-vs-candidate sign reversals (of six)',title='Hold versus canonical; StateGroup-level diagnostic')
save('fig-05-direct-sign-reversals.svg')

fig,ax=plt.subplots(figsize=(9,4));ax.scatter([r['reward_discounted_2s'] for r in rows],
             [r['bootstrap_contribution']['canonical'] for r in rows],s=15)
ax.set(xlabel='observed discounted reward Rgamma40',ylabel='gamma^40 * Vcanonical',
       title='Observed 2 s reward and conditional terminal value; runs are grouped')
save('fig-06-reward-terminal-contribution.svg')

def diagram(name,title,labels):
    fig,ax=plt.subplots(figsize=(11,2.5));ax.axis('off');ax.set_title(title)
    xs=np.linspace(.1,.9,len(labels))
    for x,label in zip(xs,labels):ax.text(x,.5,label,ha='center',va='center',fontsize=8,
                                        bbox={'boxstyle':'round','fc':'#e9f1fa'})
    for x,y in zip(xs[:-1],xs[1:]):ax.annotate('',xy=(y-.08,.5),xytext=(x+.08,.5),
                                                arrowprops={'arrowstyle':'->'})
    save(name)

diagram('fig-07-portable-input.svg','Portable pre-execution input contract',
        ['DEM+mask\n4 frames','goal + Path\n8 waypoints','Path/DEM\nfeatures','body state\nIMU+odom+joints','masked\nEvaluator'])
diagram('fig-08-task-vs-hard.svg','Continuous task heads and hard-event proposal',
        ['predict progress\nendpoint/cross-track','continuous\nuncertainty-aware','no radius/ratio\nhard gate now','verified physical\ninvalidity only'])
diagram('fig-09-next-logging-timeline.svg','Next-collection logging timing proposal',
        ['pre-Path\nanchor','command + obs\npre-step','Actor + physics\ntransition','reward terms\npost-dt','terminal pre-reset\nthen command update'])
