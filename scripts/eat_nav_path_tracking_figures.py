"""Publication-style static figures from frozen CPU audit tables."""
from __future__ import annotations
import json,pathlib,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eat_nav_path_tracking_stages import DOC,ROOT,UNIFIED,read,gzlines

def save(name):
    path=DOC/name
    plt.tight_layout();tmp=path.with_name(path.stem+'.tmp.png');plt.savefig(tmp,dpi=130);plt.close();tmp.replace(path)

def main():
    series=list(gzlines(ROOT/'tracking-series.jsonl.gz'));metrics=read(DOC/'path-tracking-metrics.json')['runs'];byid={r['run_id']:r for r in metrics}
    states=sorted({r['state_id'] for r in series});colors={'nomad':'#2767a5','bank':'#d17b1e','nominal':'#444444'}
    for kind in ('nomad','bank','nominal'):
        fig,ax=plt.subplots(2,4,figsize=(14,7));
        for k,sid in enumerate(states):
            a=ax.flat[k];rows=[r for r in series if r['state_id']==sid and r['kind']==kind]
            for r in rows:
                path=np.asarray(r['world_path']);real=np.asarray(r['world_positions']);anchor=np.asarray(r['anchor_world_xy'])
                a.plot(path[:,0]-anchor[0],path[:,1]-anchor[1],color=colors[kind],alpha=.45,lw=1)
                a.plot(real[:,0]-anchor[0],real[:,1]-anchor[1],color=colors[kind],alpha=.35,lw=.7,ls='--')
            a.scatter([0],[0],c='black',s=8);a.set_title(sid+' '+byid[rows[0]['run_id']]['terrain'],fontsize=8);a.set_aspect('equal');a.grid(alpha=.2)
        ax.flat[-1].axis('off');fig.suptitle(kind+' Path (solid) and realized 6 s (dashed), anchor-relative XY')
        save(f'planned-realized-{kind}.png')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for i,field in enumerate(('min_distance_m','final_distance_m')):
        for j,kind in enumerate(('nomad','bank')):
            vals=[r['endpoint'][field] for r in metrics if r['generator']==kind]
            ax[i].scatter(np.full(len(vals),j)+np.random.default_rng(123+j).uniform(-.13,.13,len(vals)),vals,s=9,alpha=.5)
        ax[i].set_xticks([0,1],['NoMaD','bank']);ax[i].set_title(field);ax[i].axhline(.2,color='red',lw=.8)
    save('endpoint-distance.png')
    fig,ax=plt.subplots(figsize=(7,4))
    for j,kind in enumerate(('nomad','bank','nominal')):
        vals=[r['two_second']['progress_ratio'] for r in metrics if r['generator']==kind]
        ax.boxplot(vals,positions=[j]);ax.scatter(np.full(len(vals),j)+np.random.default_rng(200+j).uniform(-.1,.1,len(vals)),vals,s=6,alpha=.3)
    ax.set_xticks(range(3),['NoMaD','bank','nominal']);ax.set_ylabel('2 s projected progress / Path length');save('progress-ratio.png')
    fig,ax=plt.subplots(figsize=(7,4))
    for j,kind in enumerate(('nomad','bank','nominal')):
        vals=[r['two_second']['cross_track_p95_m'] for r in metrics if r['generator']==kind]
        ax.boxplot(vals,positions=[j]);ax.scatter(np.full(len(vals),j)+np.random.default_rng(300+j).uniform(-.1,.1,len(vals)),vals,s=6,alpha=.3)
    ax.set_xticks(range(3),['NoMaD','bank','nominal']);ax.set_ylabel('0–2 s Path cross-track P95 (m)');save('cross-track-p95.png')
    fig,ax=plt.subplots(figsize=(8,4));groups=['nomad','bank','nominal']
    timing=read(DOC/'endpoint-reach-timing.json')['groups']
    for j,g in enumerate(groups):
        ax.bar(j-.17,timing[g]['reached_by_2s'],width=.32,label='by 2 s' if j==0 else None,color='#2a9d8f')
        ax.bar(j+.17,timing[g]['reached_by_6s'],width=.32,label='by 6 s' if j==0 else None,color='#e9c46a')
    ax.set_xticks(range(3),groups);ax.set_ylabel('entered 0.2 m endpoint region');ax.legend();save('reach-two-vs-six.png')
    dist=read(DOC/'path-command-realized-distances.json')['states_and_sets'];fig,ax=plt.subplots(figsize=(10,4))
    for j,name in enumerate(('P','C','R')):
        vals=[next(x for x in dist if x['state_id']==sid and x['set']=='nomad')['cluster_count'][name] for sid in states]
        ax.bar(np.arange(7)+(.25*j-.25),vals,width=.25,label=name)
    ax.set_xticks(range(7),states,rotation=45,ha='right');ax.set_ylabel('NoMaD K8 cluster count');ax.legend();save('pcr-clusters.png')
    trans=read(DOC/'mode-transition-table.json')['states'];fig,ax=plt.subplots(figsize=(10,4))
    keys=['P_different_C_same','C_different_R_same','P_C_R_all_different']
    for j,key in enumerate(keys):ax.bar(np.arange(7)+(.25*j-.25),[s['nomad_pair_counts'][key] for s in trans],width=.25,label=key)
    ax.set_xticks(range(7),states,rotation=45,ha='right');ax.set_ylabel('within-state NoMaD pairs (descriptive)');ax.legend(fontsize=7);save('pcr-mode-transitions.png')
    outcomes={r['run_id']:r for r in read(UNIFIED/'analysis/per-candidate-outcomes.json')};fig,ax=plt.subplots(figsize=(7,5))
    for sid in states:
        x=[r['two_second']['cross_track_p95_m'] for r in metrics if r['state_id']==sid and r['run_id'] in outcomes]
        y=[outcomes[r['run_id']]['six_second']['delta_v'] for r in metrics if r['state_id']==sid and r['run_id'] in outcomes]
        ax.scatter(x,y,s=14,alpha=.6,label=sid)
    ax.set_xlabel('0–2 s cross-track P95 (m)');ax.set_ylabel('6 s ΔV');ax.legend(fontsize=6);save('delta-v-vs-tracking.png')
    evaluator=read(DOC/'evaluator-tracking-strata.json')['states'];classes={s['state_id']:s['classification'] for s in read(DOC/'collapse-classification.json')['states']}
    fig,ax=plt.subplots(figsize=(9,4))
    for j,state in enumerate(evaluator):
        rows=state['all_candidates'];oracle=max(rows,key=lambda r:r['delta_v']);pred=max(rows,key=lambda r:r['score'])
        ax.scatter(j,oracle['delta_v'],marker='o',color='#2a9d8f');ax.scatter(j,pred['delta_v'],marker='x',color='#e76f51')
        ax.plot([j,j],[oracle['delta_v'],pred['delta_v']],color='gray',lw=.7)
    ax.set_xticks(range(7),[s['state_id']+'\n'+classes[s['state_id']].replace('_',' ') for s in evaluator],rotation=30,ha='right',fontsize=7)
    ax.set_ylabel('6 s ΔV');ax.set_title('Oracle best (circle) vs frozen GT(P) top-1 (cross)');save('evaluator-misrank-by-collapse.png')
if __name__=='__main__':main()
