"""Offline goal support, response, and diversity figures; no rollout."""
from __future__ import annotations
import pathlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eat_nav_goal_ood_prepare import DOC,RAW,read,classify

def save(path):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.png');plt.tight_layout();plt.savefig(tmp,dpi=125);plt.close();tmp.replace(path)
def main():
    train=np.load(RAW/'train-derived.npz');goals=train['goals'];d=read(DOC/'train-goal-distribution.json');support=read(DOC/'goal-support-thresholds.json')
    states=read(DOC/'state-selection.json')['states'];input_rows=read(DOC/'goal-sweep-inputs.json')['inputs']
    adherence=read(DOC/'goal-adherence-results.json');sens=read(DOC/'goal-sensitivity-results.json');div=read(DOC/'goal-diversity-results.json')
    fig,ax=plt.subplots(figsize=(6,5));h=ax.hexbin(goals[:,0],goals[:,1],gridsize=75,bins='log',cmap='viridis');fig.colorbar(h,ax=ax,label='log count');ax.set(xlabel='robot-local x forward (m)',ylabel='y left (m)',title='Complete-only train goal XY density');ax.set_aspect('equal');save(DOC/'train-goal-xy-density.png')
    radius=np.linalg.norm(goals,axis=1);angle=np.degrees(np.arctan2(goals[:,1],goals[:,0]));fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist(radius,bins=50);ax[0].set(xlabel='saved goal radius (m)',ylabel='windows');ax[1].hist(angle,bins=72);ax[1].set(xlabel='saved goal angle (deg)',ylabel='windows');save(DOC/'train-goal-radius-angle.png')
    fig,ax=plt.subplots(figsize=(9,4));terrains=list(d['terrain']);bottom=np.zeros(len(terrains))
    for label in ('ID','near_ood','far_ood','unsupported'):
        values=np.array([d['terrain'][t]['support_classes'].get(label,0) for t in terrains]);ax.bar(range(len(terrains)),values,bottom=bottom,label=label);bottom+=values
    ax.set_xticks(range(len(terrains)),terrains,rotation=40,ha='right');ax.set_ylabel('complete train windows');ax.legend();save(DOC/'terrain-goal-support.png')
    xy=np.stack(np.meshgrid(np.linspace(-1,1,110),np.linspace(-1,1,110)),axis=-1).reshape(-1,2)
    labels,_=classify(xy,goals,support);palette={'ID':0,'near_ood':1,'far_ood':2,'unsupported':3}
    z=np.asarray([palette[x] for x in labels]).reshape(110,110);fig,ax=plt.subplots(figsize=(6,5));ax.imshow(z,extent=(-1,1,-1,1),origin='lower',cmap=matplotlib.colors.ListedColormap(['#3a9d75','#e8c547','#d66a53','#aaa']),alpha=.7);ax.set(xlabel='x forward (m)',ylabel='y left (m)',title='Train-only goal support classes');ax.set_aspect('equal');save(DOC/'goal-support-regions.png')
    fig,ax=plt.subplots(figsize=(7,4));data=adherence['states'];labels=['ID','near_ood','far_ood']
    for j,split in enumerate(('train','val')):
        for k,label in enumerate(labels):
            vals=[s['classes'][label]['error']['median'] for s in data if s['split']==split];ax.boxplot(vals,positions=[j*4+k],widths=.7)
    ax.set_xticks([0,1,2,4,5,6],[f'{s}\n{l}' for s in ('train','val') for l in labels]);ax.set_ylabel('state median endpoint-goal error (m)');ax.axhline(.2,color='red',lw=.8);save(DOC/'goal-endpoint-error.png')
    fig,ax=plt.subplots(figsize=(6,5));colors={'ID':'#3a9d75','near_ood':'#e8a547','far_ood':'#d66a53'}
    for label in labels:
        x=[s['classes'][label]['sampling_effect_rms_m']['median'] for s in sens['states']]
        y=[s['classes'][label]['goal_effect_rms_m']['median'] for s in sens['states']]
        ax.scatter(x,y,s=17,alpha=.6,label=label,color=colors[label])
    ax.plot([0,.25],[0,.25],ls='--',color='black',lw=.7);ax.set(xlabel='same goal, different seed Path RMS (m)',ylabel='different goal, same seed Path RMS (m)');ax.legend();save(DOC/'goal-vs-sampling-effect.png')
    fig,ax=plt.subplots(figsize=(8,4));means=[]
    for label in labels:
        values=[np.mean(s['classes'][label]['different_goal_same_seed']) for s in div['states']];means.append(values)
    ax.boxplot(means);ax.set_xticks(range(1,4),labels);ax.set_ylabel('different goal, same seed Path modes');save(DOC/'goal-class-path-modes.png')
    for index,state in enumerate(states):
        p=np.load(RAW/'inference'/f"{state['state_id']}.npz")['paths'];rows=input_rows[index*13:(index+1)*13]
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(goals[::30,0],goals[::30,1],s=2,color='#bbb',alpha=.16)
        for j,row in enumerate(rows):
            g=np.asarray(row['goal_xy']);ends=p[j,:,-1,:]
            ax.scatter([g[0]],[g[1]],marker='x',s=15,color=colors.get(row['declared_class'],'black'))
            ax.scatter(ends[:,0],ends[:,1],s=4,color=colors.get(row['declared_class'],'black'),alpha=.6)
            m=ends.mean(axis=0);ax.plot([g[0],m[0]],[g[1],m[1]],color=colors.get(row['declared_class'],'black'),lw=.7,alpha=.8)
            if not row['geometry_screen_valid']:ax.scatter([g[0]],[g[1]],facecolors='none',edgecolors='red',s=60)
        ax.set(xlabel='goal / endpoint x (m)',ylabel='y (m)',title=f"{state['state_id']} {state['terrain']} response map");ax.set_aspect('equal');ax.grid(alpha=.2)
        save(RAW/'response-maps'/f"{state['state_id']}.png")
        if index in (0,32):
            fig,ax=plt.subplots(3,4,figsize=(12,8))
            for j in range(1,13):
                a=ax.flat[j-1];g=np.asarray(rows[j]['goal_xy'])
                for path in p[j]:a.plot(np.r_[0,path[:,0]],np.r_[0,path[:,1]],alpha=.5,lw=.7)
                a.scatter(g[0],g[1],c='red',marker='x',s=18);a.set_title(rows[j]['goal_id'],fontsize=8);a.grid(alpha=.2);a.set_aspect('equal')
            fig.suptitle(f"{state['state_id']} eight paired-seed Paths per goal")
            save(DOC/f"representative-paths-{state['split']}.png")
            # The same response map is also a compact Git figure.
            p2=np.load(RAW/'inference'/f"{state['state_id']}.npz")['paths'];fig,ax=plt.subplots(figsize=(6,5))
            for j,row in enumerate(rows):
                g=np.asarray(row['goal_xy']);ends=p2[j,:,-1,:];m=ends.mean(axis=0);col=colors.get(row['declared_class'],'black')
                ax.scatter(g[0],g[1],marker='x',s=20,color=col);ax.scatter(ends[:,0],ends[:,1],s=5,color=col,alpha=.5);ax.plot([g[0],m[0]],[g[1],m[1]],color=col,lw=.7)
            ax.set(xlabel='goal/endpoint x (m)',ylabel='y (m)',title=f"{state['state_id']} goal-to-endpoint map");ax.set_aspect('equal');ax.grid(alpha=.2)
            save(DOC/f"goal-response-map-{state['split']}.png")
    print('FIGURES_COMPLETE',len(states),flush=True)
if __name__=='__main__':main()
