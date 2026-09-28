"""Post-inference interpretation of frozen casebook outputs; no model calls."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/tmp/eatnav-train/casebook')
OUT=ROOT/'output'

def read(name):return json.loads((OUT/name).read_text())
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def rms(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
def ci(values,episodes,seed=613,repeats=1000):
    values=np.asarray(values,dtype=float);unique=list(dict.fromkeys(episodes));rng=np.random.default_rng(seed)
    if len(unique)<2:return None
    groups={e:np.flatnonzero(np.asarray(episodes)==e) for e in unique}
    res=[]
    for _ in range(repeats):
        draw=rng.choice(len(unique),len(unique),replace=True)
        ix=np.concatenate([groups[unique[int(j)]] for j in draw]);res.append(float(values[ix].mean()))
    return np.quantile(res,[.025,.975]).tolist()

def main():
    pred=np.load(ROOT/'predictions.npz');source=np.load(ROOT/'selected.npz');co=read('casebook-cohorts.json');summary=read('casebook-summary.json');goal=read('goal-provenance.json')
    csv_path=OUT/'casebook-metrics.csv'
    csv_path.write_bytes(csv_path.read_bytes().replace(b'\r\n',b'\n'))
    with csv_path.open() as f:rows={int(r['index']):r for r in csv.DictReader(f)}
    ix={int(i):j for j,i in enumerate(pred['indices'])};sx={int(i):j for j,i in enumerate(source['indices'])};qvals=summary['q_values']
    cohort={name:[int(i) for i in co[name]] for name in ('A','B','D')}
    cohort['B_control']=[int(x['high_index']) for x in co['B_controls']]
    cohort['C']=list(dict.fromkeys(int(x[k]) for x in co['C'] for k in ('low_index','high_index')))
    cohort['C_control']=list(dict.fromkeys(int(x[k]) for x in co['C_same_q_controls'] for k in ('low_index','high_index')))
    analyses={};boot={};episodes={i:rows[i]['episode_key'] for i in rows}
    for name,ids in cohort.items():
        if not ids:continue
        low=[];high=[];sampling=[];effect=[];enable=[];disable=[];progress=[];range_=[];mixed=[]
        for i in ids:
            j=ix[i];teacher=np.cumsum(source['action'][sx[i]],axis=0);x=pred['pred'][j];b=pred['baseline'][j]
            low.append(np.mean([rms(y,teacher) for y in x[:,0]]));high.append(np.mean([rms(y,teacher) for y in x[:,-1]]))
            sampling.append(float(rows[i]['sampling_rms_m']));effect.append(float(rows[i]['q_effect_rms_m']))
            enable.append(float(rows[i]['enabled_teacher_rms_m']));disable.append(float(rows[i]['disabled_teacher_rms_m']))
            ps=x[:,2,-1,0];progress.append(float(ps.mean()));range_.append(float(np.ptp(ps)));mixed.append(bool(np.any(ps<0) and np.any(ps>0)))
        analyses[name]={'n':len(ids),'q_low_teacher_rms_mean_m':float(np.mean(low)),'q_high_teacher_rms_mean_m':float(np.mean(high)),
            'q_high_minus_low_teacher_rms_m':float(np.mean(np.asarray(high)-low)),
            'q_effect_median_m':float(np.median(effect)),'sampling_median_m':float(np.median(sampling)),
            'enabled_teacher_rms_mean_m':float(np.mean(enable)),'disabled_teacher_rms_mean_m':float(np.mean(disable)),
            'sampling_q05_progress_range_median_m':float(np.median(range_)),
            'sampling_q05_mixed_forward_sign_fraction':float(np.mean(mixed)),
            'generated_progress_mean_m':float(np.mean(progress)),
            'teacher_progress_mean_m':float(np.mean([float(rows[i]['teacher_progress_m']) for i in ids]))}
        for label,values in [('q_high_minus_low_teacher_rms_m',np.asarray(high)-low),('enabled_minus_disabled_teacher_rms_m',np.asarray(enable)-disable)]:
            analyses[name][label+'_ci95_episode']=ci(values,[episodes[i] for i in ids])
        boot[name]={k:{'count':sum(int(rows[i]['category_'+k]) for i in ids),
                       'fraction':float(np.mean([int(rows[i]['category_'+k]) for i in ids])),
                       'ci95_episode':ci([int(rows[i]['category_'+k]) for i in ids],[episodes[i] for i in ids])} for k in 'ABCDE'}
    # Paired high/low teacher preference for the same anchor, same seed bank.
    pair=read('casebook-metrics.json')['pair_teacher_distances'];pref=defaultdict(list)
    for r in pair:pref[(r['low_index'],r['high_index'],r['anchor'])].append(r)
    switches=[];correct=[]
    for group in pref.values():
        group=sorted(group,key=lambda r:r['q'])
        switches.append(group[0]['nearest']!=group[-1]['nearest'])
        correct.append(group[0]['nearest']=='low' and group[-1]['nearest']=='high')
    pair_analysis={'anchor_count':len(pref),'nearest_teacher_switch_fraction':float(np.mean(switches)) if switches else None,
                   'low_at_0p1_high_at_0p95_fraction':float(np.mean(correct)) if correct else None}
    no=cohort['B'];matched=cohort['B_control']
    goal_analysis={'no_progress_goal_mean_m':float(np.mean([float(rows[i]['goal_distance_m']) for i in no])),
                   'matched_high_progress_goal_mean_m':float(np.mean([float(rows[i]['goal_distance_m']) for i in matched])),
                   'matched_goal_distance_gap_mean_m':float(np.mean([r['goal_distance_gap_m'] for r in co['B_controls']])),
                   'goal_progress_pearson_all_heldout':goal['progress_goal_pearson'],
                   'q_goal_pearson_all_heldout':goal['q_goal_pearson'],
                   'raw_command_goal_direction_median_deg':goal['command_goal_direction_abs_deg_median'],
                   'raw_command_goal_distance_difference':goal['command_goal_distance_difference']}
    # Select replay candidates after diagnostics; never use these IDs in model selection or gate outcomes.
    latest={}
    for i in cohort['A']:
        episode=episodes[i];seconds=float(rows[i]['terminal_seconds'])
        if episode not in latest or seconds<float(rows[latest[episode]]['terminal_seconds']):latest[episode]=i
    replay=list(latest.values())
    for i in sorted(cohort['B'],key=lambda i:(float(rows[i]['enabled_teacher_rms_m']),-float(rows[i]['q_sampling_ratio']))):
        if i not in replay:replay.append(i)
        if len(replay)>=50:break
    result={'cohorts':analyses,'category_bootstrap':boot,'pair_teacher_preference':pair_analysis,
            'goal_provenance':goal_analysis,'replay_candidate_indices':replay,
            'replay_rule':'latest complete anchor from each held-out tilt episode, then B windows with closest enabled teacher RMS; max 50',
            'limitations':['static open-loop XY trajectories cannot establish contact, tipping, or closed-loop safety',
                           'dataset goal is realized endpoint and strongly coupled to teacher progress',
                           'terrain x q strata with fewer than configured support remain NA']}
    write('casebook-interpretation.json',result)
    # Supplemental plots show the required stratification without choosing examples from model output.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    names=list(analyses);x=np.arange(len(names));axs[0].bar(x-.2,[analyses[n]['q_effect_median_m'] for n in names],.4,label='q effect');axs[0].bar(x+.2,[analyses[n]['sampling_median_m'] for n in names],.4,label='sampling');axs[0].set_xticks(x,names,rotation=45);axs[0].set_ylabel('trajectory RMS [m]');axs[0].legend()
    axs[1].bar(x-.2,[analyses[n]['q_low_teacher_rms_mean_m'] for n in names],.4,label='q 0.1');axs[1].bar(x+.2,[analyses[n]['q_high_teacher_rms_mean_m'] for n in names],.4,label='q 0.95');axs[1].set_xticks(x,names,rotation=45);axs[1].set_ylabel('teacher RMS [m]');axs[1].legend();fig.tight_layout();fig.savefig(OUT/'plots'/'cohort-q-sampling-and-imitation.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));vals=[boot[n]['B']['fraction'] for n in names];ax.bar(names,vals);ax.set_ylabel('q-ignore category fraction');ax.tick_params(axis='x',rotation=45);fig.tight_layout();fig.savefig(OUT/'plots'/'cohort-categories.png',dpi=140);plt.close(fig)
    # The only held-out tilt episode has raw quaternion, allowing a measured roll/pitch timeline.
    if cohort['A']:
        raw=json.loads((ROOT/'selected-raw.json').read_text());ref=raw[str(cohort['A'][0])]['series']
        end=ref[-1]['timestamp_ns'];tt=np.asarray([(r['timestamp_ns']-end)/1e9 for r in ref])
        def angles(pose):
            w,x,y,z=pose[3:7]
            roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
            pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
            return np.degrees(roll),np.degrees(pitch)
        rp=np.asarray([angles(r['pose']) for r in ref]);fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
        axs[0].plot(tt,[r['critic_value'] for r in ref],label='Critic value');axs[0].legend()
        tq=[(raw[str(i)]['timestamp_ns']-end)/1e9 for i in cohort['A']]
        axs[1].scatter(tq,[float(rows[i]['q']) for i in cohort['A']],s=12,label='dataset q at complete anchor')
        axs[1].plot(tt,[np.linalg.norm(r['twist'][:2]) for r in ref],alpha=.6,label='speed [m/s]');axs[1].legend()
        axs[2].plot(tt,rp[:,0],label='roll [deg]');axs[2].plot(tt,rp[:,1],label='pitch [deg]');axs[2].legend()
        for ax in axs:ax.axvline(0,color='red',ls='--',label='tilt terminal');ax.grid(alpha=.25)
        axs[-1].set_xlabel('time before tilt terminal [s]');fig.tight_layout()
        fig.savefig(OUT/'plots'/'tilt-episode-timeseries.png',dpi=140);plt.close(fig)
    lines=['# Held-out trajectory casebook: findings','',f"The model and split were frozen. {summary['window_count']} unique complete held-out windows were inferred with eight paired diffusion seeds and six q values.",'',
           '## 1. Low-quality teacher imitation','',
           f"No-progress (B): enabled teacher RMS {analyses['B']['enabled_teacher_rms_mean_m']:.4f} m; tilt-before (A): {analyses['A']['enabled_teacher_rms_mean_m']:.4f} m. These are static XY distances to teachers, not failure rates.",'',
           '## 2. Effect of raising q','',
           f"In B, teacher RMS at q=0.1 is {analyses['B']['q_low_teacher_rms_mean_m']:.4f} m and at q=0.95 is {analyses['B']['q_high_teacher_rms_mean_m']:.4f} m (high minus low {analyses['B']['q_high_minus_low_teacher_rms_m']:+.4f} m). In A, the corresponding change is {analyses['A']['q_high_minus_low_teacher_rms_m']:+.4f} m. Positive means less teacher imitation at high q.",'',
           '## 3. q versus sampling','',
           f"Across all selected windows, median q-effect RMS is {summary['median_q_effect_rms_m']:.4f} m and median sampling RMS is {summary['median_sampling_rms_m']:.4f} m; median ratio {summary['median_q_sampling_ratio']:.3f}.",'',
           '## 4. Alternative samples','',
           f"At q=0.5, B has median across-seed forward-progress range {analyses['B']['sampling_q05_progress_range_median_m']:.4f} m; {analyses['B']['sampling_q05_mixed_forward_sign_fraction']:.1%} of B cases have both forward and backward endpoints among seeds. This is a static candidate-diversity measure.",'',
           '## 5. Quality-enabled versus disabled','',
           f"For B, mean enabled/disabled teacher RMS is {analyses['B']['enabled_teacher_rms_mean_m']:.4f}/{analyses['B']['disabled_teacher_rms_mean_m']:.4f} m. For A, it is {analyses['A']['enabled_teacher_rms_mean_m']:.4f}/{analyses['A']['disabled_teacher_rms_mean_m']:.4f} m. Smaller RMS means closer imitation of the teacher, not safer behavior.",'',
           '## 6. Goal provenance and redundancy','',
           f"Dataset goal is derived from the realized teacher endpoint (radius cap 0.8 m). Across all 6,208 complete held-out windows, goal distance/teacher endpoint displacement r={goal['progress_goal_pearson']:.3f}; q/goal-distance r={goal['q_goal_pearson']:.3f}. B and matched high-progress control mean goal lengths are {goal_analysis['no_progress_goal_mean_m']:.4f}/{goal_analysis['matched_high_progress_goal_mean_m']:.4f} m. Raw command is velocity, so a direct distance gap is dimensionally invalid.",'',
           '## 7. Most plausible mechanism','',
           'The realized-endpoint goal encodes much of the observed progress and is a concrete source of redundancy. Teacher imitation, q response and diffusion variation must be interpreted together using the continuous metrics and case pages. Static inference alone cannot uniquely attribute causality to one component.','',
           '## 8. Replay candidates','',
           f"Candidate held-out complete-window indices (selection rule recorded in casebook-interpretation.json): {replay}. A separate 20–50-case closed-loop simulator replay would test contact and failure behavior.",'',
           '## 9. Collection decision','',
           'Full collection remains stopped. A replay or explicit design decision is needed before changing dataset goal, model, loss, or collection scope.','',
           '## Support and limits','',
           'Sparse terrain × q cells are reported as insufficient support/NA in casebook-summary.json. Episode-cluster bootstrap intervals for cohort category fractions and key paired differences are in casebook-interpretation.json. No claim is made for all seven terrain types or physical safety from XY trajectories.','']
    (OUT/'casebook-report.md').write_text('\n'.join(lines))
    print('ANALYSIS_DONE',len(replay),flush=True)
if __name__=='__main__':main()
