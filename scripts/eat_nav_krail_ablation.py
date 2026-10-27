"""Approved three-condition ablation; immutable model and shared switching rules."""
from copy import deepcopy
import numpy as np
from eat_nav_risk_guard import RiskGuard,CONFIG,guarded_command
from eat_nav_risk_navigation_v2 import select_candidate,CANDIDATE_IDS
from visualnav_transformer.eat_nav.fall_stall_targets import cycle_measurements,body_vector

CONDITIONS={'A0':[0.,0.,0.,0.],'A1':[0.,0.,1.,.5],'A2':[2.,.5,1.,.5]}
SEEDS=[2191202837,3777004980,1710704769]
ORDER=[(s,m) for s,ms in zip(SEEDS,[['A0','A1','A2'],['A1','A2','A0'],['A2','A0','A1']]) for m in ms]

class AblationGuard(RiskGuard):
    def __init__(self,condition):
        if condition not in CONDITIONS:raise ValueError('unknown condition')
        super().__init__();self.condition=condition;self.episode_causes=[]
        self.trip_causes=[];self.release_candidate_available=True;self.release_blockers=[];self.exclusion_matrix=[[False]*3 for _ in range(4)]

    def snapshot(self):
        return dict(schema='eatnav_ablation_guard_v1',config=deepcopy(CONFIG),state=deepcopy(vars(self)))

    @classmethod
    def from_state(cls,saved):
        if saved['schema']!='eatnav_ablation_guard_v1' or saved['config']!=CONFIG:raise ValueError('state contract')
        result=cls(saved['state']['condition'])
        if set(vars(result))!=set(saved['state']):raise ValueError('state keys')
        result.__dict__.update(deepcopy(saved['state']));return result

    def _stop(self,step,reason):
        causes=['measured_tilt'] if reason=='measured_tilt' else [name for j,name in enumerate(['stall','tip','near']) if any(row[j] for row in self.exclusion_matrix)]
        for cause in causes:
            if cause not in self.episode_causes:self.episode_causes.append(cause)
            if cause not in self.trip_causes:self.trip_causes.append(cause)
        super()._stop(step,reason)

    def observe_tilt(self,tilt_deg,step):
        if step!=self.last_observation_step:self.trip_causes=[]
        running=super().observe_tilt(tilt_deg,step)
        self._update_blockers(tilt_deg,step)
        return running

    def _update_blockers(self,tilt_deg,step):
        blockers=[]
        if self.stopped:
            if tilt_deg>35:blockers.append('tilt')
            if not self.release_candidate_available:blockers.append('no_release_candidate')
            if step-self.stop_since<40:blockers.append('minimum_stop')
            if self.resume_hits<2:blockers.append('confirmation')
        self.release_blockers=blockers

    def decide(self,probabilities,progress,goal_body,step,tilt_deg):
        raw=select_candidate(probabilities,progress,goal_body)  # validate all heads, including disabled ones
        p=np.asarray(probabilities,dtype=float).copy();q=np.asarray(progress,dtype=float)
        if self.condition=='A0':p[:]=0.
        elif self.condition=='A1':p[:,:2]=0.
        stall=((p[:,2]>=.95)|(p[:,3]>=.95))&(q[:,3]<=.05)
        self.exclusion_matrix=np.column_stack([stall,p[:,0]>=.7,p[:,1]>=.7]).tolist()
        release=(p[:,0]<.5)&(p[:,1]<.5)&~stall
        self.release_candidate_available=bool(release.any())
        d=super().decide(p,q,goal_body,step,tilt_deg)
        self._update_blockers(tilt_deg,step)
        blockers=list(self.release_blockers)
        causes=list(self.episode_causes)
        if not self.stopped:self.episode_causes=[]
        self.release_blockers=blockers
        d.update(risk_weights=CONDITIONS[self.condition],probabilities=raw['probabilities'],progress_m=raw['progress_m'],condition=self.condition,
                 episode_causes=causes,trip_causes=list(self.trip_causes),release_blockers=blockers,
                 exclusion_matrix=deepcopy(self.exclusion_matrix),guard_state_after=self.snapshot())
        return d


def terminal_outcome(distance,terminated,completed_steps,limit):
    entered=distance<=.25
    return dict(goal_entered=entered,normal_success=entered and not terminated,
        reason='natural_termination' if terminated else 'goal_reached' if entered else 'horizon_complete' if completed_steps>=limit else 'running')


def window_labels(rows,body_path):
    if len(rows)!=40 or any(r['guard_stopped'] for r in rows) or len({r['selected_candidate'] for r in rows})!=1:
        return dict(linear=None,turn=None)
    adapted=[]
    for row in rows:
        r=dict(row)
        if r.get('terminal'):r['terminal']=dict(pre_reset_state=row['terminal'])
        root=row['terminal']['root_state'] if row.get('terminal') else r['post_root_state'];r['body_angular_velocity_radps']=body_vector(root,root[10:13])
        adapted.append(r)
    cycles,_,_=cycle_measurements(adapted,body_path)
    return {h:None if cycles[0][h+'_stall']['label'] is None else bool(cycles[0][h+'_stall']['label']) for h in ['linear','turn']}


def exposure_partition(rows,labels):
    stop=sum(r['guard_stopped'] for r in rows)
    evaluable=sum(sum(not r['guard_stopped'] for r in rows[i*40:(i+1)*40]) for i,v in enumerate(labels) if v is not None)
    return dict(stop_steps=stop,evaluable_running_steps=evaluable,unknown_running_steps=len(rows)-stop-evaluable,total_steps=len(rows))
