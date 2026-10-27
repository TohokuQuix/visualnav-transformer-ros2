"""R5 retains linear risk and adds turn risk; no measured-angle STOP."""
from copy import deepcopy
import numpy as np
from eat_nav_r4 import R4Guard
from eat_nav_r4_star import candidate_paths,CANDIDATE_IDS,DIRECTIONS,CONFIG as PREVIOUS_CONFIG
CONFIG=dict(deepcopy(PREVIOUS_CONFIG),schema='eatnav_r5_stall_only_no_pose_stop_v1',active_risk_weights=[0.,0.,1.,.5])

def select_candidate(probabilities,progress,goal_body):
    p=np.asarray(probabilities,dtype=float);q=np.asarray(progress,dtype=float);goal=np.asarray(goal_body,dtype=float)
    if p.shape!=(6,4) or q.shape!=(6,4) or not np.isfinite(p).all() or not np.isfinite(q).all() or not ((p>=0)&(p<=1)).all():raise ValueError('six candidate/head predictions required')
    if goal.shape!=(2,) or not np.isfinite(goal).all():raise ValueError('invalid goal')
    endpoints=candidate_paths()[:,-1].astype(float);unit=endpoints/np.linalg.norm(endpoints,axis=1,keepdims=True)
    travel=np.clip(q[:,3],0.,10.)
    proxy=np.linalg.norm(goal)-np.linalg.norm(goal[None]-unit*travel[:,None],axis=1)
    penalty=p@np.asarray([0.,0.,1.,.5]);scores=proxy-penalty;selected=int(np.argmax(scores))
    return dict(selected_index=selected,selected_candidate=CANDIDATE_IDS[selected],goal_only_index=int(np.argmax(proxy)),
        scores=scores.tolist(),goal_progress_proxy_m=proxy.tolist(),risk_penalty=penalty.tolist(),risk_weights=[0.,0.,1.,.5],
        probabilities=p.tolist(),progress_m=q.tolist(),scoring='goal_distance_reduction_proxy_minus_weighted_risk; first_frozen_candidate_tie')

class R5Guard(R4Guard):
    condition='R5'

    def snapshot(self):
        return dict(schema='eatnav_r5_guard_state_v1',config=deepcopy(CONFIG),state=deepcopy(vars(self)))

    @classmethod
    def from_state(cls,saved):
        if saved.get('schema')!='eatnav_r5_guard_state_v1' or saved.get('config')!=CONFIG:raise ValueError('R5 state contract')
        result=cls()
        if set(saved['state'])!=set(vars(result)):raise ValueError('R5 state keys')
        result.__dict__.update(deepcopy(saved['state']));return result

    def decide(self,probabilities,progress,goal_body,step,tilt_deg):
        raw=np.asarray(probabilities,dtype=float)
        if raw.shape!=(6,4):raise ValueError('six candidate/head probabilities required')
        p=np.zeros((6,4));p[:,2]=raw[:,2];p[:,3]=raw[:,3];q=np.asarray(progress,dtype=float)
        before=self.snapshot();base=select_candidate(p,q,goal_body)
        if self.last_decision_step is not None and step<=self.last_decision_step:raise ValueError('guard decision time must increase')
        self.observe_tilt(tilt_deg,step);self.last_decision_step=step
        stalled=((p[:,2]>=CONFIG['stall_trip'])|(p[:,3]>=CONFIG['stall_trip']))&(q[:,3]<=CONFIG['stall_progress_max_m'])
        available=np.flatnonzero(~stalled);score=np.asarray(base['scores'])
        if not len(available):self._stop(step,'all_candidates_high_risk')
        reason=self.stop_reason if self.stopped else 'best_safe_candidate'
        if self.stopped:
            ready=(available.size>0 and step-self.stop_since>=CONFIG['minimum_stop_steps'])
            self.resume_hits=self.resume_hits+1 if ready else 0
            if self.resume_hits>=CONFIG['resume_confirmations']:
                best=int(available[np.argmax(score[available])]);self.stopped=False;self.stop_reason=None;self.stop_since=None;self.resume_hits=0
                self._choose(best,step);reason='resume_confirmed'
        else:
            best=int(available[np.argmax(score[available])])
            if self.current is None:self._choose(best,step)
            elif stalled[self.current]:self._choose(best,step);reason='unsafe_current_override'
            elif best==self.current:self.pending_direction=None;self.pending_hits=0;reason='keep_best'
            elif step-self.last_switch_step<CONFIG['minimum_candidate_dwell_steps']:
                self.pending_direction=None;self.pending_hits=0;reason='minimum_dwell'
            elif score[best]-score[self.current]<CONFIG['switch_score_margin']:
                self.pending_direction=None;self.pending_hits=0;reason='score_margin'
            elif DIRECTIONS[best]!=DIRECTIONS[self.current]:
                direction=int(DIRECTIONS[best]);self.pending_hits=self.pending_hits+1 if direction==self.pending_direction else 1
                self.pending_direction=direction;reason='direction_confirmation'
                if self.pending_hits>=CONFIG['direction_confirmations']:self._choose(best,step);reason='direction_switch_confirmed'
            else:self._choose(best,step);reason='score_switch'
        index=None if self.stopped else self.current
        return dict(base,condition='R5',unfiltered_selected_index=base['selected_index'],selected_index=index,
            selected_candidate='STOP' if index is None else CANDIDATE_IDS[index],guard_reason=reason,guard_stopped=self.stopped,
            guard_unsafe=stalled.tolist(),guard_state_before=before,guard_state_after=self.snapshot(),measured_tilt_deg=float(tilt_deg),
            measured_tilt_stop_enabled=False,measured_tilt_release_enabled=False,
            raw_probabilities=[[float(v) if np.isfinite(v) else None for v in row] for row in raw])
