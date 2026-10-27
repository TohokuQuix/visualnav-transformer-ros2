"""Exploratory stateful risk stop and candidate hysteresis; no model changes."""
from copy import deepcopy
import math
import numpy as np
from eat_nav_risk_navigation_v2 import select_candidate,CANDIDATE_IDS

CONFIG=dict(schema='eatnav_risk_guard_config_v1',tip_trip=.7,near_trip=.7,
    tip_release=.5,near_release=.5,stall_trip=.95,stall_progress_max_m=.05,
    tilt_trip_deg=45.,tilt_release_deg=35.,minimum_stop_steps=40,resume_confirmations=2,
    minimum_candidate_dwell_steps=80,switch_score_margin=.15,direction_confirmations=2)


class RiskGuard:
    def __init__(self):
        self.stopped=False;self.stop_reason=None;self.stop_since=None
        self.current=None;self.last_switch_step=None;self.resume_hits=0
        self.pending_direction=None;self.pending_hits=0
        self.last_decision_step=None;self.last_observation_step=None

    def snapshot(self):
        return dict(schema='eatnav_risk_guard_state_v1',config=deepcopy(CONFIG),state=deepcopy(vars(self)))

    @classmethod
    def from_state(cls,saved):
        if saved.get('schema')!='eatnav_risk_guard_state_v1' or saved.get('config')!=CONFIG:
            raise ValueError('guard state schema/config mismatch')
        result=cls()
        if set(saved['state'])!=set(vars(result)):raise ValueError('guard state keys mismatch')
        result.__dict__.update(deepcopy(saved['state']))
        return result

    def _stop(self,step,reason):
        if not self.stopped:self.stop_since=step
        self.stopped=True;self.stop_reason=reason;self.current=None
        self.resume_hits=0;self.pending_direction=None;self.pending_hits=0

    def observe_tilt(self,tilt_deg,step):
        if not isinstance(step,int) or step<0 or not math.isfinite(tilt_deg) or not 0<=tilt_deg<=180:
            raise ValueError('invalid tilt/step')
        if self.last_observation_step is not None and step<self.last_observation_step:
            raise ValueError('guard observation time reversed')
        self.last_observation_step=step
        if tilt_deg>=CONFIG['tilt_trip_deg']:self._stop(step,'measured_tilt')
        return not self.stopped

    def _choose(self,index,step):
        self.current=int(index);self.last_switch_step=step
        self.pending_direction=None;self.pending_hits=0

    def decide(self,probabilities,progress,goal_body,step,tilt_deg):
        before=self.snapshot()
        base=select_candidate(probabilities,progress,goal_body)  # validates finite predictions
        if self.last_decision_step is not None and step<=self.last_decision_step:
            raise ValueError('guard decision time must increase')
        self.observe_tilt(tilt_deg,step);self.last_decision_step=step
        p=np.asarray(probabilities);progress=np.asarray(progress)
        stalled=((p[:,2]>=CONFIG['stall_trip'])|(p[:,3]>=CONFIG['stall_trip']))&(
            progress[:,3]<=CONFIG['stall_progress_max_m'])
        unsafe=(p[:,0]>=CONFIG['tip_trip'])|(p[:,1]>=CONFIG['near_trip'])|stalled
        available=np.flatnonzero(~unsafe);score=np.asarray(base['scores'])
        if not len(available):self._stop(step,'all_candidates_high_risk')
        reason=self.stop_reason if self.stopped else 'best_safe_candidate'
        if self.stopped:
            release=(p[:,0]<CONFIG['tip_release'])&(p[:,1]<CONFIG['near_release'])&~stalled
            ready=(tilt_deg<=CONFIG['tilt_release_deg'] and release.any() and
                   step-self.stop_since>=CONFIG['minimum_stop_steps'])
            self.resume_hits=self.resume_hits+1 if ready else 0
            if self.resume_hits>=CONFIG['resume_confirmations']:
                choices=np.flatnonzero(release);best=int(choices[np.argmax(score[choices])])
                self.stopped=False;self.stop_reason=None;self.stop_since=None;self.resume_hits=0
                self._choose(best,step);reason='resume_confirmed'
        else:
            best=int(available[np.argmax(score[available])])
            if self.current is None:self._choose(best,step)
            elif unsafe[self.current]:self._choose(best,step);reason='unsafe_current_override'
            elif best==self.current:
                self.pending_direction=None;self.pending_hits=0;reason='keep_best'
            elif step-self.last_switch_step<CONFIG['minimum_candidate_dwell_steps']:
                self.pending_direction=None;self.pending_hits=0;reason='minimum_dwell'
            elif score[best]-score[self.current]<CONFIG['switch_score_margin']:
                self.pending_direction=None;self.pending_hits=0;reason='score_margin'
            elif (best==3)!=(self.current==3):
                direction=-1 if best==3 else 1
                self.pending_hits=self.pending_hits+1 if direction==self.pending_direction else 1
                self.pending_direction=direction;reason='direction_confirmation'
                if self.pending_hits>=CONFIG['direction_confirmations']:
                    self._choose(best,step);reason='direction_switch_confirmed'
            else:self._choose(best,step);reason='score_switch'
        index=None if self.stopped else self.current
        return dict(base,unfiltered_selected_index=base['selected_index'],selected_index=index,
            selected_candidate='STOP' if index is None else CANDIDATE_IDS[index],
            guard_reason=reason,guard_stopped=self.stopped,guard_unsafe=unsafe.tolist(),
            guard_state_before=before,guard_state_after=self.snapshot(),measured_tilt_deg=float(tilt_deg))


def guarded_command(guard,follower,pose,stamp_ns):
    if guard.stopped:
        return (0.,0.,0.),dict(guard_stop=True,guard_reason=guard.stop_reason)
    if follower is None:raise ValueError('running guard requires active follower')
    return follower.step_diagnostic(*pose,stamp_ns)
