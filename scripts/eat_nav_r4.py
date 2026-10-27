"""R4: linear-stall only; common switching and prediction-stop timing retained."""
from copy import deepcopy
import math
import numpy as np
from eat_nav_risk_guard import RiskGuard,CONFIG as COMMON_CONFIG

CONFIG=dict(deepcopy(COMMON_CONFIG),schema='eatnav_r4_linear_only_config_v1',
    tip_trip=None,near_trip=None,tip_release=None,near_release=None,
    tilt_trip_deg=None,tilt_release_deg=None,active_risk_weights=[0.,0.,1.,0.],
    measured_tilt_stop_enabled=False,measured_tilt_release_enabled=False)

class R4Guard(RiskGuard):
    condition='R4'

    def snapshot(self):
        return dict(schema='eatnav_r4_guard_state_v1',config=deepcopy(CONFIG),state=deepcopy(vars(self)))

    @classmethod
    def from_state(cls,saved):
        if saved.get('schema')!='eatnav_r4_guard_state_v1' or saved.get('config')!=CONFIG:raise ValueError('R4 state contract')
        result=cls()
        if set(saved['state'])!=set(vars(result)):raise ValueError('R4 state keys')
        result.__dict__.update(deepcopy(saved['state']));return result

    def observe_tilt(self,tilt_deg,step):
        # Record/check time and finite tilt, without any angle-dependent control.
        if not isinstance(step,int) or step<0 or not math.isfinite(tilt_deg) or not 0<=tilt_deg<=180:raise ValueError('invalid tilt/step')
        if self.last_observation_step is not None and step<self.last_observation_step:raise ValueError('guard observation time reversed')
        self.last_observation_step=step
        return not self.stopped

    def decide(self,probabilities,progress,goal_body,step,tilt_deg):
        raw=np.asarray(probabilities,dtype=float)
        if raw.shape!=(4,4):raise ValueError('four candidate/head probabilities required')
        effective=np.zeros((4,4));effective[:,2]=raw[:,2]
        self.observe_tilt(tilt_deg,step)
        # Parent keeps the exact dwell/margin/direction/min-stop/confirmation
        # implementation. It sees disabled head values0 and an angle0 so its
        # release predicate cannot retain a35degree gate. Our observe_tilt
        # override also disables the45degree trip. No global config is changed.
        result=super().decide(effective,progress,goal_body,step,0.)
        raw_json=[[float(v) if np.isfinite(v) else None for v in row] for row in raw]
        result.update(condition='R4',risk_weights=[0.,0.,1.,0.],raw_probabilities=raw_json,
            measured_tilt_deg=float(tilt_deg),measured_tilt_stop_enabled=False,measured_tilt_release_enabled=False)
        return result
