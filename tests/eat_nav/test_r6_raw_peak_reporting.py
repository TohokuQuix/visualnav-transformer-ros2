import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest
path=Path(__file__).resolve().parents[2]/'docs/eat-nav/r6-2s-pose-20261030/reporting/raw_peak_metrics.py'
spec=importlib.util.spec_from_file_location('r6_raw_peak_reporting',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def root(roll=0):
 a=math.radians(roll)/2
 return [0,0,.3,math.cos(a),math.sin(a),0,0,0,0,0,0,0,0]

def rows(spike=10,angle=30,terminal=False):
 out=[]
 for i in range(40):
  row=dict(control_step=i,logical_sim_time_ns=1_000_000_000+i*50_000_000,post_timestamp_ns=1_000_000_000+(i+1)*50_000_000,root_state=root(),post_root_state=root(angle if i==spike else 0))
  if terminal and i==spike:row['terminal']={'pre_reset_state':{'root_state':row['post_root_state'],'termination_reasons':['tilt_over']}}
  out.append(row)
  if row.get('terminal'):break
 return out

def predicted(roll=0):
 a=np.zeros((20,9));a[:,4]=1;a[:,6]=math.cos(math.radians(roll));a[:,5]=math.sin(math.radians(roll));a[:,8]=1
 return a

def test_20hz_peak_between_10hz_samples_is_preserved():
 r=m.peak_record(predicted(),rows())
 assert r['full_window'] and r['observed_rows']==40
 assert r['observed_peak_tilt_deg']==pytest.approx(30)
 assert r['observed_grid_peak_tilt_deg']==pytest.approx(0)
 assert r['full_window_absolute_error_deg']==pytest.approx(30)

def test_terminal_peak_is_observed_but_unknown_future_not_mae():
 r=m.peak_record(predicted(),rows(angle=110,terminal=True))
 assert not r['full_window'] and r['observed_rows']==11 and r['tip_observed']
 assert r['observed_peak_tilt_deg']==pytest.approx(110)
 assert r['full_window_absolute_error_deg'] is None
 assert r['underprediction_lower_bound_deg']==pytest.approx(110)
 r=m.peak_record(predicted(140),rows(angle=110,terminal=True))
 assert r['full_window_absolute_error_deg'] is None and r['underprediction_lower_bound_deg']==0

def test_override_excludes_later_pose_and_rejects_time_gaps():
 a=rows(angle=110);a[5]['goal_hold']=True
 r=m.peak_record(predicted(),a)
 assert r['observed_rows']==5 and not r['full_window'] and r['observed_peak_tilt_deg']==0
 a=rows();a[1]['post_timestamp_ns']+=1
 with pytest.raises(ValueError,match='contiguous'):m.peak_record(predicted(),a)
