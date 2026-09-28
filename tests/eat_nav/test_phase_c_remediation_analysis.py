import numpy as np
from visualnav_transformer.eat_nav.phase_c_runner import gate_c0_status
from scripts.eat_nav_phase_c_remediation_analysis import signal_report

def test_first_divergence_is_not_reported_below_tolerance():
    x=np.zeros((4,3,2));x[1:,1,0]=0.01;x[2:,2,1]=0.02
    report=signal_report(x,0.005)
    assert report['first_step']==1
    assert report['max_abs']==0.02
    assert report['previous_step_rms']==0.0
    assert len(report['branch_rms'])==2
    assert signal_report(x,0.03)['first_step'] is None

def test_epoch_cap_without_early_stop_remains_inconclusive():
    assert gate_c0_status([0.2,0.19],300,300)=='inconclusive'
