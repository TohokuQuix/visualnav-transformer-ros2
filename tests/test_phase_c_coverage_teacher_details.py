import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_command_mode_labels_use_fixed_thresholds():
    from eat_nav_phase_c_coverage_teacher_details import command_mode_labels
    c=np.array([[0.,0.,0.],[.01,0.,.01],[.1,0.,.1]])
    labels=command_mode_labels(c,.03,.05)
    assert labels[0]==labels[1] and labels[2]!=labels[0]
