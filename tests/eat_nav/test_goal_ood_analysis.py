import pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
from eat_nav_goal_ood_analyze import path_rms,mode_count,distribution,wrap

def test_goal_effect_and_seed_effect_are_separate():
    base=np.zeros((8,2));changed=np.ones((8,2))*.1
    assert path_rms(base,base)==0
    assert abs(path_rms(base,changed)-.1)<1e-12
    assert mode_count([base,base],threshold=.05)==1
    assert mode_count([base,changed],threshold=.05)==2

def test_angle_wrap_and_unsupported_na():
    assert abs(wrap(np.pi+.1)-(-np.pi+.1))<1e-9
    assert distribution([])['median'] is None
