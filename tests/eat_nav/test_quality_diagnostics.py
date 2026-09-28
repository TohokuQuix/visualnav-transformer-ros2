import numpy as np

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.quality_diagnostics import diagnose_quality_overlap


def window(ep, step, q, terrain='flat', goal=(.2,0.)):
    d=np.ones((4,2,20,20),dtype=np.float32)
    d[:,0]=.01
    return {'episode_key':(ep,), 'step':step, 'dem':d, 'goal':np.array(goal), 'quality':q, 'terrain_type':terrain}


def test_overlap_detects_multiple_bins_and_deduplicates_adjacent_anchors():
    windows=[]
    for ep in range(60):
        windows += [window(ep,0,.1 if ep%2 else .9),window(ep,5,.1 if ep%2 else .9)]
    report=diagnose_quality_overlap(windows,EATConfig())
    assert report.multi_bin_anchors==120
    assert report.distinct_anchors==60
    assert report.gate_pass
    assert report.bin_counts[0]==60 and report.bin_counts[3]==60
    assert report.by_terrain['flat']['distinct_anchors']==60


def test_one_to_one_goal_quality_fails_gate():
    windows=[window(ep,0,.1 if ep%2 else .9,goal=(0.,0.) if ep%2 else (.8,0.)) for ep in range(60)]
    assert not diagnose_quality_overlap(windows,EATConfig()).gate_pass


def test_insufficient_windows_remain_inconclusive():
    windows=[window(1,0,.1),window(2,0,.9)]
    report=diagnose_quality_overlap(windows,EATConfig())
    assert not report.gate_pass
    assert report.distinct_anchors==2
