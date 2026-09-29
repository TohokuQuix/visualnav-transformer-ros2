"""Frozen map Paths must use the anchor pose from the one anchor-only run."""
import math
import numpy as np
from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor


def test_map_paths_from_anchor_reproduces_nominal_and_rotates_candidates():
    yaw=math.pi/2
    anchor={'map_path':[[2.,3.],[2.,4.]],
            'anchor':{'root_state':[0.,0.,0.,math.cos(yaw/2),0.,0.,math.sin(yaw/2)]}}
    base={'bank-nominal':[[0.,0.],[1.,0.]],
          'nomad-seed-1001':[[0.,0.],[0.,1.]]}
    result=map_paths_from_anchor(anchor,base)
    assert np.allclose(result['bank-nominal'],anchor['map_path'],atol=1e-8)
    assert np.allclose(result['nomad-seed-1001'],[[2.,3.],[1.,3.]],atol=1e-8)
