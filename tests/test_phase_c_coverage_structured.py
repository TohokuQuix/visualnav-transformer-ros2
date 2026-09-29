import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_basis_and_candidate_preserve_endpoint():
    from eat_nav_phase_c_coverage_structured import fit_basis, proposals
    paths=np.array([[[1.,1.],[2.,2.],[3.,3.]],[[1.,2.],[2.,3.],[3.,3.]],[[1.,0.],[2.,1.],[3.,3.]],[[1.,.5],[2.,2.],[3.,3.]]])
    basis,scale=fit_basis(paths,1,.75)
    raw=np.stack([paths[0],paths[1]])
    candidate=proposals(raw,basis,scale)
    assert np.allclose(basis[0,-1],0,atol=1e-8)
    assert np.allclose(candidate[2:,-1],raw[0,-1],atol=1e-8)
    assert len(candidate)==4


def test_proposal_selection_ignores_evaluator():
    from eat_nav_phase_c_coverage_structured import farthest_raw_index
    paths=np.array([[[0.,0.]],[[1.,0.]],[[2.,0.]]])
    assert farthest_raw_index(paths)==2
