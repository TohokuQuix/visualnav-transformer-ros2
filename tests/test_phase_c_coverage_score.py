import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_primary_composite_uses_three_train_standardized_heads():
    from eat_nav_phase_c_coverage_score import primary_standardized
    p=np.array([[[2.,4.,6.,100.]]])
    scalers={h:{'mean':0.,'std':2.} for h in ('delta_v','advantage_h','drawdown')}
    assert np.isclose(primary_standardized(p,scalers)[0,0],2.)
