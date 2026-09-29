import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_serial_outcome_uses_final_value_and_discounted_rewards():
    from eat_nav_phase_c_coverage_serial_analyze import outcome
    data={'critic_value':np.array([[[10.]],[[8.]]]),'reward':np.array([[1.],[2.]]),'critic_value_end':np.array([[12.]])}
    result=outcome(data)
    assert np.isclose(result['delta_v'],2)
    assert np.isclose(result['drawdown'],-2)
    assert np.isclose(result['recovery'],4)
    assert np.isclose(result['advantage_h'],1+.99*2+.99**2*12-10)
