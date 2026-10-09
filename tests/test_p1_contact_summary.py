import numpy as np
import pytest

from visualnav_transformer.eat_nav.p1_contact_summary import summarize_contact_arrays


def test_contact_summary_reads_counts_starts_and_active_force():
    capacity = 4
    force = np.zeros((4, 1)); force[2, 0] = 12
    sep = np.zeros((4, 1)); sep[2, 0] = -.003
    arrays = [force, np.zeros((4, 3)), np.zeros((4, 3)), sep,
              np.array([[1]]), np.array([[2]])]
    assert summarize_contact_arrays(arrays, capacity) == {
        'contact_count': 1, 'force_max_n': 12.0,
        'min_separation_m': -.003, 'bounds_valid': True,
        'buffer_full': False,
    }


def test_contact_summary_rejects_overflow():
    arrays = [np.zeros((4, 1)), np.zeros((4, 3)), np.zeros((4, 3)),
              np.zeros((4, 1)), np.array([[2]]), np.array([[3]])]
    with pytest.raises(ValueError, match='bounds'):
        summarize_contact_arrays(arrays, 4)
