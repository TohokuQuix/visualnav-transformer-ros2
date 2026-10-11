import math

from visualnav_transformer.eat_nav.p1_paired_integrity import assess_step


def test_normal_state_passes():
    row = {'post_root_state': [0, 0, 0, 1, 0, 0, 0, 0.3, 0, 0, 0, 0, 0],
           'applied_command': [0.3, 0, 0.1], 'contact_net_forces_w': [[0, 0, 1]]}
    assert assess_step(row, roll_pitch_cap_rad=1.0, speed_cap_mps=5.0) == []


def test_roll_and_nonfinite_abort():
    angle = 1.1
    row = {'post_root_state': [0, 0, 0, math.cos(angle/2), math.sin(angle/2), 0, 0,
                               0.3, 0, 0, 0, 0, 0],
           'applied_command': [0.3, 0, 0.1], 'contact_net_forces_w': [[0, 0, 1]]}
    assert 'roll_pitch_cap' in assess_step(row, roll_pitch_cap_rad=1.0, speed_cap_mps=5.0)
    row['contact_net_forces_w'] = [[float('nan'), 0, 0]]
    assert 'nonfinite_contact' in assess_step(row, roll_pitch_cap_rad=1.0, speed_cap_mps=5.0)
