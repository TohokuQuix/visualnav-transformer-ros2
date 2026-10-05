"""Offline Critic input and rank contracts."""
from pathlib import Path

import numpy as np
import pytest

from eat_nav_g40_boundary_schema import derive_schema, replace_command_only

ENV=Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml')


def test_command_slice_comes_from_saved_term_order_and_dimensions():
    schema=derive_schema(ENV, action_dim=6, command_dim=3)
    assert schema['total_dim']==267
    assert schema['command_field']['start']==9
    assert schema['command_field']['end']==12
    assert schema['fields'][-1]['name']=='height_scan'
    assert schema['fields'][-1]['shape']==[187]


def test_command_substitution_changes_only_the_verified_slice():
    source=np.arange(267,dtype=np.float32)
    result=replace_command_only(source,9,12,[.3,0.,0.])
    assert np.array_equal(result[:9],source[:9])
    assert np.array_equal(result[12:],source[12:])
    assert result[9:12].tolist()==pytest.approx([.3,0.,0.])


def test_wrong_command_width_is_rejected():
    with pytest.raises(ValueError,match='command width'):
        replace_command_only(np.zeros(267,dtype=np.float32),9,12,[0.,0.])
