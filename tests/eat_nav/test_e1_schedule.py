import json

import pytest

from eat_nav_e1_schedule import execution_entry, next_pending, require_duplicate_before_lateral
from eat_nav_e1_prepare import ROOT


def rows():
    return json.loads((ROOT/'continuation-manifest.json').read_text())['pending_rows']


def test_exact_order_and_derived_execution_identity():
    pending=rows()
    assert len(pending)==18
    assert next_pending(pending,set())['order']==1
    assert next_pending(pending,{r['process_id'] for r in pending[:3]})['order']==4
    entry=execution_entry(pending[0])
    assert entry['state_id']=='g0e-train-000'
    assert entry['kind']=='duplicate'
    assert entry['logical_t0_ns']==2_300_000_000_000
    assert entry['map_path_sha256']==pending[0]['map_path_sha256']


def test_second_group_and_lateral_require_duplicate_gate():
    pending=rows()
    with pytest.raises(ValueError,match='duplicate'):
        require_duplicate_before_lateral(pending[3],{})
    with pytest.raises(ValueError,match='duplicate'):
        require_duplicate_before_lateral(pending[9],{})
    require_duplicate_before_lateral(pending[3],{'g0e-train-000':'pass'})
