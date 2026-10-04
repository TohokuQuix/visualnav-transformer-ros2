import json

import pytest
from pathlib import Path
from eat_nav_e1_process_ownership import ProcessRegistry

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


def test_next_baseline_does_not_append_to_previous_registry(tmp_path,monkeypatch):
    import eat_nav_e1_schedule as schedule
    monkeypatch.setattr(schedule,'ROOT',tmp_path)
    previous=tmp_path/'runs/previous/process-registry.json';previous.parent.mkdir(parents=True)
    ProcessRegistry('previous',1).save(previous);original=previous.read_bytes()
    from eat_nav_e1_artifact_recovery import seal_tree
    seal_tree(previous.parent)
    (tmp_path/'e1-safety-stops.yaml').write_text('infrastructure_stop:\n  gpu_device_total_memory_over_mib: 6656\n')
    monkeypatch.setattr(schedule,'proc_table',lambda:{})
    monkeypatch.setattr(schedule,'sample',lambda *a,**k:{'monotonic_ns':1,
        'cgroup_memory':{'events':{}}})
    monkeypatch.setattr(schedule,'gate_sample',lambda *a,**k:{'device_total_used_mib':0})
    monkeypatch.setattr(schedule,'baseline_gate',lambda *a,**k:None)
    monkeypatch.setattr(schedule,'limits',lambda *a,**k:{})
    ticks=iter([0.,11.,22.]);monkeypatch.setattr(schedule.time,'monotonic',lambda:next(ticks))
    result=schedule.prelaunch_baseline({'process_id':'next'},object(),object())
    assert result['status']=='pass'
    assert previous.read_bytes()==original
    assert (tmp_path/'baselines/next.jsonl').is_file()
