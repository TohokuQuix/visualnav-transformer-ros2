import json
from pathlib import Path

import pytest

from eat_nav_expanded_feasibility_resume import (
    audit_frozen_inputs, router_process_identity, require_router_log,
)
from eat_nav_expanded_feasibility_runtime import patch_e1_sim_source, map_path_exact


def test_frozen_stopped_root_has_50_unchanged_rows_and_zero_runs():
    result=audit_frozen_inputs()
    assert result['stopped_stage']=='X17_STOPPED_ROUTER_UNAVAILABLE'
    assert result['old_attempt_count']==0
    assert result['run_manifest_count']==50
    assert result['new_state_count']==5
    assert result['new_candidate_count']==35


def test_router_identity_requires_canonical_binary_and_start_tick(tmp_path):
    proc=tmp_path/'123';(proc/'fd').mkdir(parents=True)
    (proc/'stat').write_text('123 (rmw_zenohd) S '+' '.join(['1']*18+['456']+['1']*30))
    (proc/'cmdline').write_bytes(b'/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd\0')
    (proc/'exe').symlink_to('/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd')
    result=router_process_identity(123,proc_root=tmp_path)
    assert result['pid']==123
    assert result['start_ticks']==456
    assert result['command']==['/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd']


def test_missing_router_log_stops_before_candidate_free_smoke(tmp_path):
    with pytest.raises(ValueError,match='router log evidence'):
        require_router_log(None)
    log=tmp_path/'router.log';log.write_text('Started Zenoh router\n')
    assert require_router_log(log)['status']=='pass'
    log.write_text('bind failed\n')
    with pytest.raises(ValueError,match='router log warning'):
        require_router_log(log)


def test_sim_adapter_changes_only_identity_and_anchor_path_validation():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[2]/'scripts/eat_nav_e1_sim.py').read_text()
    patched=patch_e1_sim_source(source)
    assert patched.count("state['state_id'].startswith('expanded-train-')")==1
    assert 'config_spec[\'scope\']!=\'anchor_only_then_live_candidate\'' in patched
    assert "if args.stage=='anchor':\n        fixed=np.asarray(computed_path,dtype=np.float64)" in patched
    assert source.replace("state['state_id'].startswith('g0e-train-')",
        "state['state_id'].startswith('expanded-train-')")!=source
    assert patched.count('env.step(batch)')==source.count('env.step(batch)')==1


def test_map_path_transform_matches_original_scalar_formula():
    import math
    base=[[0.,0.],[.5,.1]]
    actual=map_path_exact(base,[1.,2.,math.pi/4])
    assert actual[0]==[1.,2.]
    assert actual[1]==[float(1+math.cos(math.pi/4)*.5-math.sin(math.pi/4)*.1),
                      float(2+math.sin(math.pi/4)*.5+math.cos(math.pi/4)*.1)]
