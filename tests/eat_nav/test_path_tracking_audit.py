import copy,gzip,json,pathlib
import pytest
from visualnav_transformer.eat_nav.path_tracking_audit import (
    validate_trace_alignment, select_live_entries, identity_key,
)

DOC=pathlib.Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot')
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified')


def example():
    entry=json.loads((DOC/'schedule.json').read_text())['runs'][1]
    with gzip.open(ROOT/'runs'/entry['run_id']/'live-0.trace.json.gz','rt') as stream:
        trace=json.load(stream)
    return entry,trace


def test_real_trace_has_unique_pre_step_command_alignment():
    entry,trace=example()
    audit=validate_trace_alignment(trace,entry['logical_t0_ns'])
    assert audit['step_count']==120
    assert audit['stamp_mismatch_count']==0
    assert audit['root_step_max_abs']==0
    assert audit['command_max_abs']<1e-6


def test_one_step_stamp_shift_is_rejected_without_repair():
    entry,trace=example();bad=copy.deepcopy(trace)
    bad[5]['follower_command_stamp_ns']+=50_000_000
    with pytest.raises(ValueError,match='stamp mismatch step 5'):
        validate_trace_alignment(bad,entry['logical_t0_ns'])


def test_anchor_and_diagnostic_runs_cannot_enter_112_live_view():
    schedule=json.loads((DOC/'schedule.json').read_text())['runs']
    live,excluded=select_live_entries(schedule)
    assert len(live)==112 and len(excluded)==7
    assert all(e['kind']=='anchor' for e in excluded)
    assert len({identity_key(e) for e in live})==112
    with pytest.raises(ValueError,match='unexpected live kind'):
        select_live_entries(schedule+[{'kind':'equivalence','run_id':'diagnostic'}])


def test_audit_run_reconstructs_env_origin_and_rejects_bad_trace_sha():
    from visualnav_transformer.eat_nav.path_tracking_audit import audit_run
    entry=json.loads((DOC/'schedule.json').read_text())['runs'][1]
    ledger={r['run_id']:r for r in json.loads((DOC/'artifact-sha-ledger.json').read_text())['runs']}[entry['run_id']]
    state=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs/state-000')
    result=audit_run(entry,ROOT/'runs'/entry['run_id'],state,ledger)
    assert result['identity']['included'] is True
    assert result['identity']['path_point_count']==9
    assert result['coordinate']['env_origin_world_xy']==[4.0,-20.0]
    assert result['coordinate']['map_base_roundtrip_max_abs_m']<1e-6
    wrong=dict(ledger,trace_sha256='0'*64)
    with pytest.raises(ValueError,match='trace SHA mismatch'):
        audit_run(entry,ROOT/'runs'/entry['run_id'],state,wrong)


def test_source_hash_audit_allows_candidate_paths_but_rejects_checkpoint_drift():
    from visualnav_transformer.eat_nav.path_tracking_audit import validate_source_hashes
    hashes={'base_path':{'a','b'},'config':{'c','d'},'runner_source':{'old','fresh'},
            'rl_checkpoint':{'one'},'reference_config':{'same'},'manifest':{'m'},
            'bridge_client_source':{'b'},'capture_source':{'c'},'clock_protocol_source':{'t'},'state':{'s'}}
    validate_source_hashes(hashes)
    hashes['rl_checkpoint']={'one','drift'}
    with pytest.raises(ValueError,match='rl_checkpoint source SHA drift'):
        validate_source_hashes(hashes)


def test_recovery_log_uses_exact_stamped_pre_publish_events():
    from visualnav_transformer.eat_nav.path_tracking_audit import recovery_from_log_lines
    t0=1_000_000_000
    lines=[f'TF_RACE_JSON {json.dumps({"phase":"command_pre_publish","command_stamp_ns":t0+i*50_000_000,"linear_recovery":i==1,"angular_recovery":False})}' for i in range(3)]
    flags=recovery_from_log_lines(lines,t0,3)
    assert flags['linear_recovery_fired'] is True
    assert flags['angular_recovery_fired'] is False
    assert flags['command_events']==3
    with pytest.raises(ValueError,match='recovery log stamp mismatch'):
        recovery_from_log_lines(lines[:-1],t0,3)

def test_nominal_repeat_path_identity_within_every_state():
    schedule=json.loads((DOC/'schedule.json').read_text())['runs']
    for sid in sorted({e['state_id'] for e in schedule}):
        nominal=[e for e in schedule if e['state_id']==sid and e['kind']=='nominal']
        assert len(nominal)==3
        assert [e['repeat'] for e in nominal]==[0,1,2]
        assert len({e['base_path_sha256'] for e in nominal})==1
