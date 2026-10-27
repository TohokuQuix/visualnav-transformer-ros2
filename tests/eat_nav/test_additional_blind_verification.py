"""Real sealed fixtures exercise integrity checks while public output stays blind."""
import copy
import importlib
import json
from pathlib import Path
import pytest
from test_fall_stall_audit import source,put,sha,reseal_attempt


def api():
    name='visualnav_transformer.eat_nav.fall_stall_blind_verification'
    assert importlib.util.find_spec(name) is not None,'blind verification implementation missing'
    return importlib.import_module(name)


@pytest.fixture
def additional_fixture(source):
    pilot=json.loads((source/'pilot-manifest.json').read_text())
    rows=json.loads((source/'assignment.json').read_text())['rows']
    for i,row in enumerate(rows):
        row.update(index=i,role='reserved_test',base_group=row['state_group_id'],prepath_only=i==0,
                   output_relative=next(source.glob(f'{i:03d}-*')).name,
                   reference_relative=None if i==0 else next(source.glob('000-*')).name+'/sim-output')
    contract={'rows':rows,'manifest':{'protocol_sha256':pilot['source_sha256']['protocol']},
              'sources':{'pilot_critical_sha256':{k:v for k,v in pilot['source_sha256'].items() if k!='protocol'}},
              'repo':Path(__file__).resolve().parents[2]}
    return source,contract


def test_reserved_integrity_receipt_has_no_semantics(additional_fixture,capsys):
    root,c=additional_fixture;row=c['rows'][1];attempt=root/row['output_relative']
    receipt=api().verify_attempt_blind(attempt,row,c,root/row['reference_relative'])
    assert receipt['pass'] is True and receipt['row_index']==1 and receipt['role']=='reserved_test'
    assert set(receipt)=={'schema','row_index','role','pass','outer_seal_sha256','inner_seal_sha256'}
    assert capsys.readouterr().out==''


@pytest.mark.parametrize('mutation',['trace','identity','reference','gate','unknown_terminal','path'])
def test_tampering_is_rejected_with_allowlisted_error_only(additional_fixture,mutation,capsys):
    root,c=additional_fixture;row=c['rows'][1];attempt=root/row['output_relative'];sim=attempt/'sim-output'
    marker='PRIVATE_tip_over_tilt_89_progress_12.34'
    if mutation=='trace':(sim/'trace-env-0.jsonl').write_text(marker)
    elif mutation=='identity':
        m=json.loads((attempt/'manifest.json').read_text());m['seed']=marker;put(attempt/'manifest.json',m);reseal_attempt(attempt)
    elif mutation=='reference':
        m=json.loads((attempt/'manifest.json').read_text());m['reference_prepath_sha256']=marker;put(attempt/'manifest.json',m);reseal_attempt(attempt)
    elif mutation=='gate':put(sim/'prepath-pair-gate.json',{'pass':False,'secret':marker});reseal_attempt(attempt)
    elif mutation=='unknown_terminal':
        path=sim/'trace-env-0.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['terminal']={'termination_reason':marker};path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        complete=json.loads((sim/'complete.json').read_text());complete['trace_sha256'][0]=sha(path);put(sim/'complete.json',complete);reseal_attempt(attempt)
    else:
        m=json.loads((sim/'input-manifest.json').read_text());m['path_body_xy_m'][1][0]+=1;put(sim/'input-manifest.json',m);reseal_attempt(attempt)
    with pytest.raises(api().BlindVerificationError) as error:
        api().verify_attempt_blind(attempt,row,c,root/row['reference_relative'])
    assert error.value.code in api().ERROR_CODES
    assert marker not in str(error.value) and marker not in repr(error.value)
    captured=capsys.readouterr();assert marker not in captured.out+captured.err


def test_partial_natural_terminal_and_parallel_censor_are_accepted(additional_fixture):
    root,c=additional_fixture;row=c['rows'][1];attempt=root/row['output_relative'];sim=attempt/'sim-output'
    complete=json.loads((sim/'complete.json').read_text())
    for env in range(4):
        path=sim/f'trace-env-{env}.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()][:80]
        rows[-1]['terminal']={'termination_reason':'tilt_over','pre_reset_state':{'root_state':rows[-1]['root_state']}} if env==0 else {'termination_reason':'parallel_env_censored_by_other_env'}
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows));complete['trace_sha256'][env]=sha(path)
    complete.update(control_steps=80,trace_lengths=[80]*4,terminated_envs=[0],censored_envs=[1,2,3]);put(sim/'complete.json',complete)
    put(sim/'long-horizon-event-labels.json',{'trials':[{'window_count':80}]*4});reseal_attempt(attempt)
    assert api().verify_attempt_blind(attempt,row,c,root/row['reference_relative'])['pass']


@pytest.mark.parametrize('mutation',['numeric_terminal','nonfinite_post','nonfinite_terminal_snapshot'])
def test_numeric_failures_never_receive_a_blind_pass(additional_fixture,mutation):
    root,c=additional_fixture;row=c['rows'][1];attempt=root/row['output_relative'];sim=attempt/'sim-output'
    path=sim/'trace-env-0.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    complete=json.loads((sim/'complete.json').read_text())
    if mutation=='numeric_terminal':
        rows[-1]['terminal']={'termination_reason':'numeric_failure','pre_reset_state':{'root_state':rows[-1]['root_state']}}
        complete['terminated_envs']=[0]
    elif mutation=='nonfinite_post':rows[-1]['post_root_state'][0]=float('nan')
    else:
        rows[-1]['terminal']={'termination_reason':'tilt_over','pre_reset_state':{'root_state':rows[-1]['root_state'],'joint_velocity':[float('inf')]}}
        complete['terminated_envs']=[0]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));complete['trace_sha256'][0]=sha(path)
    put(sim/'complete.json',complete);reseal_attempt(attempt)
    with pytest.raises(api().BlindVerificationError) as error:api().verify_attempt_blind(attempt,row,c,root/row['reference_relative'])
    assert error.value.code=='TRACE_INVALID'
