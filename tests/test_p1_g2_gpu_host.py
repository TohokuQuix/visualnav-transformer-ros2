"""G2 host protocol tests; no Isaac, ROS or GPU process is started."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eat_nav_p1_g2_gpu_host as host


def test_g2_wrapper_command_has_frozen_direct_probe_without_g1_exit():
    command=host.build_wrapper_command(remote='/persistent/g2-attempt-0',
        container_id='container',token='a'*64,source_sha='b'*64,config_sha='c'*64)
    assert command[command.index('/isaac-sim/python.sh')+1].endswith('/owner-wrapper.py')
    assert command[command.index('--anchor-id')+1]=='first-differential'
    assert command[command.index('--batch-index')+1]=='0'
    assert '--gate1-probe' not in command


def test_frozen_diagnostic_params_change_only_logging_period(tmp_path):
    production=tmp_path/'production.yaml';derived=tmp_path/'diagnostic.yaml'
    production.write_text('linear_speed: 0.3\ndebug_control_log: true\n'
                          'debug_control_log_period_sec: 0.5\n')
    expected=production.read_text().replace('period_sec: 0.5','period_sec: 0.1')
    digest=hashlib.sha256(expected.encode()).hexdigest()
    assert host.diagnostic_params(production,derived,digest)==digest
    assert derived.read_text()==expected
    assert 'period_sec: 0.5' in production.read_text()
    with pytest.raises(ValueError,match='unavailable'):
        host.diagnostic_params(production,tmp_path/'bad.yaml','0'*64)


def _fixture(root: Path):
    output=root/'output';output.mkdir()
    names=['initial-state.json','terrain.json','prepath-gate.json',
           'path-publication.json','pilot-summary.json','contact-body-names.json']
    for name in names:(output/name).write_text('{}\n')
    (output/'pilot-summary.json').write_text(json.dumps({'recorded_steps':[600]*4})+'\n')
    keymap={'initial-state.json':'initial_state_sha256','terrain.json':'terrain_sha256',
        'prepath-gate.json':'prepath_gate_sha256','path-publication.json':'path_publication_sha256',
        'pilot-summary.json':'pilot_summary_sha256','contact-body-names.json':'contact_body_names_sha256'}
    marker={key:host.digest(output/name) for name,key in keymap.items()}
    marker.update({'Path_publications':1,'settling_control_steps':20,'control_steps':600})
    for index in range(4):
        path=output/f'trace-slot-{index}.jsonl'
        with path.open('w') as stream:
            for step in range(600):
                stamp=2_000_000_000+50_000_000*step
                protocol={'step':step,'stamp_ns':stamp,'clock_ns':stamp,'tf_ns':stamp,
                    'command_ns':stamp,'bridge_accept_ns':stamp,'physics_input_ns':stamp,
                    'events':['clock','tf','command','bridge_acceptance','physics'],
                    'command_accepted_steady_ns':step,'physics_begin_steady_ns':step+1,
                    'episode_length_before':20+step,'episode_length_after':21+step,
                    'terminated':False,'truncated':False,'reset_called':False}
                stream.write(json.dumps({'control_step':step,'terminal':None,
                                         'timing_protocol':protocol})+'\n')
    marker['trace_sha256']=[host.digest(output/f'trace-slot-{i}.jsonl') for i in range(4)]
    (output/'complete.json').write_text(json.dumps(marker)+'\n')


def test_g2_output_gate_requires_four_complete_same_episode_traces(tmp_path,monkeypatch):
    _fixture(tmp_path)
    def fake_run(command,**_):
        name=Path(command[-1]).name
        return SimpleNamespace(stdout=f'{host.digest(tmp_path/"output"/name)}  {name}\n')
    monkeypatch.setattr(host.subprocess,'run',fake_run)
    result=host.validate_g2_output(tmp_path,'/remote',{})
    assert result['trace_steps_per_env']==[600]*4
    assert result['logical_last_ns']-result['logical_first_ns']==599*50_000_000
    path=tmp_path/'output/trace-slot-2.jsonl'
    rows=path.read_text().splitlines()
    bad=json.loads(rows[400]);bad['timing_protocol']['reset_called']=True
    rows[400]=json.dumps(bad)
    path.write_text('\n'.join(rows)+'\n')
    marker=json.loads((tmp_path/'output/complete.json').read_text())
    marker['trace_sha256'][2]=host.digest(path)
    (tmp_path/'output/complete.json').write_text(json.dumps(marker)+'\n')
    with pytest.raises(ValueError,match='episode_termination_or_reset'):
        host.validate_g2_output(tmp_path,'/remote',{})
