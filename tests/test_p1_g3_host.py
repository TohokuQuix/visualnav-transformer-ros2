"""Offline tests for the frozen G3 batch and duplicate decisions."""
import copy
import ast
import json
from pathlib import Path

import eat_nav_p1_g3_pilot_host as host


def _rows(arrival=48, final_x=.58):
    rows=[]
    for step in range(90):
        rows.append({'label_exposure':step<=arrival+40,
                     'root_state':[.05+step*.005,0.,0.]+[0.]*10,
                     'post_root_state':[.05+step*.005,0.,0.]+[0.]*10,
                     'follower_diagnostic':{'goal_reached':step>=arrival,
                                            'linear_recovery':False,
                                            'angular_recovery':False,'mode':'default'}})
    return rows


def test_g3_command_uses_frozen_four_slot_batch():
    command=host.build_wrapper_command(remote='/persistent/batch-0',container_id='container',
        token='a'*64,source_sha='b'*64,config_sha='c'*64,
        anchor_id='second-graded',batch_index=1)
    assert command[command.index('--anchor-id')+1]=='second-graded'
    assert command[command.index('--batch-index')+1]=='1'
    assert '--gate1-probe' not in command


def test_direct_duplicate_gate_accepts_matching_outcome_and_rejects_arrival_spread():
    traces=[_rows() for _ in range(4)]
    summary={'arrivals':[{'arrival_step':48,'right_censored':False}]*4}
    terrain={'world_tile_origins':[[0.,0.]]*4}
    publication={'trial_paths':[{'path_sha256':'direct','map_xy':[[0.,0.],[.75,0.]]}]*4}
    result=host.duplicate_gate(traces,summary,terrain,publication)
    assert result['pass']
    changed=copy.deepcopy(summary)
    changed['arrivals'][1]={'arrival_step':52,'right_censored':False}
    result=host.duplicate_gate(traces,changed,terrain,publication)
    assert not result['pass'] and 'arrival_time_spread' in result['failures']


def test_g3_source_has_single_path_publication_and_640_step_cap():
    source=Path(__file__).resolve().parents[1]/'scripts/eat_nav_p1_g3_pilot_sim.py'
    tree=ast.parse(source.read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and
           isinstance(node.func,ast.Attribute) and node.func.attr=='request']
    operations=[node.args[0].value for node in calls if node.args and
                isinstance(node.args[0],ast.Constant)]
    assert operations.count('reset')==1
    assert 'update_path' not in operations
    assert operations.count('step')==1
    text=source.read_text()
    assert "steps=g3_contract['max_control_steps']" in text
    assert "cfg.episode_length_s=g3_contract['episode_length_s']" in text
    assert "conditions != ['A']*4" in text


def test_simulator_error_marker_is_failure_even_with_zero_native_exit(tmp_path):
    output=tmp_path/'output';output.mkdir()
    (output/'diagnostic-error.json').write_text(json.dumps({'type':'ValueError',
        'message':'direct DEM numeric gate','Path_publications':0}))
    assert host.simulator_artifact_failure(output,0)==\
        'simulator_diagnostic_error:ValueError:direct DEM numeric gate'
    assert host.simulator_artifact_failure(output,1).startswith('simulator_diagnostic_error:')


def test_terminal_gpu_disappearance_is_distinguished_from_live_loss(tmp_path):
    output=tmp_path/'output';output.mkdir()
    failure='gpu_attribution:owned_start_tick_mismatch,simulator_disappeared_in_bracket'
    assert host.classify_monitor_violation(failure,None,output)==failure
    assert host.classify_monitor_violation(failure,0,output)=='simulator_completion_marker_missing'
    (output/'complete.json').write_text('{}')
    assert host.classify_monitor_violation(failure,0,output) is None
    (output/'diagnostic-error.json').write_text(json.dumps({'type':'ValueError',
        'message':'numeric gate'}))
    assert host.classify_monitor_violation(failure,0,output).startswith('simulator_diagnostic_error:')


def test_dem_evidence_is_durable_before_frozen_gate_in_source():
    source=Path(__file__).resolve().parents[1]/'scripts/eat_nav_p1_g3_pilot_sim.py'
    text=source.read_text()
    assert 'persist_then_gate(args.output,dem_history,direct_dem_gate)' in text
    assert 'gate[\'direct_dem\']=direct_dem_gate(dem_history)' not in text
    stage=Path(__file__).resolve().parents[1]/'scripts/eat_nav_p1_g3_stage.py'
    assert "shutil.copy2(dem_capture,batch/'src/visualnav_transformer/eat_nav/p1_dem_capture.py')" in stage.read_text()
