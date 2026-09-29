import pytest
from eat_nav_clock_diagnostic_analyze import parse_clock_line,first_difference,nearest_previous


def test_parse_follower_clock_line():
    row=parse_clock_line('[INFO] [100.000] [follower]: CLOCK_DIAG pre seq=41 node_ns=2000000000 window_elapsed=20.1 ready=1 cmd_dist=5.929 recovery=0')
    assert row['event']=='pre' and row['seq']==41 and row['node_ns']==2000000000
    assert row['window_elapsed']==pytest.approx(20.1) and row['ready']==1
    assert parse_clock_line('unrelated') is None


def test_first_difference_reports_first_step_not_maximum():
    a=[{'command':[.3,0,0]},{'command':[.3,0,0]},{'command':[.3,0,0]}]
    b=[{'command':[.3,0,0]},{'command':[-.3,0,0]},{'command':[.3,0,0]}]
    assert first_difference(a,b,'command',1e-6)==1
    assert first_difference(a,a,'command',1e-6) is None


def test_nearest_previous_clock_sample():
    events=[{'node_ns':100,'seq':1},{'node_ns':200,'seq':2}]
    assert nearest_previous(events,199,'node_ns')['seq']==1
    assert nearest_previous(events,201,'node_ns')['seq']==2
    assert nearest_previous(events,50,'node_ns') is None


def test_bridge_command_response_maps_to_next_trace_step():
    import json
    from pathlib import Path
    repo=Path(__file__).resolve().parents[2]
    result=repo/'docs/quality-identifiability-gate/candidate-coverage/clock-domain-diagnostic'
    routing=json.loads((result/'clock-command-routing-audit.json').read_text())
    assert len(routing['rows'])==3
    assert all(row['checked_step_count']==119 and row['next_step_max_abs_error']==0.0
               for row in routing['rows'])
    compact=json.loads((result/'clock-alignment-summary.json').read_text())
    assert all(row['command_applied_step']==row['sim_step']+1 for row in compact['selected_steps'])
