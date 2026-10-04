from pathlib import Path

from eat_nav_e1_run_one import simulator_command
from eat_nav_e1_attempt2 import cli_file_paths, audit_identity
import pytest


def test_formal_command_uses_frozen_input_root_for_both_paths():
    entry = dict(kind='duplicate', logical_t0_ns=2300000000000, state_index=0,
                 candidate_id='g0e-train-000-nominal_duplicate-00',
                 local_path_sha256='a'*64, map_path_sha256='b'*64,
                 run_id='g0e-train-000-nominal_duplicate-00')
    root = Path('/tmp/e1-frozen-input')
    cmd = simulator_command(entry, '/tmp/e1-fresh/runs/first',
                            input_root=root, simulator='/tmp/e1-fresh/eat_nav_e1_sim.py',
                            timing_telemetry=True)
    assert cmd[cmd.index('--base-path')+1] == str(root/'inputs/state-000/g0e-train-000-nominal_duplicate-00-base-path.json')
    assert cmd[cmd.index('--fixed-map-path')+1] == str(root/'inputs/state-000/g0e-train-000-nominal_duplicate-00-map-path.json')
    assert cmd[cmd.index('--output')+1] == '/tmp/e1-fresh/runs/first'
    assert '--timing-telemetry' in cmd


def test_cli_file_enumeration_rejects_missing_path_flag():
    entry = dict(kind='duplicate', logical_t0_ns=2300000000000, state_index=0,
                 candidate_id='g0e-train-000-nominal_duplicate-00',
                 local_path_sha256='a'*64, map_path_sha256='b'*64,
                 run_id='g0e-train-000-nominal_duplicate-00')
    cmd=simulator_command(entry,'/tmp/out',timing_telemetry=True)
    assert '--fixed-map-path' in cli_file_paths(cmd)
    cmd.remove('--fixed-map-path')
    with pytest.raises(ValueError,match='missing/duplicate'):
        cli_file_paths(cmd)
