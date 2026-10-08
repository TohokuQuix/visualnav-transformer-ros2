"""RED→GREEN tests for the terminal-aware P1 GPU lifecycle gate."""
from __future__ import annotations

import copy

import pytest

from eat_nav_p1_terminal_lifecycle import TerminalLifecycle, signal_if_exact, completion_envelope


OWNER = {'host_pid': 900, 'start_ticks': 77}
TOKEN = 'a' * 64
SOURCE = 'b' * 64
CONFIG = 'c' * 64
OUTPUT = '/tmp/fresh/run'


def bracket(before=True, after=True, *, failures=(), before_tick=77,
            after_tick=77, start=200, end=300, tables=True, gpu=()):
    a = {900: {'host_pid': 900, 'start_ticks': before_tick}} if before else {}
    b = {900: {'host_pid': 900, 'start_ticks': after_tick}} if after else {}
    return {'process_table_before': a if tables else None,
            'process_table_after': b if tables else None,
            'process_table_before_wall_ns': start,
            'process_table_after_wall_ns': end,
            'failures': list(failures), 'pass': not failures,
            'owned_gpu_pids': list(gpu), 'raw_nvml_rows': [],
            'wall_ns': start - 10}


def marker(**change):
    base = {'schema': 'eatnav_p1_terminal_marker_v1', 'run_token': TOKEN,
            'host_pid': 900, 'start_ticks': 77, 'source_sha256': SOURCE,
            'config_sha256': CONFIG, 'output_root': OUTPUT,
            'completion_state': 'complete', 'physics_steps': 200,
            'control_steps': 20, 'written_wall_ns': 150,
            'marker_before_close': True}
    return {**base, **change}


def live():
    gate = TerminalLifecycle(OWNER, TOKEN, SOURCE, CONFIG, OUTPUT,
                             expected_physics_steps=200, expected_control_steps=20,
                             authorization_wall_ns=100)
    assert gate.observe(bracket(start=110, end=120)) == 'PREAUTH'
    gate.authorize()
    assert gate.observe(bracket(start=130, end=140, gpu=[900])) == 'RUNNING'
    return gate


def test_01_owner_disappears_running_without_marker_fails():
    with pytest.raises(ValueError, match='unexpected_owned_process_loss'):
        live().observe(bracket(False, False, failures=('owned_start_tick_mismatch',)))


def test_02_marker_before_close_owner_present_then_absent_passes():
    gate = live()
    result = gate.observe(bracket(True, False, failures=('simulator_disappeared_in_bracket',), start=200, end=300), marker(written_wall_ns=250))
    assert result == 'FINALIZING'
    assert gate.events[-1]['effective_failures'] == []


def test_03_marker_predates_absent_both_bracket():
    gate = live()
    assert gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch', 'simulator_disappeared_in_bracket',), start=200), marker()) == 'FINALIZING'


def test_04_owner_absent_before_marker_visible_fails():
    with pytest.raises(ValueError, match='unexpected_owned_process_loss'):
        live().observe(bracket(False, False, start=200), None)


def test_05_marker_after_disappearance_fails():
    with pytest.raises(ValueError, match='marker_after_owner_disappearance'):
        live().observe(bracket(False, False, start=200), marker(written_wall_ns=250))


@pytest.mark.parametrize('change', [{'run_token': 'z' * 64}, {'host_pid': 901}, {'start_ticks': 78}])
def test_06_wrong_token_pid_or_tick_fails(change):
    with pytest.raises(ValueError, match='completion_identity_mismatch'):
        live().observe(bracket(False, False), marker(**change))


def test_07_stale_marker_fails():
    with pytest.raises(ValueError, match='stale_completion_marker'):
        live().observe(bracket(False, False), marker(written_wall_ns=90))


def test_08_pid_reuse_after_completion_fails():
    with pytest.raises(ValueError, match='owned_pid_reused'):
        live().observe(bracket(True, True, after_tick=78), marker())


def test_09_native_exit_nonzero_despite_marker_fails():
    gate = live()
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',)), marker())
    with pytest.raises(ValueError, match='native_exit_nonzero'):
        gate.complete(native_exit=1, output_sha_valid=True, token_absent=True, descendants_absent=True)


def test_10_normal_exit_marker_cleanup_passes():
    gate = live()
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',)), marker())
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',), start=310, end=320))
    assert gate.complete(native_exit=0, output_sha_valid=True, token_absent=True, descendants_absent=True) == 'COMPLETE'


def test_11_nonreturning_close_keeps_native_success_unknown():
    gate = live()
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',)), marker())
    with pytest.raises(ValueError, match='native_exit_unknown'):
        gate.complete(native_exit=None, output_sha_valid=True, token_absent=True, descendants_absent=True)
    assert gate.phase == 'FINALIZING'


def test_12_exact_cleanup_signal_skips_reused_pid():
    sent = []
    assert signal_if_exact(OWNER, lambda pid: {'start_ticks': 78}, lambda pid: sent.append(pid)) is False
    assert sent == []
    assert signal_if_exact(OWNER, lambda pid: {'start_ticks': 77}, lambda pid: sent.append(pid)) is True
    assert sent == [900]


def test_13_final_bracket_unrelated_compute_fails():
    with pytest.raises(ValueError, match='unexpected_compute_process'):
        live().observe(bracket(False, False, failures=('owned_start_tick_mismatch', 'unexpected_compute_process')), marker())


def test_14_final_bracket_known_graphics_passes():
    gate = live()
    assert gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',)), marker()) == 'FINALIZING'


def test_15_wrapper_exit_zero_without_marker_fails():
    gate = live()
    with pytest.raises(ValueError, match='completion_marker_missing'):
        gate.complete(native_exit=0, output_sha_valid=True, token_absent=True, descendants_absent=True)


def test_16_marker_short_of_expected_steps_fails():
    with pytest.raises(ValueError, match='completion_step_mismatch'):
        live().observe(bracket(False, False), marker(physics_steps=199))


def test_17_missing_complete_tables_fails():
    with pytest.raises(ValueError, match='missing_process_table'):
        live().observe(bracket(False, False, tables=False), marker())


def test_18_terminal_absence_does_not_emit_start_tick_mismatch():
    gate = live()
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch', 'simulator_disappeared_in_bracket')), marker())
    assert 'owned_start_tick_mismatch' not in gate.events[-1]['effective_failures']


def test_complete_requires_separate_post_exit_terminal_bracket():
    gate = live()
    gate.observe(bracket(False, False, failures=('owned_start_tick_mismatch',)), marker())
    with pytest.raises(ValueError, match='post_exit_terminal_bracket_missing'):
        gate.complete(native_exit=0, output_sha_valid=True, token_absent=True, descendants_absent=True)


def test_finalizing_rejects_new_container_process_claiming_run_token():
    gate = live()
    sample = bracket(False, False, failures=('owned_start_tick_mismatch',))
    sample['process_table_before'][901] = {'host_pid': 901, 'start_ticks': 88,
        'cgroup': 'docker-current', 'command_line': f'python --run-token {TOKEN}'}
    with pytest.raises(ValueError, match='new_run_token_claimant'):
        gate.observe(sample, marker())


def test_original_frozen_completion_gets_identity_bound_envelope():
    raw = {'physics_steps': 200, 'pose_count': 4, 'physical_termination_pose_ids': [],
           'sim_source_sha256': SOURCE}
    result = completion_envelope(raw, raw_sha='d' * 64, raw_mtime_ns=150,
                                  owner=OWNER, token=TOKEN, source_sha=SOURCE,
                                  config_sha=CONFIG, output_root=OUTPUT,
                                  expected_physics_steps=200, expected_control_steps=20,
                                  before_close_source_verified=True)
    assert result == marker(original_marker_sha256='d' * 64,
                            marker_origin='frozen_sim_preclose_attestation')


def test_prior_H5_terminal_bracket_reclassifies_only_expected_exit_flags():
    import json
    from pathlib import Path
    prior = Path('/tmp/eatnav-p1-handshake-20261008-r2/smoke')
    if not (prior / 'gpu-brackets.jsonl').exists():
        pytest.skip('prior diagnostic root unavailable')
    samples = [json.loads(line) for line in (prior / 'gpu-brackets.jsonl').read_text().splitlines()]
    failed = samples[32]
    owner = json.loads((prior / 'result.json').read_text())['owner']
    intent = json.loads((prior / 'coord/launch-intent.json').read_text())
    gate = TerminalLifecycle(owner, intent['run_token'], intent['source_sha256'],
        intent['config_sha256'], intent['output_root'], 200, 20,
        authorization_wall_ns=samples[1]['wall_ns'])
    gate.observe(samples[1])
    gate.authorize()
    gate.observe(samples[31])
    m = completion_envelope({'physics_steps': 200, 'pose_count': 4,
        'physical_termination_pose_ids': [], 'sim_source_sha256': intent['source_sha256']},
        raw_sha='fb98d44b03435632841e19fa0e45f25625afea2e310858fabd86f33f0990fc09',
        raw_mtime_ns=1791468671461965396, owner=owner, token=intent['run_token'],
        source_sha=intent['source_sha256'], config_sha=intent['config_sha256'],
        output_root=intent['output_root'], expected_physics_steps=200,
        expected_control_steps=20, before_close_source_verified=True)
    failed['process_table_before_wall_ns'] = failed['wall_ns']
    failed['process_table_after_wall_ns'] = failed['wall_ns'] + 100_000_000
    assert gate.observe(failed, m) == 'FINALIZING'
    assert gate.events[-1]['effective_failures'] == []
