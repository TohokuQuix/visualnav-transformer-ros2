import hashlib
import json

import pytest

from scripts.eat_nav_motion_14_collection_host import verify_sim_seal


def write_sealed_candidate(root, *, terminal_envs=(), censored_envs=(), trace_lengths=(200, 200, 200, 200)):
    root.mkdir()
    (root/'raw-dem.npz').write_bytes(b'raw dem')
    trace_hashes = []
    for index, length in enumerate(trace_lengths):
        rows = []
        for step in range(length):
            terminal = None
            if step == length - 1 and index in terminal_envs:
                terminal = {'termination_reason': 'tilt_over',
                            'pre_reset_state': {'root_state': [0.0] * 13}}
            if step == length - 1 and index in censored_envs:
                terminal = {'termination_reason': 'parallel_env_censored_by_other_env'}
            rows.append({'control_step': step, 'terminal': terminal})
        data = ''.join(json.dumps(row) + '\n' for row in rows)
        path = root/f'trace-env-{index}.jsonl'
        path.write_text(data)
        trace_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'long-horizon-event-labels.json').write_text(json.dumps({
        'trials': [{'window_count': length} for length in trace_lengths]}))
    (root/'complete.json').write_text(json.dumps({
        'status': 'complete', 'env_count': 4, 'mode': 'candidate',
        'control_steps': max(trace_lengths), 'requested_control_steps': 200,
        'replan_every_steps': 40, 'trace_lengths': list(trace_lengths),
        'terminated_envs': list(terminal_envs), 'censored_envs': list(censored_envs),
        'raw_dem_sha256': hashlib.sha256((root/'raw-dem.npz').read_bytes()).hexdigest(),
        'trace_sha256': trace_hashes,
    }))
    expected = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob('*') if path.is_file()}
    ledger = root/'sha-ledger.json'
    ledger.write_text(json.dumps(expected))
    (root/'host-seal.json').write_text(json.dumps({
        'ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'file_count': len(expected),
    }))


def test_host_verifies_long_horizon_candidate_and_event_labels(tmp_path):
    output = tmp_path/'candidate'
    write_sealed_candidate(output, terminal_envs=(0,), censored_envs=(1,),
                           trace_lengths=(84, 84, 84, 84))

    result = verify_sim_seal(output, expected_control_steps=200,
                             expected_replan_every_steps=40)

    assert result['trace_count'] == 4
    assert result['terminated_envs'] == [0]
    assert result['censored_envs'] == [1]


def test_host_rejects_unmatched_event_label_windows(tmp_path):
    output = tmp_path/'candidate'
    write_sealed_candidate(output)
    labels = json.loads((output/'long-horizon-event-labels.json').read_text())
    labels['trials'][0]['window_count'] -= 1
    (output/'long-horizon-event-labels.json').write_text(json.dumps(labels))
    expected = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob('*')
                if path.is_file() and path.name not in ('sha-ledger.json', 'host-seal.json')}
    ledger = output/'sha-ledger.json'
    ledger.write_text(json.dumps(expected))
    (output/'host-seal.json').write_text(json.dumps({
        'ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'file_count': len(expected),
    }))

    with pytest.raises(ValueError, match='event label window/trace length mismatch'):
        verify_sim_seal(output)