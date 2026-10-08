import json
from pathlib import Path

from visualnav_transformer.eat_nav.minimal_p1_r4 import evaluate_baseline, assess_nvidia_smi_health, health_query_command


CONTRACT = {
    'schema': 'p1_local_primitive_resource_contract_v2',
    'required_gpu_uuid': 'GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26',
    'required_gpu_name': 'NVIDIA GeForce RTX 5070 Ti',
    'audited_total_memory_mib': 16303,
    'required_total_memory_mib_min': 16000,
    'baseline_device_total_hard_cap_mib': 1024,
    'gpu_device_total_hard_cap_mib': 7680,
    'unresolved_gpu_bytes_hard_cap': 0,
    'absolute_attribution_gap_hard_cap_mib': 256,
}


def sample():
    return {'uuid': CONTRACT['required_gpu_uuid'], 'name': CONTRACT['required_gpu_name'],
            'mig_mode': '[N/A]', 'total_mib': 16303, 'device_used_mib': 358.125,
            'other_processes_used_mib': 311.859375, 'unresolved_pid_bytes': 0,
            'bracket_gap_mib': 46.265625, 'temperature_c': 39,
            'driver_version': '580.173.02', 'nvml_version': '13.580.173.02'}


def graphics():
    return [{'host_pid': 2950, 'used_gpu_memory_bytes': 145645568,
             'kinds': ['graphics'], 'proc': {'start_ticks': 16047,
             'command_line': '/usr/lib/xorg/Xorg'}}]


def test_known_graphics_above_256_mib_is_allowed():
    result = evaluate_baseline(sample(), graphics(), CONTRACT)
    assert result['pass']
    assert result['known_other_process_mib'] == 311.859375
    assert result['failures'] == []


def test_r3_raw_sample_reclassified_without_altering_raw_data():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / 'docs/eat-nav/local-primitive-p1-r3-20261008/gpu-device-baseline.json').read_text())
    summary = json.loads((root / 'docs/eat-nav/local-primitive-p1-r3-20261008/gpu-baseline-gate.json').read_text())['sample']
    assert evaluate_baseline(summary, raw['deduplicated_processes'], CONTRACT)['pass']


def test_unexpected_compute_process_stops_baseline():
    rows = graphics() + [{'host_pid': 3000, 'used_gpu_memory_bytes': 1024,
                          'kinds': ['compute'], 'proc': {'start_ticks': 999,
                          'command_line': 'python train.py'}}]
    assert 'unexpected_compute_workload' in evaluate_baseline(sample(), rows, CONTRACT)['failures']


def test_missing_start_tick_stops_baseline():
    rows = graphics(); rows[0]['proc'].pop('start_ticks')
    assert 'unknown_other_process_identity' in evaluate_baseline(sample(), rows, CONTRACT)['failures']


def test_baseline_device_total_limit_is_distinct_from_rollout_cap():
    measured = sample(); measured['device_used_mib'] = 1100
    failures = evaluate_baseline(measured, graphics(), CONTRACT)['failures']
    assert 'baseline_device_over_cap' in failures
    assert 'gpu_device_over_cap' not in failures


def test_unresolved_and_bracket_gap_remain_strict():
    measured = sample(); measured['unresolved_pid_bytes'] = 1; measured['bracket_gap_mib'] = 256.1
    failures = evaluate_baseline(measured, graphics(), CONTRACT)['failures']
    assert 'unresolved_gpu_pid_memory' in failures
    assert 'gpu_attribution_gap' in failures


def test_health_query_uses_supported_full_report_and_rejects_recovery_action():
    assert health_query_command() == ['nvidia-smi', '-q']
    assert assess_nvidia_smi_health('GPU Recovery Action : None\n', 0) is False
    assert assess_nvidia_smi_health('GPU Recovery Action : Reset\n', 0) is True
    assert assess_nvidia_smi_health('Failed to parse --display/-d flags\n', 2) is True
